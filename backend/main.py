import base64
import io
import re
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_warp_document(image: np.ndarray) -> np.ndarray:
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        return four_point_transform(orig, screenCnt.reshape(4, 2))
    else:
        return orig

def validate_jumin(jumin_str: str) -> bool:
    digits = re.sub(r'[^0-9]', '', jumin_str)
    if len(digits) != 13:
        return False

    month = int(digits[2:4])
    day = int(digits[4:6])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False

    multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(digits[i]) * multipliers[i] for i in range(12))
    remainder = (11 - (total % 11)) % 10

    return remainder == int(digits[12])

def process_image_for_masking(img: np.ndarray, mask_mode: str):
    flattened_img = auto_warp_document(img)
    height, width, _ = flattened_img.shape

    gray = cv2.cvtColor(flattened_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    custom_config = r'--oem 3 --psm 3'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

    jumin_pattern = re.compile(r'\b\d{2}[0-1]\d[0-3]\d[-.\s]?\d[1-8]\d{6}\b|\b\d{13}\b')
    entire_ocr_text = pytesseract.image_to_string(thresh, config=custom_config)
    lines = entire_ocr_text.split('\n')

    valid_jumins = set()
    for line in lines:
        matches = jumin_pattern.findall(line)
        for match in matches:
            if validate_jumin(match):
                valid_jumins.add(match)

    if not valid_jumins:
        all_matches = jumin_pattern.findall(entire_ocr_text)
        for match in all_matches:
            if validate_jumin(match):
                valid_jumins.add(match)

    detected_count = len(valid_jumins)
    if detected_count == 0:
        return flattened_img, 0

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        word = data['text'][i].strip()
        if not word:
            continue

        cleaned_word = re.sub(r'[^0-9]', '', word)
        if not cleaned_word:
            continue

        matched_jumin = None
        for jumin in valid_jumins:
            clean_jumin = re.sub(r'[^0-9]', '', jumin)
            if cleaned_word in clean_jumin:
                matched_jumin = jumin
                break

        if matched_jumin:
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            if mask_mode == 'back':
                if len(cleaned_word) >= 6 and ('-' in word or '.' in word or len(cleaned_word) == 7):
                    half_w = w // 2
                    cv2.rectangle(flattened_img, (x + half_w, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), -1)
            else:
                cv2.rectangle(flattened_img, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 0), -1)

    return flattened_img, detected_count

@app.post("/api/mask")
async def mask_privacy_document(
    file: UploadFile = File(...),
    mask_mode: str = Form("full")
):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            images = convert_from_bytes(contents)
            if not images:
                return {"status": "error", "message": "PDF 파일을 읽을 수 없습니다."}
            pil_img = images[0]
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "파일을 처리할 수 없습니다."}

        height, width, _ = img.shape
        max_dim = 1600
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        processed_img, total_detected = process_image_for_masking(img, mask_mode)

        _, encoded_img = cv2.imencode('.png', processed_img)
        base64_image = "data:image/png;base64," + base64.b64encode(encoded_img).decode('utf-8')

        return {
            "status": "success",
            "detected_count": total_detected,
            "masked_image_base64": base64_image,
            "message": f"문서 보정 및 주민등록번호 {total_detected}건 마스킹 완료"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
