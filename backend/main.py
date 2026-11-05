from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
import base64

app = FastAPI(title="Privacy Document Redaction API", version="1.0")

# Enable CORS for Cloudflare Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_valid_rrn(text: str) -> bool:
    clean = re.sub(r'[^0-9]', '', text)
    if len(clean) != 13:
        return False
    mm = int(clean[2:4])
    dd = int(clean[4:6])
    gender = int(clean[6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 1 <= gender <= 4):
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    s = sum(int(clean[i]) * weights[i] for i in range(12))
    check = (11 - (s % 11)) % 10
    return check == int(clean[12])

def is_valid_phone(text: str) -> bool:
    clean = re.sub(r'[^0-9]', '', text)
    return bool(re.match(r'^01[016789]\d{7,8}$', clean))

@app.post("/api/mask")
async def mask_document(
    file: UploadFile = File(...),
    mask_mode: str = Form("full") # "full" or "back"
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        # Convert to OpenCV format (BGR)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        # Run Tesseract OCR with Korean + English support
        # Config to preserve word boxes
        ocr_data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
        
        n_boxes = len(ocr_data['text'])
        masked_image = img_bgr.copy()
        detected_count = 0

        for i in range(n_boxes):
            word = ocr_data['text'][i].strip()
            if not word:
                continue

            clean_word = re.sub(r'[^0-9]', '', word)
            
            is_rrn = is_valid_rrn(word) or is_valid_rrn(clean_word)
            is_phone = is_valid_phone(word) or is_valid_phone(clean_word)

            if is_rrn or is_phone:
                detected_count += 1
                bx, by, bw, bh = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                
                # Apply padding
                x = max(0, bx - 4)
                y = max(0, by - 4)
                box_w = min(w - x, bw + 8)
                box_h = min(h - y, bh + 8)

                if is_rrn and mask_mode == 'back':
                    # Mask only the back part (approx second half)
                    front_w = box_w * 0.48
                    back_x = int(x + front_w)
                    back_w = int(box_w - front_w)
                    cv2.rectangle(masked_image, (back_x, y), (back_x + back_w, y + box_h), (0, 0, 0), -1)
                else:
                    # Full mask
                    cv2.rectangle(masked_image, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)

        # Convert result back to JPEG bytes
        _, encoded_img = cv2.imencode('.png', masked_image)
        base64_img = base64.b64encode(encoded_img).decode('utf-8')

        return {
            "status": "success",
            "detected_count": detected_count,
            "masked_image_base64": f"data:image/png;base64,{base64_img}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def health_check():
    return {"status": "running", "service": "Privacy Scanner Backend"}
