// Render 백엔드 API 주소 고정
const BACKEND_URL = "https://privacy-scanner-backend.onrender.com";

let selectedFile = null;
const fileInput = document.getElementById('file-input');
const statusBox = document.getElementById('status-box');
const btnProcess = document.getElementById('btn-process');
const previewImg = document.getElementById('preview-img');
const placeholderText = document.getElementById('placeholder-text');
const downloadBtn = document.getElementById('download-btn');

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        statusBox.textContent = `✅ 파일 선택됨: ${selectedFile.name}`;
        btnProcess.removeAttribute('disabled');
        
        // Show raw preview temporarily
        const reader = new FileReader();
        reader.onload = (ev) => {
            previewImg.src = ev.target.result;
            previewImg.classList.remove('hidden');
            placeholderText.style.display = 'none';
        };
        reader.readAsDataURL(selectedFile);
    }
});

btnProcess.addEventListener('click', async () => {
    if (!selectedFile) return;

    const maskMode = document.querySelector('input[name="mask-mode"]:checked').value;

    statusBox.textContent = "⏳ 서버에서 AI OCR 분석 및 마스킹 처리 중입니다... (약 5~10초 소요)";
    btnProcess.setAttribute('disabled', 'true');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mask_mode', maskMode);

    try {
        const response = await fetch(`${BACKEND_URL}/api/mask`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            statusBox.textContent = `✅ 마스킹 완료! (총 민감정보 ${data.detected_count}건 감지 및 처리)`;
            previewImg.src = data.masked_image_base64;
            downloadBtn.href = data.masked_image_base64;
            downloadBtn.download = `masked_${selectedFile.name}`;
            downloadBtn.classList.remove('hidden');
        } else {
            statusBox.textContent = `⚠️ 오류 발생: ${data.message}`;
        }
    } catch (err) {
        console.error(err);
        statusBox.textContent = "⚠️ 서버 통신 실패. 백엔드 주소나 서버 상태를 확인하세요.";
    } finally {
        btnProcess.removeAttribute('disabled');
    }
});
