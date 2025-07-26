import os
import re
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

# Paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler\Library\bin"
pdf_path = "data/HSC26-Bangla1st-Paper.pdf"

def clean_text(text):
    """Enhanced cleaning for Bengali OCR output"""
    replacements = {
        "ﬂ": "ফ", "ﬁ": "ফি", "“": "\"", "”": "\"", "‘": "'", "’": "'",
        "–": "-", "—": "-", "…": "...", "●": "", "■": "", "|": "", "॥": "।",
        "\x0c": "", "্া": "া", "ো্": "ো", "ৌ্": "ৌ",
        r"\s+": " ",  # Replace multiple spaces with single space
        r"[^\u0980-\u09FF\s\w.,!?]": ""  # Remove non-Bengali characters
    }
    for wrong, correct in replacements.items():
        text = re.sub(wrong, correct, text)
    
    # Fix newlines inside sentences but keep paragraph breaks
    text = re.sub(r'(\S)\n(\S)', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def preprocess_image(image):
    """Advanced preprocessing for Bengali OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Deskew image
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with improved OCR"""
    print("🔎 Starting enhanced OCR with Bengali language support...")
    
    pages = convert_from_path(
        pdf_path,
        dpi=600,  # Increased DPI for better resolution
        poppler_path=poppler_path,
        thread_count=4
    )
    
    full_text = ""
    page_separator = "\n" + "=" * 50 + "\n\n"
    
    for i, page in enumerate(tqdm(pages, desc="📄 Processing pages")):
        image = np.array(page)
        processed_img = preprocess_image(image)
        
        # OCR with Bengali + English, optimized config
        text = pytesseract.image_to_string(
            processed_img,
            lang="ben+eng",
            config="--psm 6 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ঀঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯।!? "
        )
        
        cleaned = clean_text(text)
        full_text += cleaned + page_separator
    
    return full_text

if __name__ == "__main__":
    text = extract_text_from_pdf(pdf_path)
    os.makedirs("data", exist_ok=True)
    with open("data/cleaned_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("✅ Cleaned OCR text saved to: data/cleaned_text.txt")