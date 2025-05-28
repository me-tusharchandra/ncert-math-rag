import os
import cv2
import requests
import pytesseract
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# Directory to store downloaded PDFs and images
download_dir = 'pdfs'
image_dir = 'images'
os.makedirs(download_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

def download_pdf(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url}...")
        r = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"Already downloaded: {save_path}")

def pdf_to_images(pdf_path, output_folder):
    print(f"Converting {pdf_path} to images...")
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Get number of pages in PDF
    images = convert_from_path(pdf_path)
    num_pages = len(images)
    
    # Check if images already exist
    existing_images = []
    for i in range(1, num_pages + 1):
        img_path = os.path.join(output_folder, f"{base_filename}_page_{i}.png")
        if os.path.exists(img_path):
            existing_images.append(img_path)
        else:
            break
            
    if existing_images:
        print(f"Found {len(existing_images)} existing images for {pdf_path}")
        return existing_images
        
    # Convert PDF to images if they don't exist
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"{base_filename}_page_{i+1}.png")
        img.save(img_path, 'PNG')
        image_paths.append(img_path)
    return image_paths

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def ocr_tesseract(image_path):
    image = cv2.imread(image_path)
    image = deskew(image)
    text = pytesseract.image_to_string(image)
    return text

def ocr_paddle(image_path, ocr_engine):
    result = ocr_engine.predict(image_path)
    print(f"DEBUG PaddleOCR result for {image_path}:", result)  # Debug print
    if not isinstance(result, list) or not result:
        raise ValueError(f"PaddleOCR failed with result: {result}")
    rec_texts = result[0].get('rec_texts', [])
    rec_boxes = result[0].get('rec_boxes', [])
    text = '\n'.join(rec_texts)
    return text, rec_boxes

def draw_boxes_on_image(image_path, boxes, output_path):
    image = cv2.imread(image_path)
    for box in boxes:
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imwrite(output_path, image)

def main():
    # Read links
    with open('links.txt') as f:
        links = [line.strip() for line in f if line.strip()]

    # Download PDFs
    pdf_paths = []
    for url in links:
        filename = os.path.join(download_dir, os.path.basename(url))
        download_pdf(url, filename)
        pdf_paths.append(filename)

    # Convert PDFs to images
    all_image_paths = []
    for pdf_path in pdf_paths:
        image_paths = pdf_to_images(pdf_path, image_dir)
        all_image_paths.extend(image_paths)

    # Only process the first 2 images
    sample_image_paths = all_image_paths[:2]

    # Initialize PaddleOCR
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

    paddle_results = []
    tesseract_results = []
    annotated_dir = 'annotated'
    os.makedirs(annotated_dir, exist_ok=True)

    # OCR analysis
    for img_path in sample_image_paths:
        print(f"\n--- OCR Analysis for {img_path} ---")
        print("Tesseract OCR Result:")
        try:
            tesseract_text = ocr_tesseract(img_path)
            print(tesseract_text)
            tesseract_results.append(f"--- {img_path} ---\n{tesseract_text}\n")
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            tesseract_results.append(f"--- {img_path} ---\nTesseract OCR failed: {e}\n")
        print("\nPaddleOCR Result:")
        try:
            paddle_text, paddle_boxes = ocr_paddle(img_path, ocr_engine)
            print(paddle_text)
            paddle_results.append(f"--- {img_path} ---\n{paddle_text}\n")
            # Draw and save annotated image
            annotated_path = os.path.join(annotated_dir, os.path.basename(img_path))
            draw_boxes_on_image(img_path, paddle_boxes, annotated_path)
            print(f"Annotated image saved to {annotated_path}")
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
            paddle_results.append(f"--- {img_path} ---\nPaddleOCR failed: {e}\n")

    # Save results to files
    with open('tesseract_results.txt', 'w') as f:
        f.writelines(tesseract_results)
    with open('paddle_results.txt', 'w') as f:
        f.writelines(paddle_results)

if __name__ == "__main__":
    main()
