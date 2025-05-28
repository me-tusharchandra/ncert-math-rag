import os
import requests
from tqdm import tqdm
from pix2text import Pix2Text
from pdf2image import convert_from_path

DOWNLOAD_DIR = 'pdfs'
IMAGE_DIR = 'images'
LINKS_FILE = 'links.txt'
OUTPUT_MD_DIR = 'output-pdf-md'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MD_DIR, exist_ok=True)

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
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"{base_filename}_page_{i+1}.png")
        if not os.path.exists(img_path):
            img.save(img_path, 'PNG')
        image_paths.append(img_path)
    print(f"Saved {len(image_paths)} images for {pdf_path}")
    return image_paths

# Step 1: Download PDFs
with open(LINKS_FILE) as f:
    links = [line.strip() for line in f if line.strip()]

pdf_paths = []
for url in links:
    filename = os.path.join(DOWNLOAD_DIR, os.path.basename(url))
    download_pdf(url, filename)
    pdf_paths.append(filename)

# Step 2: Convert PDFs to images (optional, for debugging or future use)
for pdf_path in pdf_paths:
    pdf_to_images(pdf_path, IMAGE_DIR)

# Step 3: Save full-page markdown and figures for each PDF
p2t = Pix2Text.from_config()
print(f"\nSaving full-page markdown outputs to {OUTPUT_MD_DIR} ...")
for pdf_path in pdf_paths:
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(OUTPUT_MD_DIR, pdf_base)
    os.makedirs(output_dir, exist_ok=True)
    output_md_path = os.path.join(output_dir, "output.md")
    try:
        doc = p2t.recognize_pdf(pdf_path)
        doc.to_markdown(output_md_path)
        print(f"Saved markdown: {output_md_path}")
    except Exception as e:
        print(f"Error saving markdown for {pdf_path}: {e}")

print(f"\nExtraction complete. Markdown and figures saved in {OUTPUT_MD_DIR}") 