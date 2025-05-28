import pdfplumber

PDF_PATHS = [
    'pdfs/lemh101.pdf',
    'pdfs/lemh102.pdf',
]

def extract_pdf_text(pdf_path):
    print(f"\n===== Extracting from: {pdf_path} =====\n")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            print(f"--- Page {i} ---")
            if text:
                print(text)
            else:
                print("[No text extracted]")
            print("\n----------------------\n")

if __name__ == "__main__":
    for path in PDF_PATHS:
        extract_pdf_text(path) 