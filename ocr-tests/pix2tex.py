# from PIL import Image
# from pix2tex.cli import LatexOCR

# IMAGE_PATHS = [
#     "images/lemh102_page_6.png",
#     "images/lemh103_page_5.png",
#     "images/lemh104_page_22.png"
# ]

# def extract_latex_from_image(image_path, model):
#     print(f"\n===== Extracting LaTeX from: {image_path} =====\n")
#     img = Image.open(image_path)
#     latex = model(img)
#     print(latex)
#     print("\n----------------------\n")

# if __name__ == "__main__":
#     model = LatexOCR()
#     for path in IMAGE_PATHS:
#         try:
#             extract_latex_from_image(path, model)
#         except Exception as e:
#             print(f"Failed to process {path}: {e}")