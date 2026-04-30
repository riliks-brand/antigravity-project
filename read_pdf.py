import os
from pypdf import PdfReader

pdf_path = r"d:\Candlestick_Detection\Doc1.pdf"
out_path = r"d:\Candlestick_Detection\Doc1_text.txt"

try:
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully extracted text.")
except Exception as e:
    print(f"Error: {e}")
