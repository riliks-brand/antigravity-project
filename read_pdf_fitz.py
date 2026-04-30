import fitz  # PyMuPDF

pdf_path = r"d:\Candlestick_Detection\Executive Summary.pdf"
out_path = r"d:\Candlestick_Detection\Executive_Summary_text.txt"

try:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully extracted text with PyMuPDF.")
except Exception as e:
    print(f"Error: {e}")
