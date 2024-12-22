from pdf2image import convert_from_path
import pytesseract
import os
import time
import shutil
from PIL import Image

output_dir="frames"
final_text=[]
def pdf_to_text(filepath):
  path_to_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  pytesseract.tesseract_cmd = path_to_tesseract
  pdf_path = filepath
  output_dir="frames"
  pop_path=r'C:/Users/Deepak J Bhat/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin'
  os.makedirs(output_dir, exist_ok=True)
  images = convert_from_path(pdf_path,poppler_path=pop_path, output_folder=output_dir, fmt='jpeg')

  # Perform OCR and delete images after processing
  for image_file in os.listdir(output_dir):
      image_path = os.path.join(output_dir, image_file)
      with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
        final_text.append(text)
        img.close()
        return "".join(final_text)
      #os.remove(image_path)
  
def pdf_ocr_text(filepath):
  global output_dir
  text=pdf_to_text(filepath)
  '''
  with open("pdf_text","w") as file:
     file.write(text)
  shutil.rmtree(output_dir)
  '''
  return text

#pdf_ocr_text(r"C:\\Users\\Deepak J Bhat\\Downloads\\test file.pdf")