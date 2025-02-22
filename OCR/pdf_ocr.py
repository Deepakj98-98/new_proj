from pdf2image import convert_from_path
import pytesseract
import os
import time
import shutil
from PIL import Image
import re

class Pdf_ocr:
  def __init__(self):
    #self.output_dir="frames"
    self.final_text=[]
    
   
  #Function to extract images form the given PDF
  def pdf_to_text(self, filepath):
    self.final_text=[]
    path_to_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    pdf_path = filepath
    output_dir="frames"
    pop_path=r'C:/Users/Deepak J Bhat/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin'
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path,poppler_path=pop_path, output_folder=output_dir, fmt='jpeg')

    # Funciton to perform OCR
    for image_file in os.listdir(output_dir):
        image_path = os.path.join(output_dir, image_file)
        with Image.open(image_path) as img:
          text = pytesseract.image_to_string(img)
          self.final_text.append(text)
          img.close()
    return "".join(self.final_text)
        
  # function to return text and delete image directory after processing  
  def pdf_ocr_text(self,filepath):
    text=self.pdf_to_text(filepath)
    shutil.rmtree("frames")
    #Removing special characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s.,]', '', text)
    cleaned_text1 = re.sub(r'[^\x20-\x7E\n]', '', cleaned_text)
    return cleaned_text1
    
    

  #pdf_ocr_text(r"C:\\Users\\Deepak J Bhat\\Downloads\\test file.pdf")