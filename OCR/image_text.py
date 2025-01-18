import cv2
import pytesseract
import re

class Image_text_ocr: 
    # Funciton to perform OCR for images
    def image_text(self, filepath):
        pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        # Increase contrast using histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # Code to reduce background noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR
        text = pytesseract.image_to_string(binary)
        #Removing special characters
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.]', '', text)
        return cleaned_text

''' 
ocr = Image_text_ocr()
text_output = ocr.image_text("C:\\Users\\Deepak J Bhat\\new_proj\\uploads\\Image_SSS.png")
print("Extracted Text:\n", text_output)
'''

        
'''
    from PIL import Image 
    from pytesseract import pytesseract 
    def image_text(filepath): 
        # Defining paths to tesseract.exe  
        # and the image we would be using 
        path_to_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        image_path = filepath
        #tessdata_config=r'--tessdata-dir "C:/Users/Deepak J Bhat/miniconda3/envs/testenv/share"'
        
        # Opening the image & storing it in an image object 
        img = Image.open(image_path) 
        
        # Providing the tesseract  
        # executable location to pytesseract library 
        pytesseract.tesseract_cmd = path_to_tesseract 
        
        # Passing the image object to  
        # image_to_string() function 
        # This function will 
        # extract the text from the image 
        text = pytesseract.image_to_string(img) 

        # Displaying the extracted text 
        with open("image_text","w") as file:
            file.write(text)

        return text
    '''