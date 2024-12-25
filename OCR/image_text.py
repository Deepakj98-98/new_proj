import cv2
import pytesseract

class Image_text_ocr: 
    # Read and preprocess the image
    def image_text(self, filepath):
        pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Save preprocessed image (optional)
        #cv2.imwrite("processed_image.png", binary)

        # Perform OCR
        text = pytesseract.image_to_string(binary)
        #with open("image_text","w") as file:
        #    file.write(text)
        return text


#    image_text(None)
        
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