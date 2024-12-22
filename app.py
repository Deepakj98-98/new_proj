from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os
import pandas as pd
from OCR.pdf_ocr import pdf_ocr_text
from OCR.image_text import image_text
from OCR.docx_ocr_text import ocr_docx
from Transcript_Generation.trancripts_generation import transcripts_generate
#from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
total_sessions = []

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        files = request.files.getlist("files")
        uploaded_file_paths = []
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded files to the server
        for file in files:
            filename = file.filename
            print(filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(filepath)
            if not os.path.exists(filepath):
                file.save(filepath)
                uploaded_file_paths.append(filepath)
        
        print(uploaded_file_paths)
        
        # If files were uploaded, process them
        if uploaded_file_paths:
            process_files(uploaded_file_paths)
            flash("Files uploaded and processed successfully!", "success")  # Flash success message
            
        return redirect(url_for("home"))  # Redirect after POST request to avoid resubmission
    
    return render_template('index.html')  # Assuming you have a template for rendering

def process_files(filepaths):
    if len(filepaths) > 0:
        for file in filepaths:
            if file.endswith(".mp4") or file.endswith(".mp3") or file.endswith(".wav"):
                # Video processing method
                text = transcripts_generate(file)
                text_file = os.path.basename(file)
                text_file1 = list(os.path.splitext(text_file))
                filename = str(text_file1[0]) + ".txt"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                print(filepath)
                with open(filepath, "w") as f:
                    f.write(text)
            elif file.endswith(".docx"):
                # Docx processing method
                text = ocr_docx(file)
                text_file = os.path.basename(file)
                text_file1 = list(os.path.splitext(text_file))
                filename = str(text_file1[0]) + ".txt"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                print(filepath)
                with open(filepath, "w") as f:
                    f.write(text)
            elif file.endswith(".pdf"):
                # PDF processing method
                text = pdf_ocr_text(file)
                text_file = os.path.basename(file)
                text_file1 = list(os.path.splitext(text_file))
                filename = str(text_file1[0]) + ".txt"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                print(filepath)
                with open(filepath, "w") as f:
                    f.write(text)
            elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                # Image processing method
                text = image_text(file)
                text_file = os.path.basename(file)
                text_file1 = list(os.path.splitext(text_file))
                filename = str(text_file1[0]) + ".txt"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                print(filepath)
                with open(filepath, "w") as f:
                    f.write(text)

if __name__ == "__main__":
    app.run(debug=True)
