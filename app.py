from flask import Flask, render_template, request, redirect, url_for, flash
import os
from OCR.pdf_ocr import Pdf_ocr
from OCR.image_text import Image_text_ocr
from OCR.docx_ocr_text import ocr_docx
from Transcript_Generation.trancripts_generation import TranscriptProcessor
from Paraphraser.finetune_plus_keywordsBART import BARTFinetune_keywords
from Paraphraser.flant5_paraphrase import T5_flan_praraphrase
from Paraphraser.t5_paraphrase import T5_small

class FileProcessor:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        print("Now OCR")
        self.pdf_ocr=Pdf_ocr()
        self.image_ocr=Image_text_ocr()
        print("Now transcripts")
        self.transcript_processor=TranscriptProcessor()
        print("now transformers")
        self.bart_finetuned=BARTFinetune_keywords()
        print("Now t5 flan")
        self.flant5=T5_flan_praraphrase()
        print("now t5 small")
        self.t5_small=T5_small()

    def save_files(self, files):
        uploaded_file_paths = []
        for file in files:
            filename = file.filename
            filepath = os.path.join(self.upload_folder, filename)
            if not os.path.exists(filepath):
                file.save(filepath)
                uploaded_file_paths.append(filepath)
        return uploaded_file_paths

    def process_files(self, filepaths):
        for file in filepaths:
            extension = os.path.splitext(file)[1].lower()
            if extension in [".mp4", ".mp3", ".wav"]:
                text = self.transcript_processor.generate_transcripts(file)
            elif extension == ".docx":
                text = ocr_docx(file)
            elif extension == ".pdf":
                text = self.pdf_ocr.pdf_ocr_text(file)
            elif extension in [".png", ".jpg", ".jpeg"]:
                text = self.image_ocr.image_text(file)
            else:
                continue
            
            self._save_text_file(file, text)

    def _save_text_file(self, original_file, text):
        base_name = os.path.basename(original_file)
        filename_without_ext = os.path.splitext(base_name)[0]
        text_file = f"{filename_without_ext}.txt"
        text_path = os.path.join(self.upload_folder, text_file)
        with open(text_path, "w") as f:
            f.write(text)

# Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = "uploads"
file_processor = FileProcessor(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        files = request.files.getlist("files")
        uploaded_file_paths = file_processor.save_files(files)
        if uploaded_file_paths:
            file_processor.process_files(uploaded_file_paths)
            flash("Files uploaded and processed successfully!", "success")
        return redirect(url_for("home"))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
