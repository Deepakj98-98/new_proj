from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify,send_from_directory
import os
from OCR.pdf_ocr import Pdf_ocr
from OCR.image_text import Image_text_ocr
from OCR.docx_ocr_text import ocr_docx
from Transcript_Generation.trancripts_generation import TranscriptProcessor
from Paraphraser.finetune_plus_keywordsBART import BARTFinetune_keywords
from Paraphraser.flant5_paraphrase import T5_flan_praraphrase
from Paraphraser.t5_paraphrase import T5_small
from Indexing2 import QdrantChunking
from Retrieval_db import DissertationQueryProcessor
import speech_recognition as sr

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
                print("inside pdf")
                print(file)
                text = self.pdf_ocr.pdf_ocr_text(file)
                print("done")
            elif extension in [".png", ".jpg", ".jpeg"]:
                text = self.image_ocr.image_text(file)
            else:
                continue
            
            self._save_text_file(file, text)
    
    def role_based_file(self,filepaths, role, model, filter):
        for file in filepaths:
            base_name = os.path.basename(file)
            filename_without_ext = os.path.splitext(base_name)[0]
            text_file = f"{filename_without_ext}.txt"
            text_path = os.path.join(self.upload_folder, text_file)
            file1=text_path
            if model.lower()=="flant5":
                text=self.flant5.transcripts_generate(file1, role, filter)
            elif model.lower()=="finetuned_bart":
                text=self.bart_finetuned.transcripts_generate(file1,role,filter)
            elif model.lower()=="t5_small":
                text=self.t5_small.transcripts_generate(file1,role,filter)
            else:
                continue
            self._save_pp_file(file,role,text)


    def _save_text_file(self, original_file, text):
        base_name = os.path.basename(original_file)
        filename_without_ext = os.path.splitext(base_name)[0]
        text_file = f"{filename_without_ext}.txt"
        text_path = os.path.join(self.upload_folder, text_file)
        with open(text_path, "w",encoding="utf-8") as f:
            f.write(text)
    
    def _save_pp_file(self, original_file,role, text):
        base_name = os.path.basename(original_file)
        filename_without_ext = os.path.splitext(base_name)[0]+"pp"
        text_file = f"{filename_without_ext+role}.txt"
        if role=="dev":
             os.makedirs(role, exist_ok=True)
             text_path = os.path.join(role, text_file)
             with open(text_path, "w",encoding="utf-8") as f:
                f.write(text)
        if role=="ba":
            os.makedirs(role, exist_ok=True)
            text_path = os.path.join(role, text_file)
            with open(text_path, "w",encoding="utf-8") as f:
                f.write(text)
        if role=="management":
            os.makedirs(role, exist_ok=True)
            text_path = os.path.join(role, text_file)
            with open(text_path, "w",encoding="utf-8") as f:
                f.write(text)


# Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = ["uploads"]
ROLE_FOLDER=["ba","dev","management"]
file_processor = FileProcessor(UPLOAD_FOLDER[0])
qdrant_chunking = QdrantChunking()
query_processor=DissertationQueryProcessor()
r = sr.Recognizer()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        files = request.files.getlist("files")
        #role=request.form.get("role")
        role=["dev","ba","management"]
        print(role)
        model=request.form.get("models")
        print("model")
        print(model)
        filter=request.form.get("filter")
        print(filter)
        uploaded_file_paths = file_processor.save_files(files)
        if uploaded_file_paths:
            file_processor.process_files(uploaded_file_paths)
            for i in role:
                file_processor.role_based_file(uploaded_file_paths,i, model, filter)
                flash("Files uploaded and processed successfully!", "success")
                qdrant_chunking.chunk_files(i+"dissertation_collection",i)
        return redirect(url_for("home"))
    #return render_template('index.html')
    # List all transcript files
    folders=[ROLE_FOLDER, UPLOAD_FOLDER]
    transcript_files=[]
    for folder in folders:
        for i in folder:
            transcript_files.extend( [
            f for f in os.listdir(i) if f.endswith(".txt")
            ])
    return render_template('index.html', transcript_files=transcript_files)

@app.route("/download/<filename>")
def download_file(filename):
    for folder in UPLOAD_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    
    for folder in ROLE_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    
    flash("File not found!", "error")
    return redirect(url_for("home"))

@app.route("/view/<filename>")
def view_file(filename):
    # Search for the file in UPLOAD_FOLDERS
    for folder in UPLOAD_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_from_directory(folder, filename)
    
    # Search for the file in ROLE_FOLDERS
    for folder in ROLE_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_from_directory(folder, filename)

    flash("File not found!", "error")
    return redirect(url_for("home"))


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    try:
        # Get user input from the frontend
        role=request.json.get("role")
        user_input = request.json.get("user_input")
        print(f"role is {role}")
        if not user_input:
            return jsonify({"error": "Invalid input"}), 400

        # Process user query through DissertationQueryProcessor
        response = query_processor.query_and_store(user_input,role)

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
  # Chatbot page

def clear_files():
    directories=["ba","uploads","dev","management"]
    set=False
    for directory in directories:
        try:
            set=True
            if not os.path.exists(directory):
                print(f"Directory {directory} path is incorrect or does not exist")
                set=False
            if not os.path.isdir(directory):
                print(f"Directory {directory} is not a directory")
                set=False
            for file in os.listdir(directory):
                print(file)
                file_path=os.path.join(directory,file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipping {file_path} for deletion")
            set=True
        except Exception as e:
            print(f"An error occurred: {e}")
            set= False
    if set==True:
        return True
    else:
        return False

@app.route("/clear")
def delete_all_files():
    success=clear_files()
    if success:
        return redirect(url_for("home"))
    else:
        return "Failed to delete Files",500
    
@app.route('/chatbot/voice', methods=['POST'])
def voice_input():
    try:
        # Use the microphone as input
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("Listening for voice input...")
            audio2 = r.listen(source2)

            # Convert speech to text
            MyText = r.recognize_google(audio2).lower()
            print(f"Recognized: {MyText}")

            if MyText == "exit":
                return jsonify({'transcription': '', 'message': 'Exit command received'})

            return jsonify({'transcription': MyText})

    except sr.RequestError:
        return jsonify({'error': 'Speech Recognition API unavailable'}), 500
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400


if __name__ == "__main__":
    app.run(debug=True)
