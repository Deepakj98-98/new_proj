import whisper
from transformers import BartForConditionalGeneration, BartTokenizer
from pydub import AudioSegment
from pydub.utils import make_chunks
import spacy
import warnings
import re
import os
import shutil

class TranscriptProcessor:
    def __init__(self, spacy_model="en_core_web_sm", bart_model="facebook/bart-large-cnn", file_name="transcription2.txt"):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.file_name = file_name
        self.nlp = spacy.load(spacy_model)
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.whisper_model = whisper.load_model("base")

    def transcribe(self, filepath):
        #raw speech- text conversion from given file-path.
        
        audio =AudioSegment.from_file(filepath)
        chunks=make_chunks(audio,30000)
        chunk_dir="Chunks"
        os.makedirs(chunk_dir,exist_ok=True)
        last_transcirpt=""
        for i, chunk in enumerate(chunks):
            chunk_filename=os.path.join(chunk_dir,f"chunk_{i}.wav")
            chunk.export(chunk_filename,format="wav")
            result = self.whisper_model.transcribe(chunk_filename)
            last_transcirpt+=result["text"]+" "
        shutil.rmtree(chunk_dir)
        return last_transcirpt.strip()


    def process_video_audio(self, filepath):
        #Transcribe and paraphrase text from video/audio.
        text=self.transcribe(filepath)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s,!?]', '', text)  # Remove special characters except basic punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)  # Reduce repeated punctuation (e.g., "!!!" to "!")
        return text.strip()
        '''
        with open(self.file_name, "r") as file:
            text = file.read()

        paraphrased_versions = self.paraphrase_transcript(text)
        final_text = [para[0] for para in paraphrased_versions]
        return " ".join(final_text)
        '''

    def generate_transcripts(self, filepath):
        """Generate paraphrased transcripts from video/audio file."""
        return self.process_video_audio(filepath)

