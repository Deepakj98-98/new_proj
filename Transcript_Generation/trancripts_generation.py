import whisper
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
import warnings

class TranscriptProcessor:
    def __init__(self, spacy_model="en_core_web_sm", bart_model="facebook/bart-large-cnn", file_name="transcription2.txt"):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.file_name = file_name
        self.nlp = spacy.load(spacy_model)
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.whisper_model = whisper.load_model("base")

    def transcribe(self, filepath):
        """Transcribe audio from the provided filepath."""
        result = self.whisper_model.transcribe(filepath)
        with open(self.file_name, "w") as file:
            file.write(result["text"])
        return result["text"]

    def paraphrase_text(self, input_text: str, num_beams=2, num_return_sequences=1) -> list:
        """Paraphrase the given input text."""
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.bart_model.generate(
            inputs["input_ids"],
            max_length=400,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def split_into_sentences(self, text: str) -> list:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def paraphrase_transcript(self, transcript):
        """Paraphrase a full transcript sentence by sentence."""
        sentences = self.split_into_sentences(transcript)
        paraphrased_sentences = [self.paraphrase_text(sentence) for sentence in sentences]
        return paraphrased_sentences

    def process_video_audio(self, filepath):
        """Transcribe and paraphrase text from video/audio."""
        text=self.transcribe(filepath)
        return text
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

# Example usage from another file:
# from transcript_processor import TranscriptProcessor
# processor = TranscriptProcessor()
# transcript = processor.generate_transcripts("path/to/video_file.mp4")
# print(transcript)
