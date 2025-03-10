import speech_recognition as sr
from pydub import AudioSegment
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
import whisper

video = AudioSegment.from_file("C:\\Users\\Deepak J Bhat\\Downloads\\VM video.mp4", format="mp4")
audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
audio.export("audio.wav", format="wav")
# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Open the audio file
#with sr.AudioFile("audio.wav") as source:
#    audio_text = r.record(source)
# Recognize the speech in the audio
model = whisper.load_model("base")
    # Transcribe video directly
    #"C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4"
result = model.transcribe("audio.wav")
#print("Extracted Text:", result["text"])
print(result["text"])
'''
text = r.recognize_google(audio_text, language='en-US')

# Print the transcript
file_name = "transcription2.txt"

with open(file_name, "w") as file:
    # Write to the file
    file.write(text)


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"  # Pre-trained model
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def paraphrase_text(input_text: str, num_beams=4, num_return_sequences=1) -> list:
   
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=4000, truncation=True)
    
    # Generate paraphrases
    outputs = model.generate(
        inputs["input_ids"],
        max_length=4000,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    # Decode and return paraphrased outputs
    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrased_texts

def split_into_sentences_spacy(text: str) -> list:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def paraphrase_transcript(transcript):
    sentences=split_into_sentences_spacy(transcript)
    paraphrase_sentences=[paraphrase_text(sentence) for sentence in sentences]
    return paraphrase_sentences

# Example usage
if __name__ == "__main__":
    text=""
    with open(file_name, "r") as file:
        text=file.read()


    original_text = text
    paraphrased_versions = paraphrase_transcript(original_text)
    
    print("Paraphrased Versions:")
    print(paraphrased_versions)
    file_write="transcription_paraphrased"
    with open(file_write, "w") as file:
        # Write to the file
        for para in paraphrased_versions:
            file.write(para[0])
'''


