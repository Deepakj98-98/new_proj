import whisper
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

file_name = "transcription2.txt"
def transcribe(filepath):
    model = whisper.load_model("base")
    # Transcribe video directly
    #"C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4"
    result = model.transcribe(filepath)
    #print("Extracted Text:", result["text"])
    global file_name

    with open(file_name, "w") as file:
        # Write to the file
        file.write(result["text"])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"  # Pre-trained model
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def paraphrase_text(input_text: str, num_beams=2, num_return_sequences=1) -> list:
    print("Inside paraphrase method")
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate paraphrases
    outputs = model.generate(
        inputs["input_ids"],
        max_length=400,
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
def video_audio_text(filepath):
    transcribe(filepath)
    text=""
    with open(file_name, "r") as file:
        text=file.read()


    original_text = text
    paraphrased_versions = paraphrase_transcript(original_text)
    '''
    file_write="transcription_paraphrased"
    with open(file_write, "w") as file:
        # Write to the file
        for para in paraphrased_versions:
            file.write(para[0])
'''
    final_text=[]
    for para in paraphrased_versions:
        final_text.append(para[0])
    return " ".join(final_text)

def transcripts_generate(filepath):
    text =video_audio_text(filepath)
    return text
#transcripts_generate("C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4")