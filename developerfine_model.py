
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

model = BartForConditionalGeneration.from_pretrained("fine_tuned_bart")
tokenizer = BartTokenizer.from_pretrained("fine_tuned_bart")
file_name = "transcription2.txt"
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def paraphrase_text(input_text: str, num_beams=2, num_return_sequences=1) -> list:
    print("Inside paraphrase method")
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate paraphrases
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1024,
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
        txt=para[0].replace(":","")
        final_text.append(txt)
    return " ".join(final_text)

def transcripts_generate(filepath):
    text =video_audio_text(filepath)
    with open("finetuned_op.txt","w") as file:
        file.write(text)
transcripts_generate("C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4")