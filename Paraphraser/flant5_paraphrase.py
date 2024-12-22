
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import pandas as pd

# Load Flan-T5 model and tokenizer
model_name = "google/flan-t5-large"  # Choose "small", "base", "large", "xl", or "xxl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model1= SentenceTransformer('all-MiniLM-L6-v2')
file_name = "pdf_text.txt"
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
df=pd.read_excel("C:\\Users\\Deepak J Bhat\\Downloads\\software_dev_keywords.xlsx")
keywords= df["Keyword"].tolist()
keyword_embeddings = model1.encode(keywords)

def paraphrase_text(input_text: str, num_beams=2, num_return_sequences=1) -> list:
    print("Inside paraphrase method")
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    generation_config = GenerationConfig(
        max_length=400,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    outputs = model.generate(
        inputs["input_ids"],
        generation_config=generation_config
    )
    
    # Decode and return paraphrased outputs
    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrased_texts

def split_into_sentences_spacy(text: str) -> list:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def paraphrase_transcript(transcript):
    sentences=split_into_sentences_spacy(transcript)
    paraphrase_sentences=[paraphrase_text(sentence) for sentence in sentences if keyword_sentence_similarity(sentence) is not None]
    return paraphrase_sentences

def keyword_sentence_similarity(sentence):
    sentence_embedding = model1.encode([sentence])
    
    
    # Calculate cosine similarities between the sentence and each keyword
    similarities = cosine_similarity(sentence_embedding, keyword_embeddings)
    
    # Check if the sentence is relevant to any keyword (similarity > threshold)
    relevant_keywords = [keywords[i] for i in range(len(similarities[0])) if similarities[0][i] > 0.5]
    
    if relevant_keywords:
        print("returning")
        return sentence
    else:
        return None

# Example usage
def video_audio_text(filepath):
    text=""
    with open(file_name, "r") as file:
        text=file.read()


    original_text = text
    paraphrased_versions = paraphrase_transcript(original_text)

    final_text=[]
    for para in paraphrased_versions:
        txt=para[0].replace(":","")
        final_text.append(txt)
    return " ".join(final_text)

def transcripts_generate(filepath):
    text =video_audio_text(filepath)
    with open("finetuned_ba_flan.txt","w") as file:
        file.write(text)
transcripts_generate("C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4")