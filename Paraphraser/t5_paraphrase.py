
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings

import pandas as pd

class T5_small:
    def __init__(self,model="mrm8488/t5-small-finetuned-quora-for-paraphrasing", model1='all-MiniLM-L6-v2'):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model,legacy=False)
        self.model1= SentenceTransformer(model1)
        self.file_name = None
        #print(self.model.config)
        # Load spacy model
        self.nlp = spacy.load("en_core_web_sm")
        self.df=None
        self.keywords= []
        self.keyword_embeddings = []

    def paraphrase_text(self,input_text: str, num_beams=2, num_return_sequences=1) -> list:
        print("Inside paraphrase method")
        print(input_text)
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        generation_config = GenerationConfig(
            max_length=400,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        try:
            
            outputs = self.model.generate(
                inputs["input_ids"],
                generation_config=generation_config
            )

            
            # Decode and return paraphrased outputs
            paraphrased_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            return paraphrased_texts
        except:
            print("No paraphrase versions found")
            return input_text

    def split_into_sentences_spacy(self,text: str) -> list:
        doc = self.nlp(text)
        for sent in doc.sents:
            print(sent.text)
        
        return [sent.text.strip() for sent in doc.sents]

    def paraphrase_transcript(self,transcript):
        sentences=self.split_into_sentences_spacy(transcript)
        paraphrase_sentences=[self.paraphrase_text(sentence) for sentence in sentences if self.keyword_sentence_similarity(sentence) is not None]
        return paraphrase_sentences

    def keyword_sentence_similarity(self, sentence):
        sentence_embedding = self.model1.encode([sentence])
        
        
        # Calculate cosine similarities between the sentence and each keyword
        similarities = cosine_similarity(sentence_embedding, self.keyword_embeddings)
        
        # Check if the sentence is relevant to any keyword (similarity > threshold)
        relevant_keywords = [self.keywords[i] for i in range(len(similarities[0])) if similarities[0][i] > 0.5]
        
        if relevant_keywords:
            print("returning")
            return sentence
        else:
            return None

    # Example usage
    def video_audio_text(self,filepath):
        text=""
        with open(self.file_name, "r") as file:
            text=file.read()


        original_text = text
        paraphrased_versions = self.paraphrase_transcript(original_text)

        final_text=[]
        for para in paraphrased_versions:
            txt=para[0].replace(":","")
            final_text.append(txt)
        return " ".join(final_text)

    def transcripts_generate(self,filepath, excelPath):
        self.file_name=filepath
        self.df=pd.read_excel(excelPath)
        self.keywords= self.df["Keyword"].tolist()
        self.keyword_embeddings = self.model1.encode(self.keywords)
        text = self.video_audio_text(filepath)
        return text

    #transcripts_generate("C:\\Users\\Deepak J Bhat\\Downloads\\video_file.mp4")