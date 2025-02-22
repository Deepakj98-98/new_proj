import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig
import nltk
from nltk.tokenize import sent_tokenize

class DissertationQueryProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize models and clients
        self.qdrant_client = QdrantClient(
            url=os.getenv("ENV_URL"),
            api_key=os.getenv("API_KEY"),
        )
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.summarise_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

    def generate_emb(self, text):
        #Generate embedding for a given text.
        return self.model.encode(text).tolist()

    def reformulate_query(self, query):
        inputs = self.tokenizer(f"Rephrase this query: {query}", return_tensors="pt")
        outputs = self.summarise_model.generate(inputs.input_ids, max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def query_and_store(self, text):
        #Query Qdrant and generate an answer based on relevant chunks.
        #print(f"Processing: {text}")
        payload = {"source": "user_query"}
        text1=self.reformulate_query(text)
        user_query = self.generate_emb(text1)

        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name="dissertation_collection6",
            query_vector=user_query,
            limit=3,
            with_payload=True
        )

        # Combine relevant chunks
        relevant_chunks = [result.payload for result in results]
        combined_text = " ".join([chunk.get('text', '') for chunk in relevant_chunks])
        
            # Generate answer
        #prompt = f"User Query: {text1}\n Context: {sentence}\n Answer like an expert:"

        answer = self.generate_answer(combined_text)
    
        return answer
    '''
    def generate_answer_new(self, text):
        """Generate an answer for the given text using the summarization model."""

        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        generation_config = GenerationConfig(
            max_length=100,
            min_length=10,
            length_penalty=1.0,
            num_beams=6,
            early_stopping=True
        )
        outputs = self.summarise_model.generate(
            inputs["input_ids"],
            generation_config=generation_config
        )
        summarized_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Combine summarized sentences into one text
        return summarized_sentence
'''
    def generate_answer(self, text):
        #Generate an answer for the given text using the summarization model.
        sentences = sent_tokenize(text)  # Split text into sentences
        summarized_sentences = []
        
        for sentence in sentences:
            print(f"Processing sentence: {sentence}")
            inputs = self.tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding=True)
            generation_config = GenerationConfig(
                max_length=100,
                min_length=10,
                length_penalty=1.0,
                num_beams=6,
                early_stopping=True
            )
            outputs = self.summarise_model.generate(
                inputs["input_ids"],
                generation_config=generation_config
            )
            summarized_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summarized_sentences.append(summarized_sentence)
        
        # Combine summarized sentences into one text
        return " ".join(summarized_sentences)
    
    def refine_answer(self, text):
        prompt = f"Improve the readability of this response: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512)
        outputs = self.summarise_model.generate(inputs.input_ids, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


'''
if __name__ == "__main__":
    # Example usage
    processor = DissertationQueryProcessor()
    user_input = "What is software testing?"
    processor.query_and_store(user_input)
'''
