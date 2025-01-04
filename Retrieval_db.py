import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

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
        """Generate embedding for a given text."""
        return self.model.encode(text).tolist()

    def query_and_store(self, text):
        """Query Qdrant and generate an answer based on relevant chunks."""
        print(f"Processing: {text}")
        payload = {"source": "user_query"}
        user_query = self.generate_emb(text)

        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name="dissertation_collection2",
            query_vector=user_query,
            limit=5,
            with_payload=True
        )

        # Combine relevant chunks
        relevant_chunks = [result.payload for result in results]
        combined_text = " ".join([chunk.get('text', '') for chunk in relevant_chunks])
        
        # Generate answer
        answer = self.generate_answer(combined_text)
        return answer

    def generate_answer(self, text):
        """Generate an answer for the given text using the summarization model."""
        print(text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        generation_config = GenerationConfig(
            max_length=200,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        outputs = self.summarise_model.generate(
            inputs["input_ids"],
            generation_config=generation_config
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result


if __name__ == "__main__":
    # Example usage
    processor = DissertationQueryProcessor()
    user_input = "What is software testing?"
    processor.query_and_store(user_input)

