import os
from qdrant_client import QdrantClient,models
from dotenv import load_dotenv, dotenv_values
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

load_dotenv()

summarise_model=T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-large")

qdrant_client = QdrantClient(
    url=os.getenv("ENV_URL"), 
    api_key=os.getenv("API_KEY"),
)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def generate_emb(text):
    return model.encode(text).tolist()
'''
def upsert_qdrant(text, payload=None):
    if payload is None:
        payload={}

    vector = generate_emb(text)
    qdrant_client.upsert(
        collection_name="dissertation_collection",
        points=[
            models.PointStruct(
                id = abs(hash(text)),
                vector = vector,
                payload=payload
            )
        ]
    )
    print(f"text '{text[:30]}...' upserted successfullyy")
'''

def query_and_store(text):
    print(f"processing:{text}")
    payload={"source":"user_query"}
    user_query=generate_emb(text)

    results=qdrant_client.search(
        collection_name="dissertation_collection",
        query_vector=user_query,
        limit=5,
        with_payload=True
    )

    relevant_chunks=[result.payload for result in results]
    combined_text = " ".join([chunk.get('text', '') for chunk in relevant_chunks])
    answer= generate_answer(combined_text)
    print(answer)

def generate_answer(text):
    inputs=tokenizer(text, return_tensors="pt",max_length=512, truncation=True )
    generation_config=GenerationConfig(
        max_length=500, 
        min_length=30, 
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    outputs=summarise_model.generate(
        inputs["input_ids"],
        generation_config=generation_config

    )
    result=tokenizer.decode(outputs[0],skip_special_tokens=True)
    return result



if __name__ == "__main__":
    user_input="what are the methods used in API Testing?"

    query_and_store(user_input)


