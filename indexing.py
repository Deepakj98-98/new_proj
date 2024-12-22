import os
import numpy as np
from dotenv import load_dotenv, dotenv_values
from qdrant_client import QdrantClient,models
from langchain_text_splitters import SpacyTextSplitter,CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

load_dotenv()
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

qdrant_client = QdrantClient(
    url=os.getenv("ENV_URL"), 
    api_key=os.getenv("API_KEY"),
)
print("successful")
#qdrant_client.create_collection(collection_name="{dissertation_collection}",vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),)
CHUNK_FOLDER="role_folder"
texts=[]
def chunk_files():
    global texts
    filepaths=os.listdir(CHUNK_FOLDER)
    print(filepaths)
    text=""
    for file in filepaths:
        path=os.path.join(CHUNK_FOLDER,file)
        with open(path,"r") as file:
            text+=file.read()
    text_splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap = 30)

    #texts = text_splitter.create_documents(text)
    texts = text_splitter.split_text(text)
    print(texts[0])
    print(len(texts))
    into_vectors_db()

def get_embeddings_data(data_chunks):
    data_embeddings = []
    for data in data_chunks:
        data_embedding = model.encode(data)
        data_embeddings.append(data_embedding)

    return data_embeddings

def into_vectors_db():
    global texts
    data_embeddings1 = get_embeddings_data(texts)
    print(len(texts))
    batch_size=1000
    collection = "dissertation_collection"
    for i in range(0, len(data_embeddings1),batch_size):
        batch_embeddings=data_embeddings1[i:i+batch_size]
        array_embeddings = np.vstack(batch_embeddings)    
        try:
            qdrant_client.recreate_collection(
                collection_name = collection,
                vectors_config = models.VectorParams(
                    size = len(batch_embeddings[0]),
                    distance = models.Distance.COSINE
                )
            )
        except Exception as e:
            print(f"Error while creating collection: {e}")

        data_e = []
        for idx, emmbeddings in enumerate(array_embeddings):
            data_e.append({
                "id" : i+idx,
                "vector": emmbeddings.tolist(),
                "payload": {"text": texts[i+idx]}
            })

        qdrant_client.upsert(collection_name = collection, points = data_e)
    qdrant_client.close()




chunk_files()
#print(qdrant_client.get_collections())