import os
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_text_splitters import SpacyTextSplitter
from sentence_transformers import SentenceTransformer

class QdrantChunking:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        load_dotenv()
        #self.collection_name = collection_name
        self.texts = []
        self.model = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(
            url=os.getenv("ENV_URL"), 
            api_key=os.getenv("API_KEY"),
        )
        print("Qdrant client initialized successfully.")
        
    def create_collection_qdrant(self, collection_name):
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Error while creating collection (it may already exist): {e}")

    def chunk_files(self,collection_name,role):
        #Read and split text files into smaller chunks.
        self.texts = []
        self.create_collection_qdrant(collection_name)
        filepaths = os.listdir(role)
        print("Files found:", filepaths)

        text = ""
        for file in filepaths:
            path = os.path.join(role, file)
            with open(path, "r") as file:
                text += file.read()

        text_splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap=30)
        self.texts = text_splitter.split_text(text)
        #print(f"First chunk: {self.texts[0]}")
        print(f"Total chunks: {len(self.texts)}")

        self.into_vectors_db(collection_name)

    def get_embeddings_data(self, data_chunks):
        #Generate embeddings for the given chunks.
        data_embeddings = [self.model.encode(data) for data in data_chunks]
        return data_embeddings

    def into_vectors_db(self,collection_name):
        #Upsert the text chunks and their embeddings into Qdrant.
        data_embeddings = self.get_embeddings_data(self.texts)
        print(f"Number of chunks to insert: {len(self.texts)}")

        batch_size = 1000
        for i in range(0, len(data_embeddings), batch_size):
            batch_embeddings = data_embeddings[i:i + batch_size]
            array_embeddings = np.vstack(batch_embeddings)

            try:
                self.qdrant_client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=len(batch_embeddings[0]),
                        distance=models.Distance.COSINE
                    )
                )
            except Exception as e:
                print(f"Error while creating collection: {e}")

            data_points = [
                {
                    "id": i + idx,
                    "vector": embedding.tolist(),
                    "payload": {"text": self.texts[i + idx]}
                }
                for idx, embedding in enumerate(array_embeddings)
            ]

            self.qdrant_client.upsert(collection_name=collection_name, points=data_points)

        print("Data successfully inserted into Qdrant.")
        #self.qdrant_client.close()