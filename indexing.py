import os
from dotenv import load_dotenv, dotenv_values
from qdrant_client import QdrantClient,models
from langchain_text_splitters import SpacyTextSplitter,CharacterTextSplitter

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("ENV_URL"), 
    api_key=os.getenv("API_KEY"),
)
print("successful")
#qdrant_client.create_collection(collection_name="{dissertation_collection}",vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),)
CHUNK_FOLDER="role_folder"

def chunk_files():
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



chunk_files()    
#print(qdrant_client.get_collections())