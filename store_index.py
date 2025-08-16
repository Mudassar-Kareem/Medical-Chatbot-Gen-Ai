from src.helper import load_pdf_file, make_chunks, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


extracted_data=load_pdf_file(data='Data/')
chunks= make_chunks(extracted_data)
embeddings = download_hugging_face_embeddings()

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot-index"
pc.create_index(
    name = index_name,
    dimension = 384,  # Dimension of the embedding vector
    metric = "cosine",  # Similarity metric
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    )
)

# Embed each chunk and upsert the embeddings into your Pinecone index.
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings
)




