from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load the PDF files from the directory
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob = "*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# make the Chunks of the extracted data
def make_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 20)
    chunks = text_splitter.split_documents(data)
    return chunks

# download the embedding model from hugging face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings