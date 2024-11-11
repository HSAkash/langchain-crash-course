import os
from utils import embeddingModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Define the file path for the document
file_path = os.path.join(os.path.dirname(__file__), "books", "odyssey.txt")
db_path = os.path.join(os.path.dirname(__file__), "db", f"{file_path.split("/")[-1].split(".")[0]}_chroma")

# Create embedding model
print ("Creating embedding model...")
instance = embeddingModel.EmbeddingModel()
print ("Embedding model created.")


embedding_model = instance.embedding_model

# Create vector store
print ("Creating vector store...")
# Split the text into chunks for indexing
loader = TextLoader(file_path)
documents = loader.load()

# Split the text into chunks for indexing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)
instance.create_vector_store(chunks, db_path)
print ("Vector store created.")