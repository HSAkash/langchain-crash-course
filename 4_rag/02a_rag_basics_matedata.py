import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from glob import glob
from utils import embeddingModel








# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")



# Ensure the books directory exists
if not os.path.exists(books_dir):
    raise FileNotFoundError(
        f"The directory {books_dir} does not exist. Please check the path."
    )

# List all text files in the directory
book_files = glob(os.path.join(books_dir, "*.txt"))

# Read the text content from each file and store it with metadata
documents = []
for book_file in book_files:
    loader = TextLoader(book_file)
    book_docs = loader.load()
    for doc in book_docs:
        # Add metadata to each document indicating its source
        doc.metadata = {"source": book_file.split("/")[-1]}
        documents.append(doc)

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# Initialize the embedding model
instance = embeddingModel.EmbeddingModel()
# Create vector store
print ("Creating vector store...")
instance.create_vector_store(docs, persistent_directory)
print ("Vector store created.")

