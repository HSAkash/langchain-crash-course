import os

from utils import embeddingModel


# Define the file path for the document
file_path = os.path.join(os.path.dirname(__file__), "books", "odyssey.txt")
db_path = os.path.join(os.path.dirname(__file__), "db", f"{file_path.split("/")[-1].split(".")[0]}_chroma")

# Initialize the embedding model
instance = embeddingModel.EmbeddingModel()



# Load the existing vector store with the embedding function
db = instance.load_croma_db(db_path)

# Define the user's question
query = "Who is Odysseus' wife?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
