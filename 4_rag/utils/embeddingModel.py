from langchain_community.embeddings import HuggingFaceEmbeddings
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import yaml

class EmbeddingModel:
    def __init__(self):
        self.get_embedding_model()

    def get_embedding_model(self):
        with open("config.yaml", "r") as file:
            model_config = yaml.safe_load(file)
            
            # Load model in FP16 if GPU is available to save memory
            model_name = model_config['embedding_model']['model_name']
            self.embedding_model =  HuggingFaceEmbeddings(model_name=model_name)

    
    def create_vector_store(self, documents, db_path, batch_size=5):
        """
        Create a vector store from a text file and save it to disk.

        Args:
            documents (list): The list of documents.
            db_path (str): The path to save the vector store.
            batch_size (int): The number of documents to process at once.
        """

        
        # Create a vector store using the Chroma vector store
        current_doc_count = 0
        batch_size = batch_size
        if os.path.exists(db_path):
            db = Chroma(
                persist_directory=db_path,
                embedding_function=self.embedding_model
            )
            current_doc_count = db._collection.count()
        else:
            db = Chroma.from_documents(
                documents[current_doc_count:5], self.embedding_model, persist_directory=db_path)
            
        for i in tqdm(range(current_doc_count, len(documents), batch_size)):
            db.add_documents(documents[i:i+batch_size])

    
    def load_croma_db(self, db_path):
        """
        Load a vector store from disk.

        Args:
            db_path (str): The path to the vector store.

        Returns:
            Chroma: The loaded vector store.
        """
        return Chroma(persist_directory=db_path, embedding_function=self.embedding_model)


        
