import os
from typing import List
import time
import logging
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SwedishRAGSystem:
    def __init__(self, data_dir: str = "/Users/birgermoell/Documents/polymath/swedish-medical-benchmark/data", index_dir: str = "swedish_index"):
        self.data_dir = data_dir
        self.index_dir = index_dir
        logger.info(f"Initializing SwedishRAGSystem with data directory: {data_dir}")
        logger.info(f"Index will be stored in: {os.path.abspath(index_dir)}")
        
        # Use a multilingual model that works well with Swedish
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        
    def load_documents(self) -> List[Document]:
        """Load all .txt files from the data directory and its subdirectories."""
        logger.info("Starting document loading process...")
        start_time = time.time()
        
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()
        
        end_time = time.time()
        logger.info(f"Loaded {len(documents)} documents in {end_time - start_time:.2f} seconds")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        logger.info("Starting document splitting process...")
        start_time = time.time()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        split_docs = splitter.split_documents(documents)
        
        end_time = time.time()
        logger.info(f"Split into {len(split_docs)} chunks in {end_time - start_time:.2f} seconds")
        return split_docs
    
    def create_index(self, force_recreate: bool = False):
        """Create a searchable index from the documents.
        
        Args:
            force_recreate (bool): If True, recreate the index even if it exists.
                                 If False, use existing index if available.
        """
        # Check if index exists
        if os.path.exists(self.index_dir) and not force_recreate:
            logger.info(f"Using existing index from {os.path.abspath(self.index_dir)}")
            self.vector_store = FAISS.load_local(
                self.index_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return

        total_start_time = time.time()
        logger.info("Starting index creation process...")
        
        # Load and split documents
        documents = self.load_documents()
        split_docs = self.split_documents(documents)
        
        if not split_docs:
            logger.warning("No documents found in the data directory.")
            return
        
        # Create embeddings with progress bar
        logger.info("Creating embeddings and FAISS index...")
        embedding_start_time = time.time()
        
        # Create batches for progress tracking
        batch_size = 32
        batches = [split_docs[i:i + batch_size] for i in range(0, len(split_docs), batch_size)]
        
        all_embeddings = []
        with tqdm(total=len(batches), desc="Creating embeddings") as pbar:
            for batch in batches:
                texts = [doc.page_content for doc in batch]
                embeddings = self.embeddings.embed_documents(texts)
                all_embeddings.extend(embeddings)
                pbar.update(1)
        
        embedding_end_time = time.time()
        embedding_duration = embedding_end_time - embedding_start_time
        logger.info(f"Embeddings created in {embedding_duration:.2f} seconds")
        
        # Create FAISS index
        logger.info("Building FAISS index...")
        index_start_time = time.time()
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Save the index
        logger.info(f"Saving index to disk at {os.path.abspath(self.index_dir)}...")
        self.vector_store.save_local(self.index_dir)
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        logger.info(f"Index creation completed in {total_duration:.2f} seconds")
        
        # Log summary statistics
        logger.info("\nProcess Summary:")
        logger.info(f"Total documents processed: {len(documents)}")
        logger.info(f"Total chunks created: {len(split_docs)}")
        logger.info(f"Average chunk size: {sum(len(doc.page_content) for doc in split_docs) / len(split_docs):.0f} characters")
        logger.info(f"Total processing time: {total_duration:.2f} seconds")
        logger.info(f"Index location: {os.path.abspath(self.index_dir)}")
        
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search the index for relevant documents."""
        logger.info(f"Searching for: {query}")
        start_time = time.time()
        
        if not self.vector_store:
            if os.path.exists(self.index_dir):
                logger.info("Loading existing index...")
                self.vector_store = FAISS.load_local(
                    self.index_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.error("No index found. Please create an index first.")
                raise ValueError("No index found. Please create an index first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        end_time = time.time()
        logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
        return results

if __name__ == "__main__":
    # Example usage
    rag_system = SwedishRAGSystem()
    
    # To use existing index (default behavior):
    rag_system.create_index()
    
    # To force recreation of the index:
    #rag_system.create_index(force_recreate=True)
    
    # Example search
    results = rag_system.search("En 78-årig man kommer till Akutmottagningen på söndag kväll då han i förrgår noterade att synen försvann nästan helt på höger öga inom loppet av några minuter. Ögat känns annars helt normalt – ingen smärta eller irritation och det är inte rött.\nHan har väntat på att synen skulle komma tillbaka, men det har den inte gjort. Han har en tablettbehandlad hypertoni och opererades för katarakt på båda ögonen för några år sedan men är i övrigt frisk.\nHan klarar fingerräkning på som mest 1 m avstånd på höger öga, visus är 0,9 på vänster.\nVilken diagnos är mest trolig?\n\n*Välj ett alternativ:*\na) centralventrombos\nb) efterstarr\nc) iridocyklit\nd) makuladegeneration\ne) retinitis pigmentosa")  # Your search query here
    for doc in results:
        print(f"Content: {doc.page_content}\n")
