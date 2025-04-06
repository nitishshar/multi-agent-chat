from src.document_processing.markdown_processor import MarkdownProcessor
from src.vector_store.store_builder import VectorStoreBuilder
from langchain_community.document_loaders import TextLoader
import logging
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

# Get vector database path from environment variables
vector_db_path = os.getenv("VECTOR_DB_PATH", "_vector_db")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Set UTF-8 encoding for TextLoader to handle special characters
        TextLoader.autodetect_encoding = True
        
        start_time = time.time()
        logger.info("Starting markdown processing pipeline")
        
        # Run the processing
        markdown_processor = MarkdownProcessor(folder_path="./markdown_files")
        
        try:
            logger.info("Loading markdown files...")
            markdown_processor.load_markdown_files()
        except Exception as e:
            logger.error(f"Error loading markdown files: {str(e)}")
            sys.exit(1)
        
        try:
            logger.info("Extracting chunks from documents...")
            chunks = markdown_processor.extract_chunks()
            if not chunks:
                logger.warning("No chunks were extracted from the documents")
                sys.exit(0)
            logger.info(f"Successfully extracted {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error extracting chunks: {str(e)}")
            sys.exit(1)
        
        try:
            # Build and save vector store
            logger.info(f"Building vector store with {len(chunks)} chunks...")
            vector_store = VectorStoreBuilder(storage_path=vector_db_path)
            vector_store.build_and_save(chunks)
            logger.info(f"Successfully built and saved vector store to {vector_db_path}")
        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            sys.exit(1)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 