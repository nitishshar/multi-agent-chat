import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import time
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Import our modules
from src.document_processing.markdown_processor import MarkdownProcessor
from src.vector_store.store_builder import VectorStoreBuilder
from src.ui.components import UIComponents, Callbacks
from src.utils.caching import SessionState

# Configuration
VECTOR_DB_PATH = "_vector_db"

# Set page config
st.set_page_config(
    page_title="Document Processing | Multi-Agent Research",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return the path.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        Path to the temporary directory containing the files
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save files to temporary directory
    for i, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir, file.name)
        
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
            
        UIComponents.render_file_processing_progress(
            len(uploaded_files), i + 1, "Saving files"
        )
    
    return temp_dir

@st.cache_resource
def get_vector_store():
    """
    Get or create the vector store.
    
    Returns:
        Vector store instance
    """
    vector_store_builder = VectorStoreBuilder(storage_path=VECTOR_DB_PATH)
    
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    
    return vector_store_builder.load()

def process_documents(uploaded_files):
    """
    Process uploaded documents and build/update the vector store.
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    if not uploaded_files:
        return
    
    # Save uploaded files to a temporary directory
    temp_dir = save_uploaded_files(uploaded_files)
    
    # Process the markdown files
    markdown_processor = MarkdownProcessor(folder_path=temp_dir)
    
    with st.spinner("Loading documents..."):
        markdown_processor.load_markdown_files()
    
    with st.spinner("Extracting document chunks..."):
        document_chunks = markdown_processor.extract_chunks()
    
    # Build the vector store from the document chunks
    with st.spinner("Building vector store..."):
        vector_store_builder = VectorStoreBuilder(storage_path=VECTOR_DB_PATH)
        vector_store_builder.build_and_save(document_chunks)
    
    st.success(f"Successfully processed {len(uploaded_files)} documents and created {len(document_chunks)} chunks!")

def main():
    """
    Main function for the Document Processing app.
    """
    # Initialize session state
    SessionState.initialize()
    
    # Render header
    st.title("Document Processing")
    st.markdown(
        "Upload and process markdown documents to use with the Multi-Agent Research Chat system."
    )
    
    # Render sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        st.markdown(
            """
            - **💻 Home**: Document Processing (current)
            - **💬 [Chat](/Chat)**: Chat with your documents
            """
        )
        
        st.divider()
        
        st.header("About")
        st.markdown(
            """
            This application uses a multi-agent approach:
            
            1. **Analyst Agent**: Researches and synthesizes information
            2. **Reviewer Agent**: Checks and refines the analyst's work
            
            This design ensures accurate and comprehensive answers.
            """
        )
    
    # File upload section
    st.header("Upload Documents")
    st.markdown(
        """
        Upload markdown files (.md) to add to the knowledge base. The system will process
        them and make them available for the chat interface.
        """
    )
    
    uploaded_files = UIComponents.render_file_upload()
    
    if uploaded_files and st.button("Process Documents", type="primary"):
        process_documents(uploaded_files)
    
    # Vector store status
    st.divider()
    st.header("Current Document Status")
    vector_store = get_vector_store()
    
    if vector_store:
        st.success("Documents are loaded and ready for use in the Chat interface!")
        
        # Add a button to go to the chat page
        if st.button("Go to Chat Interface", type="primary"):
            st.switch_page("pages/01_Chat.py")
    else:
        st.warning("No documents have been processed yet. Please upload and process documents.")

if __name__ == "__main__":
    main() 