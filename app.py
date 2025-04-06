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
from src.tools.search_tool import VectorStoreSearchTool, AskForClarificationsTool
from src.agents.crew import AgentCrewBuilder
from src.ui.components import UIComponents, Callbacks
from src.utils.caching import SessionState

# Configuration
VECTOR_DB_PATH = "_vector_db"

# Set page config
st.set_page_config(
    page_title="Multi-Agent Research Chat",
    page_icon="🤖",
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
def get_vector_store(rebuild=False):
    """
    Get or create the vector store.
    
    Args:
        rebuild: Whether to rebuild the vector store
        
    Returns:
        Vector store instance
    """
    vector_store_builder = VectorStoreBuilder(storage_path=VECTOR_DB_PATH)
    
    if rebuild or not os.path.exists(VECTOR_DB_PATH):
        return None
    
    return vector_store_builder.load()

@st.cache_resource
def get_agent_crew(vector_store):
    """
    Get or create the agent crew.
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        Agent crew instance
    """
    # Create search tool with the vector store
    search_tool = VectorStoreSearchTool(storage_path=VECTOR_DB_PATH)
    clarification_tool = AskForClarificationsTool()
    
    # Create the agent crew
    crew_builder = AgentCrewBuilder(tools=[search_tool, clarification_tool])
    
    # Build the analyst task with template
    analyst_task_template = """# Follow these step-by-step instructions:
1. Understand the user request: Read and analyze thoroughly, taking conversation history into account.
2. [OPTIONAL] Only ask for clarifications if the request is unclear or ambiguous.
3. Use the search tool repeatedly as needed to collect relevant information.
4. Perform any necessary calculations or analyses.
5. Synthesize gathered data into a comprehensive response.
6. Provide a detailed answer supported by references.

# Conversation history:
```{conversation_history}```

# User request:
```{user_request}```

# IMPORTANT:
- You MUST only use information found via the search tool.
- Do NOT rely on external knowledge.
"""
    
    crew_builder.build_analyst_agent()
    crew_builder.build_reviewer_agent()
    crew_builder.build_analyst_task(analyst_task_template)
    crew_builder.build_reviewer_task()
    
    return crew_builder.build_crew()

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

def process_message(message: str, chat_history: List) -> Any:
    """
    Process a user message with the agent crew.
    
    Args:
        message: User message
        chat_history: Chat history
        
    Returns:
        Result from the agent crew
    """
    # Get the vector store
    vector_store = get_vector_store()
    
    if not vector_store:
        raise ValueError("No vector store found. Please upload some documents first.")
    
    # Get the agent crew
    crew = get_agent_crew(vector_store)
    
    # Format conversation history for the crew
    formatted_history = ""
    for i, (role, content) in enumerate(chat_history[:-1]):  # Exclude the current message
        formatted_history += f"{role.capitalize()}: {content}\n\n"
    
    # Process the message with the crew
    result = crew.kickoff(
        inputs={
            "conversation_history": formatted_history,
            "user_request": message
        }
    )
    
    return result

def main():
    """
    Main function for the Streamlit app.
    """
    # Initialize session state
    SessionState.initialize()
    
    # Render header
    UIComponents.render_header()
    
    # Render sidebar controls
    UIComponents.render_sidebar_controls(
        on_clear_history=Callbacks.handle_clear_history,
        on_regenerate=lambda: Callbacks.handle_regenerate_last_response(process_message)
    )
    
    # File upload section
    uploaded_files = UIComponents.render_file_upload()
    
    if uploaded_files and st.button("Process Documents"):
        process_documents(uploaded_files)
    
    # Divider
    st.divider()
    
    # Chat section
    st.subheader("Chat")
    
    # Render chat history
    UIComponents.render_chat_history()
    
    # Render chat input
    UIComponents.render_chat_input(
        on_submit=lambda msg: Callbacks.handle_send_message(msg, process_message)
    )

if __name__ == "__main__":
    main() 