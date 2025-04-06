import streamlit as st
import subprocess
import os
import sys
import time

# Set page config
st.set_page_config(
    page_title="Gradio Chat | Multi-Agent Research",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render header
st.title("Gradio Chat Interface")
st.markdown(
    """
    This page launches a Gradio chat interface that provides a more interactive experience 
    with step-by-step thinking indicators and status updates.
    
    The Gradio interface will open in a new tab. If it doesn't open automatically, 
    click the link that appears below after the server starts.
    """
)

# Render sidebar navigation
with st.sidebar:
    st.header("Navigation")
    st.markdown(
        """
        - **💻 [Home](/)**:  Document Processing
        - **💬 [Chat](/Chat)**: Streamlit Chat Interface
        - **🔍 [Run Query](/Run_Query)**: Run a specific query
        - **💬 Gradio Chat**: Interactive Chat (current)
        """
    )
    
    st.divider()
    
    st.header("About")
    st.markdown(
        """
        The Gradio chat interface provides:
        
        1. Step-by-step thinking indicators
        2. Status updates during processing
        3. Interactive chat experience
        4. Example questions to get started
        
        This interface is similar to the one shown in the article 
        [Next-Level Chatbots](https://thegenairevolution.com/next-level-chatbots-integrate-multiple-agents-to-boost-accuracy-and-clarity/).
        """
    )

# Check if vector store exists
vector_db_path = "_vector_db"
if not os.path.exists(vector_db_path):
    st.warning("No vector store found. Please process documents on the home page first.")
    st.stop()

# Function to launch Gradio in a separate process
def launch_gradio():
    gradio_script_path = os.path.join(os.getcwd(), "gradio_chat.py")
    
    if not os.path.exists(gradio_script_path):
        st.error(f"Gradio chat script not found at {gradio_script_path}")
        return None
    
    # Get Python executable path
    python_exe = sys.executable
    
    # Launch Gradio in a separate process
    process = subprocess.Popen(
        [python_exe, gradio_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    return process

# Container for Gradio server output
server_output = st.empty()

# Button to launch Gradio
if st.button("Launch Gradio Chat Interface", type="primary"):
    with server_output.container():
        st.info("Starting Gradio server... This may take a few moments.")
        
        # Launch Gradio
        process = launch_gradio()
        
        if process:
            st.info("Gradio server is starting...")
            
            # Wait for the server to start and get the URL
            server_url = None
            start_time = time.time()
            timeout = 30  # seconds
            
            while time.time() - start_time < timeout:
                line = process.stdout.readline()
                if line:
                    if "Running on local URL" in line:
                        server_url = line.split("Running on local URL:")[1].strip()
                        break
                
                # Display the line
                st.text(line.strip())
                
                time.sleep(0.1)
            
            if server_url:
                st.success(f"Gradio server started successfully!")
                st.markdown(f"**Open the chat interface:** [Gradio Chat Interface]({server_url})")
                
                # Display iframe with Gradio interface
                st.markdown(f'<iframe src="{server_url}" width="100%" height="800" frameborder="0"></iframe>', unsafe_allow_html=True)
            else:
                st.error("Failed to get Gradio server URL within timeout. Please check the server output.")
        else:
            st.error("Failed to start Gradio server.")
else:
    server_output.info("Click the button above to launch the Gradio chat interface.")

# Additional instructions
st.markdown(
    """
    ### Instructions
    
    1. Click the "Launch Gradio Chat Interface" button above to start the Gradio server
    2. When the server starts, you'll see a link to open the chat interface
    3. You can also use the embedded interface above once it's loaded
    4. If you need to restart the server, refresh this page and click the button again
    
    ### Features
    
    - **Thinking Indicators**: Shows when the AI is processing your question
    - **Status Updates**: Provides updates on the AI's thinking process
    - **Examples**: Try the example questions to see how the system works
    - **Conversation History**: Your chat history is preserved during the session
    """
) 