import argparse
import subprocess
import os
import sys
import webbrowser
import time

def run_streamlit():
    """Run the Streamlit application"""
    print("Starting Streamlit application...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for Streamlit to start
    url = None
    for line in streamlit_process.stdout:
        print(line, end="")
        if "You can now view your Streamlit app in your browser" in line:
            # Next line should contain the URL
            url_line = next(streamlit_process.stdout, "")
            if "Local URL:" in url_line:
                url = url_line.split("Local URL:")[1].strip()
                break
    
    # Open browser if URL was found
    if url:
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    return streamlit_process

def run_gradio_direct():
    """Run the Gradio application directly"""
    print("Starting Gradio application...")
    gradio_process = subprocess.Popen(
        [sys.executable, "gradio_chat.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for Gradio to start
    url = None
    for line in gradio_process.stdout:
        print(line, end="")
        if "Running on local URL:" in line:
            url = line.split("Running on local URL:")[1].strip()
            break
    
    # Open browser if URL was found
    if url:
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    return gradio_process

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Run Multi-Agent Chat Application")
    parser.add_argument("--mode", choices=["streamlit", "gradio", "both"], default="streamlit",
                      help="Which interface to run: streamlit, gradio, or both")
    
    args = parser.parse_args()
    
    # Check if vector store exists
    vector_db_path = "_vector_db"
    if not os.path.exists(vector_db_path):
        print(f"Warning: Vector store not found at {vector_db_path}")
        print("You'll need to process documents before using the chat interfaces.")
    
    processes = []
    
    # Run Streamlit
    if args.mode in ["streamlit", "both"]:
        streamlit_process = run_streamlit()
        processes.append(streamlit_process)
    
    # Run Gradio directly
    if args.mode in ["gradio", "both"]:
        gradio_process = run_gradio_direct()
        processes.append(gradio_process)
    
    # Wait for user to exit
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for process in processes:
            process.terminate()

if __name__ == "__main__":
    main() 