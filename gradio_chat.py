import gradio as gr
import time
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import our modules
from src.vector_store.store_builder import VectorStoreBuilder
from src.tools.search_tool import VectorStoreSearchTool, AskForClarificationsTool
from src.agents.crew import AgentCrewBuilder

# Load environment variables
load_dotenv()

# Configuration
VECTOR_DB_PATH = "_vector_db"

# Create singleton instances of tools and crew
search_tool = None
clarification_tool = None
agent_crew = None
vector_store = None

def initialize_tools_and_crew():
    """Initialize the tools and crew once at startup."""
    global search_tool, clarification_tool, agent_crew
    
    # Only initialize if not already done
    if search_tool is None:
        print("Initializing search tool and vector store...")
        search_tool = VectorStoreSearchTool(storage_path=VECTOR_DB_PATH)
    
    if clarification_tool is None:
        print("Initializing clarification tool...")
        clarification_tool = AskForClarificationsTool()
        
    
    if agent_crew is None:
        print("Building agent crew...")
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
- When using the search tool, only provide the query parameter.
"""
        
        # Build the crew
        analyst_agent = crew_builder.build_analyst_agent()
        analyst_task= crew_builder.build_analyst_task(analyst_task_template)
        agent_crew = crew_builder.build_crew_with_agents([analyst_agent], [analyst_task])
        print("Agent crew initialized successfully")

def get_vector_store():
    """
    Get or create the vector store.
    
    Returns:
        Vector store instance
    """
    vector_store_builder = VectorStoreBuilder(storage_path=VECTOR_DB_PATH)
    
    if not os.path.exists(VECTOR_DB_PATH):
        print("No vector store found. Please process documents first.")
        return None
    
    return vector_store_builder.load()

def get_agent_crew():
    """
    Get the singleton agent crew instance.
    
    Returns:
        Agent crew instance
    """
    # Ensure the tools and crew are initialized
    if agent_crew is None:
        initialize_tools_and_crew()
    
    return agent_crew

def format_history(chat_history):
    """Format chat history for the agent crew"""
    formatted_history = ""
    for message in chat_history:
        if len(message) == 2:  # Each message is (user, assistant) tuple
            user_msg, assistant_msg = message
            formatted_history += f"User: {user_msg}\n\n"
            if assistant_msg:  # Might be None for the latest user message
                formatted_history += f"Assistant: {assistant_msg}\n\n"
    return formatted_history

def process_message(message, history):
    """
    Process a message with the agent crew and yield incremental updates.
    
    Args:
        message: User message
        history: Chat history
        
    Returns:
        Generator yielding updates to the thinking status and final response
    """
    start_time = time.time()
    
    # Yield an initial thinking indicator
    yield "Thinking: Analyzing your question..."
    time.sleep(0.5)  # Brief pause for UI update
    
    try:
        # Get vector store
        yield "Thinking: Connecting to document database..."
        vector_store = get_vector_store()
        if not vector_store:
            yield "Error: No vector store found. Please process documents first."
            return
        
        # Get agent crew (will initialize if needed)
        yield "Thinking: Initializing research team..."
        crew = get_agent_crew()
        
        # Format conversation history for the crew
        yield "Thinking: Processing conversation history..."
        formatted_history = format_history(history)
        
        # Update thinking status
        yield "Thinking: Searching for relevant information in documents..."
        time.sleep(1)  # Brief pause for UI update
        
        # Process the message with the crew
        yield "Working: Starting research process..."
        time.sleep(1)  # Brief pause for UI update
        
        # Process the message with the crew with timeout
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def run_crew():
            try:
                result = crew.kickoff(
                    inputs={
                        "conversation_history": formatted_history,
                        "user_request": message
                    }
                )
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start crew in a thread
        thread = threading.Thread(target=run_crew)
        thread.start()
        
        # Wait for result with progress updates
        timeout_seconds = 180  # 3 minute timeout
        steps = ["Searching documents...", 
                 "Analyzing information...", 
                 "Synthesizing data...",
                 "Formulating response...",
                 "Reviewing information...",
                 "Finalizing answer..."]
        
        # Send progress updates while waiting
        step_index = 0
        loop_count = 0
        
        while thread.is_alive():
            # Check if we have a result
            try:
                status, result = result_queue.get_nowait()
                if status == "success":
                    break
                else:
                    yield f"Error: {result}"
                    return
            except queue.Empty:
                pass
            
            # Update progress message
            loop_count += 1
            if loop_count % 20 == 0:  # Change message every 10 seconds
                step_index = (step_index + 1) % len(steps)
            
            # Display current step with elapsed time
            elapsed = int(time.time() - start_time)
            yield f"Working: {steps[step_index]} (elapsed: {elapsed} seconds)"
            
            time.sleep(0.5)
            
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                yield "Error: The research is taking too long. Please try a more specific question or restart the AI system."
                return
        
        # If we get here and thread is still alive, we timed out
        if thread.is_alive():
            yield "Error: Research timed out. Please try a more specific question or restart the AI system."
            return
            
        # Get the result from the queue
        status, result = result_queue.get()
        if status == "error":
            yield f"Error: {result}"
            return
        
        # Extract the result text
        yield "Finishing: Compiling final answer..."
        result_text = ""
        if hasattr(result, "raw_output"):
            result_text = result.raw_output
        elif hasattr(result, "output"):
            result_text = result.output
        elif hasattr(result, "raw"):
            result_text = result.raw
        else:
            result_text = str(result)
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        # Yield the final result with processing time
        yield f"{result_text}\n\n_(Completed in {elapsed_time:.2f} seconds)_"
        
    except Exception as e:
        yield f"Error: {str(e)}\n\nPlease try asking a different question or restarting the chat interface."

def reset_tools_and_crew():
    """Reset the tools and crew for a fresh start."""
    global search_tool, clarification_tool, agent_crew
    search_tool = None
    clarification_tool = None
    agent_crew = None
    
    # Force reinitialization
    initialize_tools_and_crew()

# Initialize tools and crew at startup
initialize_tools_and_crew()

# Create the Gradio Chat Interface
demo = gr.ChatInterface(
    fn=process_message,
    title="Multi-Agent Research Chat",
    description="Ask questions about your documents and get comprehensive answers from our multi-agent system.",
    theme=gr.themes.Soft(),
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other", "Restart System"],
    save_history=True,
    type="messages",
    examples=[
        "Tell me about Liabilities and Deficiency in 2023",
        "What were the key findings in the Quarterly Financial Report?",
        "Summarize the financial position overview from the first quarter of 2023"
    ],
    chatbot=gr.Chatbot(
        avatar_images=("🧑", "🤖"),
        height=700,
        show_label=False,
        container=True,
        type="messages"
    ),
    analytics_enabled=False
)

# Add callback for flag button 
def handle_flag(flag_value):
    if flag_value == "Restart System":
        # Reset tools and crew
        global search_tool, clarification_tool, agent_crew
        search_tool = None
        clarification_tool = None
        agent_crew = None
        initialize_tools_and_crew()
        return "AI system has been reset. Start a new conversation."
    return f"Flagged as: {flag_value}"

# Connect the flagging callback
if hasattr(demo, 'flagging_callback'):
    demo.flagging_callback = handle_flag

if __name__ == "__main__":
    # Check if vector store exists
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Warning: Vector store not found at {VECTOR_DB_PATH}")
        print("Please process documents before using the chat interface.")
    
    # Launch the chat interface with a port range instead of a fixed port
    try:
        demo.launch(share=False, server_name="127.0.0.1", allowed_paths=["_vector_db"], server_port=7860)
    except OSError:
        print("Port 7860 is already in use. Trying a different port...")
        # Try a range of ports
        for port in range(7861, 7870):
            try:
                demo.launch(share=False, server_name="127.0.0.1", allowed_paths=["_vector_db"], server_port=port)
                break
            except OSError:
                continue 