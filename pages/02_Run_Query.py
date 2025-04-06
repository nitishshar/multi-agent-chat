import streamlit as st
import os
from dotenv import load_dotenv
import logging
from process_crew import analyst_crew
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Run Query | Multi-Agent Research",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_query(query, conversation_history=""):
    """Run a query with the analyst crew"""
    try:
        # Set up the inputs
        inputs = {
            "conversation_history": conversation_history,
            "user_request": query
        }
        
        with st.spinner("Processing your query... This may take a few moments."):
            start_time = time.time()
            
            # Run the crew with the inputs
            result = analyst_crew.run(inputs=inputs)
            
            elapsed_time = time.time() - start_time
            st.success(f"Query processed in {elapsed_time:.2f} seconds!")
        
        return result
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        logger.error(f"Error in run_query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function for the Run Query page"""
    # Render header
    st.title("Run Research Query")
    st.markdown(
        "Run a specific query against your documents using the Research Crew."
    )
    
    # Render sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        st.markdown(
            """
            - **💻 [Home](/)**:  Document Processing
            - **💬 [Chat](/Chat)**: Chat with your documents
            - **🔍 Run Query**: Run a specific query (current)
            """
        )
        
        st.divider()
        
        st.header("About")
        st.markdown(
            """
            This page allows you to run a specific research query using the Research Crew.
            
            The query will be processed by:
            1. **Research Specialist**: Finds relevant information
            2. **Reporting Analyst**: Creates a comprehensive report
            
            Results are saved to a file and displayed on this page.
            """
        )
    
    # Query input form
    st.header("Research Query")
    
    with st.form("query_form"):
        query_input = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="Example: Tell me about Liabilities and Deficiency in 2023",
            help="Be specific in your query for better results."
        )
        
        context_input = st.text_area(
            "Optional conversation context:",
            height=100, 
            placeholder="Any additional context or conversation history to consider.",
            help="This provides additional context for the query."
        )
        
        submit_button = st.form_submit_button("Run Query", type="primary", use_container_width=True)
    
    # Handle form submission
    if submit_button and query_input:
        result = run_query(query_input, context_input)
        
        if result:
            # Display the result
            st.header("Research Results")
            st.markdown(result)
            
            # Option to save to file
            col1, col2 = st.columns([3, 1])
            with col1:
                file_name = st.text_input("Save report as:", value="research_report.md")
            
            with col2:
                if st.button("Save to File", type="primary"):
                    try:
                        with open(file_name, "w", encoding="utf-8") as f:
                            f.write(result)
                        st.success(f"Report saved to {file_name}")
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")

if __name__ == "__main__":
    main() 