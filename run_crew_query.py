from process_crew import analyst_crew
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Run a specific query with the analyst crew"""
    logger.info("Running specific query with analyst crew")
    
    # Set up the inputs as requested
    inputs = {
        "conversation_history": "",
        "user_request": "Tell me about Liabilities and Deficiency in 2023"
    }
    
    try:
        # Run the crew with the inputs
        logger.info("Calling analyst_crew.run...")
        result = analyst_crew.run(inputs=inputs)
        
        # Debug information about result
        logger.info(f"Result type: {type(result)}")
        if hasattr(result, "__dict__"):
            logger.info(f"Result attributes: {result.__dict__}")
        
        # Print the result
        print("\n=== RESULT ===\n")
        print(result)
        
        # Result is already saved in the run method, but note the file path here
        logger.info("Report saved to crew_report.md")
    except Exception as e:
        logger.error(f"Error in run_crew_query.py: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 