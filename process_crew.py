from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv
import logging
import time
import os
import sys
from src.vector_store.store_builder import VectorStoreBuilder

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

# Define tools for the agents
@tool
def search_documents(query: str) -> str:
    """
    Search the vector database for relevant information based on a query.
    
    Args:
        query: The search query string
    
    Returns:
        A string containing the relevant information found
    """
    try:
        logger.info(f"Searching for: {query}")
        store = VectorStoreBuilder(storage_path=vector_db_path)
        results = store.similarity_search(query, k=5)
        
        if not results:
            return "No relevant information found."
        
        output = "Here is the relevant information I found:\n\n"
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source_filename', 'Unknown source')
            output += f"--- Result {i} (from {source}) ---\n"
            output += f"{doc.page_content}\n\n"
        
        return output
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return f"Error searching documents: {str(e)}"

class ResearchCrew:
    """A crew for researching and analyzing markdown documents"""
    
    def create_agents(self):
        """Create the agents for the crew"""
        logger.info("Creating agents for the crew...")
        
        researcher = Agent(
            role="Research Specialist",
            goal="Find and analyze relevant information from documents",
            backstory="You are an expert researcher with keen attention to detail. Your specialty is extracting valuable insights from documents.",
            verbose=True,
            allow_delegation=True,
            tools=[search_documents]
        )
        
        analyst = Agent(
            role="Reporting Analyst",
            goal="Create comprehensive reports based on research findings",
            backstory="You are a skilled analyst who can transform raw research into actionable insights and clear reports.",
            verbose=True,
            allow_delegation=False,
            tools=[search_documents]
        )
        
        return [researcher, analyst]
    
    def create_tasks(self, agents, user_request):
        """Create the tasks for the crew"""
        logger.info("Creating tasks for the crew...")
        
        research_task = Task(
            description=f"Research the following user request using the search_documents tool: '{user_request}'. Search for key themes and important information related to this request. Try multiple search queries to gather comprehensive information.",
            expected_output="A comprehensive analysis of the key information found in the documents, with specific details and insights organized by theme.",
            agent=agents[0]  # Researcher
        )
        
        reporting_task = Task(
            description=f"Create a detailed report answering the user request: '{user_request}'. The report should be well-structured with clear sections, including an executive summary, key findings, and recommendations.",
            expected_output="A well-structured markdown report summarizing the key insights from the documents",
            agent=agents[1],  # Analyst
            context=[research_task]  # This task depends on the research task
        )
        
        return [research_task, reporting_task]
    
    def run(self, inputs=None):
        """Run the crew"""
        try:
            logger.info("Starting CrewAI research process...")
            
            # Get user request from inputs or use default
            user_request = "Analyze the documents and provide insights"
            conversation_history = ""
            
            if inputs and isinstance(inputs, dict):
                user_request = inputs.get("user_request", user_request)
                conversation_history = inputs.get("conversation_history", "")
                logger.info(f"User request: {user_request}")
                if conversation_history:
                    logger.info("Conversation history is available")
            
            start_time = time.time()
            
            # Create agents
            agents = self.create_agents()
            
            # Create tasks with user request
            tasks = self.create_tasks(agents, user_request)
            
            # Create crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Run the crew
            logger.info("Running the crew...")
            
            # Add any inputs for the kickoff
            kickoff_inputs = {}
            if conversation_history:
                kickoff_inputs["conversation_history"] = conversation_history
            
            if kickoff_inputs:
                result = crew.kickoff(inputs=kickoff_inputs)
            else:
                result = crew.kickoff()
            
            # Handle CrewOutput object (newer versions of crewai)
            result_text = ""
            if hasattr(result, "raw_output"):
                # For newer versions of crewai that return CrewOutput
                result_text = result.raw_output
                logger.info("Using CrewOutput.raw_output")
            elif hasattr(result, "output"):
                # Another possible attribute name
                result_text = result.output
                logger.info("Using CrewOutput.output")
            elif isinstance(result, str):
                # For older versions that return string
                result_text = result
                logger.info("Using string result directly")
            else:
                # Fallback - convert to string
                result_text = str(result)
                logger.info("Used str() to convert result to string")
            
            # Save the result to a file
            output_file = "crew_report.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result_text)
            logger.info(f"Report saved to {output_file}")
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Crew process completed in {elapsed_time:.2f} seconds")
            
            return result_text
            
        except Exception as e:
            logger.error(f"Error running the crew: {str(e)}")
            sys.exit(1)

# Create a global analyst_crew instance for direct access
analyst_crew = ResearchCrew()

def main():
    try:
        # Run with specific inputs example
        inputs = {
            "conversation_history": "",
            "user_request": "Tell me about Liabilities and Deficiency in 2023"
        }
        result = analyst_crew.run(inputs=inputs)
        print("\n--- Result Summary ---")
        if isinstance(result, str):
            print(result[:300] + "..." if len(result) > 300 else result)
        else:
            print("Result is not a string. Type:", type(result))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 