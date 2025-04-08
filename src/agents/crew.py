from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional
import streamlit as st

class AgentCrewBuilder:
    """
    Builder for creating and managing multi-agent crews.
    """
    def __init__(self, tools: List[BaseTool] = None):
        """
        Initialize with optional tools for the agents.
        
        Args:
            tools: List of tools available to the agents
        """
        self.tools = tools or []
        self.analyst_agent = None
        self.reviewer_agent = None
        self.analyst_task = None
        self.reviewer_task = None
        self.crew = None
    
    def build_analyst_agent(self) -> Agent:
        """
        Build the analyst agent that researches and synthesizes information.
        """
        self.analyst_agent = Agent(
            role="Conversational Research and Analysis Specialist",
            goal="Interpret user requests, perform detailed research using the search tool, and compile comprehensive answers with reference documents.",
            backstory=(
                "With extensive experience in analytical research and data-driven insights, "
                "you excel at breaking down complex queries, conducting targeted searches, "
                "and synthesizing data into clear responses. Your methodical approach ensures "
                "accuracy and comprehensiveness in every answer."
            ),
            verbose=True,
            tools=self.tools
        )
        return self.analyst_agent
    
    def build_reviewer_agent(self) -> Agent:
        """
        Build the reviewer agent that checks and refines the analyst's work.
        """
        self.reviewer_agent = Agent(
            role="Quality Assurance and Final Review Specialist",
            goal="Carefully review and refine the initial responses, correct inaccuracies, and deliver a final answer meeting high standards of quality and clarity.",
            backstory=(
                "With a keen eye for detail and extensive experience in quality control and content review, "
                "you detect inconsistencies, validate sources, and polish answers. "
                "Your meticulous approach ensures every final response is accurate, clear, and insightful."
            ),
            verbose=True
        )
        return self.reviewer_agent
    
    def build_analyst_task(self, description_template: str) -> Task:
        """
        Build the task for the analyst agent with a customizable description template.
        
        Args:
            description_template: Template for the task description with {conversation_history} and {user_request} placeholders
        """
        if not self.analyst_agent:
            self.build_analyst_agent()
            
        self.analyst_task = Task(
            description=description_template,
            expected_output=(
                "A well-structured markdown-formatted answer, detailed and supported by reference documents and data. "
                "DO NOT include triple backticks around the markdown. DO NOT include additional comments. Just respond with markdown."
            ),
            agent=self.analyst_agent
        )
        return self.analyst_task
    
    def build_reviewer_task(self) -> Task:
        """
        Build the task for the reviewer agent.
        """
        if not self.reviewer_agent:
            self.build_reviewer_agent()
            
        if not self.analyst_task:
            raise ValueError("Analyst task must be built before reviewer task")
            
        self.reviewer_task = Task(
            description=(
                "Follow these step-by-step instructions:\n"
                "1. Review the Analyst's answer carefully.\n"
                "2. Identify any errors, inconsistencies, or gaps.\n"
                "3. Refine and correct the response to enhance clarity and accuracy.\n"
                "4. Extract and format all document references used in the response.\n"
                "5. Provide a polished final answer with a references section."
            ),
            expected_output=(
                "A finalized markdown-formatted answer ready for delivery, including:\n"
                "1. The main content with all corrections and improvements\n"
                "2. A 'References' section at the end listing all documents used, formatted as:\n"
                "   ```\n"
                "   ## References\n"
                "   - <a href='source_filename' target='_blank'>Document Title</a>\n"
                "   ```\n"
                "DO NOT include triple backticks around the markdown. DO NOT include additional comments. Just respond with markdown."
            ),
            agent=self.reviewer_agent,
            context=[self.analyst_task]
        )
        return self.reviewer_task
    
    def build_crew(self) -> Crew:
        """
        Build the complete crew with both agents and tasks.
        """
        if not self.analyst_agent or not self.reviewer_agent:
            self.build_analyst_agent()
            self.build_reviewer_agent()
            
        if not self.analyst_task or not self.reviewer_task:
            raise ValueError("Both analyst and reviewer tasks must be built before creating the crew")
            
        self.crew = Crew(
            agents=[self.analyst_agent, self.reviewer_agent],
            tasks=[self.analyst_task, self.reviewer_task],
            verbose=False
        )
        return self.crew
    def build_crew_with_agents(self, agents: List[Agent], tasks: List[Task]) -> Crew:
        """
        Build the crew with a list of agents and tasks.
        """
        self.crew = Crew(agents=agents, tasks=tasks, verbose=False)
        return self.crew
    
    @st.cache_resource  # Cache the crew instance
    def get_or_create_crew(self, description_template: str, tools: List[BaseTool]) -> Crew:
        """
        Static method to get or create a crew instance with caching.
        
        Args:
            description_template: Template for the analyst task description
            tools: List of tools for the agents
            
        Returns:
            Crew instance
        """
        builder = AgentCrewBuilder(tools)
        builder.build_analyst_agent()
        builder.build_reviewer_agent()
        builder.build_analyst_task(description_template)
        builder.build_reviewer_task()
        return builder.build_crew()
    
    @st.cache_resource  # Cache the crew instance
    def get_or_create_crew_with_agents(self, agents: List[Agent], tasks: List[Task]) -> Crew:
        """
        Static method to get or create a crew instance with caching.
        """
        return self.build_crew_with_agents(agents, tasks)
    
    def run_crew(self, conversation_history: str, user_request: str) -> Dict[str, Any]:
        """
        Run the crew with the given inputs.
        
        Args:
            conversation_history: History of the conversation
            user_request: Current user request
            
        Returns:
            Result of the crew's work
        """
        if not self.crew:
            self.build_crew()
            
        inputs = {
            "conversation_history": conversation_history,
            "user_request": user_request
        }
        
        result = self.crew.kickoff(inputs=inputs)
        return result 