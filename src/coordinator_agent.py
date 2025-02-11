import os
import json
import asyncio
import logging

from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv, find_dotenv

from registry_creator_agent import AgentRegistry
from swarm_intelligence_agent import SwarmAgent


class Subtask(BaseModel):
    subtask: str = Field(description="Description of the subtask")
    agent: str = Field(description="Name of the assigned agent")
    dependencies: List[str] = Field(default_factory=list, description="List of dependencies")


class Task(BaseModel):
    task: str = Field(description="Name of the main task")
    subtasks: List[Subtask] = Field(description="List of subtasks")


class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="List of tasks")


# Read LLM configuration from environment variables
_ = load_dotenv(find_dotenv())  # read local .env file
tasks_parser = PydanticOutputParser(pydantic_object=TaskPlan)


class CoordinatorAgent:
    """LLM Agent that segments tasks, assigns agents, and structures execution plans."""

    SYSTEM_PROMPT = """Segment the user's task into subtasks and define dependencies.
    - Identify which tasks can be executed in parallel and which require sequential processing.
    - Use only agents from the available agent list.
    - If no agents match a subtask, the LLM will handle it.
    - Return a structured plan in JSON format with tasks, subtasks, dependencies, and assigned agents."""

    def __init__(
            self, model_name=None, temperature=0.2, num_crews=5, run_in_parallel=True,
            agents_file="agents_registry.json", auto_register=True
    ):
        """
        Initializes the coordinator agent.
        Args:
            model_name: LLM model name.
            temperature: Controls the AI's creativity.
            agents_file: JSON file with available agents.
            auto_register: If True, automatically calls AgentRegistry if the agents file is missing.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("CoordinatorAgent")

        # Validate temperature
        if not (0.0 <= temperature <= 1.0):
            self.logger.warning("Temperature must be between 0.0 and 1.0. Using default value 0.2.")
            temperature = 0.2
        self.temperature = temperature

        # Set default model name if not provided and validate it
        self.model_name = model_name or os.getenv("OPENAI_DEPLOYMENT")
        if not self.model_name:
            error_message = """
                Model name not provided. You can define OPENAI_DEPLOYMENT environment variable or pass 
                a model name.\n Using default model: gpt-4o-mini.
            """
            self.logger.warning(error_message)
            self.model_name = "gpt-4o-mini"
        try:
            self.llm = AzureChatOpenAI(
                deployment_name=self.model_name,
                model_name=self.model_name,
                temperature=self.temperature
            )
        except Exception as e:
            error_message = f"Failed to initialize AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)

        if not (1 <= num_crews <= 10):
            self.logger.warning("Crews number must be between 1 and 10. Using default value 3.")
            num_crews = 3
        self.num_crews = num_crews

        self.run_in_parallel = run_in_parallel  # Attribute to control execution mode
        self.swarm_agent = SwarmAgent(num_crews, run_in_parallel)

        self.memory = []  # Conversation history
        self.agents_file = agents_file
        self.auto_register = auto_register
        self.agents = []

    async def initialize(self):
        """
        Asynchronous initializer for the CoordinatorAgent.
        Handles asynchronous calls to load the agents.
        """
        try:
            self.agents = await self.load_agents()
        except FileNotFoundError:
            self.logger.warning(f"Agents file not found: {self.agents_file}. Agent list will be empty.")
            self.agents = []
        except Exception as e:
            self.logger.warning(f"Error loading agents from file: {e}. Agent list will be empty.")
            self.agents = []

    def load_json_file(self):
        """Reads and parses the JSON file if it exists."""
        if os.path.exists(self.agents_file):
            try:
                with open(self.agents_file, "r") as f:
                    agents = json.load(f)
                    if not isinstance(agents, dict):
                        self.logger.warning(f"""
                            Invalid agents file format: Expected a dictionary, got {type(agents)}. 
                            Agent list will be empty.
                        """)
                        return []
                    return agents
            except json.JSONDecodeError:
                self.logger.warning("Error decoding JSON file. Check the format. Agent list will be empty.")
                return []
        else:
            self.logger.warning("Agents file not found. Agent list will be empty.")
            return []

    async def load_agents(self):
        """Loads the list of agents from a JSON file or creates it if needed."""
        # If auto_register is enabled, try to create the agents file
        if self.auto_register:
            self.logger.info("Calling Registry Creator Agent...")
            try:
                registry = AgentRegistry()
                await registry.run()
                return self.load_json_file()
            except Exception as e:
                self.logger.warning(f"Failed to create agents file: {e}. Agent list will be empty.")
                return []
        return self.load_json_file()

    def ask_user(self):
        """Requests the initial task from the user."""
        user_input = input("Enter your task: ")
        self.memory.append(HumanMessage(content=user_input))
        return user_input

    def find_relevant_agents(self, subtask):
        """Finds relevant agents for a given subtask."""
        relevant_agents = [
            agent for agent in self.agents if agent["description"].lower() in subtask.lower()
        ]
        return relevant_agents

    def segment_task(self, user_task):
        """Asks the LLM to segment the task into structured subtasks."""
        agents_info = json.dumps(self.agents)  # Convert available agents to string
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("system", "Available agents: {agents_info}"),
            ("system", "Respond with a JSON that follows this format: {format_instructions}"),
            ("human", "{user_task}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | tasks_parser
        response = chain.invoke({
            "agents_info": agents_info,
            "user_task": user_task,
            "format_instructions": tasks_parser.get_format_instructions()
        })
        return response

    def generate_subtask_plan(self, tasks):
        """Segments the task and assigns agents where applicable."""
        # Extracting subtasks from structured_plan
        subtasks_plan = [subtask for tasks in tasks.tasks for subtask in tasks.subtasks]
        self.logger.info(f"Extracted {len(subtasks_plan)} subtasks from the task plan.")
        return subtasks_plan

    async def execute_with_swarm_intelligence(self, subtasks):
        """
        Executes a task plan using swarm intelligence.
        Delegates subtasks to crews to work in parallel or sequentially.
        """
        self.logger.info("Starting swarm intelligence execution.")

        # Handle dependencies and assign tasks to crews
        crews = self._create_crews(subtasks)
        self.logger.info(f"Created {len(crews)} crews to resolve subtasks.")

        # Execute crews based on parallel or sequential mode
        if self.run_in_parallel:
            self.logger.info("Executing crews in parallel.")
            await asyncio.gather(*(self._execute_crew(crew) for crew in crews))
        else:
            self.logger.info("Executing crews sequentially.")
            for crew in crews:
                await self._execute_crew(crew)

        # Consolidate results from all crews
        results = self._consolidate_task_results(crews)
        self.logger.info(f"Swarm task execution completed. Results: {results}")
        return results

    def _create_crews(self, subtasks):
        """
        Divides the subtasks into crews for subsequent execution.
        """
        # Create crews based on the defined number (self.num_crews)
        crews = [[] for _ in range(self.swarm_agent.num_crews)]
        for i, subtask in enumerate(subtasks):
            crews[i % len(crews)].append(subtask)
        return crews

    async def _execute_crew(self, crew):
        """
        Executes a crew (list of subtasks) while respecting dependencies.
        """
        for subtask in crew:
            # Check if the subtask's dependencies are resolved:
            if not self._check_dependencies_resolved(subtask):
                self.logger.warning(
                    f"Skipping subtask {subtask.name} due to unresolved dependencies."
                )
                continue

            # Execute the subtask
            await self._execute_subtask(subtask)

    def _check_dependencies_resolved(self, subtask):
        """
        Checks if the dependencies of a subtask are resolved.
        """
        if not subtask.dependencies:
            return True
        for dependency in subtask.dependencies:
            if not dependency.get("resolved", False):
                return False
        return True

    async def _execute_subtask(self, subtask):
        """
        Executes a single subtask.
        """
        self.logger.info(f"Executing subtask {subtask.name}")
        await asyncio.sleep(1)  # Simulates actual execution
        subtask.resolved = True  # Marks the subtask as resolved
        self.logger.info(f"Subtask {subtask.name} execution completed.")

    def _consolidate_task_results(self, crews):
        """
        Consolidates the results of the subtasks executed across all crews.
        """
        results = {
            "resolved_subtasks": sum(len(crew) for crew in crews),
            "details": [
                {"name": subtask.name, "resolved": getattr(subtask, "resolved", False)}
                for crew in crews for subtask in crew
            ]
        }
        return results


    def run(self):
        """Executes the user interaction and structured task segmentation."""
        user_task = self.ask_user()
        task_plan = self.segment_task(user_task)
        self.logger.info(f"\n Generated Task Plan:\n{json.dumps(task_plan, indent=4)}")
        return task_plan


if __name__ == "__main__":
    async def main():
        # Create a sample agents registry file for testing
        # test_agents = [
        #     {"name": "ObjectDetectionAgent", "description": "Detects objects in images"},
        #     {"name": "TextSummarizationAgent", "description": "Summarizes long text"},
        #     {"name": "ImageClassificationAgent", "description": "Classifies images into categories"}
        # ]
        #
        # # Save test agents to a JSON file
        # with open("test_agents_registry.json", "w") as f:
        #     json.dump(test_agents, f, indent=4)

        # Initialize the coordinator agent with the test registry
        coordinator = CoordinatorAgent(agents_file="agents_registry.json", auto_register=True)
        await coordinator.initialize()
        print(f"Agentes inicializados: {coordinator.agents}")

        # Simulate a user task
        user_task = "Analyze an image and summarize its content."

        # Generate a plan
        plan = coordinator.segment_task(user_task)
        # Display the output
        print("\nGenerated Plan:")
        print(plan)

        coordinator.generate_subtask_plan(plan)

    asyncio.run(main())
