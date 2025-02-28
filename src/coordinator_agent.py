import os
import json
import asyncio
import logging
import copy
import random
import uuid
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from registry_creator_agent import AgentRegistry
# from swarm_intelligence_agent import SwarmAgent


class SubtaskDependency(BaseModel):
    id: str = Field(description="Unique identifier for the dependent subtask")
    name: str = Field(description="Name of the dependent subtask")


class Subtask(BaseModel):
    id: str = Field(description="Unique identifier for the subtask")
    name: str = Field(description="Description or name of the subtask")
    order: int = Field(description="Order of execution of the subtask within the task")
    agents: list = Field(description="List of the assigned agents capable of resolving the subtask")
    dependencies: List[SubtaskDependency] = Field(description="List of dependencies")
    status: str = Field(default="pending", description="Current status of the subtask (pending, in_progress, completed)")


class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    name: str = Field(description="Description or name of the main task")
    subtasks: List[Subtask] = Field(description="List of subtasks")
    status: str = Field(default="pending", description="Current status of the task")


class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="List of tasks in the plan")


class OverallState(TypedDict):
    """
    The Overall state of the LangGraph graph.
    Tracks the global execution state for tasks and subtasks.
    """
    user_task: Annotated[dict, operator.add]
    task_plan: Annotated[dict, operator.add]
    results: Annotated[dict, operator.add]  # Accumulate the results of all subtasks
    pending_tasks: Annotated[dict, operator.add]  # Pending tasks details
    completed_tasks: Annotated[dict, operator.add]  # Completed tasks details
    finished: bool  # Marks whether the entire network has been completed
    traceability: Annotated[dict, operator.add]  # Hierarchical information about nodes and connections
    user_feedback: str
    finished_flow: bool


class PrivateState(TypedDict):
    """
    Communication state for individual subtasks in LangGraph.
    Tracks crews, agents, dependencies, and results of subtasks.
    """
    crew_details: Annotated[dict, operator.add]  # Details about the crews (i.e, number, composition...)
    agents: Annotated[dict, operator.add]  # Agents assigned to each subtask
    dependencies: Annotated[dict, operator.add]  # Subtasks' dependencies
    subtask_results: Annotated[dict, operator.add]  # Individual results for each subtask
    # subtask_results indicará si el nodo ha completado la tarea y puede iniciar el siguiente dependiente
    message_exchange: Annotated[dict, operator.add]  # Messages exchanged between subtasks


def coordinator_crew(state: OverallState) -> PrivateState:
    # Actualizará el estado
    pass

def swarm_agent(state: PrivateState) -> PrivateState:
    pass

def coordinator_response(state: PrivateState) -> OverallState:
    pass

def human_feedback(state: OverallState) -> OverallState:
    pass



# Read LLM configuration from environment variables
_ = load_dotenv(find_dotenv())  # read local .env file
tasks_parser = PydanticOutputParser(pydantic_object=TaskPlan)


class CoordinatorAgent:
    """LLM Agent that segments tasks, assigns agents, and structures execution plans."""

    SYSTEM_PROMPT = """Segment the user's task into subtasks and define dependencies.
    - Identify which subtasks can be executed in parallel and which require sequential execution.
    - For each subtask, assign ALL agents from the available agent list that have the capability to solve it.
    - Use only agents from the available agent list to assign them to each subtask.
    - If no suitable agents exists for a subtask, assign the subtask to "LLM".
    - Assign unique IDs to each task and subtask using UUID format.
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
        # self.swarm_agent = SwarmAgent(num_crews, run_in_parallel)

        self.agents_file = agents_file
        self.auto_register = auto_register
        self.agents = []

        self.user_memory = []

        self.memory = MemorySaver()
        # self.graph = self.create_graph()
        # self.compiled_graph = self.graph.compile(checkpointer=self.memory)

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
        self.user_memory.append(HumanMessage(content=user_input))
        return user_input

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
        task_plan = chain.invoke({
            "agents_info": agents_info,
            "user_task": user_task,
            "format_instructions": tasks_parser.get_format_instructions()
        })
        return task_plan

    def create_crews(self, task_plan):
        """Creates heterogeneous crews, assigning agents with specific configurations to each subtask."""
        crews_plan = []

        for crew_id in range(self.num_crews):
            crew = {
                "id": str(uuid.uuid4()),
                "name": f"Crew {crew_id + 1}",
                "task_plan": {
                    "tasks": []
                }
            }

            for task in task_plan.tasks:
                structured_task = {
                    "id": task.id,
                    "name": task.name,
                    "subtasks": []
                }

                for subtask in task.subtasks:
                    subtask_agent = subtask.agents[0] if len(subtask.agents) == 1 else random.choice(subtask.agents)
                    if subtask_agent in self.agents:
                        # Select a random model from the available agent
                        selected_model = random.choice(self.agents[subtask_agent]["models"])
                        selected_hyperparameters = copy.deepcopy(selected_model["hyperparameters"])

                        # Adjust hyperparameters according to their type
                        for param, value in selected_hyperparameters.items():
                            if isinstance(value, list) and value:
                                if len(value) == 2:
                                    if all(isinstance(v, int) for v in value):
                                        selected_hyperparameters[param] = random.randint(value[0], value[1])
                                    elif any(isinstance(v, float) for v in value) and not any(isinstance(v, str) for v in value):
                                        selected_hyperparameters[param] = round(random.uniform(value[0], value[1]), 2)
                                    else:
                                        selected_hyperparameters[param] = random.choice(value)
                                elif len(value) > 2:
                                    selected_hyperparameters[param] = random.choice(value)

                        # Configure the agent with the selected model and hyperparameters
                        agent_details = {
                            "id": str(uuid.uuid4()),
                            "name": subtask_agent,
                            "model": selected_model["name"],
                            "hyperparameters": selected_hyperparameters
                        }
                    else:
                        # If no agent available, assign LLM with random configuration
                        agent_details = {
                            "id": str(uuid.uuid4()),
                            "name": "default-LLM.py",
                            "model": random.choice(["gpt-4o-mini", "gpt-4"]),
                            "hyperparameters": {
                                "temperature": round(random.uniform(0.0, 1.0), 2)
                            }
                        }

                    # Structure the subtask information
                    structured_subtask = {
                        "order": subtask.order,
                        "id": subtask.id,
                        "name": subtask.name,
                        "subtask_dependencies": [dep.id for dep in subtask.dependencies],
                        "agent": agent_details
                    }
                    structured_task["subtasks"].append(structured_subtask)
                crew["task_plan"]["tasks"].append(structured_task)
            crews_plan.append(crew)

        return crews_plan

    def ask_user_task(self, state: OverallState) -> OverallState:
        """Asks the user for a task and returns the updated state."""
        user_task = self.ask_user()
        self.logger.info(f"Received user task: {user_task}")
        state.user_task = user_task
        return state

    def task_planner(self, state: OverallState) -> OverallState:
        """Generates a task plan based on the user's task."""
        task_planner = self.segment_task(state["user_task"])
        state.task_plan = task_planner
        return state

    def initialize_crews(self, state: OverallState) -> OverallState:
        """Initializes the crews for execution."""
        crews_plan = self.create_crews(state["task_plan"])
        state.crews_plan = crews_plan
        return state

    def coordinated_response(self, state: OverallState) -> OverallState:
        """Function to handle coordinated response."""
        pass

    def human_feedback(self, state: OverallState) -> OverallState:
        """Asks the user for answer's feedback"""
        feedback = input("Are you satisfied with the answer? (yes/no): ")
        state["user_feedback"] = feedback
        state["finished_flow"] = feedback.lower() == "yes"
        return state

    def create_graph(self):
        """Create the LangGraph graph for coordinator execution."""
        self.logger.info("Starting the dynamic creation of the graph...")
        graph = StateGraph(OverallState)

        # Initial configuration for the start node
        graph.add_node("ask_user_task", self.ask_user_task)
        graph.set_entry_point("ask_user_task")

        graph.add_node("task_planner", self.task_planner)
        graph.add_node("initialize_crews", self.initialize_crews)
        graph.add_node("coordinated_response", self.coordinated_response)
        graph.add_node("human_feedback", self.human_feedback)

        for crew_num in range(self.num_crews):
            graph.add_node(f"crew_{crew_num + 1}", swarm_agent)
            graph.add_edge("initialize_crews", f"crew_{crew_num + 1}")
            graph.add_edge(f"crew_{crew_num + 1}", "coordinated_response")

        graph.add_edge("ask_user_task", "task_planner")
        graph.add_edge("task_planner", "initialize_crews")
        graph.add_edge("initialize_crews", "coordinated_response")
        graph.add_edge("coordinated_response", "human_feedback")
        graph.add_conditional_edges(
            "human_feedback",
            lambda s: END if s["finished_flow"] else "initialize_crews",
            [END, "initialize_crews"]
        )
        # graph.add_edge("human_feedback", END)

        # # Create nodes dynamically based on task's plan
        # for crew in self.crews_plan:
        #     # TODO: change task by id
        #     task_crew, task_response = task.task + " - CREW", task.task + " - RESPONSE"
        #     graph.add_node(task_crew, coordinator_crew)
        #     self.node_controllers[task_crew] = coordinator_crew
        #     graph.add_edge(START, task_crew)
        #     graph.add_node(task_response, coordinator_response)
        #     self.node_controllers[task_response] = coordinator_response
        #     graph.add_node("Human Feedback - " + task_response, human_feedback)
        #     for subtask in task.subtasks:
        #         graph.add_node(subtask.subtask, swarm_agent)
        #         self.node_controllers[subtask.subtask] = swarm_agent
        #         if not subtask.dependencies:
        #             graph.add_edge(task_crew, subtask.subtask)
        #         for dependency in subtask.dependencies:
        #             graph.add_edge(dependency, subtask.subtask)
        #         graph.add_edge(subtask.subtask, task_response)
        #     graph.add_edge(task_response, "Human Feedback - " + task_response)
        #     graph.add_conditional_edges(
        #         "Human Feedback - " + task_response,
        #         lambda s: END if s["finished_detection"] else task_crew,
        #         [END, task_crew]
        #     )
        #
        # self.logger.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph.compile(checkpointer=self.memory)

    def run(self):
        """Executes the user interaction and structured task segmentation."""
        user_task = self.ask_user()
        self.logger.info(f"Received user task: {user_task}")

        task_plan = self.segment_task(user_task)
        self.logger.info(f"\n Generated Task Plan:\n{json.dumps(task_plan, indent=4)}")

        crews_plan = self.create_crews(task_plan)
        self.logger.info(f"\n Generated Crews Plan:\n{json.dumps(crews_plan, indent=4)}")

        # Build and initialize the graph
        # graph = self.create_graph(crews_plan)



if __name__ == "__main__":
    async def main():
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
        crews_plan = coordinator.create_crews(plan)
        print("\nGenerated Crews' Plan:")
        print(crews_plan)

        graph = coordinator.create_graph()
        graph_png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        file_path = "graph_coordinator.png"
        with open(file_path, "wb") as file:
            file.write(graph_png_data)
        print(f"Graph successfully saved in: {file_path}")
    asyncio.run(main())




# # TODO: PROBADO HASTA AQUI
# async def execute_with_swarm_intelligence(self, subtasks):
#     """
#     Executes a task plan using swarm intelligence.
#     Delegates subtasks to crews to work in parallel or sequentially.
#     """
#     self.logger.info("Starting swarm intelligence execution.")
#
#     # Handle dependencies and assign tasks to crews
#     crews = self._create_crews(subtasks)
#     self.logger.info(f"Created {len(crews)} crews to resolve subtasks.")
#
#     # Execute crews based on parallel or sequential mode
#     if self.run_in_parallel:
#         self.logger.info("Executing crews in parallel.")
#         await asyncio.gather(*(self._execute_crew(crew) for crew in crews))
#     else:
#         self.logger.info("Executing crews sequentially.")
#         for crew in crews:
#             await self._execute_crew(crew)
#
#     # Consolidate results from all crews
#     results = self._consolidate_task_results(crews)
#     self.logger.info(f"Swarm task execution completed. Results: {results}")
#     return results
#
# def _create_crews(self, subtasks):
#     """
#     Divides the subtasks into crews for subsequent execution.
#     """
#     # Create crews based on the defined number (self.num_crews)
#     crews = [[] for _ in range(self.swarm_agent.num_crews)]
#     for i, subtask in enumerate(subtasks):
#         crews[i % len(crews)].append(subtask)
#     return crews
#
# async def _execute_crew(self, crew):
#     """
#     Executes a crew (list of subtasks) while respecting dependencies.
#     """
#     for subtask in crew:
#         # Check if the subtask's dependencies are resolved:
#         if not self._check_dependencies_resolved(subtask):
#             self.logger.warning(
#                 f"Skipping subtask {subtask.name} due to unresolved dependencies."
#             )
#             continue
#
#         # Execute the subtask
#         await self._execute_subtask(subtask)
#
# def _check_dependencies_resolved(self, subtask):
#     """
#     Checks if the dependencies of a subtask are resolved.
#     """
#     if not subtask.dependencies:
#         return True
#     for dependency in subtask.dependencies:
#         if not dependency.get("resolved", False):
#             return False
#     return True
#
# async def _execute_subtask(self, subtask):
#     """
#     Executes a single subtask.
#     """
#     self.logger.info(f"Executing subtask {subtask.name}")
#     await asyncio.sleep(1)  # Simulates actual execution
#     subtask.resolved = True  # Marks the subtask as resolved
#     self.logger.info(f"Subtask {subtask.name} execution completed.")
#
# def _consolidate_task_results(self, crews):
#     """
#     Consolidates the results of the subtasks executed across all crews.
#     """
#     results = {
#         "resolved_subtasks": sum(len(crew) for crew in crews),
#         "details": [
#             {"name": subtask.name, "resolved": getattr(subtask, "resolved", False)}
#             for crew in crews for subtask in crew
#         ]
#     }
#     return results
