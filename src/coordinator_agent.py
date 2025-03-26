import os
import sys
import json
import asyncio
import copy
import random
import uuid
import importlib.util
import concurrent.futures
from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import MermaidDrawMethod

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utils.setup_logger import get_agent_logger
from utils.config_loader import load_default_config
from utils.task_models import TaskPlan
from utils.state_models import OverallState, InputState, OutputState

from registry_creator_agent import AgentRegistry
from swarm_agent import SwarmAgent

# Read OpenAI configuration from environment variables
load_dotenv(find_dotenv())  # read local .env file
# Initialize tasks parser
tasks_parser = PydanticOutputParser(pydantic_object=TaskPlan)


# ==========================
# COMPONENT: ConfigManager
# ==========================
class ConfigManager:
    """Manages the coordinator agent's settings."""
    REQUIRED_FIELDS = {
        "agents": ["registry_file", "default_agent", "folder"],
        "coordinator_agent": ["model_name", "deployment_name", "temperature", "api_version", "auto_register"],
        "crews": ["num_crews", "run_in_parallel"]}

    def __init__(self, user_config=None):
        default_config = load_default_config()
        if user_config:
            default_config.update(user_config)
        self.config = default_config
        self.validate_config()

    def validate_config(self):
        """Validates required configuration fields and assigns attributes dynamically."""
        for section, fields in self.REQUIRED_FIELDS.items():
            sect = self.config.get(section, {})
            for field in fields:
                value = sect.get(field)
                if value is None:
                    raise KeyError(f"Missing required configuration: '{section}.{field}'. Check your configuration file.")
            self.config[section] = {field: sect[field] for field in fields}


# ==========================
# COMPONENT: AgentRegistryLoader
# ==========================
class AgentRegistryLoader:
    """Loads the agent registry from a JSON file, activating the Registry Creator Agent if specified."""
    def __init__(self, config, logger):
        self.registry_file = config.get("agents", {}).get("registry_file")
        self.auto_register = config.get("coordinator_agent", {}).get("auto_register")
        self.logger = logger

    async def load_agents(self):
        if self.auto_register:
            self.logger.info("Calling Registry Creator Agent...")
            try:
                registry = AgentRegistry()
                await registry.run()
            except Exception as e:
                self.logger.warning(f"Failed to create agents file: {e}")
        return self.load_json_file()

    def load_json_file(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    agents = json.load(f)
                    if not isinstance(agents, dict):
                        self.logger.warning("Invalid agents file format. Expected a dictionary.")
                        return {}
                    return agents
            except json.JSONDecodeError:
                self.logger.warning("Error decoding JSON file. Check the format.")
        else:
            self.logger.warning("Agents file not found.")
        return {}

# ==========================
# COMPONENT: TaskPlanner
# ==========================
class TaskPlanner:
    def __init__(self, llm, agents, logger):
        self.llm = llm
        self.agents = agents
        self.logger = logger

    def segment_task(self, user_task, default_agent):
        """
        Segments the user task into subtasks and dependencies, according to the agent registry and the system prompt.
        Returns a structured JSON plan.
        """
        try:
            automatic_task_segmentation = self._intelligent_task_segmentation(user_task)
            checked_tasks = self._task_agents_check(automatic_task_segmentation, default_agent)
            return self._order_tasks(checked_tasks)
        except Exception as e:
            self.logger.error(f"Error segmenting tasks: {e}")
            raise RuntimeError(f"Error segmenting tasks: {e}")

    def _intelligent_task_segmentation(self, user_task):
        """Call OpenAI to segment the user task into subtasks and dependencies."""
        messages = [
            ("system", CoordinatorAgent.SYSTEM_PROMPT),
            ("system", "Available agents: {agents_info}"),
            ("system", "Respond with a JSON that follows this format: {format_instructions}"),
            ("human", "{user_task}")]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | tasks_parser
        return chain.invoke({
            "agents_info": json.dumps(self.agents),
            "format_instructions": tasks_parser.get_format_instructions(),
            "user_task": user_task
        })

    def _task_agents_check(self, automatic_task_segmentation, default_agent):
        """Checks the validity of the agents and fix the plan if needed."""
        for task in automatic_task_segmentation.tasks:
             for subtask in task.subtasks:
                 if not subtask.agents:
                     self.logger.warning(
                         f"Subtask '{subtask.name}' does not have any associated agent. "
                         f"Selecting default agent: {default_agent}.")
                     subtask.agents = [default_agent]
                 subtask.agents = [
                     (agent if agent in self.agents else self._log_and_replace_agent(agent, default_agent))
                     for agent in subtask.agents
                 ]
        return automatic_task_segmentation

    def _log_and_replace_agent(self, agent, default_agent):
        """Logs a message and returns the default agent."""
        self.logger.warning(
            f"Agent '{agent}' not found in the registry. Selecting default agent: {default_agent}."
        )
        return default_agent

    # TODO: definir la lógica para que el LLM pregunte al usuario acerca de los inputs del agente para la tarea
    def _get_agents_task_inputs(self):
        pass

    @staticmethod
    def _order_tasks(task_plan):
        """Orders the subtasks of each task by their order attribute, returning a structured JSON plan."""
        ordered_plan = []
        for task in task_plan.tasks:
            subtasks_by_order = {
                order: [
                    {k: v for k, v in subtask.__dict__.items() if k != 'order'}
                    for subtask in task.subtasks
                    if subtask.order == order
                ]
                for order in set(s.order for s in task.subtasks)
            }
            ordered_plan.append({"id": task.id, "name": task.name, "subtasks": subtasks_by_order})
        return ordered_plan


# ==========================
# COMPONENT: CrewManager
# ==========================
class CrewManager:
    """Creates crews for execution."""
    def __init__(self, num_crews, agents, folder, task_plan, logger):
        self.num_crews = num_crews
        self.agents = agents
        self.folder = folder
        if self.folder not in sys.path:
            sys.path.append(self.folder)
        self.task_plan = task_plan
        self.logger = logger

    def initialize_agents(self):
        """Load agent modules based on task_detail. Avoid loading the same module more than once."""
        try:
            agent_modules = {}
            subtask_list = [subtask for task in self.task_plan for subtasks in task['subtasks'].values() for subtask in subtasks]
            unique_subtask_agents = list({agent for subtask in subtask_list if subtask["agents"] for agent in subtask["agents"]})
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {executor.submit(self._load_agent, agent): agent for agent in unique_subtask_agents}
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        agent_instance = future.result()
                        if agent_instance:
                            agent_modules[agent_name] = agent_instance
                            self.logger.info(f"Agent '{agent_name}' successfully loaded")
                        else:
                            self.logger.error(f"Error loading agent '{agent_name}'")
                    except Exception as exc:
                        self.logger.error(f"Error loading agent '{agent_name}': {exc}")
            return agent_modules

        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")


    def _load_agent(self, agent_name):
        """Loads an agent from its module and returns the agent instance if successful, None otherwise."""
        try:
            module = importlib.import_module(os.path.splitext(agent_name)[0])
            class_name = self.agents[agent_name]["class"]
            if not hasattr(module, class_name):
                self.logger.error(f"Class '{class_name}' not found in agent's file '{agent_name}'.")
                return None
            return getattr(module, class_name)
        except Exception as e:
            self.logger.error(f"Failed to load agent's file '{agent_name}'. Error: {e}")
            raise

    def create_crews_plan(self):
        """Creates heterogeneous crews, assigning agents with specific configurations to each subtask."""
        crews_plan = []
        for crew_id in range(self.num_crews):
            crew = {"id": str(uuid.uuid4()), "name": f"crew_{crew_id + 1}", "task_plan": {"tasks": []}}
            for task in self.task_plan:
                structured_task = {"id": task["id"], "name": task["name"], "subtasks": {}}
                for order, subtasks in task["subtasks"].items():
                    structured_subtasks = []
                    for subtask in subtasks:
                        agent_details = self.get_agent_details(subtask)
                        structured_subtask = {
                            "id": subtask.get("id"),
                            "name": subtask.get("name"),
                            "subtask_dependencies": [dep.id for dep in subtask["dependencies"]],
                            "agent": agent_details}
                        structured_subtasks.append(structured_subtask)
                    structured_task["subtasks"][order] = structured_subtasks
                crew["task_plan"]["tasks"].append(structured_task)
            crews_plan.append(crew)
        return crews_plan

    def get_agent_details(self, subtask):
        """Selects an agent for a subtask and returns its configuration."""
        selected_agent = subtask["agents"][0] if len(subtask["agents"]) == 1 else random.choice(subtask["agents"])
        agent_info = self.agents[selected_agent]
        selected_model = random.choice(agent_info.get("models", {})) if agent_info.get("models") else {}
        hyperparameters = self.randomize_hyperparameters(selected_model.get("hyperparameters", {}))
        return {
            "id": str(uuid.uuid4()),
            "name": selected_agent,
            "model": selected_model.get("name"),
            "hyperparameters": hyperparameters}

    @staticmethod
    def randomize_hyperparameters(hyperparameters):
        randomized = copy.deepcopy(hyperparameters)
        for param, value in randomized.items():
            if isinstance(value, list) and value:
                if len(value) == 2:
                    if all(isinstance(v, int) for v in value):
                        randomized[param] = random.randint(value[0], value[1])
                    elif any(isinstance(v, float) for v in value) and not any(isinstance(v, str) for v in value):
                        randomized[param] = round(random.uniform(value[0], value[1]), 2)
                    else:
                        randomized[param] = random.choice(value)
                elif len(value) > 2:
                    randomized[param] = random.choice(value)
        return randomized


# ==========================
# MAIN COMPONENT: CoordinatorAgent
# ==========================
class CoordinatorAgent:
    """High-level orchestrator that segments tasks, assigns agents, and structures execution plans."""
    SYSTEM_PROMPT = (
        "Segment the user's task into subtasks and define dependencies.\n"
        "- Identify which subtasks can be executed in parallel and which require sequential execution.\n"
        "- Assign ALL agents from the available agent list that have the capability to solve each subtask.\n"
        "- If no suitable agent exists, assign the subtask to \"default_agent\".\n"
        "- Assign unique IDs to each task and subtask using UUID format.\n"
        "- Return a structured JSON plan."
    )

    def __init__(self, user_config=None):
        self.logger = get_agent_logger("CoordinatorAgent")
        self.logger.info("Initializing CoordinatorAgent...")
        # Configuration handling
        self.config_manager = ConfigManager(user_config)
        self.config = self.config_manager.config
        self.crew_manager = None
        self.agentic_modules = None

        # Set attributes from configuration
        ca_config = self.config.get("coordinator_agent", {})
        crews_config = self.config.get("crews", {})
        agents_config = self.config.get("agents", {})

        self.deployment_name = ca_config["deployment_name"]
        self.model_name = ca_config["model_name"]
        self.temperature = ca_config["temperature"]
        self.api_version = ca_config["api_version"]
        self.auto_register = ca_config["auto_register"]
        self.num_crews = crews_config["num_crews"]
        self.run_in_parallel = crews_config["run_in_parallel"]
        self.default_agent = agents_config["default_agent"]
        self.folder = agents_config["folder"]
        self.registry_file = agents_config["registry_file"]

        # Validate numeric values
        if not (0.0 <= self.temperature <= 1.0):
            self.logger.warning("Invalid temperature. Using default value of 0.2")
            self.temperature = 0.2
        if self.num_crews < 1:
            self.logger.warning("Invalid num_crews. Using default value of 5.")
            self.num_crews = 5

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Memory and agent registry loader
        self.memory = MemorySaver()
        self.registry_loader = AgentRegistryLoader(self.config, self.logger)

        # Placeholders for runtime data
        self.user_memory = []
        self.agents = {}

        # Feedback restriction
        self.feedback_attempts = 0
        self.max_feedback_attempts = 2

    async def setup(self):
        """Performs asynchronous initialization steps."""
        await self._initialize_agents()

    async def _initialize_agents(self):
        """Loads the agent registry asynchronously."""
        try:
            self.agents = await self.registry_loader.load_agents()
        except Exception as e:
            self.logger.warning(f"Error loading agents: {e}")
            self.agents = {}

    def _initialize_llm(self):
        """Initializes the language model (LLM)."""
        try:
            return AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                api_version=self.api_version,
                temperature=self.temperature)
        except Exception as e:
            error_message = f"Failed to initialize AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)

    # ==========================
    # TASK SEGMENTATION & CREW CREATION
    # ==========================
    @staticmethod
    def ask_user_task(overall_state: OverallState) -> OverallState:
        input_state: InputState = {"user_task": input("Enter your task: ")}
        overall_state.update(input_state)
        return overall_state

    def task_planner(self, overall_state: OverallState) -> OverallState:
        self.logger.info("Starting task planner...")
        planner = TaskPlanner(self.llm, self.agents, self.logger)
        task_plan = planner.segment_task(overall_state["user_task"], self.default_agent)
        overall_state["task_plan"] = task_plan
        return overall_state

    def initialize_crews(self, overall_state: OverallState) -> OverallState:
        self.logger.info("Initializing crews...")
        if "crews_plan" not in overall_state or not overall_state["crews_plan"]:
            self.logger.info("Creating CrewManager and initializing agents...")
            self.crew_manager = CrewManager(self.num_crews, self.agents, self.folder, overall_state["task_plan"], self.logger)
            self.agentic_modules = self.crew_manager.initialize_agents()
        # Updating crews' plan
        overall_state["crews_plan"] = self.crew_manager.create_crews_plan()
        # Deleting private_states if they exist
        if overall_state.get("private_states"):
            self.logger.warning("Deleting previous private states from memory.")
            overall_state["private_states"] = []
        return overall_state

    def swarm_intelligence(self, overall_state: OverallState, config: RunnableConfig) -> OverallState | None:
        """Function to handle swarm intelligence execution."""
        crew_name = config["metadata"]["langgraph_node"]
        self.logger.info(f"Executing crew: {crew_name}")
        # Getting dict with node crew detail
        crew_detail = next((c for c in overall_state["crews_plan"] if c["name"] == crew_name), {})
        if not crew_detail:
            self.logger.warning(f"Crew {crew_name} not found.")
            return overall_state  # No state modification if crew does not exist
        self.logger.info(f"Crew {crew_name} details: {crew_detail}")
        swarm_agent = SwarmAgent(crew_detail, self.agentic_modules)
        private_state = swarm_agent.run()
        if "private_states" not in overall_state:
            overall_state["private_states"] = []
        return overall_state["private_states"].append(private_state)

    @staticmethod
    def _unify_crews_responses(overall_state):
        """
        Unifies the responses from all crews into a single dictionary.
        """
        unified_responses = []

        for crew in overall_state["private_states"]:
            crew_id = crew["id"]
            tasks = crew.get("task_plan", {}).get("tasks", [])
            for task in tasks:
                task_id = task["id"]
                subtasks = task.get("subtasks", {})
                for subtask_order, subtask_results in subtasks.items():
                    for result in subtask_results:
                        unified_responses.append({
                            "crew_id": crew_id,
                            "crew_name": crew["name"],
                            "task_id": task_id,
                            "task_name": task["name"],
                            "order": subtask_order,
                            "id": result["id"],
                            "name": result["name"],
                            "dependencies": result["subtask_dependencies"],
                            "agent": {k: v for k,v in result["agent"].items() if k in [
                                "hyperparameters", "model", "name", "id"
                            ]},
                            "status": result["agent"]["status"],
                            "result": result["agent"]["result"],
                            "timestamp": result["agent"]["timestamp"],
                        })
        return unified_responses

    def coordinated_response(self, overall_state: OverallState) -> OverallState:
        self.logger.info("Starting coordinated response...")
        # TODO: Implement coordinated response logic
        unified_responses = self._unify_crews_responses(overall_state)
        answer = f"Final results: {json.dumps(unified_responses, indent=4)}"
        overall_state["answer"] = answer
        return overall_state

    def human_feedback(self, overall_state: OverallState) -> OverallState:
        if self.feedback_attempts < self.max_feedback_attempts:
            self.logger.info(f"{self.feedback_attempts}/{self.max_feedback_attempts} feedback attempts.")
            feedback = input("Are you satisfied with the answer? (yes/no): ")
            overall_state["user_feedback"] = feedback
            overall_state["finished"] = feedback.lower().strip('\'\"') == "yes"
            self.feedback_attempts += 1
        else:
            self.logger.warning("Maximum feedback attempts reached. Ending execution.")
            overall_state["finished"] = True
        return overall_state

    # ==========================
    # GRAPH & EXECUTION FLOW
    # ==========================
    def create_graph(self, save_graph=False):
        """Creates the execution graph for coordinator execution."""
        self.logger.info("Creating execution graph...")
        graph = StateGraph(OverallState, output=OutputState)

        # Initial configuration for the start node
        graph.add_node("ask_user_task", self.ask_user_task)
        graph.set_entry_point("ask_user_task")

        graph.add_node("task_planner", self.task_planner)
        graph.add_node("initialize_crews", self.initialize_crews)
        graph.add_node("coordinated_response", self.coordinated_response)

        graph.add_node("human_feedback", self.human_feedback)
        for crew_num in range(self.num_crews):
            node_name = f"crew_{crew_num + 1}"
            graph.add_node(node_name, self.swarm_intelligence)
            graph.add_edge("initialize_crews", node_name)
            graph.add_edge(node_name, "coordinated_response")
        graph.add_edge("ask_user_task", "task_planner")
        graph.add_edge("task_planner", "initialize_crews")
        graph.add_edge("coordinated_response", "human_feedback")
        graph.add_conditional_edges(
            "human_feedback",
            lambda state: END if state.get("finished", False) else "initialize_crews",
            [END, "initialize_crews"]
        )
        compiled_graph = graph.compile(checkpointer=self.memory)
        if save_graph:
            self._save_compiled_graph(compiled_graph)
        return compiled_graph

    def _save_compiled_graph(self, compiled_graph):
        graph_png_data = compiled_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open("coordinator_agent_graph.png", "wb") as file:
            file.write(graph_png_data)
        self.logger.info(f"Coordinator Graph successfully saved")

    def run(self):
        """Execute the entire graph"""
        graph = self.create_graph()
        overall_state = OverallState(finished=False)
        thread = {"configurable": {"thread_id": "1"}}
        while not overall_state.get("finished"):
            for event in graph.stream(overall_state, thread, stream_mode="values"):
                print(event)
                overall_state = event


if __name__ == "__main__":
    async def main():
        # Initialize the coordinator agent with the test registry
        coordinator = CoordinatorAgent(user_config={})
        await coordinator.setup()
        # # Simulate a user task
        # user_task = "Detect objets in the image and summarize its content and tell me the weather in Madrid."
        coordinator.run()
    asyncio.run(main())
