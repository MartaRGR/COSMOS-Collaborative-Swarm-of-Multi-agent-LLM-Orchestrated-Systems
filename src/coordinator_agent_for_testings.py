import os
import sys
import datetime

import numpy as np
import pytz
import json
import asyncio
import copy
import random
import uuid
import importlib.util
import concurrent.futures
from collections import defaultdict

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
from utils.task_models import TaskPlan, AggResponse
from utils.state_models import OverallStateTesting, InputState, OutputState

from registry_creator_agent import AgentRegistry
from swarm_agent import SwarmAgent


# Read OpenAI configuration from environment variables
load_dotenv(find_dotenv())  # read local .env file
# Initialize tasks and response parsers
tasks_parser = PydanticOutputParser(pydantic_object=TaskPlan)
response_parser = PydanticOutputParser(pydantic_object=AggResponse)


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
            automatic_task_segmentation = self._intelligent_task_segmentation(user_task, default_agent)
            checked_tasks = self._missing_inputs_check(automatic_task_segmentation.tasks)
            return self._order_tasks(checked_tasks)
        except Exception as e:
            self.logger.error(f"Error segmenting tasks: {e}")
            raise RuntimeError(f"Error segmenting tasks: {e}")

    def _intelligent_task_segmentation(self, user_task, default_agent):
        """Call OpenAI to segment the user task into subtasks and dependencies."""
        messages = [
            ("system", CoordinatorAgent.SYSTEM_PROMPT["task_segmentation"]),
            ("system", "Available agents: {agents_info}"),
            ("system", "Respond with a JSON that follows this format: {format_instructions}"),
            ("human", "{user_task}")]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | tasks_parser
        return chain.invoke({
            "default_agent": default_agent,
            "agents_info": json.dumps(self.agents),
            "format_instructions": tasks_parser.get_format_instructions(),
            "user_task": user_task
        })

    # TODO: ahora mismo suponemos que las descripciones de los inputs de 2 o más agentes
    #  que hacen lo mismo son iguales, sino habría que sustituir esta lógica por un LLM
    @staticmethod
    def _group_missing_inputs(tasks):
        """Groups missing inputs by description and returns a dictionary of missing variables grouped by description."""
        missing_vars = defaultdict(list)
        for task in tasks:
            for subtask in task.subtasks:
                for agent in subtask.agents:
                    for req_input in agent.required_inputs:
                        if req_input.value == "missing":
                            missing_vars[req_input.description].append(req_input.variable)
        return missing_vars

    @staticmethod
    def _distribute_user_inputs(user_inputs, tasks):
        for task in tasks:
            for subtask in task.subtasks:
                for agent in subtask.agents:
                    for req_input in agent.required_inputs:
                        if req_input.value == "missing":
                            description = req_input.description
                            if description in user_inputs:
                                req_input.value = user_inputs[description]
        return tasks

    def _missing_inputs_check(self, tasks):
        """Function to check if the user task contains all necessary inputs. If not, ask the user to provide them."""
        missing_vars = self._group_missing_inputs(tasks)
        if missing_vars:
            print("The system needs additional information to proceed:\n")
            user_inputs = {}
            for description, variable in missing_vars.items():
                user_inputs[description] = input(
                    f"- {description} ({variable}):\nPlease provide the necessary details: ").strip('\'\"')
            return self._distribute_user_inputs(user_inputs, tasks)
        return tasks

    @staticmethod
    def _order_tasks(task_plan):
        """Orders the subtasks of each task by their order attribute, returning a structured JSON plan."""
        ordered_plan = []
        for task in task_plan:
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
            unique_subtask_agents = list({agent.name for subtask in subtask_list if subtask["agents"] for agent in subtask["agents"]})
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
        agent_info = self.agents[selected_agent.name]
        selected_model = random.choice(agent_info.get("models", {})) if agent_info.get("models") else {}
        hyperparameters = self.randomize_hyperparameters(selected_model.get("hyperparameters", {}))
        return {
            "id": str(uuid.uuid4()),
            "name": selected_agent.name,
            "required_inputs": {r_input.variable:r_input.value for r_input in selected_agent.required_inputs},
            "model": selected_model.get("name"),
            "hyperparameters": hyperparameters}

    @staticmethod
    def randomize_hyperparameters(hyperparameters):
        randomized = copy.deepcopy(hyperparameters)
        for param, value in randomized.items():
            if isinstance(value, list) and value:
                if len(value) == 1:
                    randomized[param] = value[0]
                elif len(value) == 2:
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
    SYSTEM_PROMPT = {
        "task_segmentation": (
            "Segment the user's task into subtasks and define dependencies.\n"
            "- Identify which subtasks can be executed in parallel and which require sequential execution.\n"
            "- Assign ALL agents from the available agent list that have the capability to solve each subtask.\n"
            "- If no suitable agent exists, assign the subtask to the default agent \"{default_agent}\".\n"
            "- Assign unique IDs to each task and subtask using UUID format.\n"
            "- For each assigned agent, extract the required inputs based on its metadata.\n"  
            "- If the user's task already provides a required input, extract its value and include it in the response.\n"  
            "- If the required input is \"task_definition\", set its value as the name of the subtask.\n"
            "- If the required input is not provided, mark it explicitly with \"value\": \"missing\".\n"
            "- Return a structured JSON plan."
        ),
        "response_aggregation": (
            "You are a reasoning engine that aggregates responses from multiple experts (agents).\n"
            "Each agent may return text, numbers, classifications, or mixed-format answers.\n"
            "Your job is to generate a final, unified answer for the task by integrating and synthesizing the information from all responses, applying the specified aggregation method: \"{aggregation_method}\".\n\n"
            "There are three aggregation methods:\n"
            "1. plurality_vote:\n"
            "   - Select the components of the response that occurs most frequently or has the highest consensus.\n"
            "   - In case of ties, explain the situation and justify your choice.\n\n"
            "2. weighted_average:\n"
            "   - If responses are numerical, compute the weighted average.\n"
            "   - If some weights are missing, assume a default weight of 1.\n"
            "   - If the values are not purely numeric, extract and average the values that make sense.\n\n"
            "3. llm_synthesis:\n"
            "   - Identify the common points of agreement among the responses.\n"
            "   - Summarize any discrepancies and explain why differences may exist.\n"
            "   - Propose a final answer that integrates the best of each perspective, in clear and natural language.\n\n"
            "IMPORTANT:\n"
            "- Do not repeat the list of individual responses or raw data.\n"
            "- Your final answer MUST be a coherent and unified text that explains the situation in a way that a human can understand.\n"
            "- For example, if the task involves object detection in an image, the answer could be:\n"
            "  'The image contains a person detected with 88.59% confidence and a sandwich detected with X% confidence. However, the classification of the sandwich is unclear, as some experts identified it as a burrito or donut with lower confidence levels.'\n\n"
            "Explain in detail how you applied the method and present the final answer clearly.\n\n"
            "ADDITIONALLY:\n"
            "- Report the level of agreement between the agents.\n"
            "- Provide traceability by indicating which agents (by model name) and crews contributed to each element of the final answer.\n"
            "- Highlight the items with low agreement or significant disagreement.\n"
            "- Optionally, assign a final confidence level (e.g., high, medium, low) based on the inter-agent consensus.\n"
            "- Your reasoning and final output should help improve the interpretability and trust in the final result."
        )
    }

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
        self.max_feedback_attempts = 100000

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
            model_config = {
                "deployment_name": self.deployment_name,
                "model_name": self.model_name,
                "api_version": self.api_version
            }
            if "mini" not in self.model_name:
                model_config["temperature"] = self.temperature
            else:
                self.logger.info(f"Model '{self.model_name}' does not allow 'temperature' configuration.")
            return AzureChatOpenAI(**model_config)
        except Exception as e:
            error_message = f"Failed to initialize AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)

    # ==========================
    # TASK SEGMENTATION & CREW CREATION
    # ==========================

    def task_planner(self, overall_state: OverallStateTesting) -> OverallStateTesting:
        self.logger.info("Starting task planner...")
        planner = TaskPlanner(self.llm, self.agents, self.logger)
        overall_state["task_plan"] = planner.segment_task(overall_state["user_task"], self.default_agent)
        self.logger.info(f"Task Plan: \n {overall_state['task_plan']}")
        return overall_state

    def initialize_crews(self, overall_state: OverallStateTesting) -> OverallStateTesting:
        self.logger.info("Initializing crews...")
        if "crews_plan" not in overall_state or not overall_state["crews_plan"]:
            self.logger.info("Creating CrewManager and initializing agents...")
            self.crew_manager = CrewManager(self.num_crews, self.agents, self.folder, overall_state["task_plan"], self.logger)
            self.agentic_modules = self.crew_manager.initialize_agents()
        # Updating crews' plan
        overall_state["crews_plan"] = self.crew_manager.create_crews_plan()
        self.logger.info(f"Crews Plan: \n {overall_state['crews_plan']}")
        # Deleting private_states if they exist
        if overall_state.get("private_states"):
            self.logger.warning("Deleting previous private states from memory.")
            overall_state["private_states"] = []
        return overall_state

    def swarm_intelligence(self, overall_state: OverallStateTesting, config: RunnableConfig) -> OverallStateTesting | None:
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
        """Unifies the responses from all crews into a single dictionary."""
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
                            # "order": subtask_order,
                            # "id": result["id"],
                            "name": result["name"],
                            # "dependencies": result["subtask_dependencies"],
                            # "agent": {k: v for k,v in result["agent"].items() if k in [
                            #     "hyperparameters", "model", "name", "id"
                            # ]},
                            # "status": result["agent"]["status"],
                            "result": result["agent"]["result"],
                            # "timestamp": result["agent"]["timestamp"],
                        })
        return unified_responses

    def _llm_aggregation(self, task_results, aggregation_method):
        """Uses LLM to aggregate responses according to the specified method."""

        def stringify_result(res):
            result = res["result"]
            if isinstance(result, (dict, list)):
                try:
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                except Exception:
                    result_str = str(result)
            else:
                result_str = str(result)
            # Escaping keys to avoid ChatPromptTemplate input errors
            return result_str.replace("{", "{{").replace("}", "}}")

        formatted_results = "\n".join(
            f"- Crew {res['crew_name']} responded for Subtask '{res['name']}':\n{stringify_result(res)}"
            for res in task_results
        )
        messages = [
            ("system", self.SYSTEM_PROMPT["response_aggregation"]),
            # ("system", "Respond with a human-readable explanation and a final answer."),
            ("system", "Respond with a JSON that follows this format: {format_instructions}"),
            ("human", f"Aggregation method: {aggregation_method}\n\nAgent responses:\n{formatted_results}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | response_parser
        try:
            return chain.invoke({
                "aggregation_method": aggregation_method,
                "format_instructions": response_parser.get_format_instructions()
            })
        except Exception as e:
            self.logger.error(f"Aggregation LLM failed: {e}")
            return {"response": "Aggregation failed", "explanation": str(e)}

    def _aggregate_responses(self, unified_responses, aggregation_methods):
        tasks_answers = []
        for agg_method in aggregation_methods:
            for task_id in {response["task_id"] for response in unified_responses}:
                task_name = {x["task_name"] for x in unified_responses if x["task_id"]==task_id}.pop()
                task_results = [response for response in unified_responses if response["task_id"] == task_id]
                task_answer = self._llm_aggregation(task_results, agg_method).model_dump()
                self.logger.info(f"Aggregation Method: {agg_method} - Answer {task_answer}")
                tasks_answers.append({"task_id": task_id, "task_name": task_name, "aggregation_method": agg_method, "result": task_answer})
        return tasks_answers

    def coordinated_response(self, overall_state: OverallStateTesting) -> OverallStateTesting:
        self.logger.info("Starting coordinated response...")
        unified_responses = self._unify_crews_responses(overall_state)
        agg_response = self._aggregate_responses(unified_responses, overall_state["aggregation_methods"])
        overall_state["answer"] = agg_response

        # Showing the final answer
        self.logger.info(">>> FINAL ANSWER <<< \n")
        for task in overall_state['answer']:
            self.logger.info(f">>> Task - {task['task_name']} ({task['task_id']})\n")
            self.logger.info(f">>> Response: \n {task['result'].get('response')}\n")
            self.logger.info(f">>> Explanation: \n {task['result'].get('explanation')}\n\n")

        # TODO: check how to return output state
        # OutputState: agg_response
        return overall_state

    def human_feedback(self, overall_state: OverallStateTesting, save_result=True) -> OverallStateTesting:
        if self.feedback_attempts < self.max_feedback_attempts:
            self.logger.info(f"{self.feedback_attempts}/{self.max_feedback_attempts} feedback attempts.")
            feedback = 'yes' #input("Are you satisfied with the answer? (yes/no): ")
            overall_state["user_feedback"] = feedback
            overall_state["finished"] = feedback.lower().strip('\'\"') == "yes"
            self.feedback_attempts += 1
        else:
            self.logger.warning("Maximum feedback attempts reached. Ending execution.")
            overall_state["finished"] = True
        if overall_state["finished"] and save_result:
            os.makedirs(overall_state["base_path"], exist_ok=True)
            file_name_overall_state = os.path.join(overall_state["base_path"], f"{overall_state['file_name']}_overall_state.json")
            file_name_responses = os.path.join(overall_state["base_path"], f"{overall_state['file_name']}_responses.json")

            if os.path.exists(file_name_overall_state):
                with open(file_name_overall_state, "r", encoding='utf-8') as f:
                    try:
                        overall_list = json.load(f)
                    except json.JSONDecodeError:
                        overall_list = []
            else:
                overall_list = []

            if os.path.exists(file_name_responses):
                with open(file_name_responses, "r", encoding='utf-8') as f:
                    try:
                        responses_list = json.load(f)
                    except json.JSONDecodeError:
                        responses_list = []
            else:
                responses_list = []

            overall_list.append(overall_state)

            data_response = {
                "id": overall_state["task_id"],
                "response": overall_state["answer"]
            }
            responses_list.append(data_response)

            with open(file_name_overall_state, "w", encoding='utf-8') as f:
                json.dump(overall_list, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Saved list of overall states to {file_name_overall_state}.")

            with open(file_name_responses, "w", encoding='utf-8') as f:
                json.dump(responses_list, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Saved list of responses to {file_name_responses}.")

            # with open(file_name_overall_state, "a") as f:
            #     f.write(json.dumps(overall_state, default=str) + "\n")
            # self.logger.info(f"Appended overall state to {file_name_overall_state}.")
            #
            # data_response = {
            #     "id": overall_state["task_id"],
            #     "response": overall_state["answer"]
            # }
            # with open(file_name_responses, "a") as f:
            #     f.write(json.dumps(data_response, default=str) + "\n")
            # self.logger.info(f"Appended response to {file_name_responses}.")
        return overall_state

    # ==========================
    # GRAPH & EXECUTION FLOW
    # ==========================
    def create_graph(self, save_graph=False):
        """Creates the execution graph for coordinator execution."""
        self.logger.info("Creating execution graph...")
        graph = StateGraph(OverallStateTesting, output=OutputState)

        graph.add_node("task_planner", self.task_planner)
        # Initial configuration for the start node
        graph.set_entry_point("task_planner")
        graph.add_node("initialize_crews", self.initialize_crews)
        graph.add_node("coordinated_response", self.coordinated_response)

        graph.add_node("human_feedback", self.human_feedback)
        for crew_num in range(self.num_crews):
            node_name = f"crew_{crew_num + 1}"
            graph.add_node(node_name, self.swarm_intelligence)
            graph.add_edge("initialize_crews", node_name)
            graph.add_edge(node_name, "coordinated_response")
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
        with open("../pruebas/coordinator_agent_for_testings_graph.png", "wb") as file:
            file.write(graph_png_data)
        self.logger.info(f"Coordinator Graph successfully saved")

    def run(self, file_name, base_path, task, aggregation_methods):
        """Execute the entire graph"""
        # TODO: change commented task_instruction
        prompt_answer_instruction = "\n IMPORTANT:\n - At the end of your explanation, **include a line exactly like this**:\n <<< FINAL ANSWER: X >>> \n where X is the option chosen (if options are available) or the final result."
        task_instruction = task["task_instruction"] + "\n Question: \n" + task["input_text"] + prompt_answer_instruction
        # task_instruction = task
        # Assign dynamic thread_id
        thread_id = str(uuid.uuid4())
        graph = self.create_graph()
        overall_state = OverallStateTesting(
            user_task=task_instruction,
            file_name=file_name,
            base_path=base_path,
            task_id=task["id"], # TODO: change when testing #"prueba",
            aggregation_methods=aggregation_methods,
            finished=False
        )
        thread = {"configurable": {"thread_id": thread_id}}
        while not overall_state.get("finished"):
            for event in graph.stream(overall_state, thread, stream_mode="values"):
                overall_state = event


if __name__ == "__main__":
    async def main():

        # Define aggregation methods
        aggregation_methods = ["llm_synthesis", "plurality_vote", "weighted_average"]
        # Load tasks from JSON file
        file_name = "logiqa.json"
        root_path = r"C:\Users\rt01306\OneDrive - Telefonica\Desktop\Doc\Doctorado\TESIS\Python_Code\prueba_langraph\pruebas\numerical_logit\logic_math_dataset"
        json_path = os.path.join(root_path, file_name)
        base_path = r"C:\Users\rt01306\OneDrive - Telefonica\Desktop\Doc\Doctorado\TESIS\Python_Code\prueba_langraph\pruebas\numerical_logit\logic_math_responses"
        with open(json_path, 'r') as f:
            tasks_input = json.load(f)
        # Initialize the coordinator agent with the test registry
        coordinator = CoordinatorAgent(user_config={})
        await coordinator.setup()
        for task in tasks_input:
            # task = "Dime el resultado de la operación matemática 2+2. Solo el resultado, sin explicaciones ni pasos intermedios. Solo el número"
            coordinator.run(file_name.split(".")[0], base_path, task, aggregation_methods)
    asyncio.run(main())
