

class ConfigManager:
    REQUIRED_FIELDS = {
        "agents": ["registry_file", "default_agent"],
        "coordinator_agent": ["model_name", "deployment_name", "temperature", "api_version", "auto_register"],
        "crews": ["num_crews", "run_in_parallel"]
    }
    DEFAULT_VALUES = {
        "model_name": "gpt-4o-mini",
        "deployment_name": "gpt-4o-mini",
        "temperature": 0.2,
        "auto_register": True,
        "num_crews": 5,
        "run_in_parallel": True,
    }

    def __init__(self, user_config=None):
        default_config = load_default_config()
        if user_config:
            default_config.update(user_config)
        self.config = default_config
        self.validate_config()

    def validate_config(self):
        for section, fields in self.REQUIRED_FIELDS.items():
            sect = self.config.get(section, {})
            for field in fields:
                value = sect.get(field, self.DEFAULT_VALUES.get(field))
                if value is None:
                    raise ValueError(f"Missing required configuration: '{section}.{field}'")
                sect[field] = value
            self.config[section] = sect

    def get(self, section, key):
        return self.config.get(section, {}).get(key)

# ==========================
# COMPONENT: AgentRegistryLoader
# ==========================
class AgentRegistryLoader:
    def __init__(self, config, logger):
        self.registry_file = config.get("agents", "registry_file")
        self.auto_register = config.get("coordinator_agent", "auto_register")
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

    def segment_task(self, user_task):
        messages = [
            ("system", CoordinatorAgent.SYSTEM_PROMPT),
            ("system", f"Available agents: {json.dumps(self.agents)}"),
            ("system", f"Respond with a JSON that follows this format: {tasks_parser.get_format_instructions()}"),
            ("human", user_task)
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | tasks_parser
        return chain.invoke({})

# ==========================
# COMPONENT: CrewManager
# ==========================
class CrewManager:
    def __init__(self, num_crews, default_agent, agents, logger):
        self.num_crews = num_crews
        self.default_agent = default_agent
        self.agents = agents
        self.logger = logger

    def create_crews(self, task_plan):
        crews_plan = []
        for crew_id in range(self.num_crews):
            crew = {"id": str(uuid.uuid4()), "name": f"crew_{crew_id + 1}", "task_plan": {"tasks": []}}
            for task in task_plan.tasks:
                structured_task = {"id": task.id, "name": task.name, "subtasks": []}
                for subtask in task.subtasks:
                    agent_details = self.get_agent_details(subtask)
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

    def get_agent_details(self, subtask):
        # Selecciona un agente de la subtask (si hay uno, o uno aleatorio)
        subtask_agent = subtask.agents[0] if len(subtask.agents) == 1 else random.choice(subtask.agents)
        if subtask_agent not in self.agents or subtask_agent == "default_agent":
            self.logger.warning(f"Agent '{subtask_agent}' not found. Using default agent: {self.default_agent}")
            agent_name = self.default_agent
        else:
            agent_name = subtask_agent
        if agent_name in self.agents:
            selected_model = random.choice(self.agents[agent_name]["models"])
            selected_hyperparameters = self.randomize_hyperparameters(selected_model["hyperparameters"])
            return {
                "id": str(uuid.uuid4()),
                "name": agent_name,
                "class": self.agents[agent_name]["class"],
                "model": selected_model["name"],
                "hyperparameters": selected_hyperparameters
            }
        # Configuración genérica si aún no se encuentra el agente
        return {
            "id": str(uuid.uuid4()),
            "name": "default-LLM.py",
            "class": "defaultLlm",
            "model": random.choice(["gpt-4o-mini", "gpt-4"]),
            "hyperparameters": {"temperature": round(random.uniform(0.0, 1.0), 2)}
        }

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


------------------------------------ ANTERIOR -------------------------------------
class CoordinatorAgent:
    """LLM Agent that segments tasks, assigns agents, and structures execution plans."""

    # ==========================
    # CONFIGURATION & INITIALIZATION
    # ==========================
    SYSTEM_PROMPT = """Segment the user's task into subtasks and define dependencies.
    - Identify which subtasks can be executed in parallel and which require sequential execution.
    - For each subtask, assign ALL agents from the available agent list that have the capability to solve it.
    - Use only agents from the available agent list to assign them to each subtask.
    - If no suitable agents exists for a subtask, assign the subtask to "default_agent".
    - Assign unique IDs to each task and subtask using UUID format.
    - Return a structured plan in JSON format with tasks, subtasks, dependencies, and assigned agents."""

    def __init__(self, user_config=None):
        """Initializes the coordinator agent."""
        self.logger = get_agent_logger("CoordinatorAgent")
        self.logger.info("Initializing CoordinatorAgent...")
        self.config = self._load_and_validate_config(user_config)
        self._set_class_attributes()
        self.user_memory = []
        self.memory = MemorySaver()

    async def setup(self):
        """Asynchronous method to complete agent's initialization"""
        await self.initialize()

    # ==========================
    # CONFIGURATION HANDLING
    # ==========================
    def _set_class_attributes(self):
        """Sets additional attributes and ensures correct data types."""
        # Temperature validation
        if not (0.0 <= self.temperature <= 1.0):
            self.logger.warning(
                f"Invalid 'temperature' value. Using default: {self.DEFAULT_MISCONFIGURED_VALUES['temperature']}.")
            self.temperature = self.DEFAULT_MISCONFIGURED_VALUES["temperature"]

        # Crews number validation
        if self.num_crews < 1:
            self.logger.warning(
                f"Invalid 'num_crews' value. Using default: {self.DEFAULT_MISCONFIGURED_VALUES['num_crews']}.")
            self.num_crews = self.DEFAULT_MISCONFIGURED_VALUES["num_crews"]

        # Auto_register validation
        if not isinstance(self.auto_register, bool):
            self.logger.warning(
                f"Invalid 'auto_register' value. Using default: {self.DEFAULT_MISCONFIGURED_VALUES['auto_register']}."
            )
            self.auto_register = self.DEFAULT_MISCONFIGURED_VALUES["auto_register"]

        # Run_in_parallel validation
        self.run_in_parallel = self.config.get("coordinator_agent", {}).get("run_in_parallel", self.DEFAULT_MISCONFIGURED_VALUES["run_in_parallel"])
        if not isinstance(self.run_in_parallel, bool):
            self.logger.warning(
                f"Invalid 'run_in_parallel' value. Using default: {self.DEFAULT_MISCONFIGURED_VALUES['run_in_parallel']}."
            )
            self.run_in_parallel = self.DEFAULT_MISCONFIGURED_VALUES["run_in_parallel"]

        # LLM validation
        self.llm = self._initialize_llm()

        # Agent's final configuration
        self.logger.info(
            f"""
    Final Configuration (some of the values may differ from the provided defaults due to user configuration or validation constraints): 
        Agents
            -registry_file: {self.registry_file}
            -default_agent: {self.default_agent} 
        Coordinator Agent
            -model_name: {self.model_name}
            -deployment_name: {self.deployment_name}
            -api_version: {self.api_version}
            -temperature: {self.temperature} 
            -auto_register: {self.auto_register}
        Crews
            -num_crews: {self.num_crews}
            -run_in_parallel: {self.run_in_parallel}
            """
        )

    def _initialize_llm(self):
        """Function to initialize the LLM based on the specified model and deployment name."""
        try:
            return AzureChatOpenAI(
                deployment_name=self.deployment_name,
                model_name=self.model_name,
                api_version=self.api_version,
                temperature=self.temperature
            )
        except Exception as e:
            error_message = f"Failed to initialize AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)



    # ==========================
    # AGENT LOADING & REGISTRATION
    # ==========================
    async def initialize(self):
        """
        Asynchronous initializer for the CoordinatorAgent.
        Handles asynchronous calls to load the agents.
        """
        try:
            self.agents = await self.load_agents()
        except FileNotFoundError:
            self.logger.warning(f"Agents registry file not found: {self.registry_file}. Agent list will be empty.")
            self.agents = []
        except Exception as e:
            self.logger.warning(f"Error loading agents from file: {e}. Agent list will be empty.")
            self.agents = []

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

    def load_json_file(self):
        """Reads and parses the JSON file if it exists."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
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

    # ==========================
    # TASK SEGMENTATION & CREW CREATION
    # ==========================
    def ask_user(self):
        """Requests the initial task from the user."""
        user_input = input("Enter your task: ")
        self.user_memory.append(HumanMessage(content=user_input))
        return user_input

    def segment_task(self, user_task):
        """Asks the LLM to segment the task into structured subtasks."""
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("system", "Available agents: {agents_info}"),
            ("system", "Respond with a JSON that follows this format: {format_instructions}"),
            ("human", "{user_task}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | tasks_parser
        return chain.invoke({
            "agents_info": json.dumps(self.agents),
            "format_instructions": tasks_parser.get_format_instructions(),
            "user_task": user_task
        })

    def create_crews(self, task_plan):
        """Creates heterogeneous crews, assigning agents with specific configurations to each subtask."""
        crews_plan = []
        for crew_id in range(self.num_crews):
            crew = {
                "id": str(uuid.uuid4()),
                "name": f"crew_{crew_id + 1}",
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
                    agent_details = self._get_agents_detail(subtask_agent)
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

    def _get_agents_detail(self, agent_name):
        """"
        Returns the agent details, selecting a random model and hyperparameters.
        If the agent is not found, falls back to the default agent.
        """
        if agent_name not in self.agents or agent_name == "default_agent":
            self.logger.warning(f"Agent '{agent_name}' not found. Using default agent: {self.default_agent}")
            agent_name = self.default_agent

        if agent_name in self.agents:
            # Select random model and hyperparameters
            selected_model = random.choice(self.agents[agent_name]["models"])
            selected_hyperparameters = self._randomize_hyperparameters(selected_model["hyperparameters"])
            return {
                "id": str(uuid.uuid4()),
                "name": agent_name,
                "class": self.agents[agent_name]["class"],
                "model": selected_model["name"],
                "hyperparameters": selected_hyperparameters
            }

        # If the agent is still unavailable, get a generic configuration
        return {
            "id": str(uuid.uuid4()),
            "name": "default-LLM.py",
            "class": "defaultLlm",
            "model": random.choice(["gpt-4o-mini", "gpt-4"]),
            "hyperparameters": {
                "temperature": round(random.uniform(0.0, 1.0), 2)
            }
        }

    @staticmethod
    def _randomize_hyperparameters(hyperparameters):
        """Generates random values for hyperparameters based on their type."""
        randomized_hyperparameters = copy.deepcopy(hyperparameters)
        for param, value in randomized_hyperparameters.items():
            if isinstance(value, list) and value:
                if len(value) == 2:
                    if all(isinstance(v, int) for v in value):
                        randomized_hyperparameters[param] = random.randint(value[0], value[1])
                    elif any(isinstance(v, float) for v in value) and not any(isinstance(v, str) for v in value):
                        randomized_hyperparameters[param] = round(random.uniform(value[0], value[1]), 2)
                    else:
                        randomized_hyperparameters[param] = random.choice(value)
                elif len(value) > 2:
                    randomized_hyperparameters[param] = random.choice(value)

    # ==========================
    # EXECUTION FLOW & FEEDBACK
    # ==========================
    def ask_user_task(self, state: OverallState) -> OverallState:
        """Asks the user for a task and returns the updated state."""
        user_task = self.ask_user()
        self.logger.info(f"Received user task: {user_task}")
        state["user_task"] = user_task
        return state

    def task_planner(self, state: OverallState) -> OverallState:
        """Generates a task plan based on the user's task."""
        self.logger.info("Starting task planner...")
        task_planner = self.segment_task(state["user_task"])
        state["task_plan"] = task_planner
        return state

    def initialize_crews(self, state: OverallState) -> OverallState:
        """Initializes the crews for execution."""
        self.logger.info("Initializing crews...")
        crews_plan = self.create_crews(state["task_plan"])
        state["crews_plan"] = crews_plan
        return state

    def swarm_intelligence(self, state: OverallState, config: RunnableConfig) -> PrivateState:
        """Function to handle swarm intelligence execution."""
        crew_name = config["metadata"]["langgraph_node"] # get node name
        self.logger.info(f"Executing crew: {crew_name}")
        # Getting dict with node crew detail
        crew_detail = next((crew_detail for crew_detail in state["crews_plan"] if crew_detail["name"] == crew_name), {})
        if not crew_detail:
            self.logger.warning(f"Crew {crew_name} not found in crews plan.")
            return PrivateState(crew_details={
                "crew_name": crew_name,
                "crew_status": {"status": "error", "detail": "Crew not found in crews plan."},
                "crew_results": {}
            })
        # Executing SwarmAgent
        swarm_agent = SwarmAgent(crew_detail)
        crew_results = swarm_agent.run()
        return PrivateState(crew_details={
            "crew_name": crew_name,
            "crew_status": {"status": "completed" if crew_results else "error", "detail": "Processing completed."},
            "crew_results": crew_results,
        })

    def coordinated_response(self, state: PrivateState) -> OverallState:
        """Function to handle coordinated response."""
        self.logger.info("Starting coordinated response...")
        pass

    def human_feedback(self, state: OverallState) -> OverallState:
        """Asks the user for answer's feedback"""
        feedback = input("Are you satisfied with the answer? (yes/no): ")
        state["user_feedback"] = feedback
        state["finished"] = feedback.lower() == "yes"
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
            node_name = f"crew_{crew_num + 1}"
            graph.add_node(node_name, self.swarm_intelligence)
            graph.add_edge("initialize_crews", node_name)
            graph.add_edge(node_name, "coordinated_response")

        graph.add_edge("ask_user_task", "task_planner")
        graph.add_edge("task_planner", "initialize_crews")
        graph.add_edge("initialize_crews", "coordinated_response")
        graph.add_edge("coordinated_response", "human_feedback")
        graph.add_conditional_edges(
            "human_feedback",
            lambda s: END if s["finished"] else "initialize_crews",
            [END, "initialize_crews"]
        )

        return graph.compile(checkpointer=self.memory)

    def run(self):
        """Executes the user interaction and structured task segmentation."""
        # Build and initialize the graph
        graph = self.create_graph()
        thread = {"configurable": {"thread_id": "1"}}
        state = OverallState(finished=False)
        while not state["finished"]:
            for event in graph.stream(state, thread, stream_mode="values"):
                print(event)
                state = event

