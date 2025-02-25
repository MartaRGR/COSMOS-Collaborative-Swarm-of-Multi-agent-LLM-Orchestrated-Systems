from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import operator
from typing import Annotated
from typing_extensions import TypedDict
from PIL import Image
from langchain_core.runnables.graph import MermaidDrawMethod


class OverallState(TypedDict):
    """
    The Overall state of the LangGraph graph.
    Tracks the global execution state for tasks and subtasks.
    """
    results: Annotated[dict, operator.add]  # Accumulate the results of all subtasks
    pending_tasks: Annotated[dict, operator.add]  # Pending tasks details
    completed_tasks: Annotated[dict, operator.add]  # Completed tasks details
    finished: bool  # Marks whether the entire network has been completed
    traceability: Annotated[dict, operator.add]  # Hierarchical information about nodes and connections


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


class GraphCreatorAgent:
    """Agent that creates a graph from a task plan"""
    def __init__(self, task_plan, n_crews=3):
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("GraphCreatorAgent")
        self.memory = MemorySaver()
        self.task_plan = task_plan
        self.n_crews = n_crews
        self.node_controllers = {}
        self.graph = self.create_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)

    def create_graph(self):
        """Create the network using the provided task plan."""
        self.logger.info("Starting the dynamic creation of the graph...")
        graph = StateGraph(OverallState)

        # Create nodes dynamically based on task's plan
        for task in self.task_plan:
            # TODO: change task by id
            task_crew, task_response = task.task + " - CREW", task.task + " - RESPONSE"
            graph.add_node(task_crew, coordinator_crew)
            self.node_controllers[task_crew] = coordinator_crew
            graph.add_edge(START, task_crew)
            graph.add_node(task_response, coordinator_response)
            self.node_controllers[task_response] = coordinator_response
            graph.add_node("Human Feedback - " + task_response, human_feedback)
            for subtask in task.subtasks:
                graph.add_node(subtask.subtask, swarm_agent)
                self.node_controllers[subtask.subtask] = swarm_agent
                if not subtask.dependencies:
                    graph.add_edge(task_crew, subtask.subtask)
                for dependency in subtask.dependencies:
                    graph.add_edge(dependency, subtask.subtask)
                graph.add_edge(subtask.subtask, task_response)
            graph.add_edge(task_response, "Human Feedback - "+ task_response)
            graph.add_conditional_edges(
                "Human Feedback - " + task_response,
                lambda s: END if s["finished_detection"] else task_crew,
                [END, task_crew]
            )

        self.logger.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph

    @staticmethod
    def is_node_completed(subtask_id: str, private_state: PrivateState) -> bool:
        """
        Check if the subtask has been completed.
        True if the node is completed, False otherwise.
        """
        results = private_state.get("subtask_results", {})
        return results.get(subtask_id, {}).get("status") == "completed"

    def can_activate_node(self, subtask_id: str, private_state: PrivateState) -> bool:
        """
        Check if a node can be activated based on whether all its dependencies are completed.
        True if all dependencies are completed, False otherwise.
        """
        dependencies = private_state.get("dependencies", {}).get(subtask_id, [])
        for dep_id in dependencies:
            if not self.is_node_completed(dep_id, private_state):
                return False  # A dependency is not completed
        return True  # All dependencies are completed

    def activate_dependent_nodes(self, subtask_id: str, private_state: PrivateState):
        """Activate nodes that depend on the given subtask, if their dependencies are resolved."""
        for dependent_id, dependencies in private_state.get("dependencies", {}).items():
            if subtask_id in dependencies and self.can_activate_node(dependent_id, private_state):
                # Create a message or update state to activate the dependent node
                private_state["message_exchange"].setdefault(dependent_id, []).append({
                    "sender": subtask_id,
                    "content": {
                        "type": "activation",
                        "data": f"Node {subtask_id} completed"
                    }
                })
                private_state["subtask_results"][dependent_id] = {
                    "status": "pending",
                    "progress": 0
                }
                self.logger.info(f"Dependent node {dependent_id} activated after completing {subtask_id}")

    def process_node(self, subtask_id: str, private_state: PrivateState):
        """Process an individual node in the graph by resolving its dependencies and executing its task."""
        # Step 1: Verify that the node can be processed (dependencies must be resolved)
        if not self.can_activate_node(subtask_id, private_state):
            self.logger.info(f"Node {subtask_id} cannot be processed yet. Unresolved dependencies.")
            return  # Exit if dependencies are not resolved

        # 2. Check if the node is currently in progress
        current_status = private_state.get("subtask_results", {}).get(subtask_id, {}).get("status")
        if current_status == "in_progress":
            self.logger.info(f"Node {subtask_id} is already in progress. Skipping.")
            return  # Do not process nodes already being worked on

        # 3. Check if the node's dependencies have been resolved
        if not self.can_activate_node(subtask_id, private_state):
            self.logger.info(f"Node {subtask_id} cannot start. Unresolved dependencies.")
            return  # Exit as dependencies are not resolved yet

        controller = self.node_controllers.get(subtask_id)  # Use external mapping
        if not controller:
            self.logger.error(f"Node {subtask_id} does not have an assigned controller.")
            return

        # 5. Mark the node as "in progress"
        private_state["subtask_results"][subtask_id] = {
            "status": "in_progress",
            "progress": 0  # Start with 0% progress
        }
        self.logger.info(f"Node {subtask_id} is now in progress.")

        # 6. Delegate execution to the node's controller
        try:
            controller_result = controller(subtask_id, private_state)  # Call the controller

            # 7. On successful execution, mark the node as completed
            private_state["subtask_results"][subtask_id] = {
                "status": "completed",
                "progress": 100,  # Fully completed
                "result": controller_result  # Store the result returned by the controller
            }
            self.logger.info(f"Node {subtask_id} has been completed successfully.")

            # 8. Activate dependent nodes
            self.activate_dependent_nodes(subtask_id, private_state)

        except Exception as e:
            # Handle any errors that occur within the controller
            private_state["subtask_results"][subtask_id] = {
                "status": "error",
                "progress": 0,
                "error": str(e)
            }
            self.logger.error(f"Error while executing node {subtask_id}: {e}")


def execute_graph(self, private_state: PrivateState):
        """
        Iterate over all nodes in the graph and process them in order based on dependencies.
        :param self: The instance of the class.
        :param private_state: The private state of the graph.
        """
        while True:
            # Fetch all nodes with pending or in-progress status
            pending_nodes = [
                node for node, result in private_state["subtask_results"].items()
                if result.get("status") not in ("completed", "in_progress")
            ]
            if not pending_nodes:
                break  # All nodes are completed, exit the loop
            for node in pending_nodes:
                self.process_node(node, private_state)
        self.logger.info("Graph execution completed.")


if __name__ == "__main__":
    import logging
    from pydantic import BaseModel, Field
    from typing import List

    class Subtask(BaseModel):
        subtask: str = Field(description="Description of the subtask")
        agent: str = Field(description="Name of the assigned agent")
        dependencies: List[str] = Field(default_factory=list, description="List of dependencies")

    class Task(BaseModel):
        task: str = Field(description="Name of the main task")
        subtasks: List[Subtask] = Field(description="List of subtasks")

    class TaskPlan(BaseModel):
        tasks: List[Task] = Field(description="List of tasks")

    tasks = [
        Task(
            task='Analyze Image',
            subtasks=[
                Subtask(subtask='Detect objects in the image', agent='object_detection_agent', dependencies=[]),
                Subtask(subtask='Summarize the content of the image based on detected objects', agent='LLM',
                        dependencies=['Detect objects in the image'])
            ]
        ),
        # Task(
        #     task='Classify image',
        #     subtasks=[
        #         Subtask(subtask='Classify whole image', agent='object_detection_agent', dependencies=[]),
        #         Subtask(subtask='Segment objects', agent='LLM', dependencies=[]),
        #         Subtask(subtask='Classify segmented objects', agent='LLM', dependencies=["Segment objects"]),
        #     ]
        # )
    ]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GraphCreatorAgentExample")

    graph_creator = GraphCreatorAgent(task_plan=tasks)
    file_path = "graph.png"
    graph_png_data = graph_creator.compiled_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(file_path, "wb") as file:
        file.write(graph_png_data)
    print(f"Graph successfully saved in: {file_path}")

    img = Image.open(file_path)
    img.show()


#
#         async def execute_with_swarm(state: TaskState, node_data):
#             """
#             Ejecuta un nodo del grafo delegando su lógica a Swarm Intelligence.
#             Args:
#                 state: El estado actual del grafo.
#                 node_data: Información asociada al nodo que se está ejecutando.
#             """
#             node_id = node_data.get("node_id")
#             self.logger.info(f"Ejecutando nodo {node_id} con Swarm Intelligence...")
#
#             # Delegar la tarea al SwarmAgent
#             result = await self.swarm_agent.run_task(node_id)
#
#             # Guardar el resultado en el estado del grafo
#             state.results[node_id] = result
#             state.completed_tasks.add(node_id)
#             self.logger.info(f"Nodo {node_id} completado con resultado: {result}")
#
#         # Crear nodos dinámicamente según TaskPlan
#         for task in task_plan.tasks:
#             for subtask in task.subtasks:
#                 state_graph.add_node(
#                     subtask.task_id,
#                     lambda state, subtask_data={"node_id": subtask.task_id}: execute_with_swarm(state, subtask_data),
#                 )
#                 # Agregar aristas entre las dependencias
#                 for dependency in subtask.dependencies:
#                     state_graph.add_edge(dependency, subtask.task_id)
#
#         # Agregar bordes START y END automáticamente
#         first_task_id = task_plan.tasks[0].subtasks[0].task_id
#         last_task_id = task_plan.tasks[-1].subtasks[-1].task_id
#         state_graph.add_edge(START, first_task_id)
#         state_graph.add_edge(last_task_id, END)
#
#         self.logger.info(f"Grafo creado con {len(state_graph.nodes)} nodos y {len(state_graph.edges)} aristas.")
#         return state_graph
#
#
# if __name__ == "__main__":
#     # Logger base para debug
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("GraphCreatorExample")
#
#
#     # Ejemplo de implementación del SwarmAgent con una lógica simulada
#     class SwarmAgent:
#         async def run_task(self, node_id):
#             """
#             Simula la ejecución de un nodo utilizando un Swarm Intelligence
#             Args:
#                 node_id: Identificador del nodo.
#             Returns:
#                 str: Resultado simulado de la ejecución.
#             """
#             logger.info(f"[SwarmAgent] Ejecutando tarea {node_id}...")
#             return f"Resultado de {node_id}"
#
#
#     # Crear tareas de ejemplo
#     from pydantic import BaseModel, Field
#     from typing import List
#
#
#     class Subtask(BaseModel):
#         task_id: str = Field(description="ID único de la subtarea")
#         dependencies: List[str] = Field(default_factory=list, description="Lista de dependencias")
#
#
#     class Task(BaseModel):
#         task: str = Field(description="Nombre de la tarea principal")
#         subtasks: List[Subtask] = Field(description="Lista de subtareas")
#
#
#     class TaskPlan(BaseModel):
#         tasks: List[Task] = Field(description="Lista de tareas")
#
#
#     subtasks = [
#         Subtask(task_id="Detect objects", dependencies=[]),
#         Subtask(task_id="Summarize image", dependencies=["Detect objects"]),
#     ]
#     tasks = [Task(task="Analyze Image", subtasks=subtasks)]
#     task_plan = TaskPlan(tasks=tasks)
#
#     # Crear el agente creador y generar el grafo
#     swarm_agent = SwarmAgent()
#     graph_creator = GraphCreatorAgent(swarm_agent=swarm_agent, logger=logger)
#
#     # Generar el grafo
#     graph = graph_creator.create_graph(task_plan)
#
#     # Compilar y ejecutar el grafo
#     compiled_graph = graph.compile()
#     compiled_graph.run()
