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
    message_exchange: Annotated[dict, operator.add]  # Messages exchanged between subtasks


def coordinator_crew(state: OverallState) -> PrivateState:
    pass

def swarm_agent(state: PrivateState) -> PrivateState:
    pass

def coordinator_response(state: PrivateState) -> OverallState:
    pass

def human_feedback(state: OverallState) -> OverallState:
    pass


class GraphCreatorAgent:
    """Agent that creates a graph from a task plan"""
    def __init__(self, task_plan):
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("GraphCreatorAgent")
        self.memory = MemorySaver()
        self.task_plan = task_plan
        self.graph = self.create_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)

    def create_graph(self):
        """Create the network using the provided task plan.
        Args:
            task_plan: Task plan containing nodes and dependencies as input.
        Returns:
            StateGraph: Task graph ready to run.
        """
        self.logger.info("Starting the dynamic creation of the graph...")
        graph = StateGraph(OverallState)

        # Create nodes dynamically based on task's plan
        for task in self.task_plan:
            task_crew, task_response = task.task + " - CREW", task.task + " - RESPONSE"
            graph.add_node(task_crew, coordinator_crew)
            graph.add_edge(START, task_crew)
            graph.add_node(task_response, coordinator_response)
            graph.add_node("Human Feedback - " + task_response, human_feedback)
            for subtask in task.subtasks:
                graph.add_node(subtask.subtask, swarm_agent)
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
        Task(
            task='Clasify image',
            subtasks=[
                Subtask(subtask='Classify whole image', agent='object_detection_agent', dependencies=[]),
                Subtask(subtask='Segment objects', agent='LLM', dependencies=[]),
                Subtask(subtask='Classify segmented objects', agent='LLM', dependencies=["Segment objects"]),
            ]
        )
    ]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GraphCreatorAgentExample")

    graph_creator = GraphCreatorAgent(logger=logger, task_plan=tasks)
    file_path = "graph.png"
    graph_png_data = graph_creator.compiled_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(file_path, "wb") as file:
        file.write(graph_png_data)
    print(f"Gráfico guardado exitosamente en: {file_path}")

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
