from typing_extensions import TypedDict, Annotated
import operator

class InputState(TypedDict):
    """
    Input state for individual tasks in LangGraph.
    """
    user_task: str

class OutputState(TypedDict):
    """
    Output state for individual tasks in LangGraph.
    """
    answer: str

class OverallState(TypedDict):
    """
    The Overall state of the LangGraph graph.
    Tracks the global execution state for tasks and subtasks.
    """
    user_task: str
    task_plan: list
    crews_plan: list
    private_states: Annotated[list, operator.add]
    answer: str  # Final answer
    finished: bool  # Marks whether the entire network has been completed
    user_feedback: str # TODO: se puede tener un detalle más específico del feedback del usuario y pasárselo al crewManager

class PrivateState(TypedDict):
    """
    Communication state for individual subtasks in LangGraph.
    Tracks crews, agents, dependencies, and results of subtasks.
    """
    id: str
    name: str
    task_plan: dict
