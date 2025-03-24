from typing_extensions import TypedDict
from typing import Annotated
import operator

from .task_models import TaskPlan

class OverallState(TypedDict):
    """
    The Overall state of the LangGraph graph.
    Tracks the global execution state for tasks and subtasks.
    """
    user_task: str
    task_plan: list
    agentic_modules: dict
    crews_plan: list
    # results: Annotated[dict, operator.add]  # Accumulate the results of all subtasks
    # pending_tasks: Annotated[dict, operator.add]  # Pending tasks details
    # completed_tasks: Annotated[dict, operator.add]  # Completed tasks details
    finished: bool  # Marks whether the entire network has been completed
    # traceability: Annotated[dict, operator.add]  # Hierarchical information about nodes and connections
    user_feedback: str


class PrivateState(TypedDict):
    """
    Communication state for individual subtasks in LangGraph.
    Tracks crews, agents, dependencies, and results of subtasks.
    """
    task_details: Annotated[dict, operator.add] # Task details
    crew_details: Annotated[dict, operator.add]  # Details about the crews (i.e, number, composition...)
    agents: Annotated[dict, operator.add]  # Agents assigned to each subtask
    dependencies: Annotated[dict, operator.add]  # Subtasks' dependencies
    subtask_results: Annotated[dict, operator.add]  # Individual results for each subtask
    message_exchange: Annotated[dict, operator.add]  # Messages exchanged between subtasks