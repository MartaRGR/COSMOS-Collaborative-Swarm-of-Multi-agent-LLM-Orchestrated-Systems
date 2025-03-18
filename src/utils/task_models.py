from typing import List, Optional
from pydantic import BaseModel, Field


class MissingInput(BaseModel):
    object: str = Field(description="Name of the missing object")
    description: str = Field(description="Description of the missing object")
    critically: str = Field(description="Level of criticality of the missing object")
    input: Optional[str] = Field(description="Input required to complete the task")

class MissingInputResponse(BaseModel):
    task: str = Field(description="Description of the task")
    missing_inputs: List[MissingInput] = Field(description="List of the missing inputs necessary to complete the task")

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
    user_input: List[MissingInput] = Field(description="List of the missing inputs necessary to complete the subtask")

class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    name: str = Field(description="Description or name of the main task")
    subtasks: List[Subtask] = Field(description="List of subtasks")
    status: str = Field(default="pending", description="Current status of the task")
    user_input: List[MissingInput] = Field(description="List of the missing inputs necessary to complete the task")

class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="List of tasks in the plan")
