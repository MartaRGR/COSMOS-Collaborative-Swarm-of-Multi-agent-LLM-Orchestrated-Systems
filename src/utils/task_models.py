from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class RequiredInput(BaseModel):
    variable: str = Field(description="Name of the variable")
    description: str = Field(description="Description of the variable")
    value: str = Field(description="Value of the variable. If not provided, 'missing' will be used as the value")

class Agent(BaseModel):
    name: str = Field(description="Name of the agent")
    required_inputs: List[RequiredInput] = Field(description="Required inputs for the agent")

class SubtaskDependency(BaseModel):
    id: str = Field(description="Unique identifier for the dependent subtask")
    name: str = Field(description="Name of the dependent subtask")

class Subtask(BaseModel):
    id: str = Field(description="Unique identifier for the subtask")
    name: str = Field(description="Description or name of the subtask")
    order: int = Field(description="Order of execution of the subtask within the task")
    agents: List[Agent] = Field(description="List of the assigned agents capable of resolving the subtask with their required inputs")
    dependencies: List[SubtaskDependency] = Field(description="List of dependencies")
    status: str = Field(default="pending", description="Current status of the subtask (pending, in_progress, completed)")

class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    name: str = Field(description="Description or name of the main task")
    subtasks: List[Subtask] = Field(description="List of subtasks")
    status: str = Field(default="pending", description="Current status of the task")
    user_input: Optional[str] = Field(default=None, description="User-provided input needed for the task")

class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="List of tasks in the plan")

# class MissingInputsResponse(BaseModel):
#     missing_inputs: Dict[str, List[RequiredInput]]


