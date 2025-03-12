import os
import sys
import importlib.util
import concurrent.futures
import re

from utils.setup_logger import get_agent_logger


class SwarmAgent:
    def __init__(self, crew_detail, default_agent="default-LLM.py", agents_folder="agents"):
        """
        Initializes the SwarmAgent with the details of the crew and its tasks.

        Arguments:
            crew_detail (dict): Detail of the tasks assigned to the group.
        """
        self.logger = get_agent_logger("SwarmAgent")
        self.logger.info("Initializing...")

        self.crew_detail = crew_detail
        self.default_agent = default_agent
        self.agents_folder = agents_folder
        self.tasks = crew_detail.get("task_plan", {}).get("tasks", [])
        self.agent_modules = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Load agent modules based on crew_detail. Avoid loading the same module more than once."""
        try:
            subtask_agent_names = [
                subtask['agent'].get("name") for task in self.tasks for subtask in task['subtasks'] if 'agent' in subtask
            ]
            for name in subtask_agent_names:
                # Avoid loading the same module twice
                if name in self.agent_modules:
                    self.logger.info(f"Agent '{name}' already loaded. Using cache.")
                    continue

                # Loading agent using the _load_agent method
                agent_instance = self._load_agent(name)
                if agent_instance:
                    self.agent_modules[name] = agent_instance
                    self.logger.info(f"Agent '{name}' successfully loaded")
                else:
                    self.logger.error(f"Error loading agent '{name}'")

        except Exception as e:
            self.logger.error(f"Error during agents' initialization: {e}")

    def _load_agent(self, agent_name):
        """Dynamically loads the specified agent module."""
        try:
            if not agent_name or agent_name == "":
                self.logger.warning(f"No agent name specified. Falling back to default agent: {self.default_agent}")
                agent_name = self.default_agent

            if self.agents_folder not in sys.path:
                sys.path.append(self.agents_folder)
            module_name = os.path.splitext(agent_name)[0]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._load_agent_from_module, module_name)
                try:
                    return future.result()
                except ModuleNotFoundError:
                    self.logger.warning(f"Module '{module_name}' not found. Falling back to default agent: '{self.default_agent}'")
                    if module_name != self.default_agent:
                        return self._load_agent(self.default_agent)
                    self.logger.error(f"Default agent '{self.default_agent}' also not found.")
                    raise ImportError(f"Cannot load module '{module_name}' or default agent '{self.default_agent}'.")
        except Exception as e:
            raise RuntimeError(f"Error loading agent '{agent_name}': {e}")

    def _load_agent_from_module(self, module_name):
        """Loads a module dynamically and return the agent class."""
        try:
            module = importlib.import_module(module_name)
            class_name = self.to_camel_case_with_agent(module_name)
            if not hasattr(module, class_name):
                self.logger.error(f"Class '{class_name}' not found in module '{module_name}'.")
                return None
            self.logger.info(f"Successfully loaded agent class '{class_name}' from module '{module_name}'.")
            return getattr(module, class_name)
        except Exception as e:
            self.logger.error(f"Failed to load module '{module_name}'. Error: {e}")
            raise

    @staticmethod
    def to_camel_case_with_agent(name):
        """
        Converts a string to CamelCase with "Agent" appended if not already present.
        Used for agent class names.
        """
        words = re.split(r'[^a-zA-Z]', os.path.splitext(name)[0])
        camel_case_name = ''.join(word.capitalize() for word in words if word)
        if not camel_case_name.endswith("Agent"):
            camel_case_name += "Agent"
        return camel_case_name

    def _process_subtask(self, subtask):
        """Process a specific subtask by running the corresponding agent."""
        self.logger.info(f"Processing subtask {subtask['id']} - {subtask['name']}...")
        try:
            # Load and configure the agent
            agent_info = subtask.get("agent")
            agent = self._load_agent(agent_info["name"])
            agent.configure(agent_info["model"], agent_info["hyperparameters"])

            # Subtasks' executing
            result = agent.run(subtask["name"])
            return subtask["id"], result
        except Exception as e:
            self.logger.error(f"Error al procesar subtarea {subtask['id']}: {e}")
            return subtask["id"], f"Error: {e}"

    def execute_task(self, task):
        """Execute all subtasks of a task, respecting dependencies and parallelizing subtasks of the same order."""
        self.logger.info(f"Executing task {task['id']} - {task['name']}...")
        subtasks = task.get("subtasks", [])
        completed = {}  # Store results of completed subtasks

        # Group subtasks in order
        subtasks_by_order = {
            order: [
                subtask for subtask in subtasks if subtask["order"] == order
            ] for order in set(s["order"] for s in subtasks)
        }

        # Run subtasks in order
        for order in sorted(subtasks_by_order.keys()):
            ready_subtasks = subtasks_by_order[order]

            # Process subtasks of the same order in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._process_subtask, subtask): subtask for subtask in ready_subtasks}
                for future in concurrent.futures.as_completed(futures):
                    subtask_id, result = future.result()
                    completed[subtask_id] = result
                    self.logger.info(f"Subtask {subtask_id} completed with result: {result}")

        return completed

    def run(self):
        """Executes all tasks and their associated subtasks in the crew."""
        results = {}

        # Run each task sequentially
        for task in self.tasks:
            task_id, task_results = task["id"], self.execute_task(task)
            results[task_id] = task_results

        self.logger.info("SwarmAgent execution completed.")
        return results

if __name__ == "__main__":
    crews_plan = {
        'id': '4f2a53f9-86d2-4acf-b329-083d4423fe67',
        'name': 'crew_1',
        'task_plan': {
            'tasks': [{
                'id': 'c1f7c5b6-5c7e-4c1a-8e9f-1c3f9e1e8f5e',
                'name': 'Detects objects in the image and Summary',
                'subtasks': [{
                    'order': 1,
                    'id': 'b1e4d1a2-4b8c-4b5a-9c8e-4b8e1c3f9e1e',
                    'name': 'Detects objects in the image',
                    'subtask_dependencies': [],
                    'agent': {
                        'id': '0678eedf-9415-48e3-8f6c-35035be2bb5a',
                        'name': 'object_detection_agent.py',
                        'class': 'ObjectDetectionAgent',
                        'model': 'resnet50',
                        'hyperparameters': {'weights': 3.65}
                    }
                },
                {
                    'order': 2,
                    'id': 'd2f8e1b4-3c7e-4c1a-8e9f-1c3f9e1e8f5f',
                    'name': 'Summarize the content',
                    'subtask_dependencies': ['b1e4d1a2-4b8c-4b5a-9c8e-4b8e1c3f9e1e'],
                    'agent': {
                        'id': '6ebadc1f-2b63-4f54-884f-9189cf24bcde',
                        'name': 'default-LLM.py',
                        'class': 'DefaultLlM',
                        'model': 'gpt-4',
                       'hyperparameters': {'temperature': 0.03}
                    }
                },
                    {
                        'order': 2,
                        'id': 'd2f8e1b4-3c7e-4c1a-8e9f-1c3f9e1e8f5f',
                        'name': 'Summarize the content2',
                        'subtask_dependencies': ['b1e4d1a2-4b8c-4b5a-9c8e-4b8e1c3f9e1e'],
                        'agent': {
                            'id': '6ebadc1f-2b63-4f54-884f-9189cf24bc----marta',
                            'name': 'hola',
                            'model': 'gpt-4',
                            'hyperparameters': {'temperature': 0.03}
                        }
                    }
                ]
            }
            ]
        }
    }

    agent = SwarmAgent(crews_plan)
    agent.run()