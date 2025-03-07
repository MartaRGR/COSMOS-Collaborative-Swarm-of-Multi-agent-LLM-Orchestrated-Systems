import importlib
import concurrent.futures
import logging


class SwarmAgent:
    def __init__(self, crew_detail):
        """
        Initializes the SwarmAgent with the details of the crew and its tasks.

        Arguments:
            crew_detail (dict): Detail of the tasks assigned to the group.
        """
        self.logger = logging.getLogger("SwarmAgent")
        self.crew_detail = crew_detail
        self.tasks = crew_detail.get("task_plan", {}).get("tasks", [])

    @staticmethod
    def _load_agent(agent_name):
        """Dynamically loads the specified agent module."""
        try:
            module_name = agent_name.replace(".py", "")
            agent_module = importlib.import_module(module_name)
            return agent_module.Agent()
        except Exception as e:
            raise RuntimeError(f"Error loading agent '{agent_name}': {e}")

    def _process_subtask(self, subtask):
        """Process a specific subtask by running the corresponding agent."""
        agent_info = subtask.get("agent")
        try:
            # Load and configure the agent
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
        subtasks = task.get("subtasks", [])
        completed = {}  # Store results of completed subtasks

        # Group subtasks in order
        subtasks_by_order = {}
        for subtask in subtasks:
            subtasks_by_order.setdefault(subtask["order"], []).append(subtask)

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
                        'model': 'gpt-4',
                       'hyperparameters': {'temperature': 0.03}
                    }
                }]
            }
            ]
        }
    }

    agent = SwarmAgent(crews_plan)
    agent.run()