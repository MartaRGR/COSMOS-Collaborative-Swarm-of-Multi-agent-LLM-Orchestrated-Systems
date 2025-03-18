import importlib
import os
import logging

import concurrent.futures

from utils.setup_logger import get_agent_logger


class SwarmAgent:
    def __init__(self, crew_detail, agentic_modules):
        """
        Initializes the SwarmAgent with the details of the crew and its tasks.

        Arguments:
            crew_detail (dict): Detail of the tasks assigned to the group.
        """
        self.crew_detail = crew_detail
        self.name = crew_detail.get("name")

        self.logger = get_agent_logger(f"SwarmAgent - Crew {self.name}")
        self.logger.info("Initializing...")

        self.tasks = crew_detail.get("task_plan", {}).get("tasks", [])
        self.modules = agentic_modules


    def _process_subtask(self, subtask):
        """Process a specific subtask by running the corresponding agent."""
        self.logger.info(f"Processing subtask {subtask['id']} - {subtask['name']}...")
        try:
            # Load and configure the agent
            agent_info = subtask.get("agent")
            agent = self.modules(agent_info["name"])
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

        # Run subtasks in order
        for order in sorted(subtasks.keys()):
            ready_subtask = subtasks[order]
            self._process_subtask(ready_subtask)

            # Process subtasks of the same order in parallel
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     futures = {executor.submit(self._process_subtask, subtask): subtask for subtask in ready_subtask}
            #     for future in concurrent.futures.as_completed(futures):
            #         subtask_id, result = future.result()
            #         completed[subtask_id] = result
            #         self.logger.info(f"Subtask {subtask_id} completed with result: {result}")

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

    def initialize_agents(crews_plan, agents):
        """Load agent modules based on task_detail. Avoid loading the same module more than once."""
        import sys
        sys.path.append("agents")
        try:
            agent_modules = {}
            subtask_list = [
                subtask
                for task in crews_plan["task_plan"]["tasks"]
                for subtasks in task['subtasks'].values()
                for subtask in subtasks
            ]
            unique_subtask_agents = list({subtask["agent"]["name"] for subtask in subtask_list if "agent" in subtask})
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {executor.submit(_load_agent, agent, agents): agent for agent in unique_subtask_agents}
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        agent_instance = future.result()
                        if agent_instance:
                            agent_modules[agent_name] = agent_instance
                            logging.info(f"Agent '{agent_name}' successfully loaded")
                        else:
                            logging.error(f"Error loading agent '{agent_name}'")
                    except Exception as exc:
                        logging.error(f"Error loading agent '{agent_name}': {exc}")
            return agent_modules

        except Exception as e:
            logging.error(f"Error initializing agents: {e}")


    def _load_agent(agent_name, agents):
        """Loads an agent from its module and returns the agent instance if successful, None otherwise."""
        try:
            module = importlib.import_module(os.path.splitext(agent_name)[0])
            class_name = agents[agent_name]["class"]
            if not hasattr(module, class_name):
                logging.error(f"Class '{class_name}' not found in agent's file '{agent_name}'.")
                return None
            return getattr(module, class_name)
        except Exception as e:
            logging.error(f"Failed to load agent's file '{agent_name}'. Error: {e}")
            raise

    agents = {
        "default-LLM.py": {
            "function": "Detect objects in an image",
            "input": "image",
            "output": "list of detected objects",
            "class": "DefaultLlmAgent",
            "models": []
        },
        "object_detection_agent.py": {
            "function": "Detect objects in an image",
            "input": "image",
            "output": "list of detected objects",
            "class": "ObjectDetectionAgent",
            "models": [
                {
                    "name": "yolo11n",
                    "hyperparameters": {
                        "classes": [
                            1,
                            5
                        ]
                    }
                },
                {
                    "name": "yolov8n",
                    "hyperparameters": {
                        "classes": [
                            5,
                            8
                        ]
                    }
                },
                {
                    "name": "resnet50",
                    "hyperparameters": {
                        "weights": [
                            1.0,
                            5.0
                        ]
                    }
                }
            ]
        }
    }

    crews_plan = {
        'id': 'f43d011b-84b5-4306-96da-7a0f3bc1e323',
        'name': 'crew_1',
        'task_plan': {'tasks': [
            {
                'id': 'e5b5c8d1-3e5e-4f4a-8e6e-2d5c8f3c1e7e',
                'name': 'Detect objects in the image and summarize its content',
                'subtasks': {
                    1: [{
                        'id': 'f1a3b2d4-2c5e-4d8b-8b3f-4f2e7c3e1e8d',
                        'name': 'Detect objects in the image',
                        'subtask_dependencies': [],
                        'agent': {
                            'id': '1ee11178-00bc-4988-bc2c-1b86c89a4134',
                            'name': 'object_detection_agent.py',
                            'class': 'ObjectDetectionAgent',
                            'model': None,
                            'hyperparameters': {}
                        }
                    }],
                    2: [{
                        'id': 'd2b1c3e5-6a8f-4d8b-bc1c-9e4f8c3e1e9e',
                        'name': 'Summarize content of detected objects',
                        'subtask_dependencies': ['f1a3b2d4-2c5e-4d8b-8b3f-4f2e7c3e1e8d'],
                        'agent': {
                            'id': '2ff56196-b422-4e95-8abb-db4309b188be',
                            'name': 'default-LLM.py',
                            'class': 'DefaultLlmAgent',
                            'model': None,
                            'hyperparameters': {}
                        }
                    }]
                }
            }
        ]}
    }

    agentic_modules = initialize_agents(crews_plan, agents)

    agent = SwarmAgent(crews_plan, agentic_modules=agentic_modules)
    agent.run()