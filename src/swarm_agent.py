import datetime
import importlib
import os
import logging
import json

import pytz
madrid_tz = pytz.timezone("Europe/Madrid")

import concurrent.futures

from utils.setup_logger import get_agent_logger
from utils.state_models import PrivateState


class SwarmAgent:
    def __init__(self, crew_detail, agentic_modules):
        """
        Initializes the SwarmAgent with the details of the crew and its tasks.

        Arguments:
            crew_detail (dict): Detail of the tasks assigned to the group.
        """
        self.crew_detail = crew_detail
        self.name = crew_detail.get("name")

        self.logger = get_agent_logger(f"SwarmAgent - {self.name}")
        self.logger.info("Initializing...")

        self.tasks = crew_detail.get("task_plan", {}).get("tasks", [])
        self.modules = agentic_modules

        self.private_state: PrivateState = {
            "id": crew_detail.get("id", "unknown"),
            "name": crew_detail.get("name", "unknown"),
            "task_plan": crew_detail.get("task_plan", {}),
        }

    def get_dependency_input(self, dependency_ids, processed_subtasks_by_order):
        """
        Collects the results of the subtasks whose IDs are in dependency_ids.
        Returns a dictionary where each key is the dependency and the value is its result.
        """
        dependency_inputs = {}
        for dep_id in dependency_ids:
            found = False
            for order, results_list in processed_subtasks_by_order.items():
                for res in results_list:
                    if res["id"] == dep_id:
                        dependency_inputs[dep_id] = res.get("agent", {}).get("result")
                        found = True
                        break
                if found:
                    break
            if not found:
                dependency_inputs[dep_id] = None
        return dependency_inputs

    def _process_subtask(self, subtasks, processed_subtasks_by_order):
        """Process the ordered subtasks by running the corresponding agents."""
        # TODO: se puede paralelizar tareas con mismo orden
        processed_subtasks = []
        for subtask in subtasks:
            self.logger.info(f"Processing subtask {subtask['id']} - {subtask['name']}...")
            # subtask info
            processed = {
                "id": subtask["id"],
                "name": subtask["name"],
                "subtask_dependencies": subtask.get("subtask_dependencies", [])
            }
            agent_info = subtask.get("agent")
            agent_name = agent_info.get("name")
            agent_module = self.modules.get(agent_name)
            if not agent_module:
                self.logger.error(f"Module for agent {agent_name} not found.")
                processed["agent"] = {
                    **agent_info,
                    "status": "error",
                    "result": None,
                    "timestamp": datetime.datetime.now(madrid_tz).strftime("%Y-%m-%d %H:%M:%S")
                }
                processed_subtasks.append(processed)
                continue

            # Agent instance with its configurations
            agent_instance = agent_module(
                config={
                    "model": agent_info.get("model"),
                    "hyperparameters": agent_info.get("hyperparameters")
                },
                crew_id=self.name
            )

            # Checking subtask's dependencies
            required_inputs = agent_info.get("required_inputs", {})
            if subtask.get("subtask_dependencies"):
                dependency_input = self.get_dependency_input(subtask["subtask_dependencies"], processed_subtasks_by_order)
                # Aggregate dependencies and required_inputs
                input_data = {
                    **required_inputs,
                    **{key: value for dep in dependency_input.values() for key, value in dep.items()}
                }
                # input_data = list(dependency_input.values())[0]
                self.logger.info(f"Subtask {subtask['id']} has dependencies; using dependency results as input.")
            else:
                input_data = required_inputs

            try:
                # Execution of Agent's logit
                input_data["task_definition"] = subtask["name"]
                agent_result = agent_instance.run(input_data)
                processed["agent"] = {
                    **agent_info,  # id, name, required_inputs, model, hyperparameters included
                    "status": "completed",
                    "result": agent_result,
                    "timestamp": datetime.datetime.now(madrid_tz).strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                self.logger.error(f"Error processing subtask {subtask['id']}: {e}")
                processed["agent"] = {
                    **agent_info,
                    "status": "error",
                    "result": str(e),
                    "timestamp": datetime.datetime.now(madrid_tz).strftime("%Y-%m-%d %H:%M:%S")
                }
            processed_subtasks.append(processed)
        return processed_subtasks

    def execute_task(self, task):
        """Execute all subtasks of a task, respecting dependencies and parallelizing subtasks of the same order."""
        self.logger.info(f"Executing task {task['id']} - {task['name']}...")
        task_id = task["id"]

        subtasks_dict = task.get("subtasks", {})
        processed_subtasks_by_order = {}
        for order in sorted(subtasks_dict.keys(), key=lambda o: int(o)):
            group = subtasks_dict[order]
            processed_group = self._process_subtask(group, processed_subtasks_by_order)
            processed_subtasks_by_order[str(order)] = processed_group

        # Updating the structure of the task_plan.
        for t in self.private_state["task_plan"]["tasks"]:
            if t["id"] == task_id:
                t["subtasks"] = processed_subtasks_by_order

        self.logger.info(f"Task {task_id} completed.")
        return processed_subtasks_by_order

    def _write_private_state_logs(self):
        timestamp = datetime.datetime.now(madrid_tz).strftime("%Y%m%d_%H%M%S")
        log_filename = f"private_state_{self.name}_{timestamp}.json"
        try:
            with open(log_filename, "w") as f:
                json.dump(self.private_state, f, indent=2, default=str)
            self.logger.info(f"Private state logged in {log_filename}")
        except Exception as e:
            self.logger.error(f"Error writing private state to file: {e}")

    def run(self, save_state=False) -> PrivateState:
        """Executes all tasks and their associated subtasks in the crew."""
        for task in self.tasks:
            self.execute_task(task)
        self.logger.info("SwarmAgent execution completed.")
        # final_state = self.private_state
        if save_state:
            self.logger.info("Saving private state...")
            self._write_private_state_logs()

        return self.private_state


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
                            'required_inputs': {'image_path': 'istockphoto-1346064470-612x612.jpg'},
                            'model': "yolov8n",
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
                            'required_inputs': {'task_definition': 'Summarize the content of the detected objects'},
                            'model': "gpt-4o-mini",
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