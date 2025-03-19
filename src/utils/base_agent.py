from abc import ABC, abstractmethod


AGENT_METADATA = {
    "function": "", # Replace by the agent's purpose. For example: Detect objects in an image.
    "input": "", # Replace by the agent's input type. For example: image, text, etc.
    "output": "",#Replace by the agent's output type. For example: list of detected objects, text, etc.
    "class": "", #Replace by the agent's class name, in this example, BaseAgent.
    "models": [ # Replace by the list of available agent's models
        {
            "name": '', # Replace by the name of one of the available agent's model, i.e., yolo11n.pt.
            "hyperparameters": {
                # Replace by the dictionary of the available hyperparameters
            }
        }
    ]
}

class BaseAgent(ABC):
    def __init__(self, crew_id: int, config: dict):
        """
        Initialize the agent with the provided configuration.
        The config dictionary must contain at least:
        - 'model': Name or identifier of the model to use.
        - 'hyperparameters': Dictionary containing the hyperparameters.
        """
        self.logger = None
        self.crew_id = crew_id
        self.config = config
        self.model_name = config.get("model")
        self.hyperparameters = config.get("hyperparameters", {})
        self.model = None
        self.device = None
        self._setup_agent(self.model_name, self.crew_id)

    @abstractmethod
    def _setup_agent(self, model_name: str, crew_id: int):
        """
        Configures the initial settings of the agent (for example, model selection based on name or identifier).
        This method must be implemented by each specific agent, as its loading logic may vary.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Executes the agent's main logic.
        This method must be implemented by each specific agent.
        """
        pass
