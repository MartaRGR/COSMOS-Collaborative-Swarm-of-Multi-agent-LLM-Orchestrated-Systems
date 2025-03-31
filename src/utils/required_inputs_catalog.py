class RequiredInput:
    def __init__(self, variable: str, description: str):
        self.variable = variable
        self.description = description

    def to_dict(self):
        return {"variable": self.variable, "description": self.description}

# Required inputs dictionary
# TODO: change or complete if needed
REQUIRED_INPUTS_CATALOG = {
    "image_path": {
        "variable": "image_path",
        "description": "The path or file of the image to be analyzed."
    },
    "task_definition": {
        "variable": "task_definition",
        "description": "The description of the task to be solved by the LLM agent."
    },
    "video_path": {
        "variable": "video_path",
        "description": "The path of the video file to process."
    }
}

def get_required_inputs(*keys):
    """Returns a list of required inputs from the catalog."""
    return [REQUIRED_INPUTS_CATALOG[key].to_dict() for key in keys if key in REQUIRED_INPUTS_CATALOG]
