# Required inputs dictionary
# TODO: change or complete if needed
REQUIRED_INPUTS_CATALOG = {
    "task_definition": {
        "variable": "task_definition",
        "description": "The description of the task to be solved by the LLM agent."
    },
    "image_path": {
        "variable": "image_path",
        "description": "The path or file of the image to be analyzed."
    },
    "video_path": {
        "variable": "video_path",
        "description": "The path of the video file to process."
    },
    "audio_path": {
        "variable": "audio_path",
        "description": "The path of the audio file to process."
    },
    "text": {
        "variable": "text",
        "description": "The path, file or text to be analyzed."
    },
    "structured_data": {
        "variable": "structured_data",
        "description": "The structured data to be processed."
    }
}
