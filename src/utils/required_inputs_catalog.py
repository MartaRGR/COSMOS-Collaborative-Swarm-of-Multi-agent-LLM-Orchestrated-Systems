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
    "query": {
        "variable": "query",
        "description": "The question or search query to be answered based on the provided input file or text."
    },
    "structured_data": {
        "variable": "structured_data",
        "description": "The structured data to be processed."
    },
    "occupancy_time": {
        "variable": "occupancy_time",
        "description": "Text with the time at which the occupancy image is taken in the following format YYYY-MM-DD HH:MM:SS."
    },
    "forecast_horizon": {
        "variable": "forecast_horizon",
        "description": "The horizon number in hours for the forecast (integer)."
    },
    "initial_temperature": {
        "variable": "initial_temperature",
        "description": "The initial temperature of the room in Celsius (float)."
    },
    "outside_temperature": {
        "variable": "outside_temperature",
        "description": "The outside temperature in Celsius (float)."
    },
    "room_volume": {
        "variable": "room_volume",
        "description": "The volume of the room in cubic meters (float)."
    }
}
