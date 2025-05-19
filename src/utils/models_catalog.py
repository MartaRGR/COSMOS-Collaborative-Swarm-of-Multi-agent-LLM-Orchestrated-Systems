# TODO: change or complete if needed
MODELS_CATALOG = {
    ### Natural Language ###
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "api_version": "2023-03-15-preview",
            "deployment_name": "gpt-4o-mini"
        }
    },
    "Phi-4-mini-instruct": {
        "name": "Phi-4-mini-instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
            "frequency_penalty": [-1.0, 1.0]
        }
    },
    "Llama-3.3-70B-Instruct": {
        "name": "Llama-3.3-70B-Instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
            "frequency_penalty": [-1.0, 1.0]
        }
    },
    "DeepSeek-R1": {
        "name": "DeepSeek-R1",
        "hyperparameters": {}
    },
    "qwen/qwen2.5-coder-32b-instruct": {
        "name": "qwen/qwen2.5-coder-32b-instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0]
        }
    },
    ### Multimodal ###
    "Phi-3.5-vision-instruct": {
        "name": "Phi-3.5-vision-instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
            "frequency_penalty": [-1.0, 1.0]
        }
    },
    "Llama-3.2-11B-Vision-Instruct": {
        "name": "Llama-3.2-11B-Vision-Instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
            "frequency_penalty": [-1.0, 1.0]
        }
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "api_version": "2024-12-01-preview",
            "deployment_name": "gpt-4o"
        }
    },
    "Phi-4-multimodal-instruct": {
        "name": "Phi-4-multimodal-instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
            "frequency_penalty": [-1.0, 1.0]
        }
    },
    "mistral-medium-3-instruct": {
        "name": "mistral-medium-3-instruct",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0],
            "presence_penalty": [-2.0, 2.0],
        }
    },
    ### Logic Reasoning ###
    "o3-mini": {
        "name": "o3-mini",
        "hyperparameters": {
            "api_version": "2024-12-01-preview",
            "deployment_name": "o3-mini"
        }
    },
    "nvidia/llama-3.1-nemotron-nano-8b-v1": {
        "name": "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "hyperparameters": {
            "temperature": [0.0, 1.0],
            "top_p": [0.1, 1.0]
        }
    },
    ### Objects recognition ###
    "yolo11n": {
        "name": "yolo11n",
        "hyperparameters": {
            "classes": [],
            "confidence": [0.5, 0.7]
        }
    },
    "yolov8n": {
        "name": "yolov8n",
        "hyperparameters": {
            "classes": [],
            "confidence": [0.5, 0.7]
        }
    },
    "resnet50": {
        "name": "resnet50",
        "hyperparameters": {
            "weights": []
        }
    }
}