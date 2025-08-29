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
            "presence_penalty": [0.0, 2.0],
            "frequency_penalty": [0.0, 1.0]
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
            "confidence": [0.5]
        }
    },
    "yolov8n": {
        "name": "yolov8n",
        "hyperparameters": {
            "classes": [],
            "confidence": [0.5]
        }
    },
    "resnet50": {
        "name": "resnet50",
        "hyperparameters": {
            "weights": []
        }
    },
    ### Embeddings ###
    "text-embedding-3-large": {
        "name": "text-embedding-3-large",
        "hyperparameters": {
            "chunk_size": [800, 1000],
            "chunk_overlap": [150, 250]
        }
    },
    "text-embedding-ada-002": {
        "name": "text-embedding-ada-002",
        "hyperparameters": {
            "chunk_size": [600, 1000],
            "chunk_overlap": [100, 200]
        }
    },
    "ibm-granite/granite-embedding-107m-multilingual": {
        "name": "ibm-granite/granite-embedding-107m-multilingual",
        "hyperparameters": {
            "chunk_size": [300, 600],
            "chunk_overlap": [50, 100]
        }
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "hyperparameters": {
            "chunk_size": [300, 600],
            "chunk_overlap": [50, 100]
        }
    },
    "intfloat/multilingual-e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "hyperparameters": {
            "chunk_size": [400, 512]
        }
    },
    ### Embedding-Retrievers ###
    "cosine_similarity_retriever": {
        "name": "cosine_similarity_retriever",
        "hyperparameters": {
            "top_k": [3, 5, 10],
            "similarity_threshold": [0.0, 1.0]
        }
    },
    "torch_embedding_retriever": {
        "name": "torch_embedding_retriever",
        "hyperparameters": {
            "top_k": [3, 5, 10],
            "similarity_threshold": [0.0, 1.0]
        }
    },
    ### Distance Calculation ###
    "midas": {
        "name": "midas",
        "hyperparameters": {}
    },
    "dpt_large": {
        "name": "dpt_large",
        "hyperparameters": {}
    },
    "depth_anything": {
        "name": "depth_anything",
        "hyperparameters": {}
    },
    ### Forecasting ###
    "prophet": {
        "name": "prophet",
        "hyperparameters": {
            "weekly_seasonality": [False, True, 'auto'],
            "seasonality_prior_scale": [0.01, 10.0],
            "changepoint_prior_scale": [0.01, 0.5],
            "seasonality_mode": ["additive", "multiplicative"]

        }
    },
    ### Thermodynamic Prediction ###
    "thermodynamic_prediction": {
        "name": "thermodynamic_prediction",
        "hyperparameters": {
            "c_p": [1000.0, 1010.0],  # Specific heat capacity of air in J/(kg·K)
            "air_density": [1.1, 1.3], #1.225 Air density in kg/m³ at sea level and 15°C
            "person_heat": [70, 100],  # Power generated per person in watts
            "ACH": [0.2, 0.5],  # Air Changes per Hour,
            "U": [0.3, 0.5]  # W/(m²·K), typical heat transfer coefficient
        }
    }
}