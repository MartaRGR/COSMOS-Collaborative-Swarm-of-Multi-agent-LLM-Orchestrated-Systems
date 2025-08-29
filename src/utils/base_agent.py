from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


AGENT_METADATA = {
    "function": "", # Replace by the agent's purpose. For example: Detect objects in an image.
    "required_inputs": "", # Replace by the agent's required inputs. Use the required input catalog, for example, image_path, task_definition, etc.
    "output": "", # Replace by the agent's output type. For example: list of detected objects, text, etc.
    "class": "", # Replace by the agent's class name, in this example, BaseAgent.
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
        self.device = None

        self.model = None
        self.system_message = None
        self.human_message = None

        # Initialize the dispatcher for different model types
        self.dispatcher = {
            "gpt": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "o3": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "deepseek": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "nvidia": {
                "init": self._init_nvidia,
                "run": self._run_nvidia
            },
            "llama": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "phi": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "qwen": {
                "init": self._init_nvidia,
                "run": self._run_nvidia
            },
            "mistral": {
                "init": self._init_mistral,
                "run": self._run_mistral
            },
            "yolo": {
                "init": self._init_yolo,
                "run": self._run_yolo
            },
            "resnet": {
                "init": self._init_resnet,
                "run": self._run_resnet
            },
            "text-embedding": {
                "init": self._init_openai_embedding,
                "run": self._run_openai_embedding
            },
            "granite-embedding": {
                "init": self._init_hf_embedding,
                "run": self._run_hf_embedding
            },
            "all-minilm": {
                "init": self._init_hf_embedding,
                "run": self._run_hf_embedding
            },
            "e5-small": {
                "init": self._init_hf_embedding,
                "run": self._run_hf_embedding
            },
            "midas": {
                "init": self._init_midas,
                "run": self._run_midas
            },
            "dpt": {
                "init": self._init_midas,
                "run": self._run_midas
            },
            "depth_anything": {
                "init": self._init_depth_anything,
                "run": self._run_depth_anything
            },
            "prophet": {
                "init": '',
                "run": self._run_prophet
            }
        }

        # Initialize the agent
        self._setup_agent(self.model_name, self.crew_id)

    # Dispatcher logic
    def _get_dispatch_entry(self):
        for key in self.dispatcher:
            if key in self.model_name.lower():
                return self.dispatcher[key]
        raise ValueError(f"No dispatcher found for model: {self.model_name}")

    @staticmethod
    def _read_file(input_path):
        """Read the content of a file."""
        import os
        ext = os.path.splitext(input_path)[-1].lower()

        if ext in [".txt", ".json", ".csv"]:
            import chardet
            with open(input_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(input_path, 'r', encoding=encoding) as f:
                return f.read()

        elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            import base64
            with open(input_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            image_format = "jpeg" if ext == ".jpg" else "png"
            return f"data:image/{image_format};base64,{image_data}"

        elif ext in [".mp3", ".wav"]:
            import base64
            with open(input_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:audio/{ext[1:]};base64,{audio_data}"

        elif ext in [".mp4", ".mov"]:
            import base64
            with open(input_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:video/{ext[1:]};base64,{video_data}"

        elif ext == ".docx":
            from docx import Document
            doc = Document(input_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif ext == ".pdf":
            import pymupdf
            doc = pymupdf.open(input_path)
            return "\n\n".join([page.get_text() for page in doc])

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _read_input_data(self, input_data):
        from .required_inputs_catalog import REQUIRED_INPUTS_CATALOG
        import os
        for key in REQUIRED_INPUTS_CATALOG.keys():
            if key in input_data and isinstance(input_data[key], str) and os.path.isfile(input_data[key]):
                input_data[key] = self._read_file(input_data[key])
        return input_data

    @staticmethod
    def get_input_key(input_dict):
        from .required_inputs_catalog import REQUIRED_INPUTS_CATALOG
        return next(
            (key for key in REQUIRED_INPUTS_CATALOG.keys() if key != "task_definition" and key in input_dict),
            None
        )

    def find_text_key(self, data, key_name):
        if key_name in data:
            return data[key_name]
        for key, value in data.items():
            if isinstance(value, dict):
                result = self.find_text_key(value, key_name)
                if result is not None:
                    return result
        return None

    def chunk_text(self, text):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.hyperparameters.get("chunk_size", 600),
            chunk_overlap=self.hyperparameters.get("chunk_overlap", 100),
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)

    def cosine_similarity(self, query_embedding, chunk_embeddings, chunks, top_k=5):
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = F.normalize(torch.tensor([x["embedding"] for x in query_embedding]), p=2, dim=-1)
        if not isinstance(chunk_embeddings, torch.Tensor):
            chunk_embeddings = F.normalize(torch.tensor([x["embedding"] for x in chunk_embeddings]), p=2, dim=-1)

        # Score (cosine similarity) and best calculation
        scores = torch.matmul(chunk_embeddings, query_embedding.unsqueeze(-1)).squeeze(-1)
        top_scores, top_indices = torch.topk(scores, k=top_k)
        return {"text": [{"chunk": chunks[i], "score": score} for i, score in zip(top_indices.squeeze(0).tolist(), top_scores.squeeze(0).tolist())]}
        # return {"text": [chunks[i] for i in top_indices.squeeze(0).tolist()], "best_scores": top_scores.squeeze(0).tolist()}

    # Init models' functions
    # # # # #
    def _init_openai(self):
        from langchain_openai import AzureChatOpenAI
        model_config = {
            "deployment_name": self.hyperparameters.get("deployment_name", self.model_name),
            "model_name": self.model_name,
            "api_version": self.hyperparameters.get("api_version", "2024-08-01-preview")
        }
        if "mini" not in self.model_name:
            model_config["temperature"] = self.hyperparameters.get("temperature", 0.2)
        else:
            self.logger.info(f"Model '{self.model_name}' does not allow 'temperature' configuration.")
        return AzureChatOpenAI(**model_config)

    @staticmethod
    def _init_foundry():
        import os
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY")),
        )

    @staticmethod
    def _init_nvidia():
        import os
        from openai import OpenAI
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
        )

    @staticmethod
    def _init_mistral():
        return "https://integrate.api.nvidia.com/v1/chat/completions"

    def _init_yolo(self):
        from ultralytics import YOLO
        return YOLO(f"{self.model_name}.pt").to(self.device)

    def _init_resnet(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        model.eval()
        return model

    def _init_openai_embedding(self):
        import os
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.inference import EmbeddingsClient
        return EmbeddingsClient(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")+f'/openai/deployments/{self.model_name}',
            credential=AzureKeyCredential(os.getenv("AZURE_OPENAI_API_KEY"))
        )

    def _init_hf_embedding(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained(self.model_name)

    def _init_midas(self):
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
        return self.depth_model

    def _init_depth_anything(self):
        from transformers import pipeline
        depth_pipeline = pipeline(
            task="depth-estimation",
            model=self.hyperparameters.get(
                "hf_model_name",
                "depth-anything/Depth-Anything-V2-Small-hf" # other models Base-hf, Large-hf
            ),
            device=0 if self.device == "cuda" else -1
        )
        return depth_pipeline


    # Run models' functions
    # # # # #
    def _run_openai(self, input_data):
        from langchain.schema.messages import SystemMessage, HumanMessage
        input_data = self._read_input_data(input_data)
        messages = [SystemMessage(content=self.system_message)]

        image_path = self.find_text_key(input_data, "image_path")
        text = self.find_text_key(input_data, "text")

        if image_path:
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": input_data.get("task_definition")},
                    {"type": "image_url", "image_url": {"url": input_data.get("image_path")}},
                ])
            )
        elif text:
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": self.human_message.format(
                        user_task=input_data.get("task_definition"),
                        input_data=input_data.get("text", '')
                    )},
                ])
            )
        return self.model.invoke(messages).content

    def _run_foundry(self, input_data):
        import re
        from azure.ai.inference.models import (
            SystemMessage, UserMessage, TextContentItem, ImageUrl, ImageContentItem
        )

        input_data = self._read_input_data(input_data)
        messages = [SystemMessage(content=self.system_message)]
        image_path = self.find_text_key(input_data, "image_path")
        text = self.find_text_key(input_data, "text")

        if image_path:
            messages.append(
                UserMessage(content=[
                    TextContentItem(text=input_data["task_definition"]),
                    ImageContentItem(image_url=ImageUrl(url=image_path))
                ])
            )
        elif text:
            messages.append(
                UserMessage(content=self.human_message.format(
                    user_task=input_data["task_definition"],
                    input_data=text
                ))
            )
        else:
            messages.append(
                UserMessage(content=self.human_message.format(
                    user_task=input_data["task_definition"],
                    input_data=""
                ))
            )
        # TODO: poner otros formatos de archivo

        response = self.model.complete(
            messages=messages,
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1),
            presence_penalty=self.hyperparameters.get("presence_penalty", 0.0),
            frequency_penalty=self.hyperparameters.get("frequency_penalty", 0.0),
            model=self.model_name
        )
        content = response.choices[0].message.content
        return re.sub(
            r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
        ) if "deepseek" in self.model_name.lower() else content

    def _run_nvidia(self, input_data):
        import re
        input_data = self._read_input_data(input_data)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.human_message.format(
                    user_task=input_data.get("task_definition"),
                    input_data=input_data.get(self.get_input_key(input_data))
                )}
            ],
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1)
        )
        content = response.choices[0].message.content
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

    def _run_mistral(self, input_data, stream=False):
        import os
        import requests

        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC')}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        image_path = self.find_text_key(input_data, "image_path")
        if image_path:
            input_data = self._read_input_data(input_data)
            input_data["text_input"] = '<img src="' + input_data["image_path"] + '" />'

        payload = {
            "model": 'mistralai/mistral-medium-3-instruct',
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user","content": self.human_message.format(
                    user_task=input_data["task_definition"], input_data=input_data.get("text_input", ''))}
            ],
            "temperature": self.hyperparameters.get("temperature", 0.2),
            "top_p": self.hyperparameters.get("top_p", 0.1),
            "presence_penalty": self.hyperparameters.get("presence_penalty", 0.0),
            "stream": stream
        }

        response = requests.post(self.model, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

    def _run_yolo(self, input_data, save_tagged_img=False):
        import cv2

        def extract_objects(results):
            """
            Extracts detected objects from YOLO results.
            Args:
                results: YOLO prediction results.
            Returns:
                detected_objects: List of detected objects with their information.
            """
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    # Extract rectangle coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width, height = x2 - x1, y2 - y1
                    area = width * height

                    # Extract class and probability
                    class_id = int(box.cls[0])
                    predicted_obj = result.names[class_id]
                    confidence = float(box.conf[0])

                    detected_objects.append({
                        "class": predicted_obj,
                        "confidence": confidence,
                        "area": area,
                        "bbox": [x1, y1, x2, y2]
                    })
            return detected_objects

        def plot_bboxes(frame, detected_objects, rectangle_thickness=2, text_thickness=1):
            import cv2
            """
            Draws bounding boxes and labels on the image.
            Args:
                frame: The image on which results will be drawn.
                detected_objects: List of detected objects.
            Returns:
                frame: The image with drawn results.
            """
            for obj in detected_objects:
                if "bbox" in obj:  # Only draw if there is a bounding box (YOLO results)
                    x1, y1, x2, y2 = obj['bbox']
                    confidence = obj['confidence']
                    predicted_obj = obj['class']

                    # Draw rectangle and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
                    cv2.putText(
                        frame, f"{predicted_obj} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness
                    )
            return frame

        def save_results(frame, detected_objects, save_path):
            """Saves the image with results to the specified path."""
            tagged_img = plot_bboxes(frame, detected_objects)
            cv2.imwrite(save_path, tagged_img)
            print(f"Results saved in {save_path}")

        image_path = input_data.get("image_path")
        if not image_path:
            self.logger.error("No image path provided.")
            return None
        frame = cv2.imread(image_path)
        self.logger.info(f"Image loaded: {image_path}")
        self.logger.info(">>> Running YOLO object detection...")
        predict_args = {"source": frame, "conf": self.hyperparameters.get("confidence", 0.5)}
        classes = self.hyperparameters.get("classes")
        if classes:
            predict_args["classes"] = classes
        results = self.model.predict(**predict_args)
        objects = extract_objects(results)
        self.logger.info(f"Detected objects: {objects}")
        if save_tagged_img:
            save_results(frame, objects, "results.jpg")
        return {"text": objects}

    def _run_resnet(self, input_data):
        import cv2

        def _fetch_imagenet_classes():
            """Fetch ImageNet class labels from the defined URL."""
            import requests
            return requests.get(
                "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"  # Imagenet Classes URL
            ).text.splitlines()

        def _preprocess_image(frame):
            """Preprocess the input frame into a tensor suitable for the model."""
            from PIL import Image
            import cv2

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image_pil = Image.fromarray(image)  # Convert to PIL Image
            return transform(image_pil).unsqueeze(0).to(self.device)

        self.logger.info(">>> Running ResNet50 object classification...")

        image_path = input_data.get("image_path")
        if not image_path:
            self.logger.error("No image path provided.")
            return None
        frame = cv2.imread(image_path)

        # Load ImageNet class labels
        imagenet_classes = _fetch_imagenet_classes()

        # Preprocess the input frame
        image_tensor = _preprocess_image(frame)

        # Perform prediction
        with torch.no_grad():
            model_outputs = self.model(image_tensor)
            confidence, predicted_class_idx = torch.nn.functional.softmax(model_outputs, dim=1).max(1)

        predicted_label = imagenet_classes[predicted_class_idx.item()]
        objects = [{"class": predicted_label, "confidence": confidence.item()}]
        self.logger.info(f"Detected objects: {objects}")
        return {"text": objects}

    def _run_openai_embedding(self, input_data):
        input_data = self._read_input_data(input_data)
        # Embedding query
        query = self.model.embed(input=input_data["query"], model=self.hyperparameters.get("model"))
        # Chunking and embedding text
        chunks = self.chunk_text(input_data["text"])[:25]
        text = self.model.embed(
            input=chunks,
            model=self.hyperparameters.get("model"),
        )
        # Best chunks and scores
        return self.cosine_similarity(query["data"], text["data"],  chunks)


    def _run_hf_embedding(self, input_data):
        from transformers import AutoTokenizer

        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        def embed_texts(texts, tokenizer, model):
            batch_dict = tokenizer(
                texts,
                max_length=self.hyperparameters.get("chunk_size", 512),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

        input_data = self._read_input_data(input_data)
        chunks = self.chunk_text(input_data["text"])[:25]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chunk_embeddings = embed_texts(['passage: ' + chunk for chunk in chunks], tokenizer, self.model)
        query_embedding = embed_texts(['query: ' + input_data["query"]], tokenizer, self.model)[0]

        # Best chunks and scores
        return self.cosine_similarity(query_embedding, chunk_embeddings, chunks)

    def _run_midas(self, input_data):
        from PIL import Image
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("No image_path provided for depth estimation")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
        return predicted_depth

    def _run_depth_anything(self, input_data):
        from PIL import Image
        import numpy as np

        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("No image_path provided for depth estimation")

        image = Image.open(image_path).convert("RGB")
        result = self.depth_pipeline(image)
        depth = np.array(result["depth"])
        return depth

    def _run_prophet(self, input_data):
        import pandas as pd
        new_occupancy = input_data["occupancy"]
        self.logger.info(f"New Occupancy {new_occupancy}")
        if isinstance(new_occupancy, dict):
            new_occupancy = [new_occupancy]

        # Converting occupancy data into dataframe
        new_data = pd.DataFrame(new_occupancy)
        new_data['ds'] = pd.to_datetime(new_data['ds'])

        # Concat historical data with the new one
        combined_data = pd.concat([self.df, new_data]).drop_duplicates(subset="ds").sort_values("ds").reset_index(drop=True)

        for col in [col for col in combined_data.columns if col not in ["ds", "y"]]:
            self.model.add_regressor(col)
        self.model.fit(combined_data)

        # Make future dataframe according to input data dataframe
        forecast_horizon = int(input_data["forecast_horizon"])
        future = self.model.make_future_dataframe(periods=forecast_horizon, freq='h')
        forecast = self.model.predict(future[-forecast_horizon:])
        # Round to nearest integer
        forecast[['base_occupancy_prediction', 'pessimistic_occupancy_prediction', 'optimistic_occupancy_prediction']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].round()
        return {"occupancy_forecast": forecast[['ds', 'base_occupancy_prediction', 'pessimistic_occupancy_prediction', 'optimistic_occupancy_prediction']].to_string(index=False)}

    # Abstract methods
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
