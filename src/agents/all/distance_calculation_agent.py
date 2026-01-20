import torch
import numpy as np
import itertools

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Estimate distances to detected objects from an image to determine what is the nearest or farthest objects.",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"]
    ],
    "output": "list of detected objects augmented with distance (meters)",
    "class": "DistanceCalculationAgent",
    "models": [
        MODELS_CATALOG["midas"],
        MODELS_CATALOG["dpt_large"],
        MODELS_CATALOG["depth_anything"]
    ]
}


class DistanceCalculationAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Distance Calculation Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are a distance estimation assistant. "
            "Given an image and detections (or a reference point), return a JSON array of objects "
            "with estimated distance in meters (if possible) or relative depth otherwise. "
            "Return ONLY a valid JSON array. No text before or after."
        )
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"

        self.use_intrinsics = self.hyperparameters.get("use_intrinsics", False)

    def _initialize(self):
        return self._get_dispatch_entry()["init"]()

    @staticmethod
    def _bbox_centroid(bbox):
        xmin, ymin, xmax, ymax = bbox
        return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

    @staticmethod
    def _sample_depth(depth_map, x, y, k=3):
        h, w = depth_map.shape[:2]
        x0, x1 = max(0, x - k), min(w - 1, x + k)
        y0, y1 = max(0, y - k), min(h - 1, y + k)

        if not (0 <= x < w) or not (0 <= y < h):
            # If the centroid is outside, return a reasonable value (i.e: global median, 0, or np.nan)
            return float(np.median(depth_map))

        if x0 > x1 or y0 > y1:
            return float(depth_map[y, x])

        patch = depth_map[y0:y1 + 1, x0:x1 + 1]
        if patch.size == 0:
            return float(depth_map[y, x])
        return float(np.median(patch))

    # @staticmethod
    # def _pixel_to_camera(u, v, Z, intrinsics):
    #     fx = intrinsics["fx"]
    #     fy = intrinsics["fy"]
    #     cx = intrinsics["cx"]
    #     cy = intrinsics["cy"]
    #     X = (u - cx) * Z / fx
    #     Y = (v - cy) * Z / fy
    #     return [X, Y, Z]

    @staticmethod
    def _euclidean_distance_3d(coord1, coord2):
        """
        Calculate the 3D Euclidean distance between two points.
        coord1, coord2: lists or tuples with (X, Y, Z) coordinates
        """
        return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))

    def compute_pairwise_distances(self, detections_with_3d):
        """Compute pairwise Euclidean distances between all detected objects that have valid 3D coordinates."""
        distances = []
        # Create indexed list to identify objects uniquely
        indexed_detections = list(enumerate(detections_with_3d))

        for (idx1, det1), (idx2, det2) in itertools.combinations(indexed_detections, 2):
            coords1 = det1.get("coords3d")
            coords2 = det2.get("coords3d")
            if coords1 is None or coords2 is None:
                continue
            dist = self._euclidean_distance_3d(coords1, coords2)
            distances.append({
                "obj1_class": det1.get("class"),
                "obj1_index": idx1,
                "obj2_class": det2.get("class"),
                "obj2_index": idx2,
                "distance_m": dist
            })
        return distances

    def run(self, input_data):
        try:
            # Relative depth estimation
            depth_map = self._get_dispatch_entry()["run"](input_data)

            # Retrieve detections and camera intrinsics
            detections = input_data.get("text", [])
            if not detections:
                self.logger.warning("No detections provided in input_data.")
                return []

            # intrinsics = input_data.get("camera_intrinsics", None)
            # if not intrinsics and self.use_intrinsics:
            #     h, w = depth_map.shape[:2]
            #     intrinsics = {
            #         "fx": 800.0,
            #         "fy": 800.0,
            #         "cx": w / 2.0,
            #         "cy": h / 2.0
            #     }
            #     self.logger.info(f"Using default intrinsics: {intrinsics}")
            #
            # # Distances calculations
            # results = []
            # for det in detections:
            #     bbox = det.get("bbox")
            #     cx, cy = self._bbox_centroid(bbox)
            #     rel_d = self._sample_depth(depth_map, cx, cy)
            #     coords3d = None
            #     distance_m = None
            #     if intrinsics:
            #         coords3d = self._pixel_to_camera(cx, cy, rel_d, intrinsics)
            #         distance_m = float(np.linalg.norm(coords3d))
            #     results.append({
            #         **det,
            #         "relative_depth": rel_d,
            #         "coords3d": coords3d,
            #         "distance_m": distance_m
            #     })
            #
            # # Pairwise distances calculation
            # pairwise_distances = self.compute_pairwise_distances(results)
            # relative depth calculations per objects
            for det in detections:
                cx, cy = self._bbox_centroid(det["bbox"])
                det["relative_depth"] = self._sample_depth(depth_map, cx, cy)

            # Pairwise relative distances calculation
            pairwise_distances = []
            for i, obj1 in enumerate(detections):
                for j, obj2 in enumerate(detections):
                    if i >= j:
                        continue
                    dist = abs(obj1["relative_depth"] - obj2["relative_depth"])
                    pairwise_distances.append({
                        "obj1": obj1["class"],
                        "obj1_depth": obj1["relative_depth"],
                        "obj2": obj2["class"],
                        "obj2_depth": obj2["relative_depth"],
                        "relative_distance": dist
                    })


            results = {"text": {
                "detections": detections,
                "pairwise_relative_distances": pairwise_distances
            }}

            self.logger.info(f">>> Task result:\n{results}")
            return results

            # # 3) Generar relaciones con referencia
            # ref_class = input_data.get("reference_class")
            # if not ref_class:
            #     self.logger.warning("No reference_class provided, returning detections only.")
            #     return detections
            #
            # ref_objs = [d for d in detections if d["class"] == ref_class]
            # if not ref_objs:
            #     self.logger.warning(f"No reference object '{ref_class}' found.")
            #     return detections
            #
            # ref_depth = ref_objs[0]["relative_depth"]
            #
            # relations = []
            # for det in detections:
            #     if det["class"] == ref_class:
            #         continue
            #     diff = det["relative_depth"] - ref_depth
            #     if diff < 0:
            #         relation = f"{det['class']} está más cerca de {ref_class}"
            #     elif diff > 0:
            #         relation = f"{det['class']} está más lejos de {ref_class}"
            #     else:
            #         relation = f"{det['class']} está a la misma distancia que {ref_class}"
            #     relations.append({
            #         "object": det["class"],
            #         "relative_depth": det["relative_depth"],
            #         "relation_to_ref": relation
            #     })
            #
            # return {
            #     "reference_object": ref_class,
            #     "relations": relations
            # }

            self.logger.info(f">>> Task result:\n{results}")
            return pairwise_distances

        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    config = {
        "model": "depth_anything", # midas or "dpt_large", "depth_anything"
        "hyperparameters": {
            "use_intrinsics": True
        }
    }
    agent = DistanceCalculationAgent("crew_1", config)

    # Simulated detections
    detections = [
      {
        "class": "refrigerator",
        "confidence": 0.9214143753051758,
        "area": 8262,
        "bbox": [
          297,
          74,
          351,
          227
        ]
      },
      {
        "class": "oven",
        "confidence": 0.8934513926506042,
        "area": 4012,
        "bbox": [
          137,
          125,
          196,
          193
        ]
      },
      {
        "class": "dining table",
        "confidence": 0.6184790134429932,
        "area": 11236,
        "bbox": [
          81,
          175,
          293,
          228
        ]
      }
    ]

    result = agent.run({
        "image_path": r"C:\Users\rt01306\OneDrive - Telefonica\Desktop\Doc\Doctorado\TESIS\Python_Code\prueba_langraph\pruebas\objects_recognition\coco\sample_images\000000037777.jpg",
        "text": detections,
        "reference_class": None
    })
    print(result)