import numpy as np
from ultralytics import YOLO

class DetectionModel:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 1280,
        verbose: bool = False,
    ):
        self.model = YOLO(model=model_path, task="detect")
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.verbose = verbose
        
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, verbose=False, conf=conf_threshold, imgsz=imgsz, classes=[0])

    def infer(self, image: np.ndarray) -> list[dict]:
        results = self.model(
            image,
            verbose=self.verbose,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            classes=[0],
        )[0]
        detections = []
        for box in results.boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                }
            )
        return detections
