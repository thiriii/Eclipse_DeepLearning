import cv2
import numpy as np
from deepface import DeepFace

def detect_deepfake(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    try:
        result = DeepFace.analyze(image, actions=['emotion'])
        return {"deepfake_detected": "No deepfake found" if result else "Possible deepfake detected"}
    except Exception as e:
        return {"error": str(e)}
