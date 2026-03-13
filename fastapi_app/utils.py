import numpy as np
from PIL import Image
from io import BytesIO
from loguru import logger
import time
from config import settings


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image into model input tensor
    """

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))

        img = np.array(image).astype(np.float32)

        # CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise


def calculate_latency(start_time: float) -> float:
    """
    Calculate request latency in milliseconds
    """
    return (time.time() - start_time) * 1000


def format_predictions(output: np.ndarray, top_k: int = 5):
    """
    Convert raw model output into structured predictions
    """

    probs = output.flatten()

    top_indices = probs.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "class_id": int(idx),
            "confidence": float(probs[idx])
        })

    return results