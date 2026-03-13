import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger

from schemas import PredictionResponse, HealthResponse, ErrorResponse
from utils import preprocess_image, calculate_latency, format_predictions
from triton_client import infer
from config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint
    """
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def predict(image: UploadFile = File(...)):
    """
    Perform inference on uploaded image
    """

    start_time = time.time()

    try:
        if image.content_type not in settings.SUPPORTED_IMAGE_FORMAT:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Read image bytes
        image_bytes = await image.read()

        # Preprocess image
        img_tensor = preprocess_image(image_bytes)

        # Send to Triton server
        output = infer(img_tensor)

        # Format predictions
        predictions = format_predictions(output)

        # Calculate latency
        latency = calculate_latency(start_time)

        logger.info(f"Inference completed in {latency:.2f} ms")

        return PredictionResponse(
            predictions=predictions,
            latency_ms=latency
        )

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Inference service failed"
        )