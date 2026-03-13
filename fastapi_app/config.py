from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str = "CV Inference API"
    APP_VERSION: str = "1.0.0"

    TRITON_URL: str = "triton:8000"
    MODEL_NAME: str = "resnet50"

    IMAGE_SIZE: int = 224
    TENSOR_NAME: str = "resnetv17_dense0_fwd"
    MODEL_NAME: str= "resnet50"
    INFER_INPUT_DATATYPE:str = "FP32"
    INFER_INPUT_NAME:str = "data"
    SUPPORTED_IMAGE_FORMAT:list = ["image/jpeg", "image/png"]

    LOG_LEVEL: str = "INFO"
    TRITON_TIMEOUT: int = 5

    class Config:
        env_file = ".env"


settings = Settings()