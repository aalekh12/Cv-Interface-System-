import tritonclient.http as httpclient
import numpy as np
import config as config

client = httpclient.InferenceServerClient(url=config.settings.TRITON_URL)

def infer(image):

    inputs = httpclient.InferInput(
        config.settings.INFER_INPUT_NAME,
        image.shape,
        config.settings.INFER_INPUT_DATATYPE
    )

    inputs.set_data_from_numpy(image)

    outputs = httpclient.InferRequestedOutput(config.settings.TENSOR_NAME)

    result = client.infer(
        model_name=config.settings.MODEL_NAME,
        inputs=[inputs],
        outputs=[outputs]
    )

    return result.as_numpy(config.settings.TENSOR_NAME)