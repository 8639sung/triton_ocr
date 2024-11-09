import tritonclient.grpc as grpcclient
import numpy as np

import app.config as config


class OCRClient:
    def __init__(self, server_url=config.INFERENCE_SERVER_URL):
        self.client = grpcclient.InferenceServerClient(url=server_url)

    def preprocess(self, image):
        return image  

    def detect_text(self, image):
        processed_image = self.preprocess(image)
        inputs = [grpcclient.InferInput('input', processed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(processed_image)
        outputs = [grpcclient.InferRequestedOutput('output')]
        response = self.client.infer(model_name="det_model", inputs=inputs, outputs=outputs)
        detection_result = response.as_numpy('output')
        return detection_result  

    def recognize_text(self, image):
        processed_image = self.preprocess(image)
        inputs = [grpcclient.InferInput('input', processed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(processed_image)
        outputs = [grpcclient.InferRequestedOutput('output')]
        response = self.client.infer(model_name="rec_model", inputs=inputs, outputs=outputs)
        recognition_result = response.as_numpy('output')
        return recognition_result 