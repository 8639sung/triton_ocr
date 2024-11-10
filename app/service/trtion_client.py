import tritonclient.grpc as grpcclient
import config as config


class TritonClient:
    def __init__(self, server_url=config.INFERENCE_SERVER_GRPC_URL):
        self.client = grpcclient.InferenceServerClient(url=server_url)

    def detect_text(self, image):
        inputs = [grpcclient.InferInput('x', image.shape, "FP32")]
        inputs[0].set_data_from_numpy(image)
        outputs = [grpcclient.InferRequestedOutput('sigmoid_0.tmp_0')]
        response = self.client.infer(model_name="det_model", inputs=inputs, outputs=outputs)
        detection_result = response.as_numpy('sigmoid_0.tmp_0')
        return detection_result  

    def recognize_text(self, image):
        inputs = [grpcclient.InferInput('x', image.shape, "FP32")]
        inputs[0].set_data_from_numpy(image)
        outputs = [grpcclient.InferRequestedOutput('softmax_2.tmp_0')]
        response = self.client.infer(model_name="rec_model", inputs=inputs, outputs=outputs)
        recognition_result = response.as_numpy('softmax_2.tmp_0')
        return recognition_result 