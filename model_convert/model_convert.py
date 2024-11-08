import paddle
import paddle2onnx

paddle_detection_model_path = "detection_model/inference.pdmodel"
paddle_detection_params_path = "detection_model/inference.pdiparams"
onnx_detection_model_path = "model_repository/det_model/1/det_model.onnx"

paddle2onnx.export(model_file=paddle_detection_model_path, save_file=onnx_detection_model_path, params_file=paddle_detection_params_path)

paddle_recognition_model_path = "recognition_model/inference.pdmodel"
paddle_recognition_params_path = "recognition_model/inference.pdiparams"
onnx_recognition_model_path = "model_repository/rec_model/1/rec_model.onnx"

paddle2onnx.export(model_file=paddle_recognition_model_path, save_file=onnx_recognition_model_path, params_file=paddle_recognition_params_path)