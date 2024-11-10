import onnx
import paddle2onnx

paddle_detection_model_path = "detection_model/inference.pdmodel"
paddle_detection_params_path = "detection_model/inference.pdiparams"
onnx_detection_model_path = "../model_repository/det_model/1/model.onnx"

paddle2onnx.export(model_filename=paddle_detection_model_path, save_file=onnx_detection_model_path, params_filename=paddle_detection_params_path)
model = onnx.load(onnx_detection_model_path)
print("Inputs:")
for inp in model.graph.input:
    name = inp.name
    dims = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Type: {dtype}")

print("\nOutputs:")
for out in model.graph.output:
    name = out.name
    dims = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    dtype = out.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Type: {dtype}")

paddle_recognition_model_path = "recognition_model/inference.pdmodel"
paddle_recognition_params_path = "recognition_model/inference.pdiparams"
onnx_recognition_model_path = "../model_repository/rec_model/1/model.onnx"

paddle2onnx.export(model_filename=paddle_recognition_model_path, save_file=onnx_recognition_model_path, params_filename=paddle_recognition_params_path)
model = onnx.load(onnx_recognition_model_path)
print("Inputs:")
for inp in model.graph.input:
    name = inp.name
    dims = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Type: {dtype}")

print("\nOutputs:")
for out in model.graph.output:
    name = out.name
    dims = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    dtype = out.type.tensor_type.elem_type
    print(f"Name: {name}, Shape: {dims}, Type: {dtype}")