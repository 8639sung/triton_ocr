deploy_triton:
	docker run -d -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
