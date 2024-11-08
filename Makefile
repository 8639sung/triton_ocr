run_triton:
	docker run -it --gpus all --shm-size=512m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:24.10-py3

deploy_model:
	tritonserver --model-repository=./model_repository