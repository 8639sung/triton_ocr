from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from PIL import Image
import numpy as np
import io

from service.trtion_client import TritonClient

router = APIRouter(prefix="/ocr")
triton_client = TritonClient()

def img_process(files: List[UploadFile]):
    np_images = []
    for file in files:
        try:
            image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
            image = image.resize((960, 960)) 
            np_image = np.array(image).astype("float32") / 255.0
            np_image = np.transpose(np_image, (2, 0, 1))
            np_images.append(np_image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")
    np_images = np.stack(np_images, axis=0)
    return np_images

def bbox_process(results):
    all_bboxes = []
    for result in results:
        bboxes = []
        for bbox in result:
            bboxes.append(bbox.tolist())
        all_bboxes.append(bboxes)
    return all_bboxes

def rec_resize(np_images):
    resized_images = []
    H = 48
    for np_image in np_images:
        _, original_height, original_width = np_image.shape
        ratio = original_width / original_height
        W = int(H * ratio)
        resized_image = np.resize(np_image, (3, H, W))
        resized_images.append(resized_image)
    resized_images = np.stack(resized_images, axis=0)
    return resized_images

@router.post("/detection")
async def detect_text(files: List[UploadFile] = File(...)):
    np_images = img_process(files)
    results = triton_client.detect_text(np_images)
    print(results)
    bboxes = bbox_process(results)
    return {
        "images": [img.tolist() for img in np_images],
        "bboxes": bboxes
    }

@router.post("/recognition")
async def recognize_text(images: List[List[List[float]]], bboxes: List[List[List[int]]]):
    recognition_results = []

    for i, image in enumerate(images):
        cropped_images = []
        for bbox in bboxes[i]:  
            x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            cropped_image = image[:, y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        recognized_texts = []
        for cropped_image in cropped_images:
            target_height = 48
            _, original_height, original_width = cropped_image.shape
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)
            resized_image = np.resize(cropped_image, (3, target_height, target_width))
            recognized_text = triton_client.recognize_text(np.expand_dims(resized_image, axis=0))
            recognized_texts.append(recognized_text[0].tolist())
        recognition_results.append(recognized_texts)

    return {
        "recognition_results": recognition_results
    }