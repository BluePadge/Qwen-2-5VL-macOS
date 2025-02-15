from fastapi import FastAPI, UploadFile, File, Form
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="mps",
        low_cpu_mem_usage=True)
    app.state.processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=False)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(image: UploadFile = File(...), text: str = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image",
            "image": pil_image
        }, {
            "type": "text",
            "text": text
        }]
    }]
    text_prompt = app.state.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = app.state.processor(text=[text_prompt],
                                 images=image_inputs,
                                 videos=video_inputs,
                                 padding=True,
                                 return_tensors="pt")
    inputs = inputs.to(app.state.model.device)
    generated_ids = app.state.model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = app.state.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return {"output": output_text}
