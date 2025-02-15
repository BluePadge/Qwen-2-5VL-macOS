from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="mps",
    low_cpu_mem_usage=True)

# min_pixels = 256 * 28 * 28
# max_pixels = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                          use_fast=False)
# min_pixels=min_pixels,
# max_pixels=max_pixels)

messages = [{
    "role":
    "user",
    "content": [
        {
            "type": "image",
            "image": "assets/invoice.jpg",
        },
        {
            "type":
            "text",
            "text":
            "Extract from this invoice the invoiced items as a list, output to JSON"
        },
    ],
}]

# Preparation for inference
text = processor.apply_chat_template(messages,
                                     tokenize=False,
                                     add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids_trimmed,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
print(output_text)
