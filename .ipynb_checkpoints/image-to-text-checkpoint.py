"""
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("testing.png")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe what's happening in this image(including background) without saying 'image.' (60 word limit)"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=70)
print(processor.decode(output[0]).replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Describe what's happening in this image(including background) without saying 'image.' (60 word limit)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "").replace("<|eot_id|>", ""))
                                           
"""


from transformers import pipeline, MllamaForConditionalGeneration, AutoProcessor
from PIL import Image 
import jsonlines
import os 
import torch


totalImgs = len(os.listdir("formatted"))
metaList = [0]*totalImgs


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe what's happening in this image(including background) without saying 'image.' (60 word limit)"}
    ]}
]


imgNum = 0
for file in os.listdir("formatted"):
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        Image.open(f"formatted/{file}"),
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=70)
    metaList[imgNum] = {"file_name": file, "text": "R3E4AL, " + processor.decode(output[0]).replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Describe what's happening in this image(including background) without saying 'image.' (60 word limit)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "").replace("<|eot_id|>", "")}
    imgNum += 1
    print(imgNum)

with jsonlines.open('metadata.jsonl', 'w') as writer:
    writer.write_all(metaList)

