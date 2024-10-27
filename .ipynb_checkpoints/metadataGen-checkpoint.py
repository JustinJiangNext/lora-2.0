from transformers import pipeline
from PIL import Image 
import jsonlines
import os 


# open method used to open different extension image file 
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0, max_new_tokens = 20)


totalImgs = len(os.listdir("formatted"))
metaList = [0]*totalImgs

imgNum = 0
for file in os.listdir("formatted"):

    textDescription = captioner(Image.open(f"formatted/{file}"))[0]["generated_text"]
    metaList[imgNum] = {"file_name": file, "text": textDescription}
    
    imgNum += 1
    print(imgNum)

with jsonlines.open('metadata.jsonl', 'w') as writer:
    writer.write_all(metaList)