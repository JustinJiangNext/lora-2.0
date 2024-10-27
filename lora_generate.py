import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_images(prompt, modifiers, num_images, batch_size, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_id = "stabilityai/stable-diffusion-2-1"
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.load_lora_weights("LORA", weight_name="pytorch_lora_weights.safetensors")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    name_offset = 0
    # Generate the specified number of images
    index = 0
    while index < num_images:
        current_modifier = modifiers[(index // 4) % len(modifiers)]
        full_prompt = f"{prompt}, {current_modifier}"
        print(full_prompt + "    " + str(index))
        images = pipe(full_prompt, 
                      height = 512, 
                      width = 512, 
                      num_inference_steps = 50, 
                      num_images_per_prompt=batch_size).images
        
        for j, image in enumerate(images):
            if index >= num_images:
                break
            image.save(os.path.join(output_folder, f"image_{index+1+name_offset}.png"))
            index += 1

# Example usage


def fulltaskworker(promptNumber):
    prompts = [
        "R3E4AL, a photograph of an airplane", 
        "R3E4AL, a photograph of an automobile", 
        "R3E4AL, a photograph of a bird", 
        "R3E4AL, a photograph of a cat",
        "R3E4AL, a photograph of a deer", 
        "R3E4AL, a photograph of a dog", 
        "R3E4AL, a photograph of a frog", 
        "R3E4AL, a photograph of a horse", 
        "R3E4AL, a photograph of a ship", 
        "R3E4AL, a photograph of a truck"
    ]
    modifiers = [
        ["aircraft", "airplane", "fighter", "flying", "jet", "plane"],
        ["family", "new", "sports", "vintage"], 
        ["flying", "in a tree", "indoors", "on water", "outdoors", "walking"], 
        ["indoors", "outdoors", "walking", "running", "eating", "jumping", "sleeping", "sitting"], 
        ["herd", "in a field", "in the forest", "outdoors", "running", "wildlife photography"], 
        ["indoors", "outdoors", "walking", "running", "eating", "jumping", "sleeping", "sitting"], 
        ["European", "in the forest", "on a tree", "on the ground", "swimming", "tropical", "wildlife photography"],
        ["herd", "in a field", "in the forest", "outdoors", "running", "wildlife photograpahy"], 
        ["at sea", "boat", "cargo", "cruise", "on the water", "river", "sailboat", "tug"], 
        ["18-wheeler", "car transport", "fire", "garbage", "heavy goods", "lorry", "mining", "tanker", "tow"]
    ]
    output_folder = [
        "loraDataSet1/SD21Airplane",
        "loraDataSet1/SD21Automobile",
        "loraDataSet1/SD21Bird",
        "loraDataSet1/SD21Cat",
        "loraDataSet1/SD21Deer",
        "loraDataSet1/SD21Dog",
        "loraDataSet1/SD21Frog",
        "loraDataSet1/SD21Horse",
        "loraDataSet1/SD21Ship",
        "loraDataSet1/SD21Truck" 
    ]


    batch_size = 16
    num_images = 6000
    generate_images(prompts[promptNumber], modifiers[promptNumber], num_images, batch_size, output_folder[promptNumber])

workerID = int(input("enter worker id: "))
fulltaskworker(workerID)
fulltaskworker(workerID + 1)
fulltaskworker(workerID + 2)
fulltaskworker(workerID + 3)
fulltaskworker(workerID + 4)

