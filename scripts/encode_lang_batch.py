import os
import json
import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder

# Specify the GPU index to use
GPU = 0

# Path to the pretrained T5 model (symbolic link in the 'google' directory)
MODEL_PATH = "google/t5-v1_1-xxl"

# Path to the configuration file
CONFIG_PATH = "configs/base.yaml"

# Path to your dataset directory
TARGET_DIR = "/media/syp/新加卷/rdt_data/hdf5_kuavo_dataset"

# If your GPU has less than 24GB VRAM, consider enabling offloading by specifying an offload directory.
# Ensure the directory exists if you choose to use it.
OFFLOAD_DIR = None  # e.g., "/media/syp/新加卷/rdt_data/models/t5_offload"

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # Collect all task directories
    task_paths = []
    for sub_dir in os.listdir(TARGET_DIR):
        middle_dir = os.path.join(TARGET_DIR, sub_dir)
        if os.path.isdir(middle_dir):
            for task_dir in os.listdir(middle_dir):
                task_path = os.path.join(middle_dir, task_dir)
                if os.path.isdir(task_path):
                    task_paths.append(task_path)

    # Process each task to generate embeddings
    for task_path in tqdm(task_paths):
        # Load the instruction JSON file
        instruction_file = os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json')
        with open(instruction_file, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        
        # Combine different instruction types into a single list
        instructions = [instruction_dict['instruction']] + \
                       instruction_dict['simplified_instruction'] + \
                       instruction_dict['expanded_instruction']
        
        # Tokenize the instructions
        tokenized_res = tokenizer(
            instructions, return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)
        
        # Generate embeddings without computing gradients
        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()
        
        attn_mask = attn_mask.cpu().bool()

        # Save each embedding separately
        for i in range(len(instructions)):
            text_embed = text_embeds[i][attn_mask[i]]
            save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
            torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
