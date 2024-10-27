import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig

# Add project root to sys path to import image_utils
import os
import sys

current_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from image_utils import (
    load_image_to_base64,
    download_image_to_base64,
    load_base64_to_PILImage,
    convert_image_base64_to_patches,
    visualize_patches
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.model.modeling_solo import SoloForCausalLM, SoloConfig
MODEL_NAME = "YangyiYY/SOLO-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

config = SoloConfig.from_pretrained(MODEL_NAME)

model = SoloForCausalLM(config)

model.load_state_dict(base_model.state_dict(), strict=False)

DEVICE = "cuda:0"
model = model.to(DEVICE)

B_INST, E_INST = "[INST]", "[/INST]"

def prepare_inputs(inputs: list, device: str):
    NON_VISION_TOKEN = -1
    
    tokens = []
    attention_masks = []
    vision_patch_indices = []
    vision_patches = []
    
    for i in inputs:
        if isinstance(i, torch.Tensor):
            # this is patches
            patches = i
            n_rows, n_cols = patches.shape[:2]
            n_patches = n_rows * n_cols
            patches = patches.view(n_patches, -1)
            
            img_tokens = ["<vision>"]
            cur_patch_indices = [NON_VISION_TOKEN]
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx != 0 and col_idx == 0: # when new row starts
                        img_tokens.append("<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)
                    img_tokens.append("<vpatch>")
                    cur_patch_indices.append(len(img_tokens) - 1)
            
            cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
            cur_attention_mask = [1] * len(cur_tokens)
            
            tokens.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend(cur_patch_indices)
            vision_patches.extend(patches.numpy().astype(np.float16))

        elif isinstance(i, str):
            i = tokenizer.bos_token + f"{B_INST} {i.strip()} {E_INST}"
            _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized["input_ids"].squeeze(0)
            cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
            print(f"cur_tokens: {cur_tokens}")
            print(f"cur_attention_mask: {cur_attention_mask}")

            tokens.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

    tokens = torch.Tensor(tokens).long()
    attention_masks = torch.Tensor(attention_masks).long()
    if len(vision_patches) > 0:
        vision_patches = torch.Tensor(np.array(vision_patches)).bfloat16()
    else:
        vision_patches = None
    vision_patch_indices = torch.Tensor(np.array(vision_patch_indices)).long()

    # Replace any -1 with valid index pointing to a dummy zero embedding (if needed)
    if vision_patches is not None:
        dummy_patch = torch.zeros_like(vision_patches[0]).unsqueeze(0)
        vision_patches = torch.cat([vision_patches, dummy_patch], dim=0)
        vision_patch_indices = torch.clamp(vision_patch_indices, min=0, max=vision_patches.shape[0] - 1)
    
    # Ensure vision_patch_indices matches the shape of tokens
    if vision_patch_indices.shape[0] > tokens.shape[0]:
        vision_patch_indices = vision_patch_indices[:tokens.shape[0]]
    elif vision_patch_indices.shape[0] < tokens.shape[0]:
        padding = torch.full((tokens.shape[0] - vision_patch_indices.shape[0],), NON_VISION_TOKEN, dtype=torch.long)
        vision_patch_indices = torch.cat([vision_patch_indices, padding], dim=0)
    
    # move to device
    tokens = tokens.to(device)
    attention_masks = attention_masks.to(device)
    vision_patch_indices = vision_patch_indices.to(device)
    if vision_patches is not None:
        vision_patches = vision_patches.to(device)

    return tokens, attention_masks, vision_patches, vision_patch_indices



def visualize_outputs(inputs, tokens, outputs):
    for idx, s in enumerate(inputs):
        if isinstance(s, str):
            if idx == len(inputs) - 1:
                print(s, end=" [")
                print(tokenizer.decode(outputs[0, len(tokens):], skip_special_tokens=True) + "]")
            else:
                print(s)
        else:
            visualize_patches(s, figsize=(4, 4))

def run_inference_and_print_outputs(
    inputs,
    do_sample=True,
    top_p=0.95,
    max_new_tokens=30,
):
    tokens, attention_masks, vision_patches, vision_patch_indices = prepare_inputs(inputs, device=DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokens.unsqueeze(0),
            attention_mask=attention_masks.unsqueeze(0),
            vision_patches=vision_patches,
            vision_patch_indices=vision_patch_indices.unsqueeze(0),
            generation_config=GenerationConfig(
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                suppress_tokens=[i for i in range(32000, len(tokenizer))],
            ),
        )
    visualize_outputs(inputs, tokens, outputs)

img_base64 = load_image_to_base64(f"{project_root}/westai0019/HOLO/testIMG.png")
img_patches = convert_image_base64_to_patches(img_base64)
print(f"patch shape: {img_patches.shape}")
visualize_patches(img_patches)

inputs = [
    img_patches,
    "This is a"
]
run_inference_and_print_outputs(inputs)

fig1_img_base64 = load_image_to_base64(f"{project_root}/westai0019/HOLO/testIMG.png")
fig1_img_patches = convert_image_base64_to_patches(fig1_img_base64)
print(f"patch shape: {fig1_img_patches.shape}")
visualize_patches(fig1_img_patches)

inputs = [
    fig1_img_patches,
    "Which option describe the object relationship in the image correctly? Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book."
]
run_inference_and_print_outputs(inputs)
