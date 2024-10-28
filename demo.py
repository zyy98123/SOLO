import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig

# Add project root to sys path to import image_utils
import os
import sys
from tqdm import tqdm


from image_utils import (
    load_image_to_base64,
    download_image_to_base64,
    load_base64_to_PILImage,
    convert_image_base64_to_patches,
    visualize_patches
)

from transformers import SoloForCausalLM

MODEL_PATH = "YangyiYY/SOLO-7B"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = SoloForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

DEVICE = "cuda:0"
model = model.to(DEVICE)

# Check tokenizer for reserved token for vision
print(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1], reverse=True)[1020:1030])

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
            
            # ---
            img_tokens = ["<vision>"]
            cur_patch_indices = [NON_VISION_TOKEN]
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx != 0 and col_idx == 0: # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)
                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(len(vision_patches) + row_idx * n_cols + col_idx)
            img_tokens.append("</vision>")
            cur_patch_indices.append(NON_VISION_TOKEN)
            
            # ---
            # NOTE tokenizer(xxx) will NOT work here
            cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
            cur_attention_mask = [1] * len(cur_tokens)
            # print(f"cur_tokens: {cur_tokens}")
            # print(f"cur_attention_mask: {cur_attention_mask}")
            # print(f"cur_patch_indices: {cur_patch_indices}")
            assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
            
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
        vision_patches = torch.Tensor(vision_patches).bfloat16()
    else:
        vision_patches = None
    vision_patch_indices = torch.Tensor(vision_patch_indices).long()

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
    do_sample=False,
    top_p=0.95,
    max_new_tokens=30,
):
    tokens, attention_masks, vision_patches, vision_patch_indices = prepare_inputs(inputs, device=DEVICE)

    with torch.no_grad():
        # 初始化进度条
        with tqdm(total=max_new_tokens, desc="Generating", unit="token") as pbar:
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
                )
            )
            # 每次生成一个新token，更新进度条
            for _ in range(len(outputs[0]) - len(tokens)):
                pbar.update(1)
    
    visualize_outputs(inputs, tokens, outputs)
    
dog_img_base64 = load_image_to_base64(f"{project_root}/HOLO/testIMG.png")
dog_img_patches = convert_image_base64_to_patches(dog_img_base64)
print(f"patch shape: {dog_img_patches.shape}")
# visualize_patches(dog_img_patches)

inputs = [
    dog_img_patches,
    "Which option describe the object relationship in the image correctly? Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book."
]
run_inference_and_print_outputs(inputs)