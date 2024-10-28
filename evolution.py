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
model = SoloForCausalLM.from_pretrained(MODEL_NAME)


DEVICE = "cuda:0"
model = model.to(DEVICE)

B_INST, E_INST = "[INST]", "[/INST]"

B_INST, E_INST = "[INST]", "[/INST]"

def prepare_inputs(inputs: list, device: str):
    NON_VISION_TOKEN = -1

    tokens = []
    attention_masks = []
    vision_patch_indices = []
    vision_patches = []

    for i in inputs:
        if isinstance(i, torch.Tensor):
            # 生成视觉补丁部分
            #import pdb; pdb.set_trace()  # 插入断点检查补丁生成逻辑
            
            patches = i
            n_rows, n_cols = patches.shape[:2]
            n_patches = n_rows * n_cols
            patches = patches.view(n_patches, -1)

            # ---
            img_tokens = ["<vision>"]
            cur_patch_indices = [NON_VISION_TOKEN]

            # 初始化视觉补丁索引
            current_patch_index = 0

            for row_idx in range(n_rows):
                if row_idx != 0:
                    img_tokens.append("<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                for col_idx in range(n_cols):
                    img_tokens.append("<vpatch>")
                    cur_patch_indices.append(current_patch_index)
                    current_patch_index += 1  # 按顺序增加补丁索引

            img_tokens.append("</vision>")
            cur_patch_indices.append(NON_VISION_TOKEN)


            # ---
            cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
            cur_attention_mask = [1] * len(cur_tokens)

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

            tokens.extend(cur_tokens)
            attention_masks.extend(cur_attention_mask)
            vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))

    tokens = torch.Tensor(tokens).long()
    attention_masks = torch.Tensor(attention_masks).long()
    if len(vision_patches) > 0:
        vision_patches = torch.Tensor(vision_patches).float()
    else:
        vision_patches = None
    vision_patch_indices = torch.Tensor(vision_patch_indices).long()

    # 检查长度是否一致
    #import pdb; pdb.set_trace()  # 在这里插入断点，检查tokens和vision_patch_indices的长度和内容

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
    # generate text
    vision_patches = torch.Tensor(vision_patches).float()
    with torch.no_grad():

        outputs = model.generate(
            input_ids=tokens.unsqueeze(0),
            # model_kwargs
            attention_mask=attention_masks.unsqueeze(0),
            vision_patches=vision_patches,
            vision_patch_indices=vision_patch_indices.unsqueeze(0),
            generation_config=GenerationConfig(
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                # repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                # anything above 32000 (vision related tokens) will be suppressed
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
