import requests
import uuid
import time
import subprocess

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os.path
import os

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from .utils import get_query_from_input, get_text_emb

ID = "" # todo

# APP_PATH = "/Users/me/app/"
APP_PATH = "/app/"

DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

#DIALOGUE_DICT = {}

# bad_words_ids = tokenizer(["\nUser: ", "\n Bot:",], add_special_tokens=False).input_ids
bad_words_ids = [
    [29871, 13, 2659, 29901, 29871],
    [29871, 13, 11273, 29901],
]

gen_params = {
    "do_sample": False,
    "max_new_tokens": 80,
    "early_stopping": True,
    "num_beams": 1,
    "remove_invalid_values": True,
    "eos_token_id": 29889,
    "pad_token_id": 29889,
    "forced_eos_token_id": 29889,
    "use_cache": True,
    "bad_words_ids": bad_words_ids,
    "num_return_sequences": 1,
}

# --

@torch.no_grad()
def gen_answer(model, tokenizer, query, history=None):

    query = torch.cat([history, query], dim=1)

    out = model.generate(
        inputs_embeds=query,
        **gen_params,
    )

    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)

    print("\n=== gen_answer ===\n", generated_texts[0])

    return generated_texts[0]

# --

def imagebind_huge(pretrained=False):
    model = imagebind_model.ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        model.load_state_dict(torch.load(APP_PATH+ ".checkpoints/imagebind_huge.pth"))

    return model


# --- SETUP ---

# Function that returns model and tokenizer that will be used during the generation
def setup_model_and_tokenizer():

    # todo: allow re-entrant

    # debug
    print("\nStarting LLaMAZoo... ", APP_PATH + "llamazoo")
    llamazoo = subprocess.Popen([
        APP_PATH + "llamazoo", 
        "--server"
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)

    print("\nWaiting for a minute...")
#    time.sleep(10) # debug

    tokenizer = None
    model = None

    tokenizer = AutoTokenizer.from_pretrained(APP_PATH + "Llama-2-7B-fp16", padding_side="left", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(APP_PATH +  "Llama-2-7B-fp16", torch_dtype=torch.float16).eval().to(device=DEVICE)

    # Instantiate model for image and audio embeddings
    model_imagebind = imagebind_huge(pretrained=True).eval().to(device=DEVICE)
    model_imagebind.query_dict = {}
        
    EMB_DIM = 4096
    N_MODALITY_EMBS = 32
    ENC_DIM = model_imagebind.modality_heads[ModalityType.VISION][-1].out_features

    projection = nn.Linear(ENC_DIM, N_MODALITY_EMBS * EMB_DIM).to(device=model.device, dtype=model.dtype).eval()
    workdir = os.getcwd()

    print("\n workdir = ", workdir)

    img_tokens_emb = None
    img_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/IMG_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

    audio_tokens_emb = None
    audio_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/AUDIO_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

    projection = None
#    projection = torch.load(
#        f"{workdir}/team_code/ckpts/projection_LLaMa-7b-EN-Linear-ImageBind",
#        map_location=model.device,
#    )
    projection = torch.load(
        APP_PATH + "projection_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

    return [
        model,
        model_imagebind,
        img_tokens_emb,
        audio_tokens_emb,
        projection,
    ], tokenizer

# --- GENERATE ---

# Function that generates the responses for dialodues queries w.r.t. history.
def generate_text(model, tokenizer, cur_query_list, history_tensor=None):

    # -- handle history

    if history_tensor is not None:
        try:
            history_tensor = torch.concat(
                [history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])],
                dim=1,
            )
        except:
            print("[ERR] Exception with history_tensor")

    else:
        # If the current history is empty
        # it is assigned to the system prompt
        PROMPT = "This is a dialog with AI assistant.\n"
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    # -- redirect simple text questions to LLaMAZoo

    prompt = None
    response = None

    if len(cur_query_list) == 1 and cur_query_list[0]["type"] == "text":

        try:

            id = ID
            prompt = cur_query_list[0]["content"]
            status = ""

            if id == "":
                id = str(uuid.uuid4()) # todo 

            r = requests.post("http://127.0.0.1:8888/jobs", json={
                "id": id,
                "prompt": prompt 
            })

            print(f"Status Code: {r.status_code}")

            while r.status_code == 200 and status != "finished":

                time.sleep(10) # debug
                r = requests.get("http://127.0.0.1:8888/jobs/" + id)
                print(r.json()) # debug
                status = r.json()["status"]
            
            response = r.json()["output"]
            print("\n=== RESPONSE ===\n", response)

        except Exception as error:

            print("\n=== EXCEPTION ===\n", error)

    # -- otherwise handle with baseline

    prompt = get_query_from_input(model, tokenizer, cur_query_list).to(DEVICE)

    # if response == None or response == "":

    response = gen_answer(model[0], tokenizer, prompt, history=history_tensor)
    print("\n=== BASELINE RESPONSE ===\n", response)

    # -- update history and return results    

 #   history_tensor = torch.concat([history_tensor, prompt], dim=1)
    tmp = [ history_tensor, prompt ]
    history_tensor = torch.Tensor(tmp) # debug

    return response, history_tensor


# --- PPL ---

def get_ppl(model, tokenizer, cur_query_tuple, history_tensor=None):
    if history_tensor is not None:
        history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
    else:
        # If the current history is empty
        # it is assigned to the system prompt
        PROMPT = "This is a dialog with AI assistant.\n"
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    current_query = get_query_from_input(model, tokenizer, cur_query_tuple[0])
    current_answer = get_text_emb(model[0], tokenizer, cur_query_tuple[1])

    # Input dialogue query with history
    dialogue_emb = torch.concat([history_tensor, current_query], dim=1).to(DEVICE)
    inputs_embeds=torch.concat([dialogue_emb, current_answer], dim=1)
    
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        out_logits = model[0](inputs_embeds=inputs_embeds).logits

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb
