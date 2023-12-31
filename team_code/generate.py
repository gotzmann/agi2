import requests
import uuid
import time
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import os.path
import os

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# from .utils import get_query_from_input, get_text_emb

DEBUG = False # True
PROMPT = "You are smart AI assistant. Please read the dialog and answer the question. Be short and precise!\n"

DEVICE = "cuda:0"
EMB_DIM = 4096
N_MODALITY_EMBS = 32
APP_PATH = "/app/"
# DEVICE = torch.device("cuda:0")

# USER = "\nUSER: "
# ASSISTANT = "\nASSISTANT:"

USER =  "\nUser: " # "\nUser: "
ASSISTANT = "\n Bot:" # "\n Bot: "

DIALOGUE_DICT = {}

# bad_words_ids = tokenizer(["\nUser: ", "\n Bot:",], add_special_tokens=False).input_ids
bad_words_ids = [
    [29871, 13, 2659, 29901, 29871],
    [29871, 13, 11273, 29901],
]

gen_params = {
    "do_sample": False,
    "max_new_tokens": 80,
    "early_stopping": True,
    "num_beams": 4, # 1,
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
    num = len(history[0])
    # query = torch.cat([history, query], dim=1)
    query = torch.cat([history[0][num-1]["embd"], query], dim=1)
#    print("\n\n=== gen_answer :: query ===\n\n", query)

    out = model.generate(
        inputs_embeds=query,
        **gen_params,
    )

#    print("\n\n=== gen_answer :: out 1 ===\n\n", out)
    out = out[:, 1:] # remove BOS token
#    print("\n\n=== gen_answer :: out 2 ===\n\n", out)

    generated_texts = tokenizer.batch_decode(out)
#    print("\n\n=== gen_answer :: generated_texts ===\n\n", generated_texts)
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
        #model.load_state_dict(torch.load(APP_PATH + ".checkpoints/imagebind_huge.pth"))

    return model


# --- SETUP ---

# Function that returns model and tokenizer that will be used during the generation
def setup_model_and_tokenizer():

    print("\n=== SuperMachina v0.12 ===\n")

    workdir = os.getcwd()
    # print("\nWORKDIR = ", workdir)

    configPath = APP_PATH + "config.yaml"
    freshConfig = workdir + "/config.yaml"
    if os.path.exists(freshConfig):
        configPath = freshConfig

    # todo: allow re-entrant
    # Reset GPU: nvidia-smi --gpu-reset
    # Find and kill processes: lsof | grep /dev/nvidia

    print("\nStarting LLaMAZoo... ", APP_PATH + "llamazoo --config", configPath)
    llamazoo = subprocess.Popen([
            APP_PATH + "llamazoo",
            "--server",
            "--config",
            configPath,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    print("\nWaiting for 5 minutes...")
    time.sleep(300) # debug

#    tokenizer = None
#    model = None

    tokenizer = AutoTokenizer.from_pretrained(APP_PATH + "Llama-2-7B-fp16", padding_side="left", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(APP_PATH +  "Llama-2-7B-fp16", torch_dtype=torch.float16).eval().to(DEVICE)

    # Instantiate model for image and audio embeddings
    model_imagebind = imagebind_huge(pretrained=True).eval().to(DEVICE)
    model_imagebind.query_dict = {}

#    EMB_DIM = 4096
#    N_MODALITY_EMBS = 32
    ENC_DIM = model_imagebind.modality_heads[ModalityType.VISION][-1].out_features

    projection = nn.Linear(ENC_DIM, N_MODALITY_EMBS * EMB_DIM).to(device=model.device, dtype=model.dtype).eval()

#    img_tokens_emb = None
    img_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/IMG_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

#    audio_tokens_emb = None
    audio_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/AUDIO_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

#    projection = None
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

    num = 0 # number of current iteration in history

    # -- handle history

    # If the current history is empty - it is assigned to the system prompt
    if history_tensor is None:
#        PROMPT = "This is a dialog with AI assistant.\n"
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        # history_tensor = prompt_embeddings
# debug        history_tensor = get_text_emb(model[0], tokenizer, PROMPT)
        # debug
        history_tensor = ([
            {
                "id": "",
                "session": "",
                "prompt": "",
                "response": "",
                "embd": prompt_embeddings
            }
        ], "")

    else:
        # print("\n === GET TEXT HISTORY ===\n", history_tensor) # debug
        num = len(history_tensor[0])
        embd = torch.concat(
            [
                history_tensor[0][num-1]["embd"],
                get_text_emb(model[0], tokenizer, history_tensor[1])
            ], dim=1)
        history_tensor[0].append(
            {
                "id": "",
                "session": "",
                "prompt": "",
                "response": "",
                "embd": embd
            })
#        try:
        #history_tensor = torch.concat(
        #    [
        #        history_tensor[0],
        #        get_text_emb(model[0], tokenizer, history_tensor[1])
        #    ], dim=1)

#        except Exception as error:
#            print("\n=== [ERR] === Exception with history_tensor ===\n", error)

    # debug

    #print("\n\n=== tokenizer.batch_decode(history_tensor) ===\n\n")
    #tokenizer.batch_decode(history_tensor)

    # -- redirect simple text questions to LLaMAZoo

    prompt = None
    response = None

    for part in cur_query_list:
        if part["type"] == "text":
            prompt = part["content"]
            history_tensor[0][num]["prompt"] = prompt

    # -- Generate answer with 70B model only if:
    #    1) query consist only of one text question and
    #    2) there is no multi-modal history

    if len(cur_query_list) == 1 and cur_query_list[0]["type"] == "text" and (num == 0 or history_tensor[0][num-1]["session"] != ""):

        try:

            id = str(uuid.uuid4())
            session = str(uuid.uuid4())
            if num != 0:
                session = history_tensor[0][num-1]["session"]
            history_tensor[0][num]["id"] = id
            history_tensor[0][num]["session"] = session

            r = requests.post("http://127.0.0.1:8888/jobs", json={
                "id": id,
                "session": session,
                "prompt": prompt
            })

            if DEBUG:
                print(f"=== HTTP STATUS | {r.status_code} ===")

            # todo: timer watchdog?
            status = ""
            while r.status_code == 200 and status != "finished":

                time.sleep(10) # debug
                r = requests.get("http://127.0.0.1:8888/jobs/" + id)
                # print(r.json()) # debug
                status = r.json()["status"]

            try:
                if r.status_code == 200:
                    response = r.json()["output"]
            except Exception as error:
                print("\n=== JSON EXCEPTION ===\n", error)

            history_tensor[0][num]["response"] = response

            if DEBUG:
                print("\n=== LLAMAZOO RESPONSE ===\n", response)

        except Exception as error:
            print("\n=== EXCEPTION ===\n", error)

    # -- otherwise handle with baseline

    prompt = get_query_from_input(model, tokenizer, cur_query_list).to(DEVICE)
    baselineResponse = gen_answer(model[0], tokenizer, prompt, history=history_tensor)
    if DEBUG:
        print("\n=== BASELINE RESPONSE ===\n", baselineResponse)

    if response is None or response == "":
        response = baselineResponse

    # -- update history and return results

    history_tensor[0][num]["response"] = response

    #history_tensor = torch.concat([history_tensor, prompt], dim=1)
    history_tensor[0][num]["embd"] = torch.concat([history_tensor[0][num]["embd"], prompt], dim=1)

    return response, history_tensor[0]


# --- PPL ---

def get_ppl(model, tokenizer, cur_query_tuple, history_tensor=None):

    # print("\n === PPL HISTORY ===\n", history_tensor) # debug

    if history_tensor is None:

        # If the current history is empty - it is assigned to the system prompt
        prompt = "This is a dialog with AI assistant.\n" # todo
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    else:
        num = len(history_tensor[0])
        #history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
        history_tensor = torch.concat([history_tensor[0][num-1]["embd"], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)

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

# === UTILS.py ===

# utils function that parses the format of the input query to a single sequence
def get_query_from_input(model, tokenizer, input_list):

    base_model = model[0]
    model_imagebind = model[1]
    img_tokens_emb = model[2]
    audio_tokens_emb = model[3]
    projection = model[4]

    all_emb = []

    ai_ids = tokenizer.encode(ASSISTANT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ai_embeddings = base_model.model.embed_tokens(ai_ids)

    # prompt = USER
    prompt_ids = tokenizer.encode(USER, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    prompt_embeddings = base_model.model.embed_tokens(prompt_ids)

    all_emb.append(prompt_embeddings)

    for el in input_list:

        if el["type"] == "text":

            query = el["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(DEVICE)
            query_embeddings = base_model.model.embed_tokens(query_ids)
            all_emb.append(query_embeddings)

#            print("\n === TEXT SIZE ===\n\n", len(all_emb))
#            print("\n === TEXT EMB ===\n\n", all_emb)

        elif el["type"] == "image":

            modality_start_emb, modality_end_emb = img_tokens_emb
            filepath = f"{el['content']}"

            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_image(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs

            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM),
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )

#            print("\n === IMAGE SIZE ===\n\n", len(all_emb))
#            print("\n === IMAGE EMB ===\n\n", all_emb)

        else:

            modality_start_emb, modality_end_emb = audio_tokens_emb
            filepath = f"{el['content']}"

            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_audio(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs

            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM),
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )

#            print("\n === AUDIO SIZE ===\n\n", len(all_emb))
#            print("\n === AUDIO EMB ===\n\n", all_emb)

        all_emb.append(ai_embeddings)

        embeddings = torch.cat(
            all_emb,
            dim=1,
        )

    return embeddings


def get_text_emb(model, tokenizer, text):
    if text is None or len(text) == 0:
        text = "I don't know.\n"
    text_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    text_embeddings = model.model.embed_tokens(text_ids)
    return text_embeddings


@torch.no_grad()
def encode_audio(model_imagebind, audio_paths, normalize=True):
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths=audio_paths, device=DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.AUDIO].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings


@torch.no_grad()
def encode_image(model_imagebind, image_paths, normalize=True):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.VISION].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings
