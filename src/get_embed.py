# create embeddings for actions in resources/allowed_actions.json
import os
import argparse
import json
from pathlib import Path
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils import load_from_file, save_to_file, convert_to_nl

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3:
    def __init__(self, engine, temp=0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text):  # engines must match when compare two embeddings
        text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance
        embedding = openai.Embedding.create(
            input=[text],
            model=self.engine  # change for different embedding dimension
        )["data"][0]["embedding"]
        return embedding

def generate_embeds(embed_model, save_dpath, action_list, embed_engine=None, update_embed=True):
    """
    Generate a database of known landmarks and their embeddings.
    :param embed_model: model used to generate embeddings.
    :param save_dpath: folder to save generated embeddings.
    :param action_list: known actions.
    :param keep_keys: filter semantic information of landmarks used to construct embeddings.
    :param embed_engine: embedding engine to use with embedding model , e.g., text-embedding-ada-002
    :param exp_name: experiment ID used in file name.
    :param update_embed: if to append new embeddings to existing embeddings, and overwrite if same landmark name.
    """
    # Load existing embeddings
    embed_dpath = os.path.join(save_dpath, "action_embeds")
    os.makedirs(embed_dpath, exist_ok=True)
    lmk_fname = "vh" # Path(lmk2sem).stem if isinstance(lmk2sem, str) else exp_name
    if embed_model == "gpt3":
        save_fpath = os.path.join(embed_dpath, f"act2embed_{lmk_fname}_{embed_model}-{embed_engine}_spot.pkl")
    else:
        save_fpath = os.path.join(embed_dpath, f"act2embed_{lmk_fname}_{embed_model}.pkl")

    act2embed = {}
    if os.path.isfile(save_fpath):
        act2embed = load_from_file(save_fpath)

    # Generate new embeddings if needed
    action_list = load_from_file(action_list) if isinstance(action_list, str) else action_list

    if embed_model == "gpt3":
        embed_module = GPT3(embed_engine)
    else:
        raise ValueError(f"ERROR: embedding module not recognized: {embed_model}")

    for idx, action in enumerate(action_list):
        if idx % 100 == 0:
            print(f"{idx}/{len(action_list)} done")
        if action not in act2embed:
            act2embed[action] = embed_module.get_embedding(action)

    save_to_file(act2embed, save_fpath)
    return act2embed, save_fpath

def build_allowed_actions(acts, objs):
    allowed_actions = []
    for act in acts:
        for obj in objs:
            allowed_actions.append(convert_to_nl(act, obj))
    return allowed_actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt3", choices=["gpt3", "llama"])
    parser.add_argument("--embed_engine", type=str, default="text-embedding-ada-002")
    args = parser.parse_args()
## action embedding for robot demo
    # generate allowed actions
    two_arg_acts = ['putin', 'puton']
    one_arg_acts = ['walk', 'open', 'close', 'grab', 'touch', 'lookat']
    zero_arg_acts = ['standup', 'sitdown']
    obj_movable = ["mail", "phone"]
    obj_loc = ['television', 'mail_room', 'bedroom', 'office_table', 'mail_box', 'bed']

    allowed_actions = []
    for act in one_arg_acts + two_arg_acts:
        if act == "walk":
            allowed_actions += build_allowed_actions([act], [[obj] for obj in obj_loc])
        elif act == "grab":
            allowed_actions += build_allowed_actions([act], [[obj] for obj in obj_movable])
        elif act == "putin":
            objs = [(obj_1, obj_2) for obj_1 in obj_movable for obj_2 in obj_loc]
            allowed_actions += build_allowed_actions([act], objs)
    breakpoint()
    # save allowed action
    action_fpath = "virtualhome_v2.3.0/resources/allowed_actions_spot.json"
    save_to_file(allowed_actions, action_fpath)
    # generate and save embeddings
    save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    _, save_fpath = generate_embeds(args.model, save_fpath, action_fpath, args.embed_engine)

## action embedding for VH
    # # env_dpath = os.path.join("data", args.env)
    # # lmk_dpath = os.path.join(env_dpath, "lmks")
    # # lmk_fpaths = [os.path.join(lmk_dpath, fname) for fname in os.listdir(lmk_dpath) if "json" in fname]
    # action_fpath = "virtualhome_v1.0.0/resources/allowed_actions.json" 
    # save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    # # for idx, lmk_fpath in enumerate(lmk_fpaths):
    #     # print(f"generating landmark embedding for {lmk_fpath}")
    #     # _, save_fpath = generate_embeds(args.model, env_dpath, lmk_fpath, keep_keys, args.embed_engine)
    #     # print(f"{idx}: embeddings generated by model: {args.model}-{args.embed_engine}\nstored at: {save_fpath}\n")
    # _, save_fpath = generate_embeds(args.model, save_fpath, action_fpath, args.embed_engine)

## obj embedding for VH
