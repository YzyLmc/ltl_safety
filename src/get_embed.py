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

def generate_embeds(embed_model, save_dpath, action_list, embed_engine=None, update_embed=True, custom_filename=None):
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
    # if os.path.isfile(save_fpath):
    #     act2embed = load_from_file(save_fpath)

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
    if custom_filename:
        save_to_file(act2embed, custom_filename)
    else:
        save_to_file(act2embed, save_fpath)
    return act2embed, save_fpath

# def build_allowed_actions(acts, objs):
#     allowed_actions = []
#     for act in acts:
#         for obj in objs:
#             allowed_actions.append(convert_to_nl(act, obj))
#     return allowed_actions

def build_allowed_actions(obj_dict):
    grab_objs = [obj for obj, state in obj_dict.items() if state["grabbable"]]
    allowed_actions = []
    for obj, state in obj_dict.items():
        allowed_actions.append(convert_to_nl('walk', [obj])) # walk always executable
        if state["receptacle"]:
            obj_tuples = [(grab_obj, obj) for grab_obj in grab_objs]
            allowed_actions.extend([convert_to_nl('puton', obj_tuple) for obj_tuple in obj_tuples])
        if state["touchable"]:
            allowed_actions.append(convert_to_nl('touch', [obj]))
        if state["grabbable"]:
            allowed_actions.append(convert_to_nl('grab', [obj]))
            allowed_actions.append(convert_to_nl('find', [obj]))
        if state["openable"]:
            allowed_actions.append(convert_to_nl('open', [obj]))
            allowed_actions.append(convert_to_nl('close', [obj]))
    return allowed_actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt3", choices=["gpt3", "llama"])
    parser.add_argument("--embed_engine", type=str, default="text-embedding-ada-002")
    args = parser.parse_args()

## action embedding for robot demo
    # generate allowed actions
    # two_arg_acts = ['putin', 'puton']
    # one_arg_acts = ['walk', 'open', 'close', 'grab', 'touch', 'lookat']
    # zero_arg_acts = ['standup', 'sitdown']
    # obj_movable = ["mail", "phone"]
    # obj_loc = ['television', 'mail_room', 'bedroom', 'office_table', 'mail_box', 'bed']

    # allowed_actions = []
    # for act in one_arg_acts + two_arg_acts:
    #     if act == "walk":
    #         allowed_actions += build_allowed_actions([act], [[obj] for obj in obj_loc])
    #     elif act == "grab":
    #         allowed_actions += build_allowed_actions([act], [[obj] for obj in obj_movable])
    #     elif act == "putin":
    #         objs = [(obj_1, obj_2) for obj_1 in obj_movable for obj_2 in obj_loc]
    #         allowed_actions += build_allowed_actions([act], objs)
    # breakpoint()
    # # save allowed action
    # action_fpath = "virtualhome_v2.3.0/resources/allowed_actions_spot.json"
    # save_to_file(allowed_actions, action_fpath)
    # # generate and save embeddings
    # save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    # _, save_fpath = generate_embeds(args.model, save_fpath, action_fpath, args.embed_engine)

    # objs = ['television', 'bookshelf', 'bedroom', 'ironing_room', 'hallway', 'office_table', 'mail_box', 'coffee_machine', 'pantry', \
    #     'sink', 'entrance', 'sofa', 'kitchen', 'pantry_1', 'fridge', 'origin', 'lamp', 'mail_room', 'ironing_board', 'door', 'sofa_side_table', 'statue']
    # obj_dict = {"television":{"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': False, 'puton':False},
    #                 'bookshelf':{"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': True, 'puton':True},
    #                 'bedroom': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'ironing_room': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'hallway': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'office_table': {"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': False, 'puton':True},
    #                 'mail_box': {"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': True, 'puton':False},
    #                 'coffee_machine': {"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': False, 'puton':True},
    #                 'pantry_1': {"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': True, 'puton':False},
    #                 'sink': {"walk": True, 'close': False, 'grab': False, 'touch': True, 'lookat': True, 'putin': True, 'puton':False},
    #                 'entrance': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'sofa': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':True},
    #                 'kitchen': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':True},
    #                 'fridge': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'lamp': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'mail_room': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'ironing_board': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':True},
    #                 'door': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 'sofa_side_table': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':True},
    #                 'statue': {"walk": True, 'close': False, 'grab': False, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 "mail":  {"walk": True, 'close': False, 'grab': True, 'touch': False, 'lookat': True, 'putin': False, 'puton':False},
    #                 "phone":  {"walk": True, 'close': False, 'grab': True, 'touch': False, 'lookat': True, 'putin': False, 'puton':False}
    #                 }
    # allowed_actions = []

    # for obj, states in obj_dict.items():
    #     for act in states.keys():
    #         if states[act] == True:
    #             if act in one_arg_acts:
    #                 allowed_actions += build_allowed_actions([act], [[obj]])
    #             elif act in two_arg_acts:
    #                 obj_tuples = [(grab_obj, obj) for grab_obj in obj_dict if obj_dict[grab_obj]["grab"]]
    #                 allowed_actions += build_allowed_actions([act], obj_tuples)
    # allowed_actions.extend(zero_arg_acts)

    # objs = ['pantry', 'mail_box', 'lamp', 'fridge', 'book_shelf', 'entrance', 'hallway', 'couch', 'coffee_machine', 'office_table', 'mail_room', 'bedside_table', 'statue', 'door', 'television', 'ironing_room', 'origin', 'sink']
    # obj_dict = {
    #     "pantry": {"location": 0, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     "mail_box": {"location": 1, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     "lamp": {"location": 2, "grabbable":False, "receptacle":False, "touchable": True, "openable":False},
    #     'fridge': {"location": 3, "grabbable":False, "receptacle":False, "touchable":False, "openable":False},
    #     'book_shelf': {"location": 4, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'entrance': {"location": 5, "grabbable":False, "receptacle":False, "touchable":False, "openable":False},
    #     'hallway': {"location": 6, "grabbable":False, "receptacle":False, "touchable":False, "openable":False},
    #     'couch': {"location": 7, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'coffee_machine': {"location": 8, "grabbable":False, "receptacle":False, "touchable":True, "openable":False, "onable": True},
    #     'office_table': {"location": 9, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'mail_room': {"location": 10, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'bedside_table': {"location": 11, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'statue': {"location": 12, "grabbable":False, "receptacle":False, "touchable": False, "openable":False},
    #     'door': {"location": 13, "grabbable":False, "receptacle":False, "touchable":False, "openable":False},
    #     'television': {"location": 14, "grabbable":False, "receptacle":False, "touchable":False, "openable":False, "onable":True},
    #     'ironing_room': {"location": 15, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},
    #     'origin': {"location": 16, "grabbable":False, "receptacle":False, "touchable":False, "openable":False},
    #     'sink': {"location": 17, "grabbable":False, "receptacle":True, "touchable":False, "openable":False},

    #     'book': {"location": 11, "grabbable":True, "receptacle":False, "touchable":False, "openable":False},
    #     'phone': {"location": 9, "grabbable":True, "receptacle":False, "touchable":False, "openable":False},
    #     'mail': {"location": 10, "grabbable":True, "receptacle":False, "touchable":False, "openable":False},
    # }

    # allowed_actions = build_allowed_actions(obj_dict)
    # obj_state_fpath = "virtualhome_v2.3.0/resources/obj_states_spot.json"
    # save_to_file(obj_dict, obj_state_fpath)
    # breakpoint()
    # action_fpath = "virtualhome_v2.3.0/resources/allowed_actions_spot.json"
    # save_to_file(allowed_actions, action_fpath)
    # # generate and save embeddings
    # save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    # _, save_fpath = generate_embeds(args.model, save_fpath, action_fpath, args.embed_engine)
    
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

## obj embedding for VH env 0
    # obj_list_0 = ['bathroom', 'bedroom', 'livingroom', 'kitchen', 'rug', 'curtains', 'ceilinglamp', 'walllamp', 'bathtub', 'towelrack', 'wallshelf', 'stall', 'toilet', 'bathroomcabinet', 'bathroomcounter', 'faucet', 'door', 'doorjamb', 'towel', 'perfume', 'deodorant', 'hairproduct', 'facecream', 'plate', 'toothpaste', 'painkillers', 'waterglass', 'toothbrush', 'barsoap', 'candle', 'window', 'lightswitch', 'washingmachine', 'tablelamp', 'nightstand', 'bookshelf', 'chair', 'desk', 'bed', 'coffeetable', 'closet', 'hanger', 'closetdrawer', 'clothesshirt', 'clothespants', 'clothespile', 'mouse', 'mousemat', 'keyboard', 'computer', 'cpuscreen', 'radio', 'wallpictureframe', 'orchid', 'pillow', 'cellphone', 'photoframe', 'book', 'box', 'mug', 'cupcake', 'wineglass', 'slippers', 'folder', 'garbagecan', 'tvstand', 'kitchentable', 'bench', 'kitchencabinet', 'kitchencounter', 'kitchencounterdrawer', 'sink', 'powersocket', 'wallphone', 'tv', 'clock', 'washingsponge', 'dishwashingliquid', 'fryingpan', 'cutleryknife', 'cutleryfork', 'dishbowl', 'condimentbottle', 'condimentshaker', 'paper', 'stovefan', 'fridge', 'coffeemaker', 'coffeepot', 'toaster', 'breadslice', 'stove', 'oventray', 'microwave', 'bananas', 'whippedcream', 'pie', 'bellpepper', 'salmon', 'chips', 'candybar', 'chocolatesyrup', 'crackers', 'creamybuns', 'cereal', 'sofa', 'cabinet', 'apple', 'lime', 'peach', 'plum', 'remotecontrol']
    # obj_list_fpath = "virtualhome_v2.3.0/resources/obj_list_0.json"
    # save_to_file(obj_list_0, obj_list_fpath)
    # # generate and save embeddings
    # save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    # _, save_fpath = generate_embeds(args.model, save_fpath, obj_list_fpath, args.embed_engine,custom_filename=f"{save_fpath}obj_embeds/0_vh.pkl")
## obj embedding for VH env 1
    # obj_list_0 = ['bathroom', 'bedroom', 'livingroom', 'kitchen', 'rug', 'curtains', 'ceilinglamp', 'walllamp', 'bathtub', 'towelrack', 'wallshelf', 'stall', 'toilet', 'bathroomcabinet', 'bathroomcounter', 'faucet', 'door', 'doorjamb', 'towel', 'perfume', 'deodorant', 'hairproduct', 'facecream', 'plate', 'toothpaste', 'painkillers', 'waterglass', 'toothbrush', 'barsoap', 'candle', 'window', 'lightswitch', 'washingmachine', 'tablelamp', 'nightstand', 'bookshelf', 'chair', 'desk', 'bed', 'coffeetable', 'closet', 'hanger', 'closetdrawer', 'clothesshirt', 'clothespants', 'clothespile', 'mouse', 'mousemat', 'keyboard', 'computer', 'cpuscreen', 'radio', 'wallpictureframe', 'orchid', 'pillow', 'cellphone', 'photoframe', 'book', 'box', 'mug', 'cupcake', 'wineglass', 'slippers', 'folder', 'garbagecan', 'tvstand', 'kitchentable', 'bench', 'kitchencabinet', 'kitchencounter', 'kitchencounterdrawer', 'sink', 'powersocket', 'wallphone', 'tv', 'clock', 'washingsponge', 'dishwashingliquid', 'fryingpan', 'cutleryknife', 'cutleryfork', 'dishbowl', 'condimentbottle', 'condimentshaker', 'paper', 'stovefan', 'fridge', 'coffeemaker', 'coffeepot', 'toaster', 'breadslice', 'stove', 'oventray', 'microwave', 'bananas', 'whippedcream', 'pie', 'bellpepper', 'salmon', 'chips', 'candybar', 'chocolatesyrup', 'crackers', 'creamybuns', 'cereal', 'sofa', 'cabinet', 'apple', 'lime', 'peach', 'plum', 'remotecontrol']
    obj_list_1 = ['bathroom', 'bedroom', 'kitchen', 'livingroom', 'toilet', 'stall', 'bathroomcabinet', 'bathroomcounter', 'sink', 'faucet', 'curtains', 'toothbrush', 'waterglass', 'barsoap', 'deodorant', 'facecream', 'hairproduct', 'toothpaste', 'toiletpaper', 'rug', 'wallpictureframe', 'walllamp', 'ceilinglamp', 'doorjamb', 'door', 'lightswitch', 'washingmachine', 'window', 'nightstand', 'desk', 'chair', 'bookshelf', 'bed', 'sofa', 'coffeetable', 'cabinet', 'computer', 'cpuscreen', 'keyboard', 'mouse', 'mousemat', 'radio', 'mug', 'book', 'photoframe', 'box', 'paper', 'papertray', 'cellphone', 'folder', 'apple', 'bananas', 'lime', 'peach', 'plum', 'dishbowl', 'pillow', 'tablelamp', 'wallphone', 'powersocket', 'cutleryknife', 'knifeblock', 'fryingpan', 'cookingpot', 'plate', 'dishwashingliquid', 'condimentshaker', 'clothespile', 'garbagecan', 'candle', 'bench', 'kitchentable', 'tvstand', 'kitchencabinet', 'kitchencounter', 'kitchencounterdrawer', 'stovefan', 'fridge', 'stove', 'oventray', 'dishwasher', 'coffeemaker', 'coffeepot', 'toaster', 'breadslice', 'microwave', 'chicken', 'cutlets', 'creamybuns', 'chips', 'chocolatesyrup', 'poundcake', 'closet', 'tv', 'orchid', 'hanger', 'clothesshirt', 'clothespants', 'remotecontrol']
    obj_list_fpath = "virtualhome_v2.3.0/resources/obj_list_1.json"
    save_to_file(obj_list_1, obj_list_fpath)
    # generate and save embeddings
    save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    _, save_fpath = generate_embeds(args.model, save_fpath, obj_list_fpath, args.embed_engine,custom_filename=f"{save_fpath}obj_embeds/vh_1.pkl")

## obj embedding for robot demo
    # # objs_spot_0 = ['pantry', 'mail_box', 'lamp', 'fridge', 'book_shelf', 'entrance', 'hallway', 'couch', 'coffee_machine', 'office_table', 'mail_room', 'bedside_table', 'statue', 'door', 'television', 'ironing_room', 'origin', 'sink']
    # objs_spot_0 = list(obj_dict.keys())
    # obj_list_fpath = "virtualhome_v2.3.0/resources/obj_list_spot.json"
    # save_to_file(objs_spot_0, obj_list_fpath)
    # # generate and save embeddings
    # save_fpath = "/users/zyang157/data/zyang157/virtualhome/"
    # _, save_fpath = generate_embeds(args.model, save_fpath, obj_list_fpath, args.embed_engine,custom_filename=f"{save_fpath}obj_embeds/0_spot.pkl")