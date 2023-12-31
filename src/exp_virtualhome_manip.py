import sys
sys.path.append("virtualhome_v2.3.0/")
import random
import numpy as np
from collections import OrderedDict, defaultdict
import argparse

import os
import openai
from openai.embeddings_utils import cosine_similarity
from utils import program2example, load_from_file, save_to_file, GPT4, get_action_and_obj, convert_old_program_to_new, read_pose, prop_level_traj, ltl2digraph, validate_next_action, progress_ltl, reprompt, state_change_by_step, convert_rooms, state_change_by_step_manipulation, manipulation_dict, evaluate_completeness, program2nl
from program_conversion import replace_new_obj_id
from get_embed import GPT3
from constraint_module import constraint_module

openai.api_key = os.getenv("OPENAI_API_KEY")

from simulation.unity_simulator import comm_unity

random.seed(123)
np.random.seed(123)

def main():
    # load dataset
    dataset = load_from_file(args.dataset_fpath)
    task_dict = dataset[args.exp][args.env_num][args.example_fname]

    if args.diverse:
        constraint_name = f"{args.constraint_num}_diverse"
    else:
        constraint_name = args.constraint_num

    constraints = task_dict["constraints"][constraint_name] # 1 - 5

    # start vh simulator
    EXEC_FNAME= "/users/zyang157/data/zyang157/virtualhome/exec_v2.3.0/linux_exec.v2.3.0.x86_64"
    comm = comm_unity.UnityCommunication(file_name=EXEC_FNAME, no_graphics=True, logging=False, port=args.port)
    comm.reset(args.env_num)
    
    # spawn agent
    s, g = comm.environment_graph()
    init_room = task_dict['init_room']
    rooms = [node for node in g['nodes'] if node["category"] == "Rooms"]
    assert init_room in [room["class_name"] for room in rooms]
    init_position = [node for node in g['nodes'] if node["class_name"] == init_room][-1]['bounding_box']['center']
    # comm.add_character('Chars/Female2', position=init_position, initial_room=init_room)
    comm.add_character('Chars/Female2', initial_room=init_room)
    s, g = comm.environment_graph()

    room2id = {room["class_name"]: room["id"] for room in rooms}
    obj2id = {node["class_name"]: node["id"] for node in g["nodes"]}
    # save translation
    translation_result = load_from_file(args.translation_result_fpath)
    if args.exp not in translation_result:
        translation_result[args.exp] = {}
    if args.env_num not in translation_result[args.exp]:
        translation_result[args.exp][args.env_num] = {}
    if args.example_fname not in translation_result[args.exp][args.env_num]:
        translation_result[args.exp][args.env_num][args.example_fname] = {}
    if constraint_name not in translation_result[args.exp][args.env_num][args.example_fname]:
        translation_result[args.exp][args.env_num][args.example_fname][constraint_name] = {}
    if translation_result[args.exp][args.env_num][args.example_fname][constraint_name]:
        trans = translation_result[args.exp][args.env_num][args.example_fname][constraint_name]
        pred_mapping = trans["unified_trans"]["predicate"]
        obj_mapping = trans["unified_trans"]["object"]
        grounded_pred_mapping = trans["unified_trans"]["grounded_pred"]
        input_ltl = trans["unified_trans"]["unified_ltl"]
        sym_utts = trans["sub_trans"]["sym_utt"]
        sym_ltls = trans["sub_trans"]["sym_ltl"]
        placeholder_maps = trans["sub_trans"]["placeholder"]
    else:
        obj_embed_fpath = f"{args.obj_embed_prefix}vh_{args.env_num}.pkl"
        cm = constraint_module()
        sym_utts, sym_ltls, out_ltls, placeholder_maps, input_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraints, log_subformulas=True, obj_embed_fpath=obj_embed_fpath)
        grounded_pred_mapping = {}
        for prop, pred in pred_mapping.items():
            for placeholder, obj in obj_mapping.items():
                pred = pred.replace(placeholder, obj)
            grounded_pred_mapping[prop] = pred
        trans = {"sub_trans":{"sym_utt": sym_utts, "sym_ltl": sym_ltls, "placeholder": placeholder_maps}, "unified_trans":{"unified_ltl":input_ltl, "grounded_pred":grounded_pred_mapping, "object":obj_mapping, "predicate":pred_mapping} }
        if not args.no_log:
    # take_num = len(translation_result[args.example_fname][args.constraint_num]
            translation_result[args.exp][args.env_num][args.example_fname][constraint_name] = trans
            save_to_file(translation_result, args.translation_result_fpath)

    print('sym_utts', sym_utts)
    print('sym_ltls', sym_ltls)
    print('placeholder_maps', placeholder_maps)
    
    grounded_pred_mapping = {}
    for prop, pred in pred_mapping.items():
        for placeholder, obj in obj_mapping.items():
            pred = pred.replace(placeholder, obj)
        grounded_pred_mapping[prop] = pred
    
    # load task
    TASK_PREFIX = "virtualhome_v2.3.0/dataset/ltl_safety/tasks/"
    lines = load_from_file(f"{TASK_PREFIX}{args.example_fname}.txt")
    # construct prompt
    task_description = load_from_file(args.planning_ts_fpath)
    if args.append_obj_list:
        filter_list = ['character', 'floor', 'wall', 'ceiling']
        obj_list = [obj for obj in obj2id if (obj not in room2id.keys()) and (obj not in filter_list)]
        task_description += f"\nObject: {obj_list}".replace("[", "{").replace("]", "}")

    example = program2example(lines)
    example = "\n".join(example)
    prompt = task_description+ "\n\n" + example + "\n"

    # give goal according to each prompt
    dummy_query = task_dict["instruction"]
    # generate description at first step
    if args.safety_level == "bad":
        prompt += f"\n{dummy_query}\nConstraints: {constraints}\nDescription:"
    else:
        prompt += f"\n{dummy_query}\nDescription:"
    print(prompt)
    gpt4 = GPT4(stop=['\n', 'Clean the kitchen', 'Plan'])
    embed_engine = GPT3(engine="text-embedding-ada-002")
    # first response is description
    output = gpt4.generate(prompt)[0]
    print(output)
    # load graph and embeddings
    prompt = prompt + f"\n{output}"
    allowed_actions = load_from_file("virtualhome_v2.3.0/resources/allowed_actions.json")
    act2embed = load_from_file(f"{args.act_embed_prefix}_0.pkl")

    n_try = 0
    program = []
    valid_action2states = []
    invalid_action_mem = None
    stopped = False
    idx = 0
    reprompted =False
    while n_try < args.max_step: # max step + replan time <=10
        n_try += 1
        prompt += f"\n{idx}."
        output = gpt4.generate(prompt)[0]
        print(output)

        if "DONE" in output: 
            stopped = True
            action_string = "STOP" # try to stop
        else:
            action_string = output.strip(".").split(". ")[-1]
            if not action_string in allowed_actions:
                action_string_embed = embed_engine.get_embedding(action_string)
                sims = {o: cosine_similarity(e, action_string_embed) for o, e in act2embed.items()}
                sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
                action_string = list(dict(sims_sorted[:1]).keys())[0]
                print('action_string', action_string)
            act, obj_ls = get_action_and_obj(action_string)
            obj_ls = [f"{obj} (0)" for obj in obj_ls]
            dummy_program_line = f"{act} {' '.join(obj_ls)}"
            dummy_program_line = convert_rooms(dummy_program_line)
            # ground output action string into program
            try:
                grounded_program_line = replace_new_obj_id([dummy_program_line], g)[0]
                program.extend(convert_old_program_to_new([grounded_program_line]))
            except:
                raise Exception(f"probably parsing problem: {dummy_program_line}")
            # masking unsafe action out
            if invalid_action_mem == grounded_program_line:
                continue
        if args.safety_level == "full":
            # this step ensure output is executable and renderable (?)
            success, state_list, manip_dict = state_change_by_step_manipulation(comm, program, input_ltl, obj2id, room2id, obj_mapping, pred_mapping, init_position, init_room, stopped=stopped, env_num=args.env_num)
            prompt += f" {program2nl([grounded_program_line])[0]}" if not output == "DONE" else f" {output}"
            if not success:
                invalid_action = action_string
                invalid_state_list = state_list
                reprompted_plan = f"\nError: {reprompt(gpt4, valid_action2states, invalid_action, invalid_state_list, constraints, grounded_pred_mapping)} Maybe try walking to other objects in the object list first. The correct plan would be:"
                program = program[:-1] if not "DONE" in output else program
                print("handling error by reprompting")
                print(reprompted_plan)
                prompt += reprompted_plan
                invalid_action_mem = grounded_program_line
                stopped = False
                reprompted = True
            else:
                if "DONE" in output: break
                valid_action2states.append((action_string,state_list))
                invalid_action_mem = None
                idx += 1
                reprompted = False

            # breakpoint()
        else: # null or bad safety constraints
            if "DONE" in output:
                stopped = True
                success, state_list, manip_dict = state_change_by_step_manipulation(comm, program, input_ltl, obj2id, room2id, obj_mapping, pred_mapping, init_position, init_room, stopped=stopped)
                break
            prompt += f" {output}"
            idx += 1
    print("Programs:")
    [print(line) for line in program]
    print(prompt)

    # evaluate completeness
    if args.diverse: # expert formula must have been stored
        trans = translation_result[args.exp][args.env_num][args.example_fname][args.constraint_num]
        pred_mapping = trans["unified_trans"]["predicate"]
        obj_mapping = trans["unified_trans"]["object"]
        grounded_pred_mapping = trans["unified_trans"]["grounded_pred"]
        input_ltl = trans["unified_trans"]["unified_ltl"]
        comm.reset(args.env_num)
        success, state_list, manip_dict = state_change_by_step_manipulation(comm, program, input_ltl, obj2id, room2id, obj_mapping, pred_mapping, init_position, init_room, stopped=True)
    # goal_state = {'salmon': {'is_in': 'fridge'}}
    goal_state = task_dict['goal_state']
    complete = evaluate_completeness(manip_dict, goal_state)
    if not args.no_log:
        if args.diverse:
            result_path = f"{args.saved_results_fpath.split('.')[0]}_diverse.json"
        else:
            result_path = args.saved_results_fpath
        if os.path.exists(result_path):
            saved_results = load_from_file(result_path)
        else:
            saved_results = defaultdict(dict)
        result = {"constraint": constraints, "program":program, "safe": success, "completed": complete, "safety_level": args.safety_level}
        if args.exp not in saved_results:
            saved_results[args.exp] = {}
        if str(args.env_num) not in saved_results[args.exp]:
            saved_results[args.exp][str(args.env_num)] = {}
        if args.example_fname in saved_results[args.exp][str(args.env_num)]:
            saved_results[args.exp][str(args.env_num)][args.example_fname].append(result)
        else:
            saved_results[args.exp][str(args.env_num)][args.example_fname] = [result]
        save_to_file(saved_results, result_path)
    else:
        print("completed", complete)
        print("safe", success)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["rooms", "mobile_manip"])
    parser.add_argument("--env_num", type=str, default='0', choices=['0', '1', '2', '3', '4'])
    parser.add_argument("--append_obj_list", action="store_true", help="append the full list of objects in the env in the task specification")
    parser.add_argument("--safety_level", default="full", choices=["full", "bad", "null"], help="full for safety chip, bad for input everything together, null for no safety constraints")
    parser.add_argument("--dataset_fpath", type=str, default="virtualhome_v2.3.0/dataset/ltl_safety/vh/task_vh.json")
    parser.add_argument("--planning_ts_fpath", type=str, default="prompts/planning/planning_with_cons_v2.txt", help="task specification for planning")
    parser.add_argument("--example_fname", type=str, default="0_10", help="name of the text file for tasks")
    parser.add_argument("--constraint_num", type=str, default='5', help='number of constraints applied')
    parser.add_argument("--max_step", type=int, default=15, help="max step of generation")
    parser.add_argument("--saved_results_fpath", type=str, default="results/results_vh.json", help="filepath for saved experiment results")
    parser.add_argument("--translation_result_fpath", type=str, default="results/translation_vh.json")
    parser.add_argument("--act_embed_prefix", type=str, default="/users/zyang157/data/zyang157/virtualhome/action_embeds/act2embed_vh_gpt3-text-embedding-ada-002_vh")
    parser.add_argument("--act_list", type=str, default="virtualhome_v2.3.0/resources/allowed_actions.json")
    parser.add_argument("--obj_embed_prefix", default="/users/zyang157/data/zyang157/virtualhome/obj_embeds/")
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--port", type=str, default="8080", help="port number for unity, if you want to run multiple simulation in parallel")
    parser.add_argument("--diverse", action="store_true", help="evaluate use expert, inference use diverse")
    args = parser.parse_args()

    main()