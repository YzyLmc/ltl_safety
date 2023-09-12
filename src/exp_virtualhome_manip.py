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
    constraints = task_dict["constraints"][args.constraint_num] # 1 - 5


    # start vh simulator
    EXEC_FNAME= "/users/zyang157/data/zyang157/virtualhome/exec_v2.3.0/linux_exec.v2.3.0.x86_64"
    comm = comm_unity.UnityCommunication(file_name=EXEC_FNAME, no_graphics=True, logging=False)
    comm.reset(args.env_num)
    
    # spawn agent
    s, g = comm.environment_graph()
    init_room = task_dict['init_room']
    rooms = [node for node in g['nodes'] if node["category"] == "Rooms"]
    assert init_room in [room["class_name"] for room in rooms]
    init_position = [node for node in g['nodes'] if node["class_name"] == init_room][-1]['obj_transform']['position']
    comm.add_character('Chars/Female2', position=init_position, initial_room=init_room)
    s, g = comm.environment_graph()
    # breakpoint()
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
    if args.constraint_num not in translation_result[args.exp][args.env_num][args.example_fname]:
        translation_result[args.exp][args.env_num][args.example_fname][args.constraint_num] = {}
    if translation_result[args.exp][args.env_num][args.example_fname][args.constraint_num]:
        trans = translation_result[args.exp][args.env_num][args.example_fname][args.constraint_num]
        pred_mapping = trans["unified_trans"]["predicate"]
        obj_mapping = trans["unified_trans"]["object"]
        grounded_pred_mapping = trans["unified_trans"]["grounded_pred"]
        input_ltl = trans["unified_trans"]["unified_ltl"]
    else:
        obj_embed_fpath = f"{args.obj_embed_prefix}vh_{args.env_num}.pkl"
        cm = constraint_module()
        sym_utts, sym_ltls, out_ltls, placeholder_maps, input_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraints, log_subformulas=True, obj_embed_fpath=obj_embed_fpath)
        grounded_pred_mapping = {}
        for prop, pred in pred_mapping.items():
            for placeholder, obj in obj_mapping.items():
                pred = pred.replace(placeholder, obj)
            grounded_pred_mapping[prop] = pred
        if args.log:
            trans = {"sub_trans":{"sym_utt": sym_utts, "sym_ltl": sym_ltls, "placeholder": placeholder_maps}, "unified_trans":{"unified_ltl":input_ltl, "grounded_pred":grounded_pred_mapping, "object":obj_mapping, "predicate":pred_mapping} }
        # take_num = len(translation_result[args.example_fname][args.constraint_num]
            translation_result[args.exp][args.env_num][args.example_fname][args.constraint_num] = trans
            save_to_file(translation_result, args.translation_result_fpath)
    
    # breakpoint()


    # test constraints
    # constraints = ["you have to go to living room before pick up salmon"]
    # constraints = ["you have to put pie in fridge before pick up salmon"]
    # constraints = ["you have to put pie in fridge before put salmon in the fridge"]
    # constraints = ["you have to put pie in fridge before putting salmon in fridge", "you have to enter bathroom before entering kitchen", "you have to enter living room in the future if you pick up salmon", "don't go to bedroom if you have put salmon in the fridge", "you can visit kitchen at most two times"]
    # constraints = ["you have to put pie in fridge before putting salmon in fridge", "you have to enter living room in the future if you pick up salmon"]
    # input_ltl = '& W ! c d G i b F a'
    # pred_mapping = {'a': 'agent_at(B)', 'b': 'is_grabbed(A)', 'c': 'is_in(A,D)', 'd': 'is_in(C,D)'}
    # obj_mapping = {'A': 'salmon', 'B': 'livingroom', 'C': 'pie', 'D': 'fridge'}
    # constraints = ["always avoid kitchen table", "always avoid tv stand", "always avoid chair"]
    # input_ltl = '& & G ! b G ! a G ! c'
    # pred_mapping = {'a': 'agent_at(A)', 'b': 'agent_at(B)', 'c': 'agent_at(C)'}
    # obj_mapping = {'A': 'tvstand', 'B': 'kitchentable', 'C': 'chair'}
    # constraints = ["you have to put apple in fridge before putting salmon in fridge", "you have to enter bathroom before entering kitchen", "don't go to living room if you have put apple in fridge"]
    # cm = constraint_module()
    # input_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraints)
    # breakpoint()
   
    # input_ltl = 'W ! b a'
    # obj_mapping = {'A': 'livingroom', 'B': 'salmon'}
    # pred_mapping = {'a': 'agent_at(A)', 'b': 'is_grabbed(B)'}
    # input_ltl = 'W ! b a'
    # pred_mapping = {'a': 'is_in(C,B)', 'b': 'is_grabbed(A)'}
    # obj_mapping = {'A': 'salmon', 'B': 'fridge', 'C': 'pie'}
    # input_ltl = 'W ! b a'
    # pred_mapping = {'a': 'is_in(A,B)', 'b': 'is_in(C,B)'}
    # obj_mapping = {'A': 'pie', 'B': 'fridge', 'C': 'salmon'}
    # input_ltl = '& W ! c a W ! d b'
    # pred_mapping = {'a': 'is_in(C,A)', 'b': 'agent_at(H)', 'c': 'is_in(D,A)', 'd': 'agent_at(B)'}
    # obj_mapping = {'A': 'fridge', 'B': 'kitchen', 'C': 'apple', 'D': 'salmon', 'H': 'bathroom'}
    # input_ltl = '& & W ! a b W ! d h G i b G ! c'
    # pred_mapping = {'a': 'is_in(H,J)', 'b': 'is_in(C,J)', 'c': 'agent_at(D)', 'd': 'agent_at(A)', 'h': 'agent_at(B)'}
    # obj_mapping = {'A': 'kitchen', 'B': 'bathroom', 'C': 'apple', 'D': 'livingroom', 'H': 'salmon', 'J': 'fridge'}
    # input_ltl = '& & & W ! b h W ! k j G i l F a G i c G ! d'
    # pred_mapping = {'a': 'agent_at(J)', 'b': 'is_in(D,K)', 'c': 'is_in(D,C)', 'd': 'agent_at(B)', 'h': 'is_in(L,K)', 'j': 'agent_at(A)', 'k': 'agent_at(H)', 'l': 'is_grabbed(D)'}
    # obj_mapping = {'A': 'bathroom', 'B': 'bedroom', 'C': 'fridge', 'D': 'salmon', 'H': 'kitchen', 'J': 'livingroom', 'K': 'fridge', 'L': 'pie'}
    # breakpoint()
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
    # graph_dict_path = "virtualhome_v2.3.0/env_graphs/TestScene1_graph.json"
    # graph_dict = load_from_file(graph_dict_path)
    allowed_actions = load_from_file("virtualhome_v2.3.0/resources/allowed_actions.json")
    act2embed = load_from_file(f"{args.act_embed_prefix}_{args.env_num}.pkl")

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
            act, obj_ls = get_action_and_obj(action_string)
            obj_ls = [f"{obj} (0)" for obj in obj_ls]
            dummy_program_line = f"{act} {' '.join(obj_ls)}"
            dummy_program_line = convert_rooms(dummy_program_line)
            # ground output action string into program
            try:
                # grounded_program_line = replace_new_obj_id([dummy_program_line], graph_dict)[0]
                grounded_program_line = replace_new_obj_id([dummy_program_line], g)[0]
                program.extend(convert_old_program_to_new([grounded_program_line]))
            except:
                raise Exception(f"probably parsing problem: {dummy_program_line}")
            # masking unsafe action out
            if invalid_action_mem == grounded_program_line:
                continue
        if args.safety_level == "full":
            # this step ensure output is executable and renderable(?)
            success, state_list, manip_dict = state_change_by_step_manipulation(comm, program, input_ltl, obj2id, room2id, obj_mapping, pred_mapping, init_position, init_room, stopped=stopped)
            prompt += f" {program2nl([grounded_program_line])[0]}" if not output == "DONE" else f" {output}"
            if not success:
                invalid_action = action_string
                invalid_state_list = state_list
                reprompted_plan = f"\nError: {reprompt(gpt4, valid_action2states, invalid_action, invalid_state_list, constraints, grounded_pred_mapping)} Maybe try walking to other objects in the object list first. The correct plan would be:"
                program = program[:-1]
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
    # goal_state = {'salmon': {'is_in': 'fridge'}}
    goal_state = task_dict['goal_state']
    complete = evaluate_completeness(manip_dict, goal_state)
    if args.log:
        if os.path.exists(args.saved_results_fpath):
            saved_results = load_from_file(args.saved_results_fpath)
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
        save_to_file(saved_results, args.saved_results_fpath)

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
    parser.add_argument("--max_step", type=int, default=25, help="max step of generation")
    parser.add_argument("--saved_results_fpath", type=str, default="results/results_vh.json", help="filepath for saved experiment results")
    parser.add_argument("--translation_result_fpath", type=str, default="results/translation_vh.json")
    parser.add_argument("--act_embed_prefix", type=str, default="/users/zyang157/data/zyang157/virtualhome/action_embeds/act2embed_vh_gpt3-text-embedding-ada-002_vh")
    parser.add_argument("--act_list", type=str, default="virtualhome_v2.3.0/resources/allowed_actions.json")
    parser.add_argument("--obj_embed_prefix", default="/users/zyang157/data/zyang157/virtualhome/obj_embeds/")
    parser.add_argument("--log", default=True)
    args = parser.parse_args()

    main()