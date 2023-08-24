import openai
from time import sleep
import logging
import json
import csv
import os
from pathlib import Path
import string
from string import ascii_lowercase
from collections import defaultdict
import numpy as np
import dill
import random
import networkx as nx
import spot
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append("virtualhome_v2.3.0/simulation/")
import evolving_graph.utils as utils
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
from evolving_graph.check_programs import modify_objects_unity2script, check_one_program
from evolving_graph.scripts import read_script, read_script_from_string, read_script_from_list_string, ScriptParseException

# General utils
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'pkl':
        with open(fpath, 'rb') as rfile:
            out = dill.load(rfile)
    elif ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line.strip() for line in rfile.readlines()]
    elif ftype == 'json':
        with open(fpath, 'r') as rfile:
            out = json.load(rfile)
    elif ftype == 'csv':
        with open(fpath, 'r') as rfile:
            csvreader = csv.reader(rfile)
            if noheader:
                fileds = next(csvreader)
            out = [row for row in csvreader]
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out

def save_to_file(data, fpth, mode=None):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, mode if mode else 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, mode if mode else 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, mode if mode else 'w') as wfile:
            json.dump(data, wfile, sort_keys=True,  indent=4)
    elif ftype == 'csv':
        with open(fpth, mode if mode else 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            writer.writerows(data)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")

def prompt2msg(query_prompt):
    """
    Make prompts for GPT-3 compatible with GPT-3.5 and GPT-4.
    Support prompts for
        RER: e.g., data/osm/rer_prompt_16.txt
        symbolic translation: e.g., data/prompt_symbolic_batch12_perm/prompt_nexamples1_symbolic_batch12_perm_ltl_formula_9_42_fold0.txt
        end-to-end translation: e.g., data/osm/osm_full_e2e_prompt_boston_0.txt
    :param query_prompt: prompt used by text completion API (text-davinci-003).
    :return: message used by chat completion API (gpt-3, gpt-3.5-turbo).
    """
    prompt_splits = query_prompt.split("\n\n") if type(query_prompt) == str else query_prompt
    # breakpoint()
    task_description = prompt_splits[0]
    examples = prompt_splits[1: -1]
    query = prompt_splits[-1]

    msg = [{"role": "system", "content": task_description}]
    for example in examples:
        if "\n" in example:
            example_splits = example.split("\n")
            q = '\n'.join(example_splits[0:-1])  # every line except the last in 1 example block
            a_splits = example_splits[-1].split(" ")  # last line is the response
            q += f"\n{a_splits.pop(0)}"
            a = " ".join(a_splits)
            msg.append({"role": "user", "content": q})
            msg.append({"role": "assistant", "content": a})
        else:  # info should be in system prompt, e.g., landmark list
            msg[0]["content"] += f"\n{example}"
    msg.append({"role": "user", "content": query})
    breakpoint()
    return msg

class GPT4:
    def __init__(self, engine="gpt-4-0613", temp=0.00000001, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
            # try:
            raw_responses = openai.ChatCompletion.create(
                model=self.engine,
                messages=prompt2msg(query_prompt),
                temperature=self.temp,
                n=self.n,
                stop=self.stop,
                max_tokens=self.max_tokens,
            )
            complete = True
            # except:
            #     sleep(30)
            #     logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
            #     # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
            #     logging.info("OK continue")
            #     ntries += 1
        if self.n == 1:
            responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]
        return responses
        
    def extract_re(self, query_prompt):
        outs = self.generate(query_prompt)
        # breakpoint()
        name_entities = outs[0].split(' | ')
        return name_entities

    # def translate(self, query, prompt=""):
    #     if isinstance(query, list):
    #         query = query[0]
    #     query_prompt = prompt + query
    #     outs = self.generate(query_prompt)
    #     return outs

def prefix_to_infix(formula):
    """
    :param formula: LTL formula string in prefix order
    :return: LTL formula string in infix order
    Spot's prefix parser uses i for implies and e for equivalent. https://spot.lre.epita.fr/ioltl.html#prefix
    """
    BINARY_OPERATORS = {"&", "|", "U", "W", "R", "->", "i", "<->", "e"}
    UNARY_OPERATORS = {"!", "X", "F", "G"}
    formula_in = formula.split()
    stack = []  # stack

    while formula_in:
        op = formula_in.pop(-1)
        if op == ">":
            op += formula_in.pop(-1)  # implication operator has 2 chars, ->
        if formula_in and formula_in[-1] == "<":
            op += formula_in.pop(-1)  # equivalent operator has 3 chars, <->

        if op in BINARY_OPERATORS:
            formula_out = "(" + stack.pop(0) + " " + op + " " + stack.pop(0) + ")"
            stack.insert(0, formula_out)
        elif op in UNARY_OPERATORS:
            formula_out = op + "(" + stack.pop(0) + ")"
            stack.insert(0, formula_out)
        else:
            stack.insert(0, op)

    return stack[0]

def ltl2digraph(formula: str):
    """
    Turns an LTL expression into a DFA for progression

    :params formula: ltl formula in valid postfix notation
    :returns: a DFA that represents the LTL formula
    """

    # Spot Automaton Code
    aut = spot.translate(formula, 'BA', 'complete', 'state-based')
    nodelist = defaultdict(dict)
    bdd = aut.get_dict()

    initial_state = aut.get_init_state_number()
    accepting_states = [state for state in range(aut.num_states()) if aut.state_is_accepting(state)]

    for state in range(aut.num_states()):
        for edge in aut.out(state):
            edge_formula = spot.bdd_format_formula(bdd, edge.cond)
            out_state = edge.dst
            nodelist[state][out_state] = {'edge_label': edge_formula}

    # Digraph Code
    dfa = nx.DiGraph(nodelist)

    return dfa, accepting_states, initial_state

def progress_ltl(dfa, curr_dfa_state, action):
    '''
    New progress function for action pruning
    '''
    formula_action = spot.formula(action)
    for next_state in dfa.adj[curr_dfa_state]:
        formula_edge = spot.formula(dfa.get_edge_data(curr_dfa_state, next_state, default=0)['edge_label'])
        if spot.contains(formula_edge, formula_action) or spot.are_equivalent(formula_edge, formula_action):
            return next_state
    return curr_dfa_state

def string_to_boolean(formula: str, curr_state: str) -> bool:
    """
    Returns a boolean function from a string

    :params string: a string that represents a boolean function
    :returns: a boolean function from a string
    """
    LOG_SYMBOLS = ['not', 'and', 'or', '(', ')']
    formula_list = formula.replace('&', 'and').replace('|', 'or').replace('!', 'not ').replace('(', '( ').replace(')', ' )').split()

    # print("formula_list: ", formula_list)
    for i in range(len(formula_list)):
        # if self loop, then we always take that translation (that edge will always be taken)
        if formula_list[i] == '1':
            return True
        elif formula_list[i] in LOG_SYMBOLS:
            continue
        elif formula_list[i] == curr_state:
            formula_list[i] = 'True'
        else:
            formula_list[i] = 'False'

    return eval(' '.join(formula_list))

def validate_next_action(dfa, curr_dfa_state, traj_state, accepting_states):
    """
    Progress an LTL formula through a DFA
    :params dfa: a DFA that represents the LTL formula
    :params curr_dfa_state: the current state of the DFA
    :params traj_state: the current state of the trajectory
    :returns bool:
    :returns next_state: the next state in the dfa
    """
    
    if traj_state == "stop":
        return True if curr_dfa_state in accepting_states else False, curr_dfa_state
    else:
        next_state = progress_ltl(dfa, curr_dfa_state, traj_state)
        for accepting_state in accepting_states:
            if nx.has_path(dfa, next_state, accepting_state):
                return True, next_state
            return False, next_state
    # for accepting_state in accepting_states:
    #     if nx.has_path(dfa, next_state, accepting_state):
    #         return True
    #     return False
    
def all_prop(formula, action):
    formula_ls = [char for char in formula]
    props = [f"! {char}" for char in formula_ls if char in ascii_lowercase]
    props = list(set(props))
    return '&'.join(props).replace(f"! {action}", action)

##
## VH utils

def find_node(graph, class_name):
    return [n for n in graph["nodes"] if n["class_name"] == class_name]

def read_pose(pose_list):
    '''
    Convert raw pose list read from text to a dictionary of pose of each component
    :params list(str) pose_list: list read from pd_script.txt 
    :returns dict(list(float)): 3d pose for each component of the agent
    '''
    pose_dict = defaultdict(list)
    headers = pose_list[0].split()
    for line in pose_list[1:]:
        line = [float(num) for num in line.split()]
        for idx, h in enumerate(headers):
            pose_dict[h].append(line[3*idx+1: 3*idx+4])
    return pose_dict

def hardcoded_truth_value_vh(cur_loc, obj_id, graph, radius=0.5, room=False):
    '''
    hardcoded barrier function for truth value of one object (not for rooms)
    :params cur_loc: 1 by 3 location [x, z, y] (?)
    :params obj_id: object want to avoid 
    :params graph: graph returned by comm.environment_graph()
    :returns bool: true for hitting the barrier
    '''
    def is_point_inside_rotated_rectangular(center, size, rotation_matrix, point):
        half_size = np.array(size) / 2
        local_point = np.dot(np.linalg.inv(rotation_matrix), point - center)   
        return all(-h <= p <= h for p, h in zip(local_point, half_size))

    obj = [node for node in graph['nodes'] if node['id'] == obj_id][0]
    rotation_matrix = R.from_quat(obj["obj_transform"]["rotation"]).as_matrix()
    if room:
        bb_center = np.array(obj["bounding_box"]["center"])
        bb_size = np.array(obj["bounding_box"]["size"])#  - np.array([1, 1, 1]) # make bb smaller
        return is_point_inside_rotated_rectangular(bb_center, bb_size, rotation_matrix, cur_loc)

    else:     
        obj_loc = np.array(obj["obj_transform"]["position"])
        cur_loc = np.array(cur_loc)
        dist = np.linalg.norm(cur_loc - obj_loc)
        return True if dist < radius else False

def concat_props(prop_state):
    '''
    concat props in states from prop_traj in the form of LTL
    :params dict prop_state: dictionary {prop: truth_value}
    :returns formula: truth value of props in LTL formula
    '''
    return " & ".join([prop if truth_value else f'!{prop}' for prop, truth_value in prop_state.items()])

def prop_level_state(graph, obj_ids, room_ids, mappings, radius=0.5):
    '''
    proposition-level trajectory for executing one action
    :params list(str) obj_ids: objects for props
    :params list(str) room_ids: rooms for props
    :params dict mappings: name of the obj_id/room_id to proposition character generated by Lang2LTL
    :returns list(dict) prop_state: string of propositions that spot can take
    '''
    pass

def prop_level_traj(pose_list, graph, obj_ids, room_ids, mappings, radius=0.5):
    '''
    proposition-level trajectory for executing one action. For v2.3.0 only
    :params pose_list: list of [x, z, y] poses
    :params list(str) obj_ids: objects for props
    :params list(str) room_ids: rooms for props
    :params dict mappings: name of the obj_id/room_id to proposition character generated by Lang2LTL
    :returns list(dict) prop_traj: list of dictionaries tracking all props for each state change
    '''
    init_state = {obj:hardcoded_truth_value_vh(pose_list[0], obj, graph, radius) for obj in obj_ids}
    init_state.update({room: hardcoded_truth_value_vh(pose_list[0], room, graph, radius, room=True) for room in room_ids})
    init_state_mapped = {mappings[k]:v for k, v in init_state.items()}

    prop_traj = [init_state_mapped]
    # record init truth values and only update when props changed
    state_buffer = [b for b in init_state_mapped.values()]
    for pose in pose_list[1:]:
        state = {obj:hardcoded_truth_value_vh(pose, obj, graph,radius=radius) for obj in obj_ids}
        state.update({room: hardcoded_truth_value_vh(pose, room, graph, radius=radius, room=True) for room in room_ids})
        # breakpoint()
        new_state = [b for b in state.values()]
        if not state_buffer == new_state:
            prop_traj.append({mappings[k]:v for k, v in state.items()})
            state_buffer = new_state
    return [concat_props(prop_state) for prop_state in prop_traj]

def state_change_by_step(comm, program, input_ltl, obj_ids, room_ids, mappings, init_position, init_room, env_num=0,stopped=False):
    comm.reset(env_num)
    _, g = comm.environment_graph()
    comm.add_character('Chars/Female2', position=init_position, initial_room=init_room)
    comm.render_script(program, recording=True, save_pose_data=True)
    pose_dict = load_from_file("Output/script/0/pd_script.txt")
    pose_list = read_pose(pose_dict)["Head"]
    prop_traj = prop_level_traj(pose_list, g, obj_ids, room_ids, mappings, radius=2.0)
    dfa, accepting_states, curr_state = ltl2digraph(input_ltl)
    state_list = []
    success = True
    if stopped:
        prop_traj.append("stop")
    for state in prop_traj:
        action = state
        valid, next_state = validate_next_action(dfa, curr_state, action, accepting_states)
        if valid:
            state_list.append(f"Safe: {action}")
            curr_state = next_state
        else:
            state_list.append(f"Violated: {action}")
            success = False
            # breakpoint()
            break
        # breakpoint()
    # breakpoint()
    return success, state_list

def omit_obj_id(script_lines):
    '''
    omit redundent obj index. e.g., convert (1.319) -> (319)
    :params list(str) script_lines: script lines after line[4] (program on;y)
    '''
    new_script = []
    for line in script_lines:
        while "." in line:
            idx = line.index(".")
            line = line[:idx-1] + line[idx+1:]
        new_script.append(line)
    return new_script

def convert_rooms(line):
    line = line.replace("dining_room", "kitchen")
    line = line.replace("home_office", "living_room")
    line = line.replace("livingroom", "living_room")
    line = line.replace("oven", "microwave") # :-(
    return line

def convert_old_program_to_new(script_lines):
    '''
    convert old program to new for 1
    '''
    new_lines = []
    for line in script_lines:
        line = line.replace("dining_room", "kitchen")
        line = line.replace("home_office", "living_room")
        line = "<char0> " + line
        new_lines.append(line.lower())
    return new_lines

def program2example(program_lines):
    title = program_lines[:2]
    title[1] = "Description: " + title[1]
    script = read_script_from_list_string(program_lines)
    example = []
    # breakpoint()
    for idx, script_line in enumerate(script):
        act = script_line.action.name.lower()
        params = script_line.parameters
        if act == "walk":
            example.append(f"{idx}. walk to {params[0].name}")
        elif act == "lookat":
            example.append(f"{idx}. look at {params[0].name}")
        elif act == "plug":
            example.append(f"{idx}. plug in {params[0].name}")
        elif act == "point":
            example.append(f"{idx}. point at {params[0].name}")
        elif act == "put":
            example.append(f"{idx}. put {params[0].name} on {params[1].name}")
        elif act == "putin":
            example.append(f"{idx}. put {params[0].name} in {params[1].name}")
        elif act == "switchon":
            example.append(f"{idx}. switch on {params[0].name}")
        elif act == "switchoff":
            example.append(f"{idx}. switch off {params[0].name}")
        elif act == "lie":
            example.append(f"{idx}. lie on {params[0].name}")
        elif act == "sleep":
            example.append(f"{idx}. sleep")
        else:
            assert(len(script_line.parameters) == 1)
            example.append(f"{idx}. {script_line.action.name.lower()} {script_line.parameters[0].name}.")
        # breakpoint()
    example.append(f"{idx+1}. DONE")
    return title + example

def get_action_and_obj(output_line):
    while "." in output_line:
        output_line = output_line.replace(".", "")
    str_list = output_line.split()
    if len(str_list) == 2:
        return f"[{str_list[0]}]", [f"<{str_list[1]}>"]
    elif len(str_list) == 3:
        if "at" in str_list:
            return "[lookat]", [f"<{str_list[2]}>"]
        elif "switch" in str_list:
            if "on" in str_list:
                return "[switchon]", [f"<{str_list[2]}>"]
            elif "off" in str_list:
                return "[switchoff]", [f"<{str_list[2]}>"]
        else:
            return f"[{str_list[0]}]", [f"<{str_list[2]}>"]
    elif len(str_list) > 3:
        # print(str_list)
        if "in" in str_list:
            return "[putin]", [f"<{str_list[1]}>", f"<{str_list[3]}>"]
        else:
            return "[put]", [f"<{str_list[1]}>", f"<{str_list[3]}>"]

def reprompt(translate_engine, valid_actions, invalid_action, constraints, state_lists, mappings, id2name,prompt_fpath="prompts/dialogue/explain_zeroshot_v2.txt"):

    def state_prop2eng(state_list, mappings):
        state_eng_list = []
        inverse_mappings = {v:k for k,v in mappings.items()}
        for state in state_list:
            for prop in inverse_mappings.keys():
                while prop in state:
                    state = state.replace(prop, id2name[inverse_mappings[prop]])
            state_eng_list.append(state)
        return state_eng_list

    task_description = load_from_file(prompt_fpath)
    task_description = "\n".join([task_description]) # just one line actually
    constraints = f"Constraints: {str(constraints)}"

    if valid_actions:
        valid_act = ""
        for i, action in enumerate(valid_actions):
            header = f"Valid action {i}: {action}"
            valid_act_i = header
            state_list_i = state_lists[i]
            valid_act_i += "\nState change:"
            
            state_eng = state_prop2eng(state_list_i, mappings)
            for state in state_eng:
                valid_act_i += f"\n{state}"
            valid_act += valid_act_i
    else:
        valid_act = ""
    
    invalid_act = ""
    header = f"Invalid action: {invalid_action}"
    invalid_act = header
    state_list_i = state_lists[-1]
    invalid_act += "\nState change:"
    state_eng = state_prop2eng(state_list_i, mappings)
    for state in state_eng:
        invalid_act += f"\n{state}"
    prompt = f"{task_description}\n\n{constraints}\n{valid_act}\n{invalid_act}\nReason of violation:"
    # breakpoint()
    return translate_engine.generate(prompt)[0]


