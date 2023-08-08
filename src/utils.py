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
import random
import networkx as nx
import spot

# General utils
def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line[:-1] for line in rfile.readlines()]
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
            json.dump(data, wfile, sort_keys=True)
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
    prompt_splits = query_prompt.split("\n\n")
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

    return msg

class GPT4:
    def __init__(self, engine="gpt-4", temp=0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = openai.ChatCompletion.create(
                    model=self.engine,
                    messages=prompt2msg(query_prompt),
                    temperature=self.temp,
                    n=self.n,
                    stop=self.stop,
                    max_tokens=self.max_tokens,
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
                # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info("OK continue")
                ntries += 1
        if self.n == 1:
            responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]
        return responses

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
    :returns: the next state in the dfa
    """
    next_state = progress_ltl(dfa, curr_dfa_state, traj_state)
    for accepting_state in accepting_states:
      if nx.has_path(dfa, next_state, accepting_state):
        return True
    return False
    
def all_prop(formula, action):
    formula_ls = [char for char in formula]
    props = [f"! {char}" for char in formula_ls if char in ascii_lowercase]
    props = list(set(props))
    return '&'.join(props).replace(f"! {action}", action)

# VH utils
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
    obj = [node for node in graph['nodes'] if node['id'] == obj_id][0]
    if room:
        bb_center = obj["bounding_box"]["center"]
        bb_size = obj["bounding_box"]["size"]
        for i in range(len(cur_loc)):
            if np.abs(cur_loc[i] - bb_center[i]) > bb_size[i]/2:
                return False
        return True

    else:     
        obj_loc = np.array(obj["obj_transform"]["position"])
        cur_loc = np.array(cur_loc)
        dist = np.linalg.norm(cur_loc - obj_loc)
        return True if dist < radius else False

def prop_level_traj(pose_list, graph, obj_ids, room_ids, mappings, radius=0.5):
    '''
    proposition-level trajectory for executing one action
    :params pose_list: list of [x, z, y] poses
    :params list(str) obj_ids: objects for props
    :params list(str) room_ids: rooms for props
    :params dict mappings: name of the obj_id/room_id to proposition character generated by Lang2LTL
    :returns list(dict) prop_traj: list of dictionaries tracking all props for each state change
    '''
    def concat_props(prop_state):
        '''
        concat props in states from prop_traj in the form of LTL
        :params dict prop_state: dictionary {prop: truth_value}
        :returns formula: truth value of props in LTL formula
        '''
        return " & ".join([prop if truth_value else f'!{prop}' for prop, truth_value in prop_state.items()])
    init_state = {obj:hardcoded_truth_value_vh(pose_list[0], obj, graph, radius) for obj in obj_ids}
    init_state.update({room: hardcoded_truth_value_vh(pose_list[0], room, graph, radius, room=True) for room in room_ids})
    init_state_mapped = {mappings[k]:v for k, v in init_state.items()}

    prop_traj = [init_state_mapped]
    # record init truth values and only update when props changed
    state_buffer = [b for b in init_state_mapped.values()]
    for pose in pose_list[1:]:
        state = {obj:hardcoded_truth_value_vh(pose, obj, graph,radius) for obj in obj_ids}
        state.update({room: hardcoded_truth_value_vh(pose, room, graph, radius, room=True) for room in room_ids})
        # breakpoint()
        new_state = [b for b in state.values()]
        if not state_buffer == new_state:
            prop_traj.append({mappings[k]:v for k, v in state.items()})
            state_buffer = new_state
    # return prop_traj
    return [concat_props(prop_state) for prop_state in prop_traj]

def omit_obj_id(script_lines):
    '''
    omit redundent obj index. e.g., convert (1.319) -> (319)
    '''
    new_script = []
    for line in script_lines:
        while "." in line:
            idx = line.index(".")
            line = line[:idx-1] + line[idx+1:]
        new_script.append(line)
    return new_script