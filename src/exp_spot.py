import sys
sys.path.append("virtualhome_v2.3.0/")
import random
import numpy as np
import networkx as nx
from collections import OrderedDict, defaultdict

import os
import openai
from openai.embeddings_utils import cosine_similarity
from utils import program2example, load_from_file, save_to_file, GPT4, get_action_and_obj, convert_old_program_to_new, read_pose, prop_level_traj, ltl2digraph, validate_next_action, progress_ltl, reprompt, state_change_by_step_spot
from program_conversion import replace_new_obj_id
from get_embed import GPT3
from constraint_module import constraint_module

openai.api_key = os.getenv("OPENAI_API_KEY")

from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2

graph_fpath = "robot/downloaded_graph"
with open(graph_fpath + '/graph', 'rb') as graph_file:
    # Load the graph from disk.
    data = graph_file.read()
    graph = map_pb2.Graph()
    graph.ParseFromString(data)
    print(
        f'Loaded graph has {len(graph.waypoints)} waypoints and {len(graph.edges)} edges'
    )
objs = [waypoint for waypoint in graph.waypoints if not "waypoint" in waypoint.annotations.name]
id2loc = {obj.id: [obj.waypoint_tform_ko.position.x, obj.waypoint_tform_ko.position.y, obj.waypoint_tform_ko.position.z] for obj in graph.waypoints}
obj2id = {obj.annotations.name: obj.id for obj in objs}
room2id = {}

# convert spot nva graph to nx graph for planning
connect_graph = nx.Graph()
connect_graph.add_nodes_from([wp.id for wp in graph.waypoints])
connect_graph.add_edges_from([(e.id.from_waypoint, e.id.to_waypoint) for e in graph.edges])
# call nx.dijkstra_path() to plan a trajectory

# test constraints
constraints = ["you have to go to bedroom before picking up mail"]
# cm = constraint_module()
# input_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraints)
# breakpoint()
input_ltl = 'W ! b a'
obj_mapping = {'A': 'mail', 'B': 'bedroom'}
pred_mapping = {'a': 'agent_at(B)', 'b': 'is_grabbed(A)'}
grounded_pred_mapping = {}
for prop, pred in pred_mapping.items():
    for placeholder, obj in obj_mapping.items():
        pred = pred.replace(placeholder, obj)
    grounded_pred_mapping[prop] = pred

# load task
TASK_PREFIX = "virtualhome_v2.3.0/dataset/ltl_safety/spot/"
lines = load_from_file(f"{TASK_PREFIX}test_script.txt")
# construct prompt
task_description = load_from_file("prompts/planning/planning_spot_v1.txt")
example = program2example(lines)
example = "\n".join(example)
prompt = task_description+ "\n\n" + example + "\n"

# give goal according to each prompt
dummy_query = "drop mail into the mail box"
# generate description at first step
prompt += f"\n{dummy_query}\nDescription:"
print(prompt)
gpt4 = GPT4(stop=['\n', 'Clean the kitchen', 'Plan'])
embed_engine = GPT3(engine="text-embedding-ada-002")
# first response is description
output = gpt4.generate(prompt)[0]
print(output)
# load graph and embeddings
prompt = prompt + f" {output}"
allowed_actions = load_from_file("virtualhome_v2.3.0/resources/allowed_actions_spot.json")
act2embed = load_from_file("/users/zyang157/data/zyang157/virtualhome/action_embeds/act2embed_vh_gpt3-text-embedding-ada-002_spot.pkl")

n_try = 0
program = []
valid_action2states = OrderedDict()
invalid_action_mem = None
stopped = False
idx = 0
while n_try < 10: # max step + replan time <=10
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
        program.append(dummy_program_line)

    success, state_list, state_act_idx = state_change_by_step_spot(program, input_ltl, connect_graph, obj_mapping, pred_mapping, obj2id, room2id, id2loc, stopped=stopped)
    prompt += f" {output}"

    # masking unsafe action out
    if invalid_action_mem == dummy_program_line:
        continue

    if not success:
        invalid_action = action_string
        invalid_state_list = state_list
        reprompted_plan = f"\nError: {reprompt(gpt4, valid_action2states, invalid_action, invalid_state_list, constraints, grounded_pred_mapping)} The correct plan would be:"
        program = program[:-1]
        print("handling error by reprompting")
        print(reprompted_plan)
        prompt += reprompted_plan
        invalid_action_mem = dummy_program_line
        stopped = False
    else:
        if "DONE" in output: break
        valid_action2states[action_string] = state_list
        invalid_action_mem = None
        idx += 1
        # breakpoint()
print("Programs:")
[print(line) for line in program]
print(prompt)

# save_fpath = "results/spot_results.json"
# if os.path.exists(save_fpath):
#     saved_results = load_from_file(save_fpath)
# else:
#     saved_results = defaultdict(dict)
# saved_results[0][0] = {"constraint": constraints, "program":program, "safe": success, "completed": None}
# save_to_file(saved_results, save_fpath)