import sys
sys.path.append("virtualhome_v2.3.0/")
import random
import numpy as np
import networkx as nx

import os
import openai
from openai.embeddings_utils import cosine_similarity
from utils import program2example, load_from_file, GPT4, get_action_and_obj, convert_old_program_to_new, read_pose, prop_level_traj, ltl2digraph, validate_next_action, progress_ltl, reprompt, state_change_by_step, convert_rooms
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
obj_dict = {obj.annotations.name: [obj.waypoint_tform_ko.position.x, obj.waypoint_tform_ko.position.y, obj.waypoint_tform_ko.position.z] for obj in objs}
obj2id = {obj.annotations.name: obj.id for obj in objs}

# convert spot nva graph to nx graph for planning
connect_graph = nx.Graph()
connect_graph.add_nodes_from([wp.id for wp in graph.waypoints])
connect_graph.add_edges_from([(e.id.from_waypoint, e.id.to_waypoint) for e in graph.edges])
# call nx.dijkstra_path to plan a trajectory


# # example task: put mail into mailbox.
# example_script = ["[find] <mail>",\
#                     "[grab] <mail>",\
#                     "[walk] <mailbox>",\
#                     "[putin] <mail> <mailbox>"]

# cm = constraint_module()
# constraint_strs = ["you can't go to mailbox before going to bedroom"]
# unified_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraint_strs)
# breakpoint()

task_description = load_from_file("prompts/planning/planning_with_cons_v1.txt")
example = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/spot/test_script.txt")
example = "\n".join(example)
prompt = task_description+ "\n\n" + example

dummy_query = "put mail into mailbox"
prompt += f"\n{dummy_query}\nDescription:"
print(prompt)
breakpoint()
gpt4 = GPT4(stop=['\n', 'Plan'])
embed_engine = GPT3(engine="text-embedding-ada-002")

# first response is description
output = gpt4.generate(prompt)[0]
print(output)
prompt = prompt + f"\n{output}"
n_try = 0
program = []
state_lists = []
valid_actions = []
invalid_action_mem = None
stopped = False
idx = 0
while n_try < 10: # max step + replan time <=10
    prompt += f"\n{idx}."
    output = gpt4.generate(prompt)[0]
    print(output)

    if "DONE" in output: 
        stopped = True
        action_string = "STOP"