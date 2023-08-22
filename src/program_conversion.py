import json
import sys
sys.path.append("virtualhome_v2.3.0/")
sys.path.append("virtualhome_v2.3.0/dataset_utils")
sys.path.append("virtualhome_v2.3.0/simulation/")

import evolving_graph.utils as utils
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
from evolving_graph.check_programs import modify_objects_unity2script, check_one_program
from evolving_graph.scripts import read_script, read_script_from_string, read_script_from_list_string, ScriptParseException
import add_preconds

from utils import load_from_file, save_to_file, omit_obj_id, find_node

# check executability first and then convert
def replace_objs(program_lines, obj_dict):
    new_lines = []
    for line in program_lines:
        for old_obj, new_obj in obj_dict.items():
            while old_obj in line:
                line = line.replace(old_obj, new_obj)
        new_lines.append(line)
    return new_lines

def replace_rooms(program_lines):
    room_dict = {"dining_room": "kitchen", "home_office": "living_room"}
    return replace_objs(program_lines, room_dict)

def replace_new_obj_id(script_lines, graph_dict):
    script = read_script_from_list_string(script_lines)
    for script_line in script:
        for i, obj in enumerate(script_line.parameters):
            nodes = find_node(graph_dict, obj.name)
            if not nodes: # no match node
                raise Exception(f"no matching obj for {obj.name}")
            script_line.parameters[i].instance = nodes[0]["id"]

    return [str(line)[:str(line).find("[", -5) - 1] for line in script]

def check_executability(graph_dict, script, precond, helper):
    helper.initialize(graph_dict)
    script = read_script_from_list_string(script)
    # print([str(script_line) for script_line in script])
    message, executable, final_state, graph_state_list, id_mapping, info, script = check_one_program(helper, script, precond, graph_dict, place_other_objects=False, w_graph_list=True)
    return message, executable, final_state, graph_state_list, id_mapping, info, script


if __name__ == "__main__":
    graph_dict_path = "virtualhome_v2.3.0/env_graphs/TestScene1_graph.json"
    graph_dict = load_from_file(graph_dict_path)
    script_path = "/users/zyang157/data/zyang157/virtualhome/data/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/"
    script_fpath = script_path + "file180_2.txt"
    script = load_from_file(script_fpath)
    # omit obj id; replace rooms: dining_room to kitchen, home_office to living_room
    script = replace_rooms(omit_obj_id(script))
    # replace old obj to new with obj dict, this dict is constructive collectively by manually testing conversion on multiple old programs
    obj_dict = {"<freezer>": "<fridge>", "<food_food>": "<pie>", "<light>": "<lightswitch>", "<phone>": "<cellphone>", "<oven>": "<microwave>"}
    script = replace_objs(script, obj_dict)

    script = replace_new_obj_id(script, graph_dict)
    # obj_dict = {"livingroom": "living_room"}
    # script = replace_objs(script, obj_dict)
    precond = add_preconds.get_preconds_script(script).printCondsJSON()
    for script_line in script:
        print(str(script_line))
    helper = utils.graph_dict_helper()
    message, executable, final_state, graph_state_list, id_mapping, info, script = check_executability(graph_dict, script, precond, helper)
    breakpoint()