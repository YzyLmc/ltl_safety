import sys
sys.path.append("virtualhome_v2.3.0/")
import random
import numpy as np

from simulation.unity_simulator import comm_unity
from utils import *

random.seed(123)
np.random.seed(123)

EXEC_FNAME= "/users/zyang157/data/zyang157/virtualhome/exec_v2.3.0/linux_exec.v2.3.0.x86_64"
comm = comm_unity.UnityCommunication(file_name=EXEC_FNAME, no_graphics=True, logging=False)
comm.reset(0) # env num_0
comm.add_character('Chars/Female2', initial_room='bedroom')
s, g = comm.environment_graph()
char_node = [node for node in g['nodes'] if node['category'] == 'Characters'][0]
salmon_node = [node for node in g['nodes'] if node['class_name'] == 'salmon'][0]
fridge_id = [node['id'] for node in g['nodes'] if node['class_name'] == 'fridge'][0]
glass_id = [node['id'] for node in g['nodes'] if node['class_name'] == 'waterglass'][-1]
table_id = [node['id'] for node in g['nodes'] if "table" in node["class_name"]][-1]
kitchen_id = [node['id'] for node in g['nodes'] if node["class_name"] == "kitchen"][-1]
salmon_id = salmon_node['id']
# script_1 = ['<char0> [walk] <bedroom> ({})'.format(73)]
# # script_1 = [
# #     '<char0> [walk] <coffeetable> ({})'.format(table_id),]
# # print(char_node)
# # comm.render_script(script, recording=False, skip_animation=True, save_pose_data=True)
# comm.render_script(script_1, recording=True, save_pose_data=True)
# new_s_1, new_g_1 = comm.environment_graph()
# new_char_node_1 = [node for node in new_g_1['nodes'] if node['category'] == 'Characters'][0]
# glass_1 = [node for node in new_g_1['nodes'] if node['class_name'] == 'waterglass'][-1]
# # print(new_char_node_1)
# # print(glass_1)
# # script_2 = ['<char0> [walk] <salmon> ({})'.format(salmon_id),]
# script_2 = ['<char0> [walk] <livingroom> ({})'.format(335),]
# comm.render_script(script_2, recording=True, save_pose_data=True)
# new_s_2, new_g_2 = comm.environment_graph()
# new_char_node_2 = [node for node in new_g_2['nodes'] if node['category'] == 'Characters'][0]
# glass_2 = [node for node in new_g_2['nodes'] if node['class_name'] == 'waterglass'][-1]
# # print(new_char_node_2)
# # print(glass_2)
# script_2 = ["<char0> [FIND] <towel> (1.1000)",
# "<char0> [FIND] <soap> (1.1002)",
# "<char0> [WIPE] <soap> (1.1002)",
# "<char0> [GRAB] <soap> (1.1002)",
# "<char0> [PUTOBJBACK] <soap> (1.1002)"]
# script_fpath = "/users/zyang157/data/zyang157/virtualhome/data/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/file1004_2.txt"
# script_2 = load_from_file(script_fpath)[4:]

script_2 = ["[WALK] <dining_room> ({})".format(kitchen_id),
"[FIND] <salmon> ({})".format(salmon_id),
"[GRAB] <salmon> ({})".format(salmon_id),
"[WALK] <fridge> ({})".format(fridge_id),
"[OPEN] <fridge> ({})".format(fridge_id),
"[PUTIN] <salmon> ({}) <fridge> ({})".format(salmon_id, fridge_id),
"[CLOSE] <fridge> ({})".format(fridge_id)]
script_2 = convert_old_program_to_new(script_2)
print(script_2)
# breakpoint()
comm.render_script(script_2, recording=True, save_pose_data=True)

pose_list = load_from_file("Output/script/0/pd_script.txt")
pose_dict = read_pose(pose_list)
pose_list = pose_dict["Head"]

rooms = [node for node in g['nodes'] if "room" in node["class_name"]]
tables = [node for node in g['nodes'] if "table" in node["class_name"]]

# room_ids = [11, 73, 335] # bathroom, bedroom, livingroom
room_ids = [11, 335, 205] # bathroom, livingroom, kitchen
obj_ids = [111, 231, 371] # coffeetable_1, kitchen table, coffetable_2

mappings = {11:"a", 335: "b", 205: "c", 111:"d", 231:"h", 371:"j"}
prop_traj = prop_level_traj(pose_list, g, obj_ids, room_ids, mappings, radius=2.0)
# for s in prop_traj: print(s)
input_ltl = "(! c U b) & G ! h" # enter bedroom before livingroom and always avoid coffeetable_2
dfa, accepting_states, curr_state = ltl2digraph(input_ltl)
for state in prop_traj:
    action = state
    if validate_next_action(dfa, curr_state, action, accepting_states):
        print("Safe:", action)
        curr_state = progress_ltl(dfa, curr_state, action)
    else:
        print("Violated:", action)
        break
breakpoint()

# saycan part
