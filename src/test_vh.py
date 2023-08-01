import sys
sys.path.append("virtualhome/")

from simulation.unity_simulator import comm_unity
from utils import *

EXEC_FNAME= "/users/zyang157/data/zyang157/virtualhome/exec/linux_exec.v2.3.0.x86_64"
comm = comm_unity.UnityCommunication(file_name=EXEC_FNAME, no_graphics=True, logging=False)
comm.reset(0) # env num_0
comm.add_character('Chars/Female2', initial_room='kitchen')
s, g = comm.environment_graph()
char_node = [node for node in g['nodes'] if node['category'] == 'Characters'][0]
salmon_node = [node for node in g['nodes'] if node['class_name'] == 'salmon'][0]
glass_id = [node['id'] for node in g['nodes'] if node['class_name'] == 'waterglass'][-1]
table_id = [node['id'] for node in g['nodes'] if "table" in node["class_name"]][-1]
salmon_id = salmon_node['id']
script_1 = [
    '<char0> [walk] <coffeetable> ({})'.format(table_id),]
# print(char_node)
# comm.render_script(script, recording=False, skip_animation=True, save_pose_data=True)
comm.render_script(script_1, recording=True, save_pose_data=True)
new_s_1, new_g_1 = comm.environment_graph()
new_char_node_1 = [node for node in new_g_1['nodes'] if node['category'] == 'Characters'][0]
glass_1 = [node for node in new_g_1['nodes'] if node['class_name'] == 'waterglass'][-1]
# print(new_char_node_1)
# print(glass_1)
script_2 = [
    '<char0> [walk] <salmon> ({})'.format(salmon_id),]
comm.render_script(script_2, recording=True, save_pose_data=True)
new_s_2, new_g_2 = comm.environment_graph()
new_char_node_2 = [node for node in new_g_2['nodes'] if node['category'] == 'Characters'][0]
glass_2 = [node for node in new_g_2['nodes'] if node['class_name'] == 'waterglass'][-1]
# print(new_char_node_2)
# print(glass_2)

pose_list = load_from_file("Output/script/0/pd_script.txt")
pose_dict = read_pose(pose_list)
pose_list = pose_dict["Head"]

rooms = [node for node in g['nodes'] if "room" in node["class_name"]]
tables = [node for node in g['nodes'] if "table" in node["class_name"]]
room_ids = [11, 73, 335] # bathroom, bedroom, livingroom
obj_ids = [111, 231, 371] # coffeetable_1, kitchen table, coffetable_2
mappings = {11:"a", 73: "b", 335: "c", 111:"d", 231:"h", 371:"j"}
prop_traj = prop_level_traj(pose_list, g, obj_ids, room_ids, mappings, radius=2.0)
# for s in prop_traj: print(s)
input_ltl = "(! c U b) & G ! j" # enter bedroom before livingroom and always avoid coffeetable_2
dfa, accepting_states, initial_state = ltl2digraph(input_ltl)
for state in prop_traj:
    action = state
    if validate_next_action(dfa, initial_state, action, accepting_states):
        print("Safe:", action)
    else:
        print("Violated:", action)
        break
breakpoint()
