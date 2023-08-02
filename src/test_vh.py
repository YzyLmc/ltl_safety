import os
os.system("/users/zyang157/data/zyang157/virtualhome/exec/exec_linux_07_01.x86_64")

from simulation.unity_simulator import comm_unity

comm.reset(0) # env num_0
comm.add_character('Chars/Female2', initial_room='kitchen')
s, g = comm.environment_graph()
char_node = [node for node in g['nodes'] if node['category'] == 'Characters'][0]
salmon_node = [node for node in g['nodes'] if node['class_name'] == 'salmon'][0]
glass_id = [node['id'] for node in g['nodes'] if node['class_name'] == 'waterglass'][-1]
table_id = [node['id'] for node in g['nodes'] if "table" in node["class_name"]][-1]
salmon_id = salmon_node['id']