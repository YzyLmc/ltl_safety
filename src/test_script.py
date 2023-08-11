'''
filter executable scripts
'''
import sys
import os

sys.path.append("virtualhome_v1.0.0/dataset_utils")
sys.path.append("virtualhome_v1.0.0/simulation/")
sys.path.append("src/")

from utils import load_from_file, save_to_file, omit_obj_id
import evolving_graph.utils as utils
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
from evolving_graph.scripts import read_script, read_script_from_string, read_script_from_list_string, ScriptParseException
from evolving_graph.check_programs import modify_objects_unity2script, check_one_program
import add_preconds

name_equivalence = utils.load_name_equivalence()
helper = utils.graph_dict_helper(max_nodes=500)
# graph_dict_fpath = 'virtualhome_v1.0.0/example_graphs/TrimmedTestScene1_graph.json'
graph_dict_fpath = "virtualhome_v1.0.0/example_graphs/TestScene1_graph_corrected.json"
# graph_dict_fpath = "/users/zyang157/data/zyang157/virtualhome/data/programs_processed_precond_nograb_morepreconds/init_and_final_graphs/TrimmedTestScene1_graph/results_intentions_march-13-18/"
# script_name = "/users/zyang157/data/zyang157/virtualhome/data/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/file7_1.txt"
script_path = "/users/zyang157/data/zyang157/virtualhome/data/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/"
scripts = os.listdir(script_path)
n_count = 0
executable_scripts = []
for script_name in scripts:
    # graph_dict = load_from_file(f"{graph_dict_fpath}/{script_name[:-4]}.json")
    graph_dict = load_from_file(graph_dict_fpath)
    helper.initialize(graph_dict)
    # graph = EnvironmentGraph(graph_dict)
    # executor = ScriptExecutor(graph, name_equivalence)
    script_fpath = os.path.join(script_path, script_name)
    script_lines = load_from_file(script_fpath)[4:]
    script_lines = omit_obj_id(script_lines)
    # print(script_lines)
    # script = ['[Walk] <television> (1)', '[SwitchOn] <television> (1)', 
    #           '[Walk] <sofa> (1)', '[Find] <controller> (1)',
    #           '[Grab] <controller> (1)']
    precond = add_preconds.get_preconds_script(script_lines).printCondsJSON()
    # print(precond)

    script = read_script_from_list_string(script_lines)
    # script, precond = modify_objects_unity2script(helper, script, precond)
    # executable, final_state, graph_state_list = executor.execute(script, w_graph_list=True)
    message, executable, final_state, graph_state_list, id_mapping, info, script = check_one_program(helper, script, precond, graph_dict, w_graph_list=True)
    if not executable:
        print("not executable", script_lines)
        n_count += 1
    else:
        executable_scripts.append(script_name)
print(f"executable/total: {n_count}/{len(scripts)}={n_count/len(scripts)}")
save_to_file(executable_scripts, "executable_scripts.json")
    # breakpoint()
