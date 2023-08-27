import sys
sys.path.append("virtualhome_v2.3.0/")
import random
import numpy as np

import os
import openai
from openai.embeddings_utils import cosine_similarity
from utils import program2example, load_from_file, GPT4, get_action_and_obj, convert_old_program_to_new, read_pose, prop_level_traj, ltl2digraph, validate_next_action, progress_ltl, reprompt, state_change_by_step, convert_rooms
from program_conversion import replace_new_obj_id
from get_embed import GPT3

openai.api_key = os.getenv("OPENAI_API_KEY")

from simulation.unity_simulator import comm_unity

random.seed(123)
np.random.seed(123)
# start vh simulator
EXEC_FNAME= "/users/zyang157/data/zyang157/virtualhome/exec_v2.3.0/linux_exec.v2.3.0.x86_64"
comm = comm_unity.UnityCommunication(file_name=EXEC_FNAME, no_graphics=True, logging=False)
comm.reset(0) # env num_0
# init_room = "kitchen"
init_room = "bedroom"
# init_room = "bathroom"
s, g = comm.environment_graph()
init_position = [node for node in g['nodes'] if node["class_name"] == init_room][-1]['obj_transform']['position']
comm.add_character('Chars/Female2', position=init_position, initial_room=init_room)
s, g = comm.environment_graph()
# breakpoint()
# get useful objs
rooms = [node for node in g['nodes'] if node["category"] == "Rooms"]
tables = [node for node in g['nodes'] if "table" in node["class_name"]]
kitchen_id = [node['id'] for node in g['nodes'] if node["class_name"] == "kitchen"][-1]
livingroom_id = [node['id'] for node in g['nodes'] if node["class_name"] == "livingroom"][-1]
bathroom_id = [node['id'] for node in g['nodes'] if node["class_name"] == "bathroom"][-1]
bedroom_id = [node['id'] for node in g['nodes'] if node["class_name"] == "bedroom"][-1]
coffeetable1_id = [node['id'] for node in g['nodes'] if node["class_name"] == "coffeetable"][-1]
coffeetable2_id = [node['id'] for node in g['nodes'] if node["class_name"] == "coffeetable"][-2] # two coffee tables
kitchentable_id = [node['id'] for node in g['nodes'] if node["class_name"] == "kitchentable"][-1]

room_ids = [bathroom_id, bedroom_id, kitchen_id, livingroom_id]
# obj_ids = [coffeetable1_id, kitchentable_id, coffeetable2_id]
obj_ids = []
# mappings = {bathroom_id: "a", bedroom_id: "b", kitchen_id: "c", coffeetable1_id:"d", kitchentable_id:"h", coffeetable2_id:"j"}
mappings = {bathroom_id: "A", bedroom_id: "B", kitchen_id: "C", livingroom_id: "D"}
id2name = {bathroom_id: "bathroom", bedroom_id: "bedroom", kitchen_id: "kitchen", livingroom_id: "living_room"}
# 0_1
# input_ltl = "(! A W D)"
# constraints = ["you have to enter living room before bathroom"]
input_ltl = "G(B -> F D)"
constraints = ["you have to go to living room afterwards if you have entered bedroom"]
# input_ltl = "(! A U D) & (! D U B)" # you have to enter living room before bathroom
# constraints = ["you have to enter living room before bathroom", "you have to enter bedroom before going into living room" ]
# input_ltl = "(! A U D) & (! D U B) & (G B -> X C)" # i only works with prefix
# constraints = ["you have to enter living room before bathroom", "you have to enter bedroom before going into living room", "you have to go to kitchen right after living room" ]
# input_ltl = "(! A U D) & (! D U B) & (G B -> X C) & (G D -> F C) & !(F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F(C)))))))))))))))))" # i only works with prefix
# constraints = ["you have to enter living room before bathroom", "you have to enter bedroom before going into living room", "you have to go to kitchen right after living room", "you can only go to kitchen twice"]
# input_ltl = "(! A U D) & (! D U B) & (G B -> X C) & (G D -> F C) & !(F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F(C))))))))))))))))) & (!(F((B & (B U (!(B) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B)))))))))))))" # i only works with prefix
# constraints = ["you have to enter living room before bathroom", \
#                 "you have to enter bedroom before going into living room", \
#                 "you have to go to kitchen right after living room", \
#                 "go to living room means you have to go to kitchen in the future", \
#                 "you can only go to kitchen twice", \
#                 "don't go to bedroom more than two times"]

# 0_10
# input_ltl = "(! C U A)"
# constraints = ["you have to enter bathroom before kitchen"]
# input_ltl = "(! C U A) & (G C -> X D)" 
# constraints = ["you have to enter bathroom before kitchen", "you have to enter living room right after entering kitchen" ]
# input_ltl = "(! C U A) & (G C -> X D) & (G A -> F D)"
# constraints = ["you have to enter bathroom before kitchen", "you have to enter living room right after entering kitchen", "entering bathroom means you have to visit living room once"]
# input_ltl = "(! C U A) & (G C -> X D) & (G A -> F D) & (!(F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F(C)))))))))))))"
# constraints = ["you have to enter bathroom before kitchen", "you have to enter living room right after entering kitchen", "entering bathroom means you have to visit living room once", "you can only go to kitchen twice"]
# input_ltl = "(! C U A) & (G C -> X D) & (G A -> F D) & (!(F((C & (C U (!(C) & (!(C) U F((C & (C U (!(C) & (!(C) U F(C))))))))))))) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B)))))))"
# constraints = ["you have to enter bathroom before kitchen", "you have to enter living room right after entering kitchen", "entering bathroom means you have to visit living room once", "you can only go to kitchen twice", "don't go to bedroom more than two times"]

# 0_9
# input_ltl = "(! C U A) & (G A -> X B)" 
# constraints = ["you have to enter bathroom before kitchen", "you have to enter bedroom right after entering bathroom" ]
# input_ltl = "(! C U A) & (G A -> X B) & (G A -> G! D)" 
# constraints = ["you have to enter bathroom before kitchen", "you have to enter bedroom right after entering bathroom", "entering bathroom means you cannot visit living room even once" ]
# input_ltl = "(! C U A) & (G A -> X B) & (G A -> G! D) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B)))))))" 
# constraints = ["you have to enter bathroom before kitchen", "you have to enter bedroom right after entering bathroom", "entering bathroom means you cannot visit living room even once", "don't go to bedroom more than two times" ]
# input_ltl = "(! C U A) & (G A -> X B) & (G A -> G! D) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B))))))) & (G ! D)" 
# constraints = ["you have to enter bathroom before kitchen", "you have to enter bedroom right after entering bathroom", "entering bathroom means you cannot visit living room even once", "don't go to bedroom more than two times", "always avoid living room" ]

# 0_2 
# input_ltl = "(! D U A)"
# constraints = ["you have to enter bathroom before living room"]
# input_ltl = "(! D U A) & (G A -> X B)"
# constraints = ["you have to enter bathroom before living room", "you have to go to bedroom right after bathroom" ]
# input_ltl = "(! D U A) & (G A -> X B) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B)))))))"
# constraints = ["you have to enter bathroom before living room", "you have to go to bedroom right after bathroom" , "you can go to bedroom at most twice"]
# input_ltl = "(! D U A) & (G A -> X B) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B))))))) & (G C -> G ! A)"
# constraints = ["you have to enter bathroom before living room", "you have to go to bedroom right after bathroom" , "you can go to bedroom at most twice", "you cannot go to bathroom if you have visited kitchen"]
# input_ltl = "(! D U A) & (G A -> X B) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B))))))) & (G C -> G ! A)"
# constraints = ["you have to enter bathroom before living room", "you have to go to bedroom right after bathroom" , "you can go to bedroom at most twice", "you cannot go to bathroom if you have visited kitchen", "don't enter living room if you haven't enter kitchen"]

# 0_5
# input_ltl = "(! D U A)"
# constraints = ["you have to enter bathroom before living room"]
# input_ltl = "(! D U A) & !(F((A & (A U (!(A) & (!(A) U F(A)))))))"
# constraints = ["you have to enter bathroom before living room", "you can only go to bathroom once"]
# input_ltl = "(! D U A) & !(F((A & (A U (!(A) & (!(A) U F(A))))))) & (G A -> F C) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B)))))))"
# constraints = ["you have to enter bathroom before living room", "you can only go to bathroom once", "go to bathroom means you have to visit kitchen", "you have to visit bedroom at most two times"]
# input_ltl = "(! D U A) & !(F((A & (A U (!(A) & (!(A) U F(A))))))) & (G A -> F C) & (!(B) U F((B & (B U (!(B) & (!(B) U F(B))))))) & (G C -> G(! A))"
# constraints = ["you have to enter bathroom before living room", "you can only go to bathroom once", "go to bathroom means you have to visit kitchen", "you have to visit bedroom at most two times", "you cannot go to bathroom if kitchen is visited"]

# 0_7
# input_ltl = "(F C)"
# constraints = ["you have to visit kitchen at least once"]
# input_ltl = "(F C) & (G C -> X B)"
# constraints = ["you have to visit kitchen at least once", "you have to go to bedroom right after kitchen"]
# input_ltl = "(F C) & (G C -> X B) & (G A -> F B)"
# constraints = ["you have to visit kitchen at least once", "you have to go to bedroom right after kitchen", "you have to visit bedroom if you ever been to bathroom"]

# load tasks
task_description = load_from_file("prompts/planning/planning_with_cons_v1.txt")
# breakpoint()
# task_description = "\n".join(task_description)
lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_1.txt")
# lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_10.txt")
# lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_9.txt")
# lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_2.txt")
# lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_5.txt")
example = program2example(lines)
example = "\n".join(example)
prompt = task_description+ "\n\n" + example + "\n"
# prompt = [line + '\n' for line in prompt]
# [print(line) for line in prompt]

# give goal according to each prompt
dummy_query = "Go to toilet" # for 0_1
# dummy_query = "Put salmon in Fridge"
# dummy_query = "Cook a pie"
# dummy_query = "Type on computer"
# dummy_query = "Browse on computer by my desk"

prompt += f"\n{dummy_query}\nDescription:"
# prompt = ''.join(prompt)
# [print(line) for line in prompt]
print(prompt)
gpt4 = GPT4(stop=['\n', 'Clean the kitchen', 'Plan'])
embed_engine = GPT3(engine="text-embedding-ada-002")

# first response is description
output = gpt4.generate(prompt)[0]
print(output)

prompt = prompt + f"\n{output}"
graph_dict_path = "virtualhome_v2.3.0/env_graphs/TestScene1_graph.json"
graph_dict = load_from_file(graph_dict_path)
allowed_actions = load_from_file("virtualhome_v2.3.0/resources/allowed_actions.json")
act2embed = load_from_file("/users/zyang157/data/zyang157/virtualhome/action_embeds/act2embed_vh_gpt3-text-embedding-ada-002.pkl")
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

        # print(dummy_program_line)
        try:
            # breakpoint()
            grounded_program_line = replace_new_obj_id([dummy_program_line], graph_dict)[0]
            program.extend(convert_old_program_to_new([grounded_program_line]))
        except:
            raise Exception(f"probably parsing problem: {dummy_program_line}")
        if invalid_action_mem == grounded_program_line:
            continue

    partial_program = program
    # this step ensure output is executable and renderable(?)
    success, state_list = state_change_by_step(comm, program, input_ltl, obj_ids, room_ids, mappings, init_position, init_room, stopped=stopped)
    # prompt += f" {output}" if prompt[-2:] == f"{idx}." else f"\n{output}"
    prompt += f" {output}"
    if not success:
        invalid_action = action_string
        invalid_state_lists = state_lists + [state_list]
        reprompted_plan = f"\nError: {reprompt(gpt4, valid_actions, invalid_action, constraints, invalid_state_lists, mappings, id2name)} The correct plan would be:"
        program = program[:-1]
        print("handling error by reprompting")
        print(reprompted_plan)
        prompt += reprompted_plan
        invalid_action_mem = grounded_program_line
        stopped = False
    else:
        if "DONE" in output: break
        state_lists.append(state_list)
        valid_actions.append(action_string)
        invalid_action_mem = None
        idx += 1
    # breakpoint()
print("Programs:")
# old_program = program
[print(line) for line in program]
# print(old_program)
print(prompt)