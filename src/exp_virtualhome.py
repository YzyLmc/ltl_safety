import os
import openai
from openai.embeddings_utils import cosine_similarity
from utils import program2example, load_from_file, GPT4, get_action_and_obj
from program_conversion import replace_new_obj_id
from get_embed import GPT3

openai.api_key = os.getenv("OPENAI_API_KEY")

task_description = load_from_file("prompts/planning/planning_nn_v1.txt")
lines = load_from_file("virtualhome_v2.3.0/dataset/ltl_safety/tasks/0_1.txt")
example = program2example(lines)

prompt = task_description+ ["\n\n"] + example
[print(line) for line in prompt]

dummy_query = "Go to apple"

prompt.append("\n" + dummy_query + "\nDescription:")
prompt = ''.join(prompt)
# [print(line) for line in prompt]
# print(prompt)
gpt4 = GPT4(stop=['\n'])
embed_engine = GPT3(engine="text-embedding-ada-002")

# # first response is description
# des = gpt4.generate(''.join(prompt))[0]
# print("This is a description:", des)

# prompt_1 = prompt + des + '\n'
# act_0 = gpt4.generate(''.join(prompt_1))
# print("first action:", act_0)

# first response is description
output = gpt4.generate(''.join(prompt))[0]
print("This is a description:", output)

prompt = prompt + output + '\n'
graph_dict_path = "virtualhome_v2.3.0/env_graphs/TestScene1_graph.json"
graph_dict = load_from_file(graph_dict_path)
allowed_actions = load_from_file("virtualhome_v2.3.0/resources/allowed_actions.json")
act2embed = load_from_file("/users/zyang157/data/zyang157/virtualhome/action_embeds/act2embed_vh_gpt3-text-embedding-ada-002.pkl")
n_try = 0
program = []
while n_try < 6:
    output = gpt4.generate(''.join(prompt))[0]
    if "DONE" in output: break
    action_string = output.strip(".").split(". ")[-1]
    if not action_string in allowed_actions:
        action_string_embed = embed_engine.get_embedding(action_string)
        sims = {o: cosine_similarity(e, action_string_embed) for o, e in act2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        action_string = list(dict(sims_sorted[:1]).keys())[0]
    # breakpoint()
    act, obj_ls = get_action_and_obj(action_string)
    obj_ls = [f"{obj} (0)" for obj in obj_ls]
    dummy_program_line = f"{act} {' '.join(obj_ls)}"
    # print(dummy_program_line)
    # try:
    # breakpoint()
    grounded_program_line = replace_new_obj_id([dummy_program_line], graph_dict)[0]
    program.append(dummy_program_line)
    # except:
    #     raise Exception(f"probably parsing problem: {grounded_program_line}")
    print(grounded_program_line)
    prompt = prompt + output + '\n'
    # breakpoint()
breakpoint()