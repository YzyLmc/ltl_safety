import os
import sys
sys.path.append("src/")
import spot

from utils import *
from lang2ltl import rer, ground_res, ground_utterances, build_placeholder_map, substitute, translate_modular, unify_formula, sub_predicate, name_to_prop
from get_embed import GPT3
from openai.embeddings_utils import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")

class constraint_module():
    def __init__(self,constraints={}, constraint_strs=[], placeholder_map=None):
        self.translator = GPT4()
        self.constraints = constraints
        self.placeholder_map = placeholder_map
        self.embed_engine = GPT3(engine="text-embedding-ada-002")
        if constraint_strs:
            constraints = self.encode_constraints(constraint_strs)
            self.dfa, self.accepting_states, self.init_state = ltl2digraph(self. encode_constraints(constraints))

    def encode_constraints(self, constraint_strs, manipulation=True, prompt_fpath="prompts/translation/rer_general.txt", trans_modular_prompt_fpath="prompts/translation/symbolic_trans_manipulation_v2.txt", obj_embed_fpath="/users/zyang157/data/zyang157/virtualhome/obj_embeds/0_spot.pkl"):
        '''
        convert a list of language constraints and returns final formula with prop map
        '''
        # rer
        prompt = load_from_file(prompt_fpath)
        obj_embeds = load_from_file(obj_embed_fpath)
        names, utt2res = rer(self.translator, prompt, constraint_strs)
        # resolution is skipped temporarily
        re2grounds = {}
        for obj in names:
            obj_embed = self.embed_engine.get_embedding(obj)
            sims = {o: cosine_similarity(e, obj_embed) for o, e in obj_embeds.items()}
            sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            re2grounds[obj] = list(dict(sims_sorted[:1]).keys())
            # re2grounds[obj] = [grounded_obj]
        # re2grounds = {obj: [name_to_prop(obj)] for obj in names}
        # build mappings
        ground_utts, objs_per_utt = ground_utterances(constraint_strs, utt2res, re2grounds)
        sym_utts, sym_ltls, out_ltls, placeholder_maps = translate_modular(ground_utts, self.translator, objs_per_utt, trans_modular_prompt_fpath=trans_modular_prompt_fpath)
        unified_ltl, obj_mapping = unify_formula(out_ltls, placeholder_maps)
        self.constraint = unified_ltl
        self.obj_mapping = obj_mapping
        if manipulation:
            unified_ltl, pred_mapping = sub_predicate(unified_ltl)
            self.pred_mapping = pred_mapping
        return unified_ltl, obj_mapping, pred_mapping if manipulation else None

    def action_pruning(self, start, trajs):
        '''
        validate prop level trajectory
        :params str start:
        :params list(str) trajs: 
        '''
        dfa, accepting_states, curr_state = ltl2digraph(input_ltl)
        state_list = []
        success = True
        for state in prop_traj:
            action = state
            stopped = True if state == prop_traj[-1] else False
            if validate_next_action(dfa, curr_state, action, accepting_states,stopped=stopped):
                state_list.append(f"Safe: {action}")
                curr_state = progress_ltl(dfa, curr_state, action)
            else:
                state_list.append(f"Violated: {action}")
                success = False
                break
        pass

    def agent_query(self, trajectory, invalid_action, agent_prompt):
        pass

    def human_query(self, trajectory, human_prompt):
        pass

if __name__ == "__main__":
    cm = constraint_module()
    constraint_strs = ["don't go to kitchen until go to bathroom",\
                        "always avoid coffeetable",\
                        "if you go to kitchen, you have to go to living room later"]
    """
    & & W ! a b G ! c G i a F d
    {'A': 'living room', 'B': 'coffeetable', 'C': 'bathroom', 'D': 'kitchen'}
    {'a': 'agent_at(D)', 'b': 'agent_at(C)', 'c': 'agent_at(B)', 'd': 'agent_at(A)'}
    """
    unified_ltl, obj_mapping, pred_mapping = cm.encode_constraints(constraint_strs)
    print(unified_ltl, obj_mapping, pred_mapping)