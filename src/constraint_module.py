import os
import sys
sys.path.append("src/")
import spot

from utils import *
from lang2ltl import rer, ground_res, ground_utterances, build_placeholder_map, substitute, translate_modular, unify_formula

openai.api_key = os.getenv("OPENAI_API_KEY")

class constraint_module():
    def __init__(self,constraints={}, constraint_strs=[], placeholder_map=None):
        self.translator = GPT4()
        self.constraints = constraints
        self.placeholder_map = placeholder_map
        if constraint_strs:
            constraints = self.encode_constraints(constraint_strs)
            self.dfa, self.accepting_states, self.init_state = ltl2digraph(self. encode_constraints(constraints))

    def encode_constraints(self, constraint_strs, prompt_fpath="prompts/translation/rer_general.txt"):
        '''
        convert a list of language constraints and returns final formula with prop map
        '''
        # rer
        prompt = load_from_file(prompt_fpath)
        names, utt2res = rer(self.translator, prompt, constraint_strs)
        # resolution is skipped temporarily
        re2grounds = {obj: [obj] for obj in names}
        # build mappings
        ground_utts, objs_per_utt = ground_utterances(constraint_strs, utt2res, re2grounds)
        sym_utts, sym_ltls, out_ltls, placeholder_maps = translate_modular(ground_utts, self.translator, objs_per_utt)
        # breakpoint()
        unified_ltl, unified_mapping = unify_formula(out_ltls, placeholder_maps)
        self.constraint = unified_ltl
        self.mapping = unified_mapping
        return unified_ltl, unified_mapping

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
    constraints = cm.encode_constraints(constraint_strs)
    print(constraints)