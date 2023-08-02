import spot

from utils import GPT4, ltl2digraph, validate_next_action

class constraint_module():
    def __init__(self,constraints=None):
        self.translator = GPT4()
        if constraints:
            self.dfa, self.accepting_states, self.init_state = ltl2digraph(self. encode_constraints(constraints))
        
    @staticmethod
    def encode_constraints(self, constraints, translator):
        '''
        convert a list of language constraints and returns final formula with prop map
        '''
        input_ltl = "(! c U b) & G ! h" # enter bedroom before livingroom and always avoid coffeetable_2
        dfa, accepting_states, curr_state = ltl2digraph(input_ltl)
        pass

    def action_pruning(self, start, trajs):
        '''
        validate prop level trajectory
        :params str start:
        :params list(str) trajs: 
        '''
        pass

    def agent_query(self, trajectory, invalid_action, agent_prompt):
        pass

    def human_query(self, trajectory, human_prompt):
        pass