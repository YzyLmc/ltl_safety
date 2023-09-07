
class RobotAndObjectStates:

    def __init__(self, origin_location, robot_standing_start=True, reprompt_type='causal'):

        self.parametrizable_robot_skills = {'walk':'[walk] {}', 'find':'[walk] {}', 'puton':'[puton] {} ({})', 'putin':'[putin] {} ({})', 'grab':'[grab] {}', 'lookat':'[lookat] {}', 'touch':'[touch] {}', 'standup':'[standup]', 'sitdown':'[sitdown]', 'open': '[open] {}', 'close': '[close] {}', 'switchon':'[switchon] {}', 'switchoff':'[switchoff] {}'}
        self.object_states = {}

        self.robot_states = {'location': origin_location, 'holding':None, 'standing':robot_standing_start}

        self.check_preconditions = {'walk': self.walkto_precondition, 'find': self.walkto_precondition, 'puton': self.putdown_precondition, 'putin': self.putdown_precondition, 'grab': self.grab_precondition, 'lookat': self.look_precondition, 'touch': self.touch_precondition, 'standup': self.stand_precondition, 'sitdown': self.sit_precondition, 'open': self.open_precondition, 'close': self.close_precondition, 'switchon':self.turnon_precondition, 'switchoff':self.turnoff_precondition}
        self.modify_postconditions = {'walk': self.walkto_postcondition, 'find': self.walkto_postcondition, 'puton': self.putdown_postcondition, 'putin': self.putdown_postcondition, 'grab': self.grab_postcondition, 'lookat': self.look_postcondition, 'touch': self.touch_postcondition,'standup': self.stand_postcondition, 'sitdown': self.sit_postcondition, 'open': self.open_postcondition, 'close': self.close_postcondition, 'switchon':self.turnon_postcondition, 'switchoff':self.turnoff_postcondition}

        reprompt_types = {'success-only':self.success_only_reprompt_format, 'inference':self.inference_reprompt_format, 'causal':self.causal_reprompt_format}
        self.reprompt_function = reprompt_types[reprompt_type]
    
    def success_only_reprompt_format(self, prompt, natural_action):
        return 'Error: Task failed. A correct step would be to'
    
    def inference_reprompt_format(self, prompt, natural_action):
        return f'Error: I cannot {natural_action}. A correct step would be to'
    def causal_reprompt_format(self, prompt, natural_action):
        return prompt

    def populate_object_states(self, object_attribute_dict):

        for obj in object_attribute_dict.keys():

            if obj not in self.object_states:
                self.object_states[obj] = {}


            self.object_states[obj]['location'] = object_attribute_dict[obj]['location']
            
            if object_attribute_dict[obj]['receptacle']:
                self.object_states[obj]['contains'] = set()

                for obj1 in object_attribute_dict.keys():
                    if object_attribute_dict[obj]['location'] == object_attribute_dict[obj1]['location'] and obj!=obj1:
                        self.object_states[obj]['contains'].add(obj1)
            

            if object_attribute_dict[obj]['openable']:
                self.object_states[obj]['open'] = False

            if 'onable' in object_attribute_dict[obj].keys():
                self.object_states[obj]['on'] = False

    def run_step(self, action, natural_action, modify=True):

        skill = action[action.index('[')+1 : action.index(']')]

        try:
            args = action[action.index(']')+1:].strip()
            while "<" in args or ">" in args:
                args = args.replace('<','')
                args = args.replace('>','')
            args = args.split(' ')
        except:
            args = ['']

        precondition_check = self.check_preconditions[skill]
        postcondition_function = self.modify_postconditions[skill]

        preconds_satisfied, message = precondition_check(*args)
        
        if preconds_satisfied and modify:
            postcondition_function(*args)
        else:
            message = self.modify_message(message, natural_action)

        return not preconds_satisfied, message

    def modify_message(self, message, natural_action):

        return self.reprompt_function(message, natural_action)
    
    def turnon_precondition(self, obj):
        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing'] == True

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'

        return cond1 and cond2 and cond3, message
    
    def turnoff_precondition(self, obj):
        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing'] == True

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'

        return cond1 and cond2 and cond3, message

    def walkto_precondition(self, location):
        #location = args[0]
        cond1 = self.robot_states['standing']==True
        
        cond2 = self.robot_states['holding']!=location
        
        message = None
        

        if not cond1:
            message = 'Error: I am sitting. A correct step would be to'
        if not cond2:
            message = f'Error: I am already holding the {location}. A correct step would be to'
        return cond1 and cond2, message

    def putdown_precondition(self, obj, recepticle):
        # obj = args[0]; recepticle = args[1]
        message = None 
        cond1 = self.robot_states['location']==self.object_states[recepticle]['location']
        
        cond2 = self.robot_states['holding']==obj

        cond3 = self.robot_states['standing']==True
        
        cond4 = 'open' not in self.object_states[recepticle] or ('open' in self.object_states[recepticle] and self.object_states[recepticle]['open'])

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(recepticle)
        if not cond2:
            message = 'Error: I am not holding the {}. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'
        if not cond4:
            message = 'Error: the {} is not open. A correct step would be to'.format(recepticle)

        return cond1 and cond2 and cond3 and cond4, message
        
    def grab_precondition(self, obj):

        # obj = args[0]
        message = None
        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing']==True
        
        potential_receptacle = None
        for obj1 in self.object_states:

            if obj1!=obj and self.object_states[obj]['location']==self.object_states[obj1]['location'] and 'open' in self.object_states[obj1].keys():
                potential_receptacle = obj1
                break
        
        cond4 = potential_receptacle is None or (potential_receptacle is not None and self.object_states[obj1]['open'] == True)

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2: 
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'
        if not cond4:
            message = 'Error: the {} is not open. A correct step would be to'.format(potential_receptacle)

        return cond1 and cond2 and cond3 and cond4, message
    
    def look_precondition(self, obj):
        # obj = args[0]

        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing']==True

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'

        return cond1 and cond2 and cond3, message

    def touch_precondition(self, obj):
        # obj = args[0]

        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing'] == True

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'

        return cond1 and cond2 and cond3, message

    def stand_precondition(self, args):

        message = None

        cond1 = self.robot_states['standing'] == False

        if not cond1:
            message = 'Error: I am already standing. A correct step would be to'.format(cond1)

        return cond1, message
    
    def sit_precondition(self, args):

        message = None

        cond1 = self.robot_states['standing'] == True

        if not cond1:
            message = 'Error: I am already sitting. A correct step would be to'.format(cond1)

        return cond1, message

    def open_precondition(self, obj):
        # obj = args[0]

        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing']==True
        cond4 = self.object_states[obj]['open']==False

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'
        
        if not cond4:
            message = 'Error: the {} is already open. A correct step would be to'.format(obj)

        return cond1 and cond2 and cond3, message

    def close_precondition(self, obj):
        # obj = args[0]

        message = None

        cond1 = self.robot_states['location']==self.object_states[obj]['location']
        cond2 = self.robot_states['holding'] == None
        cond3 = self.robot_states['standing']==True
        cond4 = self.object_states[obj]['open']==True

        if not cond1:
            message = 'Error: I am not near the {}. A correct step would be to'.format(obj)

        if not cond2:
            message = 'Error: I am already holding the {}. My hands are full. A correct step would be to'.format(obj)
        if not cond3:
            message = 'Error: I am sitting. A correct step would be to'
        
        if not cond4:
            message = 'Error: the {} is already closed. A correct step would be to'.format(obj)

        return cond1 and cond2 and cond3, message
    
    def walkto_postcondition(self, obj):

        # obj = args[0]

        self.robot_states['location'] = self.object_states[obj]['location']

    def putdown_postcondition(self, obj, recepticle):
        # obj = args[0]
        # recepticle = args[1]

        self.robot_states['holding'] = None
        self.object_states[obj]['location'] = self.object_states[recepticle]['location']
        self.object_states[recepticle]['contains'].add(obj)
    
    def grab_postcondition(self, obj):

        # obj = args[0]

        self.robot_states['location'] = None
        self.robot_states['holding'] = obj
        self.object_states[obj]['location'] = None
        
        for obj1 in self.object_states.keys():

            if 'contains' in self.object_states[obj1] and obj in self.object_states[obj1]['contains']:
                self.object_states[obj1]['contains'].remove(obj)
    
    def look_postcondition(self, obj):
        # obj = args[0]
        pass
    
    def touch_postcondition(self, obj):
        # obj = args[0]
        self.robot_states['location'] = None
    def turnon_postcondition(self, obj):
        self.object_states[obj]['on']=True
        self.robot_states['location'] = None
    def turnoff_postcondition(self, obj):
        self.object_states[obj]['on']=False
        self.robot_states['location'] = None
    def stand_postcondition(self, args):
        self.robot_states['standing'] = True
    
    def sit_postcondition(self, args):
        self.robot_states['standing'] = False
    
    def open_postcondition(self, obj):
        # obj = args[0]
        
        self.robot_states['location'] = None
        self.object_states[obj]['open'] = True
    
    def close_postcondition(self, obj):
        # obj = args[0]

        self.robot_states['location'] = None
        self.object_states[obj]['open']=False

