import re

def get_choice_at(sched,s):
    return sched.get_choice(s).get_deterministic_choice()

def get_states_from_choice(model,s,a):
    return model.states[int(re.findall('\\d+', str(s.actions[int(a)].transitions))[0])]

def get_choice_label_at(model,sched,s):
    if len(get_choice_labels(model,s))>1:
        return get_choice_labels(model,s)[get_choice_at(sched,s)]
    return 'gentle'

def get_choice_labels(model, s):

    return [model.choice_labeling.get_labels_of_choice(model.get_choice_index(s, a_i)).pop() for a_i in
            range(get_no_actions(model,s)) if model.choice_labeling.get_labels_of_choice(model.get_choice_index(s, a_i))]

def get_no_actions(model,s):
    return len(model.states[s].actions)
