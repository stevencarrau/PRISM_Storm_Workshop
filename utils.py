import re

def get_choice_at(sched,s):
    return sched.get_choice(s).get_deterministic_choice()

def get_next_fsc(model,s,sched):
    a = get_choice_at(sched,s)
    j_out = model.state_valuations.get_json(get_states_from_choice(model,s,a))
    return int(j_out['fsc'])

def get_states_from_choice(model,s,a):
    return model.states[int(re.findall('\\d+', str(s.actions[int(a)].transitions))[0])]

def get_choice_label_at(model,sched,s):
    if len(get_choice_labels(model,s))>1:
        return get_choice_labels(model,s)[get_choice_at(sched,s)]
    return 'west'

def choice_to_vel(label):
    if label=='east':
        return (1,0)
    elif label=='west':
        return (-1,0)
    elif label=='north':
        return (0,1)
    elif label=='south':
        return (0,-1)
    else:
        print('Check output for vel')
        return (0,0)

def xy_to_idx(x,y,grid):
    return y*grid + x%grid

def idx_to_xy(idx,grid):
    return idx%grid,int(idx/grid)

def goal_to_int(goal,s=0):
    if goal == 'true':
        return 1
    elif goal == 'false':
        return 0
    else:
        return print('Error')

def goal_update(s_tup=[0,0,0]):
    state_type =  s_tup[0]
    return (1,) if state_type==35 else (s_tup[2],)

def storm_in_to_idx(model,s,grid_size):
    j_out = model.state_valuations.get_json(s)
    return xy_to_idx(int(j_out["dx"]),int(j_out["dy"]),grid_size), xy_to_idx(int(j_out["ax"]),int(j_out["ay"]),grid_size), goal_to_int(j_out['pickup'])

def storm_out_to_idx(model,s,grid_size,s_in_tup=[0,0,0]):
    state_type = s_in_tup[0]
    j_out = model.state_valuations.get_json(s)
    return xy_to_idx(int(j_out["dx"]),int(j_out["dy"]),grid_size), 1 if state_type==35 else s_in_tup[2]

def storm_to_stage(model,s):
    j_out = model.state_valuations.get_json(s)
    return int(j_out['fsc'])

def get_choice_labels(model, s):

    return [model.choice_labeling.get_labels_of_choice(model.get_choice_index(s, a_i)).pop() for a_i in
            range(get_no_actions(model,s)) if model.choice_labeling.get_labels_of_choice(model.get_choice_index(s, a_i))]

def get_no_actions(model,s):
    return len(model.states[s].actions)

def grid_to_idx(model,s,grid_size):
    j_out = model.state_valuations.get_json(s)
    return xy_to_idx(int(j_out["ax"]),int(j_out["ay"]),grid_size)