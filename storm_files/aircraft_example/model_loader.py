import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import re
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import utils


def load_model(path):
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "Rmin=? [F \"goal\" & !\"broken\" ]"
    # formula_str = 'Pmax=? [!\"broken\" U \"goal\"]'
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    options = stormpy.BuilderOptions([f.raw_formula for f in properties])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    model = stormpy.build_sparse_parametric_model_with_options(prism_program,options)
    parameters = model.collect_probability_parameters()
    numstate = model.nr_states
    # print("Number of states after bisim:", numstate)
    return parameters, model, properties

# Rebuilds model for new initial position and damage conditions
def reload_model(path,x,y,dam1,dam2):
    prism_str = path.replace('prism','template')
    with open(prism_str,'r') as prism_file:
        prism_raw = prism_file.readlines()
    ## Hard coded x,y and damage entries:
    prism_raw[58] = prism_raw[58].replace('initX',str(x))
    prism_raw[59] = prism_raw[59].replace('initY',str(y))
    prism_raw[27] = prism_raw[27].replace('init_damage',str(dam1))
    prism_raw[33] = prism_raw[33].replace('init_damage2', str(dam2))
    with open(path.replace('prism','temp'),'w') as prism_temp:
        prism_temp.writelines(prism_raw)
    prism_program = stormpy.parse_prism_program(path.replace('prism','temp'))
    formula_str = "Rmin=? [F \"goal\" & !\"broken\" ]"
    # formula_str = 'Pmax=? [!\"broken\" U \"goal\"]'
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    options = stormpy.BuilderOptions([f.raw_formula for f in properties])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    model = stormpy.build_sparse_parametric_model_with_options(prism_program,options)
    parameters = model.collect_probability_parameters()
    numstate = model.nr_states
    # print("Number of states after bisim:", numstate)
    return parameters, model, properties

def simulate(model,policy,steps):
    simulator = stormpy.simulator.create_simulator(model)
    simulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)
    # simulator.set_observation_mode(stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
    total_reward = 0
    state = 0
    act_list = []
    for i in range(steps):
        actions = simulator.available_actions()
        act_list.append(actions[utils.get_choice_at(policy,state)])
        state,reward,labels = simulator.step(actions[utils.get_choice_at(policy,state)])
        total_reward += reward[0]
    # out_array = [output['x'],output['y'],output['x']]
    return total_reward,state,act_list,simulator.is_done()