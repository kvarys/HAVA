import time
from hava import AlignmentValue
import itertools
from itertools import cycle
from pathlib import Path
import os
import argparse
import random
from agent_wrapper import AgentWrapper
from constants import JUNCTION_NAMES_TO_SEEDS_1
from environment import SimulatorController
import traci
import numpy as np

def make_model_dirs(model_name):
    cwd = Path.cwd()
    from_scratch = True
    try:
        os.mkdir(f"{cwd}/models/{model_name}")
        os.mkdir(f"{cwd}/models/{model_name}/training")
        os.mkdir(f"{cwd}/models/{model_name}/training/priority")
    except FileExistsError as e:
        from_scratch = False
    priority_training_path = f"{cwd}/models/{model_name}/training/priority/training.csv"
    priority_model_path = f"{cwd}/models/{model_name}/training/priority"
    return priority_training_path, priority_model_path, from_scratch

def log_training(path,episode,time,avg_time_100, avg_time_700,score,avg_score_100,avg_score_700):
    f = open(path, 'a+')
    f.write(f"{episode},{time},{avg_time_100},{avg_time_700},{score},{avg_score_100},{avg_score_700}" + "\n")
    f.close()

def vehicle_ids(i):
    limit = 10
    if i > limit:
        i = vehicle_ids(i-limit)
    return i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelname", help="what to call this model",type=str)
    parser.add_argument("alpha", help="what alpha to use",type=str)
    parser.add_argument("tau", help="what tau to use",type=str)
    parser.add_argument("mode", help="what norms to use? (safe, dd, mix)",type=str)
    args = parser.parse_args()
    ALPHA = float(args.alpha)
    TAU = float(args.tau)
    MODE = args.mode
    MODEL_NAME = args.modelname + f"_mode_{MODE}_alpha_{ALPHA}_tau_{TAU}"

    priority_training_path, priority_model_path, from_scratch = make_model_dirs(MODEL_NAME)

    if from_scratch:
        log_training(path=priority_training_path,episode="episode",time="time",avg_time_100="avg_time_100",avg_time_700="avg_time_700",score="score",avg_score_100="avg_score_100",avg_score_700="avg_score_700")

    random.seed(28)
    priority_scores = []
    priority_finish_times = []
    finish_times = None
    ddqn_scores = None # will be priority_scores

    positions = [x for x in range(10)]
    starting_positions = [x for x in itertools.permutations(positions, r=3)]
    starting_positions = [starting_positions[145]] # experiment with a single starting position
    cycle_starting_positions_iterable = cycle(starting_positions)

    EPISODES = 300000
    MAX_TIME = 198000 # 55 hours in seconds
    WEIGHTED = True
    GAMMA = 0.99

    junction_name = "priority_1"

    per_timestep = []

    config = "maps/threeJunctions/priority_junction_25/sumo.sumocfg"
    seed = JUNCTION_NAMES_TO_SEEDS_1[junction_name]

    epsilon = 1

    ddqn_agent = AgentWrapper(model_checkpoint_path=priority_model_path, model_name=MODEL_NAME)
    if not from_scratch: # we are continuing the training
        ddqn_agent.load_model()
        priority_finish_times = ddqn_agent.last_finish_times
        priority_scores = ddqn_agent.last_scores

    if from_scratch:
        START_TIME = time.time()
    else:
        START_TIME = ddqn_agent.initial_time

    avg_score = 0
    highest_pass = []

    for i in range(0, EPISODES):

        vehicle_id = "1.0"
        starting_position = next(cycle_starting_positions_iterable)

        current_time = time.time()
        if (current_time - START_TIME) > MAX_TIME:
            # saves the weights, hyperparams and replay buffer to enable further training.
            ddqn_agent.save_model()
            break

        done = False
        score = 0

        try:
            simulator = SimulatorController(hava=AlignmentValue(tau=TAU,alpha=ALPHA, datadriven_social_norms_filename="nn_ns_priority.keras", name="priority"),run_gui=False, mode=MODE, vehicle_id=vehicle_id, seed=seed, config_path=config, agent=ddqn_agent, junction_name=junction_name)
        except Exception as e:
            MAX_TIME = -1 # SUMO failed to stop, times up
            print(f"Simulator error:\n {e}")
            break

        current_state = None
        timestep = 0
        weights = []

        while not done:
            timestep += 1
            action = None
            acceleration = None
            done = should_control = False

            ddqn_scores = priority_scores
            finish_times = priority_finish_times
            ddqn_agent.episode_num = i + ddqn_agent.initial_episode_num

            if current_state != None:
                ddqn_agent.timestep_num = timestep
                action, acceleration = ddqn_agent.choose_action(current_state)

            (next_state, action, reward, done, should_control) = simulator.next_step(current_state, action, timestep, starting_position)
            if should_control or done:
                weighted_reward = simulator.hava.calculate_weighted_reward(reward)
                score += weighted_reward

                ddqn_agent.remember(state=current_state, action=acceleration, reward=weighted_reward, new_state=next_state, done=int(done))

                ddqn_agent.learn()

            if done:
                simulator.hava.reset()

            current_state = next_state

        traci.close()

        per_timestep = []
        ddqn_scores.append(score)
        finish_times.append(simulator.agent_timestep)

        if len(ddqn_scores) > 705:
            ddqn_scores = ddqn_scores[5:]
            finish_times = finish_times[5:]

        ddqn_agent.last_finish_times = finish_times
        ddqn_agent.last_scores = ddqn_scores

        avg_finish = np.mean(finish_times[max(0, len(finish_times)-500):])
        avg_finish_100 = np.mean(finish_times[max(0, len(finish_times)-100):])

        avg_score = np.mean(ddqn_scores)
        avg_score_100 = np.mean(ddqn_scores[max(0, len(ddqn_scores)-100):])

        log_training(path=priority_training_path,episode=(i + ddqn_agent.initial_episode_num),time=simulator.agent_timestep,avg_time_100=avg_finish_100,avg_time_700=avg_finish,score=score,avg_score_100=avg_score_100,avg_score_700=avg_score)
