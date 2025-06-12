import math
from collections import deque
import itertools
from itertools import cycle
import logging
from agent_wrapper import AgentWrapper
from junction import Junction
from hava import AlignmentValue
from utils import *
import traci


class SimulatorController:
    """
    Environment provides methods to extract information from the simulation, and to step the simulation forward.
    """

    def __init__(self, hava:AlignmentValue, agent: AgentWrapper, mode, vehicle_id, config_path, junction_name, run_gui=False, seed="23429"):
        self.run_gui = run_gui
        self.junction_name = junction_name
        self.CHECKPOINT_REWARD = 50
        self.CHECKPOINT_EVERY = 2
        self.checkpoint_next = 20
        self.past_speeds = deque([], maxlen=3)

        self._start_sumo(run_gui=run_gui, config_path=config_path, seed=seed)
        self.norms_mode = mode

        self.junction: Junction = Junction()
        self.vehicle_id = vehicle_id

        self.agent_timestep = 0
        self.distance_travelled_so_far = 0
        self.done_once = False
        self.max_reward = 100
        self.agent_passed = False

        self.state = None
        self.last_state = None

        positions = [x for x in range(10)]
        self.starting_positions = [x for x in itertools.permutations(positions, r=3)]
        self.starting_positions = [self.starting_positions[145]] # for now experiment with a single starting position
        self.starting_positions = cycle(self.starting_positions)

        self.hava = hava
        self.agent = agent

    def _start_sumo(self, run_gui, config_path, seed):
        ''' start the simulator '''
        try:
            if run_gui:
                traci.start(["sumo-gui", '-c', config_path, "--device.fcd.period", "100", "--collision-output", "results/logs/collisions.xml", "--collision.action", "warn", "--seed", seed, "--time-to-impatience", "0", "--step-length", "0.1"])
            else:
                traci.start(["sumo", '-c', config_path, "--collision-output", "results/logs/collisions.xml", "--collision.action", "warn", "--seed", seed, "--time-to-impatience", "0", "--step-length", "0.1"])
        except Exception as e:
            ''' Sumo failed to switch off, save the model'''
            self.agent.save_model()

    def _steer_vehicle_with_safety_norms_only(self, current_state, action):
        '''
        Control the vehicle
        '''
        self.past_speeds.append(action)
        reward = 0
        x = current_state[-3]
        # define checkpoints at which the agent gets rewarded
        if x > self.checkpoint_next:
            reward = 10 + np.mean(self.past_speeds)
            self.checkpoint_next += self.CHECKPOINT_EVERY
        else:
            reward = -1

        intended_speed_in_ms = self._convert_action_to_speed_50(action) # in m/s
        traci.vehicle.setSpeed(self.vehicle_id, intended_speed_in_ms)
        traci.executeMove()
        try:
            actual_speed_in_ms = traci.vehicle.getSpeed(self.vehicle_id) # in m/s
            next_state, _, _ = self.get_state()
            logging.debug(f"simulation moved by half-step: \n {current_state} \n {next_state}")
            logging.debug(f"after traci.executeMove the action was {action} intended_speed was {intended_speed_in_ms} but actual_speed is {actual_speed_in_ms}")
            actual_action = self._convert_speed_to_action_50(actual_speed_in_ms) # in km/h
            if current_state[-3] > 96:
                traci.vehicle.setSpeedMode(self.vehicle_id,32)#make the agent ignore legal and safety constraints of SUMO
            speed_difference = abs(actual_action - action)

            self.hava.calculate_new_reputation_safe_only(next_state=next_state, current_state=current_state, ddqn_action=action, rb_distance=speed_difference)

            next_state, _, _ = self.get_state() # get next_state with the latest weight
            experience = [(next_state, action, reward, False, True)]
        except traci.TraCIException as e:
            logging.info("car stopped existing half-way through the step.")
            experience = [(current_state, action, 10+ np.mean(self.past_speeds), True, False)]
        return experience

    def _steer_vehicle_with_datadriven_norms_only(self, current_state, action):
        '''
        Control the vehicle
        '''
        self.past_speeds.append(action)
        reward = 0
        x = current_state[-3]
        # define checkpoints at which the agent gets rewarded
        if x > self.checkpoint_next:
            reward = 10 + np.mean(self.past_speeds)
            self.checkpoint_next += self.CHECKPOINT_EVERY
        else:
            reward = -1

        intended_speed_in_ms = self._convert_action_to_speed(action) # in m/s
        traci.vehicle.setSpeed(self.vehicle_id, intended_speed_in_ms)
        traci.executeMove()
        try:
            next_state, _, _ = self.get_state()

            self.hava.calculate_new_reputation(current_state=current_state, ddqn_action=action, actual_action=action, rb_distance=0)

            next_state, _, _ = self.get_state() # get next_state with the latest reputation
            experience = (next_state, action, reward, False, True)
        except traci.TraCIException as e:
            logging.info("car stopped existing half-way through the step.")
            experience = (current_state, action, 10+np.mean(self.past_speeds), True, False)
        return experience

    def _steer_vehicle_with_hava(self, current_state, action):
        '''
        Control the vehicle
        '''
        experience = ()
        self.past_speeds.append(action)
        reward = 0
        x = current_state[-3]
        # define checkpoints at which the agent gets rewarded
        if x > self.checkpoint_next:
            reward = 10 + np.mean(self.past_speeds)
            self.checkpoint_next += self.CHECKPOINT_EVERY
        else:
            reward = -1

        intended_speed_in_ms = self._convert_action_to_speed_50(action) # in m/s
        traci.vehicle.setSpeed(self.vehicle_id, intended_speed_in_ms)
        traci.executeMove()
        try:
            actual_speed_in_ms = traci.vehicle.getSpeed(self.vehicle_id) # in m/s
            next_state, _, _ = self.get_state()
            actual_action = self._convert_speed_to_action_50(actual_speed_in_ms) # in km/h
            if current_state[-3] > 96: # agent has passed the junction
                traci.vehicle.setSpeedMode(self.vehicle_id,32) # make the agent ignore legal and safety constraints (the junction has been passed now)
            speed_difference = abs(actual_action - action)

            self.hava.calculate_new_reputation(current_state=current_state, ddqn_action=action, actual_action=actual_action, rb_distance=speed_difference)

            next_state, _, _ = self.get_state() # get next_state with the latest weight
            experience = (next_state, action, reward, False, True)
        except traci.TraCIException as e:
            logging.info("car stopped existing half-way through the step.")
            experience = (current_state, action, 10+np.mean(self.past_speeds), True, False)
        return experience

    def next_step(self, current_state, action, simtime, starting_position=None):
        traci.simulationStep()
        self.define_first_cars_of_the_simulation(simtime, starting_position)

        vehicle_should_be_controlled_by_the_agent = self._check_vehicle_should_be_controlled()
        experience = None
        if vehicle_should_be_controlled_by_the_agent:
            self._do_only_once()
            self.agent_timestep += 1
            self._track_vehicle(self.vehicle_id)

            _, _, leaving_vehicle_ids = self.get_state()
            self._agent_passed_the_junction(leaving_vehicle_ids)

            experience = (current_state, action, 0, False, False) # vehicle exists but not to be controlled yet
            if self.norms_mode == "mix":
                experience = self._steer_vehicle_with_hava(current_state, action)
            elif self.norms_mode == "dd":
                experience = self._steer_vehicle_with_datadriven_norms_only(current_state, action)
            elif self.norms_mode=="safe":
                experience = self._steer_vehicle_with_safety_norms_only(current_state, action)
            else:
                raise TypeError("Expected to get either mix, dd or safe as norm type.")

        elif not vehicle_should_be_controlled_by_the_agent and self.done_once:
            ''' the car stopped existing in the simulator'''
            experience = (current_state, action, np.mean(self.past_speeds), True, False)
            self.past_speeds = []

        elif not vehicle_should_be_controlled_by_the_agent and not self.done_once:
            ''' vehicle hasn't spawned yet'''
            try:
                state, _, _ = self.get_state()
                experience = (state, None, None, False, False)
            except Exception as e:
                experience = (None, None, None, False, False)

        return experience

    def _agent_passed_the_junction(self, leaving_ids):
        if self._is_the_vehicle_mentioned(self.vehicle_id, leaving_ids):
            self.agent_passed = True
        return self.agent_passed

    def _check_vehicle_should_be_controlled(self):
        try:
            distance_travelled = traci.vehicle.getDistance(self.vehicle_id)
            self.distance_travelled_so_far += distance_travelled - self.distance_travelled_so_far
            return distance_travelled > 10
        except Exception as e:
            return False

    def _track_vehicle(self, vehicle_id):
        '''
        follow the vehicle
        '''
        if self.run_gui:
            traci.gui.track(objID=vehicle_id)

    def _do_only_once(self):
        '''
        When the car spawns in the simulator do these operations only once
        '''
        if not self.done_once:
            self.done_once = True

            traci.vehicle.setColor(self.vehicle_id,(247, 45, 0, 255)) # highlight the agent's car

            traci.vehicle.setAccel(self.vehicle_id, 5)
            traci.vehicle.setDecel(self.vehicle_id, 9)

            if self.norms_mode=="safe" or self.norms_mode == "mix":
                traci.vehicle.setSpeedMode(self.vehicle_id,31) # make the agent RESPECT legal and safety constraints of SUMO
            elif self.norms_mode == "dd":
                traci.vehicle.setSpeedMode(self.vehicle_id,32) # make the agent IGNORE SUMO safety / legal norms
            else:
                raise TypeError("Expected to get either mix, dd or safe as norm type.")

    def _is_the_vehicle_mentioned(self, vehicle_id, vehicle_ids):
        '''
        Checks if the agent is in a given list of vehicle_ids.

        This method is used to find out whether our agent is a leader (first incoming vehicle) or leaver (last outgoing)
        '''
        if vehicle_id in vehicle_ids:
            return True
        return False

    def get_state(self):
        state, leading_vehicle_ids, leaving_vehicle_ids = self.junction.get_leading_incoming_and_last_outgoing_vehicle_coordinates()
        # looks like this: state[["iNLx", "iNLy", "iNTx", "iNTy", "oNLx", "oNLy", "oNTx", "oNTy", "iELx", "iELy", "iETx", "iETy", "oELx", "oELy", "oETx", "oETy", "iSLx", "iSLy", "iSTx", "iSTy", "oSLx", "oSLy", "oSTx", "oSTy", "iWLx", "iWLy", "iWTx", "iWTy", "oWLx", "oWLy", "oWTx", "oWTy"]]
        state = state.loc[0, :].values.flatten().tolist()

        state.extend([float(f'{self.hava.get_last_reputation()}')]) # current reward weight (reputation) needs to be added to the state space

        state.extend([float(f'{traci.vehicle.getPosition(self.vehicle_id)[0]:.5f}')]) # agent's X
        state.extend([float(f'{traci.vehicle.getPosition(self.vehicle_id)[1]:.5f}')]) # agent's Y
        state.extend([float(f'{traci.vehicle.getSpeed(self.vehicle_id):.5f}')]) # agent's speed
        return state, leading_vehicle_ids, leaving_vehicle_ids

    def _convert_speed_to_action_50(self, speed):
        '''
        speed in m/s is assigned to one of the discrete agent's actions
        '''
        action = speed * 3.6
        if action > 50:
            ''' we manually add a maximum speed norm '''
            action = 50
        return action

    def _convert_action_to_speed(self, action):
        '''
        discrete action needs to be converted from km/h to m/s
        '''
        return (action)/3.6

    def _convert_action_to_speed_50(self, action):
        '''
        discrete action needs to be converted from km/h to m/s
        '''
        if action > 50:
            action = 50
        return (action)/3.6

    def define_first_cars_of_the_simulation(self, timestep, starting_position=None):
        ''' populates the simulator with three vehicles at timestep 1'''
        DISTANCE_MULTIPLIER = 2.5
        acc = 2.5
        if timestep == 1:
            self.set_starting_position(starting_position)
            traci.vehicle.add(vehID="iNoS", routeID="N-to-S", typeID='myveh', depart='now', departLane='first', departPos=f'{self.starting_position[0]*DISTANCE_MULTIPLIER}', departSpeed=f"{math.sqrt(2*acc*self.starting_position[0]*DISTANCE_MULTIPLIER)}")
            traci.vehicle.add(vehID="iEoS", routeID="E-to-S", typeID='myveh', depart='now', departLane='first', departPos=f'{self.starting_position[1]*DISTANCE_MULTIPLIER}', departSpeed=f"{math.sqrt(2*acc*self.starting_position[1]*DISTANCE_MULTIPLIER)}")
            traci.vehicle.add(vehID="iWoE", routeID="W-to-E", typeID='myveh', depart='now', departLane='first', departPos=f'{self.starting_position[2]*DISTANCE_MULTIPLIER}', departSpeed=f'{math.sqrt(2*acc*self.starting_position[2]*DISTANCE_MULTIPLIER)}')

    def set_starting_position(self, starting_position=None):
        '''Either you set the position yourself or it's set automatically'''
        if starting_position != None:
            self.starting_position = starting_position
        else:
            self.starting_position = next(self.starting_positions)
