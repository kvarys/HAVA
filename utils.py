import numpy as np
import random
from scipy.special import rel_entr
import traci
from pandas import DataFrame
import pandas
from scipy.stats import ks_2samp, multivariate_normal
from sklearn.linear_model import LinearRegression
import os
from constants import JUNCTION_EDGE_TO_VEH_ID, ROOT_PATH, SPEEDS

def decrement_round(car_distance, decrements):
    pocket = 0
    for i, decrement in enumerate(decrements):
        car_distance -= decrement
        pocket = i
        if car_distance < 0:
            break
    return pocket

def round_ie_in(orig_distance):
    decrements = [0.3,12,12,12,12,12,7,5,5,5,3,3,2,1.2,0.8,0.5]
    return decrement_round(abs(orig_distance - 200), decrements)

def round_agent_x(orig_distance):
    decrements = [0.3,5,5,5,5,5,5,5,5,5,5,5,5,4,3,3,3,2,2,2,2,2,2,2,2,1.2,1,0.8,0.5,0.5,0.8,1.2,2,3,5,8,12,12,12,12]
    return decrement_round(orig_distance, decrements)

def round_iw_is(orig_distance):
    decrements = [0.3,12,12,12,12,12,7,5,5,5,3,3,2,1.2,0.8,0.5]
    if orig_distance == -1:
        return orig_distance
    return decrement_round(orig_distance, decrements)

def round_outgoing_ow_os(orig_distance):
    decrements = [0.5,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2]

    return decrement_round(107.2-orig_distance, decrements)

def round_outgoing_oe_on(orig_distance):
    decrements = [0.5,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2]
    return decrement_round(orig_distance-92.8, decrements)

def round_to_base(x, base=2.5):
    return base * round(x/base)

def convert_speed_to_action(speed):
    ''' speed is expected to be in km/h'''
    return round_to_base(speed, 1)

def all_speeds_to_actions(dataset: DataFrame):
    '''
    convert speeds in km/h to actions
    '''
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "oWS" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "oES" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "oSS" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "oNS" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "iWS" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "iES" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "iSS" else x)
    dataset = dataset.apply(lambda x: convert_speed_to_action(x).astype(int) if x.name == "iNS" else x)
    return dataset

def round_agents_x_and_speed(dataset:pandas.DataFrame):
    '''
    round agent's speed and X coordinate
    '''
    dataset["agentX"] = dataset["agentX"].apply(lambda x: round_agent_x(x))
    dataset["agentSpeed"] = dataset["agentSpeed"].apply(lambda x: convert_speed_to_action(x))
    return dataset

def round_state_space(dataset:pandas.DataFrame):
    '''
    rounds the state space
    '''
    BASE = 1
    dataset["oWLx"] = dataset["oWLx"].apply(lambda x: round_outgoing_ow_os(x))
    dataset["oWLy"] = dataset["oWLy"].apply(lambda x: round_to_base(x, BASE))
    dataset["oNLx"] = dataset["oNLx"].apply(lambda x: round_to_base(x, BASE))
    dataset["oNLy"] = dataset["oNLy"].apply(lambda x: round_outgoing_oe_on(x))
    dataset["oELx"] = dataset["oELx"].apply(lambda x: round_outgoing_oe_on(x))
    dataset["oELy"] = dataset["oELy"].apply(lambda x: round_to_base(x, BASE))
    dataset["oSLx"] = dataset["oSLx"].apply(lambda x: round_to_base(x, BASE))
    dataset["oSLy"] = dataset["oSLy"].apply(lambda x: round_outgoing_ow_os(x))

    dataset["iWLx"] = dataset["iWLx"].apply(lambda x: round_iw_is(x))
    dataset["iWLy"] = dataset["iWLy"].apply(lambda x: round_to_base(x, BASE))
    dataset["iNLy"] = dataset["iNLy"].apply(lambda x: round_ie_in(x))
    dataset["iNLx"] = dataset["iNLx"].apply(lambda x: round_to_base(x, BASE))
    dataset["iELx"] = dataset["iELx"].apply(lambda x: round_ie_in(x))
    dataset["iELy"] = dataset["iELy"].apply(lambda x: round_to_base(x, BASE))
    dataset["iSLy"] = dataset["iSLy"].apply(lambda x: round_iw_is(x))
    dataset["iSLx"] = dataset["iSLx"].apply(lambda x: round_to_base(x, BASE))
    return dataset

def round_goals(dataset):
    '''
        "fromCenterToEast": (200,98.4),
        "fromCenterToNorth": (101.6, 200),
        "fromCenterToSouth": (98.4, 0),
        "fromCenterToWest": (0, 101.6)
    '''
    dataset["iWTx"] = dataset["iWTx"].apply(lambda x: round_to_base(x, 1))
    dataset["iNTx"] = dataset["iNTx"].apply(lambda x: round_to_base(x, 1))
    dataset["iETx"] = dataset["iETx"].apply(lambda x: round_to_base(x, 1))
    dataset["iSTx"] = dataset["iSTx"].apply(lambda x: round_to_base(x, 1))

    dataset[f'iNGoal']=0
    dataset[f'iEGoal']=0
    dataset[f'iWGoal']=0
    dataset[f'iSGoal']=0
    dataset.loc[dataset['iWTx'] == 200, f'iWGoal'] = 1 # going east
    dataset.loc[dataset['iWTx'] == 102, f'iWGoal'] = 2 # going north
    dataset.loc[dataset['iWTx'] == 98,  f'iWGoal'] = 3 # going south
    dataset.loc[dataset['iWTx'] == 0,   f'iWGoal'] = 4 # going west impossible

    dataset.loc[dataset['iNTx'] == 200, f'iNGoal'] = 1 # going east
    dataset.loc[dataset['iNTx'] == 102, f'iNGoal'] = 2 # going north
    dataset.loc[dataset['iNTx'] == 98,  f'iNGoal'] = 3 # going south
    dataset.loc[dataset['iNTx'] == 0,   f'iNGoal'] = 4 # going west

    dataset.loc[dataset['iETx'] == 200, f'iEGoal'] = 1
    dataset.loc[dataset['iETx'] == 102, f'iEGoal'] = 2 # going north
    dataset.loc[dataset['iETx'] == 98,  f'iEGoal'] = 3 # going south
    dataset.loc[dataset['iETx'] == 0,   f'iEGoal'] = 4 # going west

    dataset.loc[dataset['iSTx'] == 200, f'iSGoal'] = 1
    dataset.loc[dataset['iSTx'] == 102, f'iSGoal'] = 2 # going north
    dataset.loc[dataset['iSTx'] == 98,  f'iSGoal'] = 3
    dataset.loc[dataset['iSTx'] == 0,   f'iSGoal'] = 4

    dataset = dataset.drop(["iSTx", "iSTy", "iNTx", "iNTy", "iETx", "iETy", "iWTx", "iWTy", "oNTx", "oNTy", "oWTx", "oWTy", "oETx", "oETy", "oSTx", "oSTy"], axis=1)
    return dataset

def drop_never_changing_columns(dataset:pandas.DataFrame):
    dataset = dataset.drop(["iNLx", "oNLx", "iELy", "oELy", "iWLy", "oWLy", "iSLx", "oSLx"], axis=1)
    return dataset

def turn_iw_to_negative_if_leader(dataset:pandas.DataFrame):
    def to_two_decimal_places(num):
        return float(f'{num:.2f}')

    dataset["iWLx"] = dataset["iWLx"].apply(lambda x: round_to_base(x,1))

    dataset['iWS'] = np.where((round_to_base(dataset['agentX'], 1) == dataset['iWLx']), -1, dataset['iWS'])
    dataset['iWLx'] = np.where((round_to_base(dataset['agentX'], 1) == dataset['iWLx']), -1, dataset['iWLx'])
    return dataset

def remove_low_occuring_values(dataset:pandas.DataFrame):
    for x in range(0, 39):
        frequencies = dataset.loc[dataset['agentX'] == x]['agentSpeed'].value_counts(sort=True).to_dict()
        total_sum = sum(frequencies.values())
        speeds_to_delete = []
        for speed in list(frequencies.keys()):
            num_of_observations = frequencies[speed]
            portion = num_of_observations / total_sum
            if portion < 0.03:
                speeds_to_delete.append(speed)
        for speed_to_delete in speeds_to_delete:
            dataset = dataset[((dataset['agentX'] == x) & (dataset['agentSpeed'] != speed_to_delete)) | (dataset['agentX'] != x)]
    return dataset


def print_df(df):
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def find_csv_in_folder(folder):
    '''
    find all csv files in a folder
    '''
    all_files = os.listdir(folder)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    return csv_files

def record_csv_dataset(name, values):
    metrics = os.path.join("results", "{}.csv".format(name))
    with open(metrics, "a") as file:
        for row in values:
            for value in row:
                file.write(", {}".format(value))
            file.write("\n")

def get_election_winner(votes):
    '''
    votes is a dictionary
    return the key with the highest number of votes
    '''
    winner = ''
    count = 0
    for key in list(votes.keys()):
        if votes[key] > count:
            winner = key
            count = votes[key]
    return winner

def get_speeds_for_vehicles(vehicles: list):
    '''
    return speeds of chosen vehicles
    '''
    speeds = []
    for vehicle in vehicles:
        speeds.append(vehicle.get_vehicle_speed())
    return speeds

def calculate_mean(dataset:DataFrame ):
    dataset = dataset.drop(['e_speed','e_location','s_speed','s_location'], axis=1)
    mean = dataset.mean(axis=0)
    return mean

def calculate_correlation(dataset:DataFrame ):
    dataset = dataset.drop(['e_speed','e_location','s_speed','s_location'], axis=1)
    return dataset.corr(method="pearson")

def create_multivariate_normal_distribution(dataset:DataFrame):
    '''
    creates a multivariate normal RV from mean vector, covariance matrix
    '''
    rv = multivariate_normal(
            mean=calculate_mean(dataset),
            cov=calculate_correlation(dataset),
            allow_singular=False)
    return rv

def linear_regression(data, target):
    return LinearRegression().fit(data,target)

def round_to_ten(x, base=10):
    return base * round(x/base)

def round_to_base(x, base):
    return base * round(x/base)

def round_to_five(x, base=5):
    return base * round(x/base)

def print_lane_id_for_vehicle_id(vehicle_id):
    try:
        return traci.vehicle.getLaneID(vehicle_id)
    except Exception as e:
        return "leader eneexistuje"

def get_lane_length(lane_id):
    '''
    get length of a lane
    '''
    return traci.lane.getLength(lane_id)

GOAL_TO_COORDINATES = {
        "fromCenterToEast": (200,98.4),
        "fromCenterToNorth": (101.6, 200),
        "fromCenterToSouth": (98.4, 0),
        "fromCenterToWest": (0, 101.6)
}

INCOMING_DIRECTION_TO_COORDINATES = {
        "north": (98.4,200),
        "east": (200, 101.6),
        "south": (101.6, 0),
        "west": (0, 98.4),
}

OUTGOING_DIRECTION_TO_GOAL = {
        "west": GOAL_TO_COORDINATES["fromCenterToWest"],
        "south": GOAL_TO_COORDINATES["fromCenterToSouth"],
        "north": GOAL_TO_COORDINATES["fromCenterToNorth"],
        "east": GOAL_TO_COORDINATES["fromCenterToEast"],
}

INCOMING_DIRECTION_TO_GOAL = {
        "north": GOAL_TO_COORDINATES["fromCenterToSouth"],
        "south": GOAL_TO_COORDINATES["fromCenterToNorth"],
        "east": GOAL_TO_COORDINATES["fromCenterToWest"],
        "west": GOAL_TO_COORDINATES["fromCenterToEast"],
}

def get_speed_coordinates_from_vehicle_or_return_default_without_rounding(vehicle, direction, leaving=False):
    data = []
    try:
        goal_x = GOAL_TO_COORDINATES[vehicle.get_route()[1]][0]
        goal_y = GOAL_TO_COORDINATES[vehicle.get_route()[1]][1]
        speed = float(f'{vehicle.get_vehicle_speed():.5f}')
        coor_x = float(f'{traci.vehicle.getPosition(vehicle.id)[0]:.5f}')
        coor_y = float(f'{traci.vehicle.getPosition(vehicle.id)[1]:.5f}')
        data.append(speed)
        data.append(coor_x)
        data.append(coor_y)
        data.append(goal_x)
        data.append(goal_y)
    except Exception as e:
        data.append(0) # default speed
        if not leaving:
            data.append(INCOMING_DIRECTION_TO_COORDINATES[direction][0]) # default coor_x
            data.append(INCOMING_DIRECTION_TO_COORDINATES[direction][1]) # default coor_y
            data.append(INCOMING_DIRECTION_TO_GOAL[direction][0]) # default goal x
            data.append(INCOMING_DIRECTION_TO_GOAL[direction][1]) # default goal y
        else:
            data.append(OUTGOING_DIRECTION_TO_GOAL[direction][0]) # default coor_x
            data.append(OUTGOING_DIRECTION_TO_GOAL[direction][1]) # default coor_y
            data.append(OUTGOING_DIRECTION_TO_GOAL[direction][0]) # default goal x
            data.append(OUTGOING_DIRECTION_TO_GOAL[direction][1]) # default goal y
    return data
def get_speed_location_from_vehicle_or_return_default_without_rounding(vehicle, default_signal=0, default_speed=0, default_location=0, add_location=0):
    '''
    try/catch:
    try: if the vehicle exists return its roundedup speed and location. Optionally you can add add_location to location if you want to offset it (useful for outgoing vehicles).
    catch: if the vehicle does not exist return default values
    '''
    data = []
    try: # f'{pi:.2f}'
        speed = float(f'{vehicle.get_vehicle_speed():.5f}')
        location = float(f'{vehicle.get_lane_position() + add_location:.5f}')
        turning_signal = int(vehicle.get_turning_signal())
        data.append(speed)
        data.append(location)
        data.append(turning_signal)
    except Exception as e:
        data.append(default_speed)
        data.append(default_location)
        data.append(default_signal)
    return data

def get_speed_location_from_vehicle_or_return_default(vehicle, default_signal=0, default_speed=0, default_location=0, add_location=0):
    '''
    try/catch:
    try: if the vehicle exists return its roundedup speed and location. Optionally you can add add_location to location if you want to offset it (useful for outgoing vehicles).
    catch: if the vehicle does not exist return default values
    '''
    data = []
    try:
        speed = round_to_five(vehicle.get_vehicle_speed())
        location = round_to_ten(vehicle.get_lane_position() + add_location)
        turning_signal = int(vehicle.get_turning_signal())
        data.append(speed)
        data.append(location)
        data.append(turning_signal)
    except Exception as e:
        data.append(default_speed)
        data.append(default_location)
        data.append(default_signal)
    return data

def make_probs_from_pdf_for_all_speeds(pdf):
    probs = np.array([0.0]*len(SPEEDS))
    for speed in SPEEDS:
        probs[speed] = pdf(speed)[0]
    return probs / probs.sum() # normalize the array to sum to 1

def KS_test_with_PDFs(pdf_1, pdf_2):
    pvalue_votes = 0
    for _ in range(0,10):
        data1 = pdf_1.resample(size=10)
        data2 = pdf_2.resample(size=10)
        if ks_2samp(data1[0], data2[0]).pvalue > 0.05:
            pvalue_votes += 1
        else:
            pvalue_votes -= 1

    if pvalue_votes > 0:
        return 1
    return 0

def calculate_distance_between_two_pdfs(pdf_1, pdf_2):
    probs1 = make_probs_from_pdf_for_all_speeds(pdf_1)
    probs2 = make_probs_from_pdf_for_all_speeds(pdf_2)

    kl_sum = sum(rel_entr(probs1, probs2))
    return kl_sum

def KS_test(dataset_1, dataset_2):
    return ks_2samp(dataset_1, dataset_2).pvalue

def remove_files(file_names: list[str]):
    for file_name in file_names:
        os.remove("{}results/continual_learning/{}".format(ROOT_PATH, file_name))

def pick_vehicle_at_edge(junction_name, edge):
    '''
    The agent arrives at a junction from a certain direction and we want it to posses a car travelling in the same direction.
    '''
    return f"{random.choice(JUNCTION_EDGE_TO_VEH_ID[(junction_name, edge)])}"
