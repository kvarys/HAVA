import traci
from contextlib import suppress
from constants import LANE_ID_TO_POSITION_ORIENTATION, LANE_ID_TO_SPEED_ORIENTATION, SIGNALS_TO_STRAIGHT_LEFT_RIGHT


class Vehicle:
    """Class of any vehicle in the simulator used to access vehicle information such as speed, distance to other cars, neighbouring cars etc."""
    def __init__(self, id):
        self.id = id

    def get_distance_travelled_so_far(self):
        '''
        return in meters the distance travelled
        '''
        return traci.vehicle.getDistance(self.id)

    def get_turning_signal(self):
        '''
        return turning signal - ignore breaking
        '''
        return SIGNALS_TO_STRAIGHT_LEFT_RIGHT[f"{traci.vehicle.getSignals(self.id)}"]

    def set_destination(self, destination):
        '''
        change the route of the vehicle
        '''
        traci.vehicle.setRoute(self.id, [self.get_route()[0], destination])

    def get_route(self):
        '''
        get the route planned for the vehicle
        '''
        return traci.vehicle.getRoute(self.id)

    def get_leader(self):
        '''
        returns a vehicle after the current vehicle
        '''
        l = traci.vehicle.getLeader(vehID=self.id) # defaults to None
        leader = l if l is not None else ('', -1)
        return leader

    def get_follower(self):
        '''
        returns a vehicle before the current vehicle
        '''
        follower = traci.vehicle.getFollower(vehID=self.id) # defaults to ('', -1)
        return follower

    def get_map_position(self):
        '''
        where is the car located within the map
        '''
        return traci.vehicle.getPosition(self.id)

    def get_lane_position(self):
        '''
        how far down the vahicle has travelled in the current lane
        '''
        return traci.vehicle.getLanePosition(self.id)

    def get_vehicle_neighbours(self, max_seeing_distance):
        '''
        max_seeing_distance - the max distance where we assume our vehicle can spot neighbours - partial observability
        Return a list of vehicle IDs that are around this vehicle
        '''
        # default values
        rightLeader = ('',-1)
        leftLeader = ('', 1)
        rightFollower = ('',-1)
        leftFollower = ('', -1)
        # try to get info from traci
        with suppress(IndexError): rightLeader = traci.vehicle.getRightLeaders(vehID=self.id)[0]
        with suppress(IndexError): leftLeader = traci.vehicle.getLeftLeaders(vehID=self.id)[0]
        with suppress(IndexError): rightFollower = traci.vehicle.getRightFollowers(vehID=self.id)[0]
        with suppress(IndexError): leftFollower = traci.vehicle.getLeftFollowers(vehID=self.id)[0]
        leader = self.get_leader()
        follower = self.get_follower()
        neighbours_dictionary = {
            "rightLeader": rightLeader,
            "leftLeaders":leftLeader,
            "rightFollowers": rightFollower,
            "leftFollowers": leftFollower,
            "leader": leader,
            "follower": follower
        }
        neighbours_ids = []
        for direction in list(neighbours_dictionary.keys()):
            vehicle_id = neighbours_dictionary[direction][0]
            distance_from_us = neighbours_dictionary[direction][1]
            if distance_from_us < max_seeing_distance and vehicle_id != '':
                neighbours_ids.append(vehicle_id)

        vehicles = []
        for vehicle_id in neighbours_ids:
            vehicles.append(Vehicle(vehicle_id))

        return vehicles

    def get_vehicle_speed(self):
        '''
        Return the current speed in km/h
        '''
        return traci.vehicle.getSpeed(self.id) * 3.6

    def get_vehicle_speed_without_traci(self):
        '''
        the speed that the oracle would have chosen if we didn't stop it through traci setSpeed
        '''
        return traci.vehicle.getSpeedWithoutTraCI(self.id) * 3.6

    def set_speed(self, speed):
        '''
        Control the speed of this vehicle in km/h
        '''
        # <vType id="type1" accel="0.8" decel="4.5" sigma="1" length="5" maxSpeed="150"/>
        # traci.vehicle.setSpeed(vehID=self.id, speed=speed/3.6)
        traci.vehicle.slowDown(vehID=self.id, speed=speed/3.6, duration=1)

    def get_location_orientation(self):
        '''
        returns either iSS, iES, iWS, iNS, oSS, oWS, oES, oNS
        '''
        lane_id = traci.vehicle.getLaneID(self.id)
        return LANE_ID_TO_POSITION_ORIENTATION[lane_id]

    def get_speed_orientation(self):
        '''
        returns either iSS, iES, iWS, iNS, oSS, oWS, oES, oNS
        '''
        lane_id = traci.vehicle.getLaneID(self.id)
        return LANE_ID_TO_SPEED_ORIENTATION[lane_id]

    def set_speed_mode(self):
        '''
        controls what rules the car follows
        '''
        traci.vehicle.setSpeedMode(vehID=self.id,speedMode=9)
