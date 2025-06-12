from vehicle import Vehicle
from utils import *
import traci


class Junction:
    """Junction without traffic lights"""
    def __init__(self):
        self.junction_id="centerNode",
        self.inner_lanes_IDs = [[":centerNode_9_0", ":centerNode_4_0", ":centerNode_14_0"], [":centerNode_13_0", ":centerNode_16_0", ":centerNode_8_0", ":centerNode_2_0"], [":centerNode_12_0", ":centerNode_6_0", ":centerNode_1_0"], [":centerNode_0_0", ":centerNode_10_0", ":centerNode_5_0", ":centerNode_18_0"]]
        self.incoming_lanes_IDs = ["toCenterFromNorth_0", "toCenterFromEast_0", "toCenterFromSouth_0", "toCenterFromWest_0"]
        self.outgoing_lanes_IDs = ["fromCenterToNorth_0", "fromCenterToEast_0", "fromCenterToSouth_0", "fromCenterToWest_0"]
        self.directions = ["north", "east", "south", "west"]

    def _extract_vehicle_id_or_none(self, vehicle):
        if vehicle is None:
            return None
        return vehicle.id

    def get_leading_incoming_and_last_outgoing_vehicle_coordinates_no_normalization(self):
        data = []
        leading_vehicle_ids = []
        leaving_vehicle_ids = []
        for direction, in_lane_ID, inner_lane_IDs, out_lane_ID in zip(self.directions, self.incoming_lanes_IDs, self.inner_lanes_IDs, self.outgoing_lanes_IDs):
            leading_vehicle: Vehicle | None = self.get_leading_incoming_vehicle_for_lane(in_lane_ID)
            last_vehicle = None
            for inner_lane_ID in inner_lane_IDs:
                if last_vehicle is None:
                    last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(inner_lane_ID)
            if last_vehicle is None:
                last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(out_lane_ID)

            leader_data = get_speed_coordinates_from_vehicle_or_return_default_without_rounding(
                    leading_vehicle,
                    direction)

            follower_data = get_speed_coordinates_from_vehicle_or_return_default_without_rounding(
                    last_vehicle,
                    direction, leaving=True)

            data.extend(leader_data)
            data.extend(follower_data)
            leading_vehicle_ids.append(self._extract_vehicle_id_or_none(leading_vehicle))
            leaving_vehicle_ids.append(self._extract_vehicle_id_or_none(last_vehicle))

        dataset = DataFrame([data],columns=["iNS","iNLx","iNLy","iNTx","iNTy","oNS","oNLx", "oNLy", "oNTx", "oNTy","iES","iELx", "iELy", "iETx", "iETy","oES","oELx", "oELy", "oETx", "oETy","iSS","iSLx", "iSLy", "iSTx", "iSTy","oSS","oSLx", "oSLy", "oSTx", "oSTy","iWS","iWLx", "iWLy", "iWTx", "iWTy","oWS","oWLx", "oWLy", "oWTx", "oWTy"])

        return dataset, leading_vehicle_ids, leaving_vehicle_ids

    def get_leading_incoming_and_last_outgoing_vehicle_coordinates(self):
        data = []
        leading_vehicle_ids = []
        leaving_vehicle_ids = []
        for direction, in_lane_ID, inner_lane_IDs, out_lane_ID in zip(self.directions, self.incoming_lanes_IDs, self.inner_lanes_IDs, self.outgoing_lanes_IDs):
            leading_vehicle: Vehicle | None = self.get_leading_incoming_vehicle_for_lane(in_lane_ID)
            last_vehicle = None
            for inner_lane_ID in inner_lane_IDs:
                if last_vehicle is None:
                    last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(inner_lane_ID)
            if last_vehicle is None:
                last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(out_lane_ID)

            leader_data = get_speed_coordinates_from_vehicle_or_return_default_without_rounding(
                    leading_vehicle,
                    direction)

            follower_data = get_speed_coordinates_from_vehicle_or_return_default_without_rounding(
                    last_vehicle,
                    direction, leaving=True)

            data.extend(leader_data)
            data.extend(follower_data)
            leading_vehicle_ids.append(self._extract_vehicle_id_or_none(leading_vehicle))
            leaving_vehicle_ids.append(self._extract_vehicle_id_or_none(last_vehicle))

        dataset = DataFrame([data],columns=["iNS","iNLx","iNLy","iNTx","iNTy","oNS","oNLx", "oNLy", "oNTx", "oNTy","iES","iELx", "iELy", "iETx", "iETy","oES","oELx", "oELy", "oETx", "oETy","iSS","iSLx", "iSLy", "iSTx", "iSTy","oSS","oSLx", "oSLy", "oSTx", "oSTy","iWS","iWLx", "iWLy", "iWTx", "iWTy","oWS","oWLx", "oWLy", "oWTx", "oWTy"])

        return dataset, leading_vehicle_ids, leaving_vehicle_ids

    def _normalize(self, dataset: pandas.DataFrame):
        dataset['oNS'] = dataset['oNS'].div(50)
        dataset['oWS'] = dataset['oWS'].div(50)
        dataset['oES'] = dataset['oES'].div(50)
        dataset['oSS'] = dataset['oSS'].div(50)

        dataset['iNS'] = dataset['iNS'].div(50)
        dataset['iWS'] = dataset['iWS'].div(50)
        dataset['iES'] = dataset['iES'].div(50)
        dataset['iSS'] = dataset['iSS'].div(50)

        dataset['oWLx'] = dataset['oWLx'].div(200)
        dataset['oWLy'] = dataset['oWLy'].div(200)
        dataset['oWTx'] = dataset['oWTx'].div(200)
        dataset['oWTy'] = dataset['oWTy'].div(200)

        dataset['oNLx'] = dataset['oNLx'].div(200)
        dataset['oNLy'] = dataset['oNLy'].div(200)
        dataset['oNTx'] = dataset['oNTx'].div(200)
        dataset['oNTy'] = dataset['oNTy'].div(200)

        dataset['oSLx'] = dataset['oSLx'].div(200)
        dataset['oSLy'] = dataset['oSLy'].div(200)
        dataset['oSTx'] = dataset['oSTx'].div(200)
        dataset['oSTy'] = dataset['oSTy'].div(200)

        dataset['oELx'] = dataset['oELx'].div(200)
        dataset['oELy'] = dataset['oELy'].div(200)
        dataset['oETx'] = dataset['oETx'].div(200)
        dataset['oETy'] = dataset['oETy'].div(200)

        dataset['iWLx'] = dataset['iWLx'].div(200)
        dataset['iWLy'] = dataset['iWLy'].div(200)
        dataset['iWTx'] = dataset['iWTx'].div(200)
        dataset['iWTy'] = dataset['iWTy'].div(200)

        dataset['iWLx'] = dataset['iWLx'].div(200)
        dataset['iWLy'] = dataset['iWLy'].div(200)
        dataset['iWTx'] = dataset['iWTx'].div(200)
        dataset['iWTy'] = dataset['iWTy'].div(200)

        dataset['iNLx'] = dataset['iNLx'].div(200)
        dataset['iNLy'] = dataset['iNLy'].div(200)
        dataset['iNTx'] = dataset['iNTx'].div(200)
        dataset['iNTy'] = dataset['iNTy'].div(200)

        dataset['iSLx'] = dataset['iSLx'].div(200)
        dataset['iSLy'] = dataset['iSLy'].div(200)
        dataset['iSTx'] = dataset['iSTx'].div(200)
        dataset['iSTy'] = dataset['iSTy'].div(200)

        dataset['iELx'] = dataset['iELx'].div(200)
        dataset['iELy'] = dataset['iELy'].div(200)
        dataset['iETx'] = dataset['iETx'].div(200)
        dataset['iETy'] = dataset['iETy'].div(200)

        dataset['iWLx'] = dataset['iWLx'].div(200)
        dataset['iWLy'] = dataset['iWLy'].div(200)
        dataset['iWTx'] = dataset['iWTx'].div(200)
        dataset['iWTy'] = dataset['iWTy'].div(200)
        return dataset

    def get_leading_incoming_and_last_outgoing_vehicle_without_rounding_2(self):
        '''
        This method is the same as: get_leading_incoming_and_last_outgoing_vehicle but it doesnt round up the measurements
        Look at the first incoming / last outgoing vehicle of each lane of this junction.
        Return the cars speed / location or some default values.
        '''
        add_location = 0
        data = []
        leading_vehicle_ids = []
        leaving_vehicle_ids = []
        for in_lane_ID, inner_lane_IDs, out_lane_ID in zip(self.incoming_lanes_IDs, self.inner_lanes_IDs, self.outgoing_lanes_IDs):
            leading_vehicle: Vehicle | None = self.get_leading_incoming_vehicle_for_lane(in_lane_ID)
            last_vehicle = None
            for inner_lane_ID in inner_lane_IDs:
                if last_vehicle is None:
                    last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(inner_lane_ID)
            if last_vehicle is None:
                last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(out_lane_ID)
                if last_vehicle is not None:
                    add_location = get_lane_length(inner_lane_IDs[0])

            leader_data = get_speed_location_from_vehicle_or_return_default_without_rounding(
                    leading_vehicle,
                    default_speed=0,
                    default_location=0)

            follower_data = get_speed_location_from_vehicle_or_return_default_without_rounding(
                    last_vehicle,
                    default_speed=0,
                    default_location=150,
                    add_location=add_location)

            data.extend(leader_data)
            data.extend(follower_data)
            leading_vehicle_ids.append(self._extract_vehicle_id_or_none(leading_vehicle))
            leaving_vehicle_ids.append(self._extract_vehicle_id_or_none(last_vehicle))
        add_location = 0

        dataset = DataFrame([data],columns=["iNS","iNL","iNT","oNS","oNL", "oNT","iES","iEL", "iET","oES","oEL", "oET","iSS","iSL", "iST","oSS","oSL", "oST","iWS","iWL", "iWT","oWS","oWL", "oWT"])

        dataset = dataset.loc[0, :].values.flatten().tolist()
        return dataset, leading_vehicle_ids, leaving_vehicle_ids

    def get_leading_incoming_and_last_outgoing_vehicle_without_rounding(self):
        '''
        This method is the same as: get_leading_incoming_and_last_outgoing_vehicle but it doesnt round up the measurements
        Look at the first incoming / last outgoing vehicle of each lane of this junction.
        Return the cars speed / location or some default values.
        '''
        add_location = 0
        data = []
        leading_vehicle_ids = []
        leaving_vehicle_ids = []
        for in_lane_ID, inner_lane_IDs, out_lane_ID in zip(self.incoming_lanes_IDs, self.inner_lanes_IDs, self.outgoing_lanes_IDs):
            leading_vehicle: Vehicle | None = self.get_leading_incoming_vehicle_for_lane(in_lane_ID)
            last_vehicle = None
            for inner_lane_ID in inner_lane_IDs:
                if last_vehicle is None:
                    last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(inner_lane_ID)
            if last_vehicle is None:
                last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(out_lane_ID)
                if last_vehicle is not None:
                    add_location = get_lane_length(inner_lane_IDs[0])

            leader_data = get_speed_location_from_vehicle_or_return_default_without_rounding(
                    leading_vehicle,
                    default_speed=0,
                    default_location=0)

            follower_data = get_speed_location_from_vehicle_or_return_default_without_rounding(
                    last_vehicle,
                    default_speed=0,
                    default_location=150,
                    add_location=add_location)

            data.extend(leader_data)
            data.extend(follower_data)
            leading_vehicle_ids.append(self._extract_vehicle_id_or_none(leading_vehicle))
            leaving_vehicle_ids.append(self._extract_vehicle_id_or_none(last_vehicle))
        add_location = 0

        dataset = DataFrame([data],columns=["iNS","iNL","iNT","oNS","oNL", "oNT","iES","iEL", "iET","oES","oEL", "oET","iSS","iSL", "iST","oSS","oSL", "oST","iWS","iWL", "iWT","oWS","oWL", "oWT"])

        context = dataset[['iNL', 'iNT', 'oNL', 'oNT', 'iEL', 'iET', 'oEL', 'oET', 'iSL', 'iST', 'oSL', 'oST', 'iWL', 'iWT', 'oWL', 'oWT']]
        context = context.loc[0, :].values.flatten().tolist()
        return context, leading_vehicle_ids, leaving_vehicle_ids

    def get_leading_incoming_and_last_outgoing_vehicle(self):
        '''
        Look at the first incoming / last outgoing vehicle of each lane of this junction.
        Return the cars speed / location or some default values.
        '''
        add_location = 0
        data = []
        leading_vehicle_ids = []
        leaving_vehicle_ids = []
        for in_lane_ID, inner_lane_IDs, out_lane_ID in zip(self.incoming_lanes_IDs, self.inner_lanes_IDs, self.outgoing_lanes_IDs):
            leading_vehicle: Vehicle | None = self.get_leading_incoming_vehicle_for_lane(in_lane_ID)
            last_vehicle = None
            for inner_lane_ID in inner_lane_IDs:
                if last_vehicle is None:
                    last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(inner_lane_ID)
            if last_vehicle is None:
                last_vehicle: Vehicle | None = self.get_last_outgoing_vehicle_for_lane(out_lane_ID)
                if last_vehicle is not None:
                    add_location = get_lane_length(inner_lane_IDs[0])

            leader_data = get_speed_location_from_vehicle_or_return_default(
                    leading_vehicle,
                    default_speed=0,
                    default_location=0)

            follower_data = get_speed_location_from_vehicle_or_return_default(
                    last_vehicle,
                    default_speed=0,
                    default_location=150,
                    add_location=add_location)

            data.extend(leader_data)
            data.extend(follower_data)
            leading_vehicle_ids.append(self._extract_vehicle_id_or_none(leading_vehicle))
            leaving_vehicle_ids.append(self._extract_vehicle_id_or_none(last_vehicle))
        add_location = 0

        data = DataFrame([data],columns=["iNS","iNL","iNT","oNS","oNL", "oNT","iES","iEL", "iET","oES","oEL", "oET","iSS","iSL", "iST","oSS","oSL", "oST","iWS","iWL", "iWT","oWS","oWL", "oWT"])

        context = data[['iNL', 'iNT', 'oNL', 'oNT', 'iEL', 'iET', 'oEL', 'oET', 'iSL', 'iST', 'oSL', 'oST', 'iWL', 'iWT', 'oWL', 'oWT']]
        speeds = data[['iNS', 'oNS', 'iES', 'oES', 'iSS', 'oSS', 'iWS', 'oWS']]
        return context, speeds, leading_vehicle_ids, leaving_vehicle_ids

    def get_leading_incoming_vehicle_for_lane(self, lane_id):
        '''
        return the first incoming vehicle for a specified lane
        '''
        vehicles_IDs = traci.lane.getLastStepVehicleIDs(lane_id)
        leading_vehicle = None
        for vehicle_ID in vehicles_IDs:
            leader = Vehicle(vehicle_ID).get_leader()[0]
            if not leader in vehicles_IDs:
                ''' vehicle is the leader on this lane '''
                leading_vehicle = Vehicle(vehicle_ID)
        return leading_vehicle

    def get_last_outgoing_vehicle_for_lane(self, lane_id):
        '''
        return the last outgoing vehicle for a specified lane
        '''
        try:
            vehicles_IDs = traci.lane.getLastStepVehicleIDs(lane_id)

            last_vehicle = None
            for vehicle_ID in vehicles_IDs:
                follower = Vehicle(vehicle_ID).get_follower()[0]
                if not follower in vehicles_IDs:
                    ''' this is the last vehicle on this lane '''
                    last_vehicle = Vehicle(vehicle_ID)
            return last_vehicle
        except traci.TraCIException as e:
            logging.warning(f"lane with ID {lane_id} does not exist")
            return None
