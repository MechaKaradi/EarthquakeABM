from __future__ import annotations

from collections import defaultdict
from pyclbr import Class
import types
from typing import Callable, Dict, List, Tuple, Union, Any, Set

from mesa import Agent, Model
import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from functools import partial
import itertools
from operator import attrgetter

from enum import Enum
import pickle
import networkx as nx
import networkx.algorithms as nxalg
import pandas as pd
import matplotlib.pyplot as plt
import random

from agents import *
from UtilClasses import *


# Pure Utility functions
def count_agents(self: MinimalModel):
    return len(self.schedule.agents)


class Dispatcher:
    """handles the availability of information to the Responder agents
    A special role in the simulation that is not an agent.
    Dispatcher is not modeled as an agents because it is not necessarily analogous to an institution or organization.
    It may handle information retrieval in cases where the agents are acting autonomously
    In all cases, the dispatcher is not the locus at which different policies are modeled.
    """
    _assigned: Set[str]
    "A set of unique Ids of citizens that have been assigned to a responder"
    _calls_queue: List[str]
    "A list of unique Ids of citizens that have called for help"
    model: MinimalModel
    "The calling model"
    size: int
    "the capacity of the dispatcher to handle calls"

    def __init__(self, model, dispatch_size=10):
        """

        Parameters
        ----------
        dispatch_size : int
        model: MinimalModel
            the model in which the Dispatcher is called
        """
        self.model = model
        self.size = dispatch_size
        self._calls_queue = list()
        self._assigned = set()

    def injured_citizens(self):
        # list expression that creates a shuffled list of citizen unique Ids and returns them 1 by one

        injured_citizens = []
        for citizen in self.model.schedule.agents_by_type["Citizen"].values():
            if citizen.is_injured:
                if citizen.health > 0:
                    if isinstance(citizen._pos, Hospital):
                        continue
                    injured_citizens.append(citizen.unique_id)
        return injured_citizens

    def update_calls_queue(self):
        # adds random injured citizens to the calls queue
        calls = set()
        # create a set from injured citizens and then subtract the set of assigned citizens
        calls = set(self.injured_citizens()) - self._assigned
        calls = calls - set(self._calls_queue)
        # convert the set to a list and shuffle it and then get the first self.size elements
        calls = self.model.random.sample(list(calls), min(self.size, len(calls)))
        # add the calls to the queue
        self._calls_queue.extend(calls)

    def get_call(self):
        """fetches the next call in the queue
        Called by a Responder agent to get the next call in the queue
        Returns
        -------

        """
        # returns the next call in the queue
        if len(self._calls_queue) > 0:
            call: str = self._calls_queue.pop(0)
            self._assigned.add(call)
            return call
        else:
            return None


class MinimalModel(Model):
    """ Minimal Model
    The basic model of earthquake response, consisting of Citizens that get injured, and Ambulances that rush to
    pick them up in response.

    """
    dispatcher: Dispatcher
    MINIMUM_RESIDENCY: int
    """a percentage, the minimum proportion of each building which is occupied at the time of spawn"""

    G: nx.Graph
    """the street network"""

    EARTHQUAKE_EVENTS: Dict[int, float]
    """a dictionary of earthquake events. the first number is the step of the event, the second is the magnitude"""

    def __init__(self,
                 num_buildings: int,
                 num_citizens: int,
                 num_hospitals: int,
                 num_ambulances: int,
                 dispatch_size: int,
                 **kwargs
                 ):

        # Parameters
        self.num_buildings = num_buildings
        self.num_citizens = num_citizens
        self.num_hospitals = num_hospitals
        self.num_ambulances = num_ambulances

        self.dispatch_size = dispatch_size

        if 'EARTHQUAKE_EVENTS' in kwargs:
            self.EARTHQUAKE_EVENTS = kwargs['EARTHQUAKE_EVENTS']
        else:
            self.EARTHQUAKE_EVENTS: Dict[int, float] = {
                1: 8.0,  # initial earthquake
                # 600: 7.0,  # aftershock 1
                # 1200: 6.0,  # aftershock 2
            }

        if 'MINIMUM_RESIDENCY' in kwargs:
            self.MINIMUM_RESIDENCY = kwargs['MINIMUM_RESIDENCY']
        else:
            self.MINIMUM_RESIDENCY = 50  # minimum percentage of building capacity which is occupied

        # Create time
        self.schedule = StagedAndTypedTime(self)

        # Create Space
        with open('.\street_network.data', 'rb') as file:
            self.G = pickle.load(file)

        self.grid = SpatialNetwork(self.G)

        # Create Dispatcher

        self.dispatcher = Dispatcher(self, dispatch_size=self.dispatch_size)

        # Create Agents
        # Todo: Bring in the create agents logic and and parameters for numbers of agents to be created.
        # ToDo: Create Ambulances and assign them to the hospital?

        # Create Data Collector
        model_metrics = {
            "Number of Agents": count_agents
        }
        agent_metrics = {
            'Citizen': {
                "Citizen Coordinates": "position",
                "Citizen Position": "position_node",
                "Health": "health"
            },
            'Building': {
                "Building Position": "position_node",
                "Damage": "state",
            },
            'Ambulance': {
                "Ambulance Position": "position_node",
                "Ambulance Status": "status",
                "Patient": lambda a: a.patient.unique_id if a.patient is not None else None,
            },
        }
        self.datacollector = ExtendedDataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
        self.datacollector.collect(self)

        self.running = True

        # Initialise the Agents
        self.initialize_agents(num_buildings, num_citizens, num_hospitals, num_ambulances)

    # Create a closure to create agents with different classes but sequential unique ids
    def create_agents(self, agent_type):
        agent_type_str = str(agent_type.__name__)
        # set agent_id to an integer representation of the agent_type
        agent_id = 0
        agent_type_created = agent_type
        model = self

        def _create_agent(location, **kwargs):
            """ Initialize and agent and add it to the model schedule?

            Parameters
            ----------
            location : int | Agent , passed the agents spawn method

            Returns
            -------
            object: the agent object created by the method
            """
            nonlocal agent_id
            nonlocal model
            nonlocal agent_type_str
            nonlocal agent_type_created
            unique_id = f"{agent_type_str}_{agent_id}"
            a: MinimalAgent = agent_type_created(unique_id, model, **kwargs)

            model.schedule.add(a)

            agent_id += 1

            # place the agent on the grid
            a.spawn(location=location)

            return a

        return _create_agent

    def buildings_to_nodes(self, number_of_buildings):
        create_building = self.create_agents(Building)
        i = 0
        while i < number_of_buildings:
            node_id = self.random.choice(list(self.G.nodes))
            # num = self.random.randint(0, 5)
            num = 1
            for j in range(num):
                create_building(location=node_id)
            i += num
        return f'Created: {i} buildings'

    def hospitals_to_nodes(self, number_of_hospitals):
        create_hospital = self.create_agents(Hospital)
        i = 0
        while i < number_of_hospitals:
            node_id = self.random.choice(list(self.G.nodes))
            create_hospital(location=node_id)
            i += 1
        # create a representation of the voronoi regions for the hospitals
        self.assign_nearest_hospital_node()
        self.schedule.trigger_agent_action_by_type("Hospital", "get_own_voronoi_cell")

        return f'Created: {i} hospitals'

    def citizens_to_buildings(self, number_of_citizens: int):
        create_citizen = self.create_agents(Citizen)
        num = 0
        building_iterator = list(self.schedule.agents_by_type["Building"].values())
        self.random.shuffle(building_iterator)
        b = 0
        while num <= number_of_citizens:
            if b >= len(building_iterator):
                raise Exception("Not enough buildings to create citizens")
            building = building_iterator[b]
            i = len(building.residents)
            building_residents_number = round(
                building.capacity * self.random.randint(self.MINIMUM_RESIDENCY, 100) / 100, None)
            while i <= building_residents_number:
                create_citizen(location=building)
                i += 1
            b += 1
            num += i
        return f'Created: {num} citizens in {b} buildings'

    def ambulances_to_hospital(self, ambulances_per_hospital: int | Callable = 1):
        create_ambulance = self.create_agents(Ambulance)
        i = 0
        hospitals_list: List[Hospital] = list(self.schedule.agents_by_type["Hospital"].values())
        for hospital in hospitals_list:
            hospital.ambulances = list()
            ambulances_num = ambulances_per_hospital
            location_list = self.random.choices(list(hospital.service_area), k=ambulances_num)
            for location in location_list:
                hospital.ambulances.append(create_ambulance(location=location))
                create_ambulance(location=location)
                i += 1
        return f'Created: {i} ambulances'

    def assign_nearest_hospital_node(self) -> None:
        """Assign each node the nearest hospital as an attribute"""
        hospital_nodes = [(lambda x: x._pos)(x) for x in (self.schedule.agents_by_type["Hospital"].values())]
        self.hospital_voronoi = nxalg.voronoi_cells(self.G, set(hospital_nodes), weight='Length')

        for hospital in self.hospital_voronoi:
            hospital_color = self.random.choice(['red', 'blue', 'green', 'yellow', 'orange', 'purple'])
            for node in self.hospital_voronoi[hospital]:
                self.G.nodes[node]['nearest_hospital'] = hospital
                self.G.nodes[node]['nearest_hospital_color'] = hospital_color

    def earthquake(self, magnitude: float):
        """Simulate an earthquake event

        Parameters
        ----------
        magnitude : float
            The magnitude of the earthquake
        """
        self.schedule.trigger_agent_action_by_type("Building", "earthquake", magnitude=magnitude)

    # a generator to return the collapsed buildings
    def get_collapsed_building(self):
        for building in self.schedule.agents_by_type["Building"].values():
            if building.state == 3:
                yield building

    def step(self):
        # Phase 1
        """External Events:
        Check if this is an "Earthquake" event. If so, then call the "earthquake" method.
        """
        if self.schedule.steps in self.EARTHQUAKE_EVENTS:
            self.earthquake(magnitude=self.EARTHQUAKE_EVENTS[self.schedule.steps])
            """
            self.earthquake calls the earthquake method of each building 
            the building method handles applying the damage to the occupants of the buildings
            """

        # Phase 2
        """Internal Events:
        tick down the health of injured citizens not in a hospital
            #TODO: apply a heal method to Citizens for hospital, ambulance, and doctor team
            Current status: Being in Hospital will pause death, but not heal
        heal injured citizens in a hospital
            #TODO: Create Heal method in hospital/citizen
        """
        self.schedule.trigger_agent_action_by_type("Citizen", "tick_health")

        # Phase 3
        """Dispatcher observes situation and collects information
        - Get list of collapsed buildings
        ? - Get some subset of injured citizens
        """
        self.dispatcher.update_calls_queue()

        # Phase 4
        """Agent decision making:
        By Agent:
        - Citizen: make_choice() -> Return choice -> create status?
        - DoctorTeam:
        - Ambulance:
        - Hospital:
        """
        self.schedule.trigger_agent_action_by_type("Ambulance", "make_choice")
        # Phase 5

        # Phase 6

        print("This is step: " + str(self.schedule.steps))
        # self.schedule.step()

        self.datacollector.collect(self)

        self.schedule.steps += 1
        self.schedule.time += 1

    def initialize_agents(self, num_buildings, num_citizens, num_hospitals, num_ambulances):
        self.buildings_to_nodes(num_buildings)
        self.citizens_to_buildings(num_citizens)
        self.hospitals_to_nodes(num_hospitals)
        self.ambulances_to_hospital(num_ambulances)
        return None


class TriageDispatcher(Dispatcher):
    def __init__(self, model: TriageModelAlpha, dispatch_size: int) -> object:
        super().__init__(model, dispatch_size)
        self.flipper_calls = 0 | 1
        self._incidents_queue = list()
        self._priority_queue = list()
        self._triaged_queue = list()

    def _collapsed_buildings(self):
        # list expression that creates a shuffled list of citizen unique Ids and returns them 1 by one

        collapsed_buildings = []
        for building in self.model.schedule.agents_by_type["Building"].values():
            if building.state == 3:
                collapsed_buildings.append(building.unique_id)
        return collapsed_buildings

    def update_incidents_queue(self):
        """Update the incidents queue with the collapsed buildings
        Only called by Dispatcher
        """
        incidents_queue = set(self._collapsed_buildings()) - self._assigned
        incidents_queue = incidents_queue - set(self._incidents_queue)
        # add as many incidents as possible

        incidents_queue = self.model.random.sample(list(incidents_queue), min(self.size, len(incidents_queue)))
        self._incidents_queue.extend(incidents_queue)
        return len(incidents_queue)

    def get_incident(self):
        """fetches the next incident that needs to be addressed
        Called by a Doctor Agent to go perform triage
        Returns
        -------
        """
        if len(self._incidents_queue) > 0:
            incident = self._incidents_queue.pop(0)
            self._assigned.add(incident)
            return incident
        else:
            return None

    def update_priority_queue(self, citizen_id: str):
        """Update the priority queue with the citizen_id
        Only called by DoctorTeam Responders when they are done triaging a citizen
        """
        self._priority_queue.append(citizen_id)
        return None

    def update_triaged_queue(self, citizen_id: str):
        """Update the priority queue with the citizen_id
        Only called by DoctorTeam Responders when they are done triaging a citizen
        """
        self._triaged_queue.append(citizen_id)
        return None

    def get_call(self):
        """fetches the next call in the queue
        if there is an avialable priority call, it will return that
        if there are no priority calls, it will alternatively return a regular call or a triaged call
        """
        if len(self._priority_queue) > 0:
            call = self._priority_queue.pop(0)
            self._assigned.add(call)
            return call
        elif len(self._calls_queue) > 0 and self.flipper_calls == 1:
            call = self._calls_queue.pop(0)
            self._assigned.add(call)
            self.flipper_calls = 0
            return call
        elif len(self._triaged_queue) > 0 and self.flipper_calls == 0:
            call = self._triaged_queue.pop(0)
            self._assigned.add(call)
            self.flipper_calls = 1
            return call
        else:
            return None


class TriageModelAlpha(MinimalModel):
    """A model with some number of agents."""
    dispatcher: TriageDispatcher
    def __init__(self,
                 num_buildings: int,
                 num_citizens: int,
                 num_hospitals: int,
                 num_ambulances: int,
                 num_doctors: int,
                 dispatch_size: int,
                 **kwargs
                 ):
        super().__init__(num_buildings, num_citizens, num_hospitals, num_ambulances)
        self.doctors_to_hospital(num_doctors)
        self.dispatcher = TriageDispatcher(self, dispatch_size)

    def doctors_to_hospital(self, doctors_per_hospital: int | Callable = 1):
        create_doctor = self.create_agents(DoctorTeam)
        i = 0
        hospitals_list: List[Hospital] = list(self.schedule.agents_by_type["Hospital"].values())
        for hospital in hospitals_list:
            hospital.doctors = list()
            ambulances_num = doctors_per_hospital
            location_list = self.random.choices(list(hospital.service_area), k=ambulances_num)
            for location in location_list:
                hospital.ambulances.append(create_doctor(location=location))
                create_doctor(location=location)
                i += 1
        return f'Created: {i} Doctor Teams'
