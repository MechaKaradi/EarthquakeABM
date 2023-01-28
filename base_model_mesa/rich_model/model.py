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


class SpatialNetwork(space.NetworkGrid):
    def __init__(self, G) -> None:
        """

        Parameters
        ----------
        G : nx.Graph
        """
        self.G = G

        for node_id in self.G.nodes:
            G.nodes[node_id]["Building"] = list()
            G.nodes[node_id]["Hospital"] = list()
            G.nodes[node_id]["Ambulance"] = list()
            G.nodes[node_id]["DoctorTeam"] = list()
            G.nodes[node_id]["Citizen"] = list()

    def place_agent_node(self, agent, node_id):
        """Place an agent on the given node, and set its pos.
        Args:
            agent: Agent to place
            node_id: Node ID of node to place agent on

        Parameters
        ----------
        agent : MinimalAgent
        """
        # TODO: Can be refactored to use agent.family instead of isinstance
        self.G.nodes[node_id][agent.agentFamily].append(agent)
        """
        if isinstance(agent, Building):
            self.G.nodes[node_id]["Building"].append(agent)
        elif isinstance(agent, Hospital):
            self.G.nodes[node_id]["Hospital"].append(agent)
        elif isinstance(agent, Ambulance):
            self.G.nodes[node_id]["Ambulance"].append(agent)
        elif isinstance(agent, DoctorTeam):
            self.G.nodes[node_id]["DoctorTeam"].append(agent)
        else:
            self.G.nodes[node_id]["Agent"].append(agent)"""

        agent.pos = node_id

    def remove_agent(self, agent: MinimalAgent) -> None:
        """Remove the agent from the network """
        node_id = agent.position_node()
        # TODO: Can be refactored to use agent.family instead of isinstance
        self.G.nodes[node_id][agent.agentFamily].remove(agent)

    def get_node_agents(self, node_id):
        raise 'NotImplementedError'
        # return self.G.nodes[node_id]["Agent"]


class StagedAndTypedTime(time.BaseScheduler):
    agents_by_type: defaultdict[str, dict]
    """A dictionary of all the agents organised by type
    Contains a key for each agent type and as a value contains a dictionary of all agents of that type 
    The nested dictionary is keyed by the agent's unique ID and the value is the respective agent object"""

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_type = defaultdict(dict)

    def add(self, agent: MinimalAgent) -> None:
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.
        """
        super().add(agent)
        agent_class: str = agent.agentFamily
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent: MinimalAgent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class: str = agent.agentFamily
        del self.agents_by_type[agent_class][agent.unique_id]

    def trigger_agent_action_by_type(self, agent_type: str, agent_action: str, agent_filter=None, **kwargs):
        if agent_filter is None:
            def agent_filter(test_agent):
                return True

        for agent in self.agents_by_type[agent_type].values():
            if agent_filter(agent):
                getattr(agent, agent_action)(**kwargs)

    def find_agent_by_id(self, agent_id):
        agent_type = agent_id.split("_")[0]
        return self.agents_by_type[agent_type][agent_id]


class ExtendedDataCollector(DataCollector):

    def __init__(self, model_reporters=None, agent_reporters=None, tables=None):
        """Instantiate a DataCollector with lists of model and agent reporters by type.

        Args:
            model_reporters: Dictionary of reporter names and attributes/funcs
            agent_reporters: Dictionary of reporter names and attributes/funcs.
            tables: Dictionary of table names to lists of column names.

            Model reporters can take four types of arguments:
            lambda like above:
            {"agent_count": lambda m: m.schedule.get_agent_count() }
            method with @property decorators
            {"agent_count": schedule.get_agent_count()
            class attributes of model
            {"model_attribute": "model_attribute"}
            functions with parameters that have placed in a list
            {"Model_Function":[function, [param_1, param_2]]}

        """
        self.agent_types = agent_reporters.keys()

        self.model_reporters = {}
        self.model_vars = {}

        if model_reporters is not None:
            for name, reporter in model_reporters.items():
                self._new_model_reporter(name, reporter)

        self.agent_reporters = {}
        self._agent_records = {}
        self.agent_attr_index = {}
        self.agent_name_index = {}

        for agent_type in self.agent_types:
            self.agent_reporters[agent_type] = {}
            self._agent_records[agent_type] = {}
            self.agent_name_index[agent_type] = {}
        self.tables = {}

        for agent_type in self.agent_types:
            if agent_reporters[agent_type] is not None:
                for name, reporter in agent_reporters[agent_type].items():
                    self.agent_name_index[agent_type][name] = reporter
                    self._new_agent_reporter(name, reporter, agent_type)

        if tables is not None:
            for name, columns in tables.items():
                self._new_table(name, columns)

    def _new_agent_reporter(self, name, reporter, agent_type):
        """Add a new agent-level reporter to collect.

        Args:
            name: Name of the agent-level variable to collect.
            reporter: Attribute string, or function object that returns the
                      variable when given a model instance.

        """
        if type(reporter) is str:
            attribute_name = reporter
            reporter = partial(self._getattr, reporter)
            reporter.attribute_name = attribute_name
        self.agent_reporters[agent_type][name] = reporter

    def _record_agents(self, model, agent_type):
        """Record agents data in a mapping of functions and agents."""
        rep_funcs = self.agent_reporters[agent_type].values()
        if all([hasattr(rep, "attribute_name") for rep in rep_funcs]):
            prefix = ["model.schedule.steps", "unique_id"]
            attributes = [func.attribute_name for func in rep_funcs]
            self.agent_attr_index[agent_type] = {k: v for v, k in enumerate(prefix + attributes)}
            if len(self.agent_attr_index[agent_type]) == 0:
                raise ValueError("No agent attributes to record.")
            get_reports = attrgetter(*prefix + attributes)

        else:
            def get_reports(agent):
                _prefix = (agent.model.schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in rep_funcs)
                return _prefix + reports

        agent_records = map(get_reports, model.schedule.agents_by_type[agent_type].values())
        return agent_records

    def collect(self, model):
        """Collect all the data for the given model object."""
        if self.model_reporters:

            for var, reporter in self.model_reporters.items():
                # Check if Lambda operator
                if isinstance(reporter, types.LambdaType):
                    self.model_vars[var].append(reporter(model))
                # Check if model attribute
                elif isinstance(reporter, partial):
                    self.model_vars[var].append(reporter(model))
                # Check if function with arguments
                elif isinstance(reporter, list):
                    self.model_vars[var].append(reporter[0](*reporter[1]))
                else:
                    self.model_vars[var].append(self._reporter_decorator(reporter))

        for agent_type in self.agent_types:
            if self.agent_reporters[agent_type]:
                agent_records = self._record_agents(model, agent_type)
                self._agent_records[agent_type][model.schedule.steps] = list(agent_records)

    def get_agent_vars_dataframe(self, agent_type):
        """Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        """
        all_records = itertools.chain.from_iterable(self._agent_records[agent_type].values())
        rep_names = [rep_name for rep_name in self.agent_reporters[agent_type]]

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID"] + rep_names,
        )
        df = df.set_index(["Step", "AgentID"], drop=True)
        return df


class Dispatcher:
    """handles the availability of information to the Responder agents
    A special role in the simulation that is not an agent.
    Dispatcher is not modled as an agents because it is not necessarily analogous to an institution or organization.
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
    MINIMUM_RESIDENCY: int
    """a percentage, the minimum proportion of each building which is occupied at the time of spawn"""

    G: nx.Graph
    """the street network"""

    EARTHQUAKE_EVENTS: Dict[int, float]
    """a dictionary of earthquake events. the first number is the step of the event, the second is the mangnitude"""

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
        with open('street_network.data', 'rb') as file:
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
                # "Agent ID": "unique_id",
                # lambda function to get the pos attribute of the node with position agent.position
                "Agent Coordnates": "position",
                "Agent Position": "_pos",
                "Agent Colour": "color",
                "family": "agentFamily",
                "Health": "health"},
            'Ambulance': {
                "Ambulance Position": "_pos",
                "Ambulance Status": "status",
                "Patient": "patient",
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
            #TODO: apply a heal method to Citizens for hosptital, ambulance, and doctor team
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


def count_agents(self: MinimalModel):
    return len(self.schedule.agents)
