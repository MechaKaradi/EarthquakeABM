from collections import defaultdict
from pyclbr import Class
import types

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
            G.nodes[node_id]["Agent"] = list()

    def place_agent_node(self, agent, node_id):
        """Place an agent on the given node, and set its pos.
        Args:
            agent: Agent to place
            node_id: Node ID of node to place agent on
        """
        if isinstance(agent, Building):
            self.G.nodes[node_id]["Building"].append(agent)
        elif isinstance(agent, Hospital):
            self.G.nodes[node_id]["Hospital"].append(agent)
        else:
            self.G.nodes[node_id]["Agent"].append(agent)

        agent.pos = node_id

    def remove_agent(self, agent: MinimalAgent) -> None:
        """Remove the agent from the network and set its pos attribute to None."""
        node_id = agent.position_node()
        self.G.nodes[node_id]["Agent"].remove(agent)


class StagedAndTypedTime(time.BaseScheduler):
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

    def trigger_agent_action_bytype(self, agent_type: str, agent_action: str, agent_filter=None):
        if agent_filter is None:
            def agent_filter(test_agent):
                return True

        for agent in self.agents_by_type[agent_type].values():
            if agent_filter(agent):
                getattr(agent, agent_action)()


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
            print(self.agent_attr_index[agent_type])
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

    # Override the vars from agents method to ensure a steps column is accessible
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


class MinimalModel(Model):
    G: nx.Graph = None

    def __init__(self):
        # Parameters
        self.MINIMUM_RESIDENCY = 50  # minimum percentage of building capacity which is occupied

        # Create time
        self.schedule = StagedAndTypedTime(self)

        # Create Space
        with open('street_network.data', 'rb') as file:
            self.G = pickle.load(file)

        self.grid = SpatialNetwork(self.G)

        # Create Data Collector
        model_metrics = {
            "Number of Agents": count_agents
        }
        agent_metrics = {'Citizen': {
            # "Agent ID": "unique_id",
            # lambda function to get the pos attribute of the node with position agent.position
            "Agent Coordnates": "position",
            "Agent Position": "_pos",
            "Agent Colour": "color",
            "family": "agentFamily",
            "Health": "health",
        }}
        self.datacollector = ExtendedDataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
        self.datacollector.collect(self)

        self.running = True

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
            a = agent_type_created(unique_id, model, **kwargs)

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
        self.schedule.trigger_agent_action_bytype("Hospital", "get_own_voronoi_cell")

        return f'Created: {i} hospitals'

    def citizens_to_buildings(self, number_of_citizens: int):
        create_citizen = self.create_agents(Citizen)
        num = 0
        building_iterator = list(self.schedule.agents_by_type["Building"].values())
        self.random.shuffle(building_iterator)
        b = 0
        while num <= number_of_citizens:
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

    def assign_nearest_hospital_node(self) -> None:
        """Assign each node the nearest hospital as an attribute"""
        hospital_nodes = [(lambda x: x._pos)(x) for x in (self.schedule.agents_by_type["Hospital"].values())]
        self.hospital_voronoi = nxalg.voronoi_cells(self.G, set(hospital_nodes), weight='Length')

        for hospital in self.hospital_voronoi:
            hospital_color = self.random.choice(['red', 'blue', 'green', 'yellow', 'orange', 'purple'])
            for node in self.hospital_voronoi[hospital]:
                self.G.nodes[node]['nearest_hospital'] = hospital
                self.G.nodes[node]['nearest_hospital_color'] = hospital_color

    def step(self):
        print("This is step: " + str(self.schedule.steps))
        self.schedule.step()
        self.datacollector.collect(self)


"""
    def run_model(self, n):
        for i in range(n):
            self.step()
"""

""" Model metrics"""


def count_agents(self: MinimalModel):
    return len(self.schedule.agents)


"""Run Model"""
"""
model = MinimalModel()
for i in range(3):
    model.step()
"""

# Get the Pandas Dataframe from the model, by using the table name we defined in the model
"""
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()
print(model_data)
print(agent_data)
"""
