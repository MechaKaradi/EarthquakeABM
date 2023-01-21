from pyclbr import Class

from mesa import Agent, Model
import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from enum import Enum
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random

from agents import MinimalAgent, Buildings, MobileAgent, Citizen


class SpatialNetwork(space.NetworkGrid):
    def __init__(self, G) -> None:
        super().__init__(G)
        for node_id in self.G.nodes:
            G.nodes[node_id]["buildings"] = list()
    def place_agent_node(self, agent, node_id):
        """Place an agent on the given node, and set its pos.
        Args:
            agent: Agent to place
            node_id: Node ID of node to place agent on
        """
        if isinstance(agent, Buildings):
            self.G.nodes[node_id]["buildings"].append(agent)
        else:
            self.G.nodes[node_id]["agent"].append(agent)

        agent.pos = node_id


class MinimalModel(Model):
    G: nx.Graph = None
    def __init__(self):

        self.MINIMUM_RESIDENCY = 50 # minimum percentage of building capacity which is occupied
        self.schedule = time.RandomActivation(self)
        with open('street_network.data', 'rb') as file:
            self.G = pickle.load(file)
        # self.G = nx.relabel_nodes(self.G, {15012: 0}) """ No longer needed, incorporated into network creation
        # notebook"""
        self.grid = SpatialNetwork(self.G)

        model_metrics = {
            "Number of Agents": count_agents
        }

        agent_metrics = {
            "Agent ID": "unique_id",
            # lambda function to get the pos attribute of the node with position agent.position
            "Agent Position": lambda a: a.model.grid.G.nodes[a.position]["pos"],
            "Agent Colour": "color",
            "family": "agentFamily",
        }
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
        self.running = True
        self.datacollector.collect(self)

    agent_dictionary = {}

    # Create a closure to create agents with different classes but sequential unique ids
    def create_agents(self, agent_type: Class) -> classmethod:
        agent_type_str = str(agent_type.__name__)
        # set agent_id to an integer representation of the agent_type
        agent_id = 0

        model = self
        model.agent_dictionary[agent_type_str] = list()
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
            unique_id = f"{agent_type_str}_{agent_id}"
            a = agent_type(unique_id, model, **kwargs)
            model.schedule.add(a)

            agent_id += 1

            # add a to a list located in agent dictionary at the key of agent_type_str
            # if agent_type_str not in model.agent_dictionary.keys():

            model.agent_dictionary[agent_type_str].append(a)

            # place the agent on the grid
            a.spawn(location=location)

            return a

        return _create_agent

    def buildings_to_nodes(self, number_of_buildings):
        create_building = self.create_agents(Buildings)
        i = 0
        while i < number_of_buildings:
            node_id = self.random.choice(list(self.G.nodes))
            num = self.random.randint(0, 5)
            for j in range(num):
                create_building(location=node_id)
            i += num
        return f'Created: {i} buildings'

    def citizens_to_buildings(self, number_of_citizens: int):
        create_citizen = self.create_agents(Citizen)
        num = 0
        building_iterator = list(self.agent_dictionary["Buildings"])
        self.random.shuffle(building_iterator)
        b = 0
        while num <= number_of_citizens:
            building = building_iterator[b]
            i = len(building.residents)
            building_residents_number = round(building.capacity * self.random.randint(self.MINIMUM_RESIDENCY,100)/100, None)
            while i <= building_residents_number:
                create_citizen(location=building)
                i += 1
            b += 1
            num += i

        return f'Created: {num} citizens in {b} buildings'


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


def count_agents(self):
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
