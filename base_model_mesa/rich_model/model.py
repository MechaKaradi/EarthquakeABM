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

from agents import MinimalAgent, Buildings, MobileAgent

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

        self.schedule = time.RandomActivation(self)
        with open('street_network.data', 'rb') as file:
            self.G = pickle.load(file)
        # self.G = nx.relabel_nodes(self.G, {15012: 0}) """ No longer needed, incorporated into network creation
        # notebook"""
        self.grid = SpatialNetwork(self.G)
        self.num_agents = 5
        #self.spawn_agents() "This line with a # for this (AttributeError: 'MinimalModel' object has no attribute 'spawn_agents')" Lets check this error soon

        model_metrics = {
            "Number of Agents": count_agents
        }

        agent_metrics = {
            "Agent ID": "unique_id",
            "Agent Colour": "color",
            "family": "agentFamily",
        }
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
        self.running = True
        self.datacollector.collect(self)

    agent_dictionary = {}

    # Create a closure to create agents with different classes but sequential unique ids
    def create_agents(self, agent_type):
        agent_type_str = str(agent_type)
        # set agent_id to an integer representation of the agent_type
        agent_id = 0
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
            unique_id = f"{agent_type_str}_{agent_id}"
            a = agent_type(unique_id, model, **kwargs)
            model.schedule.add(a)

            agent_id += 1
            model.agent_dictionary.update({a.unique_id:a})

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
        return f'Created: {number_of_buildings}'


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
    return self.num_agents


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
