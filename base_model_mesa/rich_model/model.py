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

from agents import MinimalAgent, Buildings


class MinimalModel(Model):
    def __init__(self):
        self.schedule = time.RandomActivation(self)
        with open('street_network.data', 'rb') as file:
            self.G = pickle.load(file)
        # self.G = nx.relabel_nodes(self.G, {15012: 0}) """ No longer needed, incorporated into network creation
        # notebook"""
        self.grid = space.NetworkGrid(self.G)
        self.num_agents = 10

        list_of_random_nodes = self.random.sample(list(self.G), self.num_agents)

        for i in range(self.num_agents):
            a = MinimalAgent(i+100000, self)
            self.schedule.add(a)
            print(list_of_random_nodes[i])
            self.grid.place_agent(a, list_of_random_nodes[i])

        for i in range(self.num_agents):
            a = Buildings(i+200000, self)
            self.schedule.add(a)
            print(list_of_random_nodes[i])
            self.grid.place_agent(a, list_of_random_nodes[i])

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