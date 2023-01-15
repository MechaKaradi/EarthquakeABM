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


class MinimalModel(Model):
    def __init__(self):

        self.schedule = time.RandomActivation(self)
        with open('street_network.data', 'rb') as file:
            self.G = pickle.load(file)
        # self.G = nx.relabel_nodes(self.G, {15012: 0}) """ No longer needed, incorporated into network creation
        # notebook"""
        self.grid = space.NetworkGrid(self.G)
        self.num_agents = 5
        self.spawn_agents()

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

        def _create_agent(location):
            nonlocal agent_id
            nonlocal model
            unique_id = f"{agent_type_str}_{agent_id}"
            a = agent_type(unique_id, model)
            model.schedule.add(a)

            agent_id += 1
            model.agent_dictionary[a.unique_id] = a

            a.spawn(location=location)

            return a

        return _create_agent

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
