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


from rich_model.agents import *


class MinimalModel(Model):
    def __init__(self):
        self.schedule = time.RandomActivation(self)
        with open('street_network.data', 'rb') as file:
            self.streets = pickle.load(file)
        
        self.G = self.streets
        
        self.grid = space.NetworkGrid(self.streets)
        
        
        self.num_agents = 10
  
        for i in range(self.num_agents):
            a = MinimalAgent(i, self)
            self.schedule.add(a)
            location = random.choice(list(self.streets))
            #print(location)
            self.grid.place_agent(a, location)

        model_metrics = {
            "Number of Agents": count_agents
        }

        agent_metrics = {
            "Agent ID": "unique_id"
        }

        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        print("This is step: " + str(self.schedule.steps))
        self.schedule.step()
        self.datacollector.collect(self)

""" Model metrics"""
def count_agents(self):
    return self.num_agents


"""Run Model"""
model = MinimalModel()
for i in range(3):
    model.step()




# Get the Pandas Dataframe from the model, by using the table name we defined in the model
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()
print(model_data)
print(agent_data)
