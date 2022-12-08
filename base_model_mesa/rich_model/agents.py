from mesa import Agent, Model
import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from enum import Enum
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random


class MinimalAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.color = '#%02x%02x%02x' % (random.randrange(0,256,16),random.randrange(0,256,16),random.randrange(0,256,16))

    def step(self):
        print("Hello world! I am agent: " + str(self.unique_id) + " my node id is: " + str(self.pos) +"my color is:"+ str(self.color))