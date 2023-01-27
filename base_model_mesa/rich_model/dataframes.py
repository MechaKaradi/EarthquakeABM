import pandas
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


from agents import MinimalAgent, Building, MobileAgent, Citizen
from model import  MinimalModel

import holoviews as hv
hv.extension('matplotlib')

model = MinimalModel(10)

model.step()

model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()
print(model_data)
print(agent_data)

graph = model.G

datadump = pandas.DataFrame.from_dict(model.G._node, orient="index")


# from graph get a dictionary of nodes and their pos attribute
a = dict(model.G.nodes.data('pos'))

# Node positions defined as a dictionary mapping from node id to (x, y) tuple or networkx layout function which computes a positions dictionary
# create a dictionary consisting of x,y coordinates for each node


hv.Graph.from_networkx(G=graph, positions=a)


