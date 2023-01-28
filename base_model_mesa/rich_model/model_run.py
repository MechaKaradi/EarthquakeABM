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

from model import MinimalModel

model = MinimalModel(500, 20000, 5, 15, 10)

# Run the model
for i in [*range(500)]:
    model.step()

model.step()

model_data = model.datacollector.get_model_vars_dataframe()
citizen_data = model.datacollector.get_agent_vars_dataframe('Citizen')
building_data = model.datacollector.get_agent_vars_dataframe('Building')
