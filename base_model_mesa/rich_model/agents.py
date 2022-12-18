from mesa import Agent
import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from enum import Enum
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random


class MinimalAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agentFamily = str(type(self))
        self.color = '#%02x%02x%02x' % (random.randrange(0, 256, 16),
                                        random.randrange(0, 256, 16),
                                        random.randrange(0, 256, 16)
                                        )

    def step(self):
        print("Hello world! I am agent: " + str(self.unique_id) +
              "\n my node id is: " + str(self.pos) +
              "\n my color is: " + str(self.color)
              )
class Buildings(MinimalAgent):
    """
    Buildings exist on each node in the model
    Each building has a `damageState`:
        0: Serviceable
        1: Damaged
        2: Unsafe
        3: Collapsed

    Each building has a `capacity` which is the total number of `citizens` that can be in the building

    Each Building has a 'damageFromTremor' method which responds to an earthquake call at the beginning of a tick
    when `damageFromTremor` is called, it has a probability to increase the building damage, and a probability to
    injure or trap citizens on the node in the building.

    """
    def __init__(self, unique_id, model, initial_state = 0, base_capacity = 0):
        """
        Parameters
        ----------
        unique_id : int, passed from the 'model' object/class when the model object is initialised and the agent
        object is called
        initial_state : int
        base_capacity : int
        """
        super().__init__(unique_id,model)
        print(self.color)
        state = initial_state
        capacity = base_capacity

