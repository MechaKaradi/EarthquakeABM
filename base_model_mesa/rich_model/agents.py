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
              "\n my color is: " + str(self.color))


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

    def __init__(self, unique_id, model, initial_state=0, base_capacity=0):
        """
        Parameters
        ----------
        unique_id : int, passed from the 'model' object/class when the model object is initialised and the agent
        object is called
        initial_state : int
        base_capacity : int
        """
        super().__init__(unique_id, model)
        print(self.color)
        state = initial_state
        capacity = base_capacity

        def step(self):
            super().step(self)

class MobileAgent(MinimalAgent):
    """
    Basic agent capable of traversing the spatial network in the model
    """
    def __int__(self, unique_id, model):
        super().__int__(unique_id, model)

    def spawn(location):
        """
        Spawn the agent at an initial location on the spatial network.
        Location can be a position
        Returns -> None
        -------

        """

    def find_path(self,destination):


    pass


class Citizen(MobileAgent):
    """
    A citizen is the parent class for all agents that represent residents of the city that are potentially affected
    by the disaster and are attempting to survive the situation. Citizens are able to traverse the network,
    enter or exit buildings, and may visit hospitals. Citizens have a health value (max 13) and a trapped status(
    boolean),
    If a citizen health value goes to below 13 they become a casulality
    If the citizen health value goes to 0 they become a corpse.
    Each citizen is a 'resident' of the city and has a `Residence(Building)` which they are assigned to as their 'home.
    """

    def __init__(self, unique_id, model, health=13, trapped=False):
        """
        Parameters
        ----------
        unique_id
        model
        health
        trapped
        """
        super().__init__(unique_id, model)
        self.health = health
        self.trapped = trapped


    pass
