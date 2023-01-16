import networkx
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

    position = None

    def spawn(self, location: str = None, ) -> None:
        """
        Spawn the agent at an initial location on the spatial network.
        Default location of the agent is a node
        Args:
            location : a node ID for the spatial network. If None, select a random node ID from the model

        -------
        Returns
        None
        """

        if location == None:
            location = self.model.random.sample(list(self.model.G), 1)
            location = location[0]

        self.model.grid.place_agent(self, location)

        self.position = location
        return None

    def step(self):
        print("Hello world! I am agent: " + str(self.unique_id) +
              "\n my node id is: " + str(self.pos) +
              "\n my color is: " + str(self.color))

class StaticAgent(MinimalAgent):
    def init(self, unique_id, model):
        super().__init__(unique_id, model)

    def place(self, location):
        self.model.grid.place_agent(self, location)


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
        self.state = initial_state
        self.capacity = base_capacity

        self.strength = model.random.gauss(7,1)
        """Defines the strength of the building. 
        The strength is a random variable with a mean of 7 and a standard deviation of 1. This implies approximately 97% of buildings will have a strength between 5 and 9.  
        """

    def damage_from_tremor(self, intensity: float) -> None:
        """Determines the damage from a tremor.
        Method to simulate the damage that the building will suffer from the earthquake. The method should update the 
        building's `damageState` attribute based on the intensity of the earthquake.

        If the building is already in a collapsed state, the method should do nothing.
        If the building is in an unsafe state, the method should direcly chagne the state to collapsed.
        If the building is in a damaged state, the probability of collapsing the building should be 2 times the probability if the building is in a serviceable state.

        If the building is in a serviceable state, the probability of collapsing the building depends on the intensity.
        If the intensiity is less than the building's strength, the probability of damaging the building is 0.
         If the difference between the intensity and the building's strength is greater than 2, the building is moved to a collapsed state.
         If the difference in between the intensity and the building's strength is between 0 and 2 a random number is generated between 0 and 1. If the number is between 0 and 0.1, the building is moved to an unsafe state.
        If the number is between 0.1 and 0.5, the building is moved to a damaged state.
        If the number is between 0.5 and 1, the building is moved to a serviceable state.

        Parameters
        ----------
        intensity : the intensity of the earthquake expressed as a magnitude. Expected to be between 5 and 9

        """

        if self.state == 3: #if the building is already collapsed, do nothing
            return None

        if self.state == 2: #if the building is unsafe, it will collapse
            self.state = 3
            return None

        # if the building is serviceable, the probability of collapsing depends on the intensity
        if self.state == 0 or self.state == 1:
            if intensity < self.strength:
                return None

            if intensity - self.strength >= 2:
                self.state = 3
                return None

            if intensity - self.strength < 2:
                rand = self.model.random.random()
                rand = rand * (self.state + 1) #the probability of collapsing is 2 times the probability of damaging the building
                if rand < 0.1:
                    self.state = 2
                    return None
                elif rand < 0.5:
                    self.state = 1
                    return None
                else:
                    return None
    def is_full(self) -> bool:
        """
        Method that returns True if the building is full (i.e., the number of citizens in the building is equal to its
        capacity) and False otherwise.
        """
        pass

    def add_citizen(self, citizen) -> None:
        """
        Method to add a citizen to the building. The method should check if the building is full before adding the
        citizen.

        Parameters
        ----------
        citizen : object
        """
        pass

    def remove_citizen(self, citizen) -> None:
        """
        Method to remove a citizen from the building.
        """
        pass



class MobileAgent(MinimalAgent):
    """
    Basic agent capable of traversing the spatial network in the model
    Attributes
    model_parent : the model in which the agent is being initialized
    """
    def __int__(self, unique_id, model):
        super().__int__(unique_id, model)


    def find_path(self, destination):
        # Use the networkx library to find the shortest route from the position to the destination
        path = nx.shortest_path(self.model.G, source = self.position, target = destination)
        next_node = path[1]
        return (next_node,path)
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
