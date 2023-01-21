from __future__ import annotations

from mesa import Agent
from agents import Citizen

import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # We ensure that these are not imported during runtime to prevent cyclic
    # dependency.
    from model import MinimalModel

from enum import Enum
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random



class MinimalAgent(Agent):
    model: MinimalModel

    def __init__(self, unique_id, model: MinimalModel):
        super().__init__(unique_id, model)

        self.agentFamily = str(type(self).__name__)
        self.color = '#%02x%02x%02x' % (random.randrange(0, 256, 16),
                                        random.randrange(0, 256, 16),
                                        random.randrange(0, 256, 16)
                                        )

    position: int | Agent = None
    # positions of the agents may be a node or a building

    def spawn(self, location: int | Agent = None, ) -> None:
        """
        Spawn the agent at an initial location on the spatial network.
        Default location of the agent is a node
        Args:
            location : a node ID for the spatial network. If None, select a random node ID from the model

        -------
        Returns
        None
        """
        if isinstance(location, int):
            self.model.grid.place_agent_node(self, location)

        elif isinstance(location, Buildings):
            location.add_citizen(self)

        elif location is None:
            raise ValueError("The location of the agent is not specified")

        self.position = location
        return location

    def step(self):
        print("Hello world! I am agent: " + str(self.unique_id) +
              "\n my node id is: " + str(self.pos) +
              "\n my color is: " + str(self.color))

class Citizen(MinimalAgent):
    def __init__(self, unique_id, model, initial_state = "healthy"):
        super().__init__(unique_id, model)
        self.state = initial_state
        self.transported = False
        self.is_injured = False
        
    def set_health_status(self, status: str):
        """
        set the health status of the citizen
        """
        self.health_status = status


class Ambulance(MinimalAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.patient = None
    
    def transport_patient(self, patient: Citizen):
        self.patient = patient
        self.patient.transported = True
        # Move the ambulance to the patient's location
        self.model.grid.move_agent(self, patient.pos)
        
    def step(self):
        if self.patient is not None:
            # Move the ambulance to the hospital
            hospital = self.model.get_closest_hospital(self.patient.pos)
            self.model.grid.move_agent(self, hospital)
            self.patient.health_status = "treated"
            self.patient = None
        
        
        
# class StaticAgent(MinimalAgent):
#     def init(self, unique_id, model):
#         super().__init__(unique_id, model)
#
#     def place(self, location):
#         self.model.grid.place_agent_node(self, location)


class Buildings(MinimalAgent):
    """
    state : int; the damage state of the building
    capacity : int; the maximum number of occupants that the building can hold

    """

    def __init__(self, unique_id, model, initial_state = 0, initial_capacity = None):
        """
        Parameters
        ----------
        unique_id : int, passed from the 'model' object/class when the model object is initialized and the agent
        object is called
        initial_state : int
        initial_capacity : int | None
        """
        super().__init__(unique_id, model)
        self.state = initial_state
        self.capacity = initial_capacity if initial_capacity != None else model.random.choice([1, 5, 10, 25, 50, 100,
                                                                                             250, 500])

        self.occupants: list[Citizen] = []
        self.residents: list[Citizen] = []

        self.strength = model.random.gauss(7, 1)
        """Defines the strength of the building. The strength is a random variable with a mean of 7 and a standard 
        deviation of 1. This implies approximately 97% of buildings will have a strength between 5 and 9. """

    def damage_from_tremor(self, intensity: float) -> int:
        """Determines the damage from a tremor.
        Each building has a `damageState`:
            0: Serviceable
            1: Damaged
            2: Unsafe
            3: Collapsed

        Method to simulate the damage that the building will suffer from the earthquake. The method should update the
        building's `damageState` attribute based on the intensity of the earthquake.
        If the building is already in a collapsed state, the method should do nothing. If the building is in an
        unsafe state, the method should directly change the state to collapse. If the building is in a damaged state,
        the probability of collapsing the building should be 2 times the probability if the building is in a
        serviceable state.

        If the building is in a serviceable state, the probability of collapsing the building depends on the
        intensity. If the intensity is less than the building's strength, the probability of damaging the building
        is 0. If the difference between the intensity and the building's strength is greater than 2, the building is
        moved to a collapsed state. If the difference in between the intensity and the building's strength is between
        0 and 2 a random number is generated between 0 and 1. If the number is between 0 and 0.1, the building is
        moved to an unsafe state. If the number is between 0.1 and 0.5, the building is moved to a damaged state. If
        the number is between 0.5 and 1, the building is moved to a serviceable state.

        Parameters
        ----------
        intensity : the intensity of the earthquake expressed as a magnitude. Expected to be between 5 and 9

        Returns
        -------
        int : the increase in damage state of the building in number of levels

        """
        in_state = self.state # the initial state of the building
        x = lambda self : self.state if in_state < 3 else 0

        if self.state == 3:  # if the building is already collapsed, do nothing
            return x(self)

        if self.state == 2:  # if the building is unsafe, it will collapse
            self.state = 3
            return x(self)

        # if the building is serviceable, the probability of collapsing depends on the intensity
        if self.state == 0 or self.state == 1:
            if intensity < self.strength:
                return x(self)

            if intensity - self.strength >= 2:
                self.state = 3
                return x(self)

            if intensity - self.strength < 2:
                rand = self.model.random.random()
                rand = rand * (
                            self.state + 1)  # the probability of collapsing is 2 times the probability of damaging
                # the building
                if rand < 0.1:
                    self.state = 2
                    return x(self)
                elif rand < 0.5:
                    self.state = 1
                    return x(self)
                else:
                    return x(self)

    def earthquake(self, intensity: float) -> None:
        """Simulates the earthquake on the building.
        The method should call the `damage_from_tremor` method to determine the damage the building will suffer from
        the earthquake. It then calls the `get_injured` method of each occupant of the building to determine the
        injuries.

        Parameters
        ----------
        intensity : the intensity of the earthquake expressed as a magnitude. Expected to be between 5 and 9

        Returns
        -------
        None

        """
        damage = self.damage_from_tremor(intensity)
        for occupant in self.occupants:
            occupant.get_injured(3 + damage*3)

    # Create a list object to store the occupants of the building

    def assign_home(self, agent: Agent) -> None:
        """ Assigns the agents as a Resident of the building
        Parameters
        ----------
        agent : Agent
        """
        self.residents.append(agent)
        return None


    def is_full(self) -> bool:
        """
        Method that returns True if the building is full (i.e., the number of citizens in the building is equal to its
        capacity) and False otherwise.
        """
        return len(self.occupants) >= self.capacity

    def add_citizen(self, citizen) -> None:
        """
        Method to add a citizen to the building. The method should check if the building is full before adding the
        citizen.

        Parameters
        ----------
        citizen : object
        """
        if not self.is_full():
            self.occupants.append(citizen)

    def remove_citizen(self, citizen) -> None:
        """
        Method to remove a citizen from the building.
        """
        self.occupants.remove(citizen)


# Create a hospital class the inherits from the Buildings class
class Hospital(Buildings):
    pass


class MobileAgent(MinimalAgent):
    """
    Basic agent capable of traversing the spatial network in the model
    Attributes
    model_parent : the model in which the agent is being initialized
    """

    def __int__(self, unique_id, model):
        super().__init__(unique_id, model)

    def find_path(self, destination):
        # Use the networkx library to find the shortest route from the position to the destination
        path = nx.shortest_path(self.model.G, source=self.position, target=destination)
        next_node = path[1]
        return next_node, path
        pass


class Ambulance(MobileAgent):
    pass


class Citizen(MobileAgent):
    """
    A citizen is the parent class for all agents that represent residents of the city that are potentially affected
    by the disaster and are attempting to survive the situation. Citizens are able to traverse the network,
    enter or exit buildings, and may visit hospitals. Citizens have a health value (max 13) and a trapped status(
    boolean),
    If a citizen health value goes to below 13 they become a casualty
    If the citizen health value goes to 0 they become a corpse.
    Each citizen is a 'resident' of the city and has a `Residence(Building)` which they are assigned to as their 'home'.
    """
    home: Buildings

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

        self.home = None

    def spawn(self, location: int | Agent = None, ) -> None:
        """
        Method to spawn a citizen in the model. The method should assign the citizen to a building and add the citizen
        to the building's list of occupants. It should extend the `spawn` method of the parent class.
        If the location is not specified, the method should call the parent's spawn method.
        If the location is an integer, the method should leave the home attribute unassigned and call the parent
        class's spawn method.
        If the location is a Buildings agent, the method should assign the citizen to the building using the
        building's resident method.
        Parameters
        ----------
        location : int | Agent
        """
        if location is None:
            super().spawn()
        elif isinstance(location, int):
            super().spawn(location)
        elif isinstance(location, Buildings):
            self.home = location
            location.assign_home(self)
            super().spawn(location)

    def move(self, destination):
        """
        Method to move the agent to a new node in the network. The method should update the agent's `position` attribute
        to the new node.
        Parameters
        ----------
        destination : the node to which the agent is moving
        """
        pass


    def get_injured(self, severity):
        """reduces the health of the Citizen in response to external events

        When an earthquake occurs, a building collapses, or when there is some other external cause for damage

        Parameters ---------- severity: integer value between 0 and 12 representing the possible total value of
        damage that the citizen can suffer. Determined and passed by the calling event.

        Returns
        -------

        """
        self.health -= self.model.random.randint(0, severity)


    @property
    def deteriorate_health(self):
        """internal process of deterioration over time

        RPM , Mean time to next level in mains
        12 , 90
        11 , 90
        10 , 60
        9 , 60
        8 , 60
        7 , 30
        6 , 15
        5 , 15
        4 , 15
        3 , 15
        2 , 10
        1 , 5
        0 , 0

        Returns
        -------
        None
        """
        if self.health != 13:
            self.health -= 1
        return None
