from __future__ import annotations

from mesa import Agent

import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from typing import TYPE_CHECKING, Tuple, List, Any, Set

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
    _pos: int | Agent | None
    model: MinimalModel
    agentFamily: str
    counter: dict[str, int]

    def __init__(self, unique_id, model: MinimalModel):
        super().__init__(unique_id, model)

        self.agentFamily = str(type(self).__name__)
        self.color = '%02x%02x%02x' % (random.randrange(0, 256, 16),
                                       random.randrange(0, 256, 16),
                                       random.randrange(0, 256, 16)
                                       )

        self._pos = None


    # Handles Position of all agents
    @property
    def position(self) -> Tuple[float, float]:
        """Handles the position of the agent in the model space The hidden attribute _pos can be either a node ID or
        an agent object. The position keyword returns the x,y coordinates of the node ID - recursively if the _pos is
        an agent object The setter method allows to set the position of the agent to a node ID or an agent object The
        deleter method allows to remove the agent from the model space Both the setter and the deleter also update
        the _pos attribute and call the appropriate methods of the model space and the agent object if the _pos is a
        node ID or an agent object respectively

        Returns
        -------

        """
        if isinstance(self._pos, int):
            return self.model.grid.G.nodes[self._pos]["pos"]
        else:
            return self._pos.position

    @position.setter
    def position(self, location: int | Agent):
        """
        The setter method allows to set the position of the agent to a node ID or an agent object Based on the type
        of the location argument, the method calls the appropriate methods of the model space or of the agent object

        If the position is an ambulance, the position method does not call the ambulance method,
        because this method is called from the ambulance's method itself
        Parameters
        ----------
        location: int or Agent,
        """
        if isinstance(location, int):
            self.model.grid.place_agent_node(self, location)

        elif isinstance(location, (Building, Hospital)):
            location.add_citizen(self)

        elif isinstance(location, Ambulance):
            pass

        elif location is None:
            raise ValueError("The location of the agent is not specified")

        self._pos = location

    @position.deleter
    def position(self):
        if isinstance(self._pos, int):
            self.model.grid.remove_agent(self)
        elif isinstance(self._pos, (Building, Hospital)):
            self._pos.remove_citizen(self)
        elif isinstance(self._pos, Ambulance):
            pass
        elif self._pos is None:
            pass
        else:
            raise ValueError("The location of the agent is of a type that is not supported")

        self._pos = None

    def position_node(self):
        """
        Returns the node ID of the agent's position, recursively if the _pos is an agent object
        Returns
        -------

        """
        if isinstance(self._pos, int):
            return self._pos
        elif isinstance(self._pos, (Building, Hospital, Ambulance)):
            return self._pos.position_node()
        else:
            raise ValueError("The location of the agent is of a type that is not supported")

    def spawn(self, location: int | Agent = None, ) -> Tuple[float, float]:
        """
        Spawn the agent at an initial location on the spatial network.
        Default location of the agent is a node
        Args:
            location : a node ID for the spatial network. If None, select a random node ID from the model

        Returns
        -------
        position: attribute of the agent which is always a tuple of the x,y coords of the agent's node
        """
        self.position = location
        return self.position

    def step(self):
        pass


class Building(MinimalAgent):
    """
    state : int; the damage state of the building
    capacity : int; the maximum number of occupants that the building can hold

    """
    allowed_capacity = [1, 5, 10, 25, 50, 100, 250, 500]

    def __init__(self, unique_id, model, initial_state=0, initial_capacity=None):
        """
        Parameters
        ----------
        unique_id : int, passed from the 'model' object/class when the model object is initialized and the agent
        object is called
        initial_state : int
        initial_capacity : int | None
        """
        super().__init__(unique_id, model)
        self.agentFamily = str(type(self).__name__)
        self.state = initial_state
        if initial_capacity is not None:
            self.capacity = initial_capacity
        else:
            self.capacity = model.random.choice(self.allowed_capacity)

        self.occupants: list[Citizen] = []
        self.residents: list[Citizen] = []

        self.strength = model.random.gauss(8, 1)
        """Defines the strength of the building. The strength is a random variable with a mean of 8 and a standard 
        deviation of 1. This implies approximately 97% of buildings will have a strength between 5 and 9. """

    def damage_from_tremor(self, intensity: float) -> int:
        """Determines the damage from a tremor.
        Each building has a `damageState`:
            0: Serviceable
            1: Damaged
            2: Unsafe
            3: Collapsed

        Method to simulate the damage that the building will suffer from the earthquake. The method should update the
        building's `damageState` attribute based on the magnitude of the earthquake.
        If the building is already in a collapsed state, the method should do nothing. If the building is in an
        unsafe state, the method should directly change the state to collapse. If the building is in a damaged state,
        the probability of collapsing the building should be 2 times the probability if the building is in a
        serviceable state.

        If the building is in a serviceable state, the probability of collapsing the building depends on the
        magnitude. If the magnitude is less than the building's strength, the probability of damaging the building
        is 0. If the difference between the magnitude and the building's strength is greater than 2, the building is
        moved to a collapsed state. If the difference in between the magnitude and the building's strength is between
        0 and 2 a random number is generated between 0 and 1. If the number is between 0 and 0.1, the building is
        moved to an unsafe state. If the number is between 0.1 and 0.5, the building is moved to a damaged state. If
        the number is between 0.5 and 1, the building is moved to a serviceable state.

        Parameters
        ----------
        intensity : the magnitude of the earthquake expressed as a magnitude. Expected to be between 5 and 9

        Returns
        -------
        int : the increase in damage state of the building in number of levels

        """
        in_state = self.state  # the initial state of the building

        def x(building):
            var = building.state if in_state < 3 else 0
            return var

        if self.state == 3:  # if the building is already collapsed, do nothing
            return x(self)

        if self.state == 2:  # if the building is unsafe, it will collapse
            self.state = 3
            return x(self)

        # if the building is serviceable, the probability of collapsing depends on the magnitude
        if self.state == 0 or self.state == 1:
            if intensity < self.strength:
                # if the magnitude is less than the building's strength, nothing happens
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

    def earthquake(self, magnitude: float) -> None:
        """Simulates the earthquake on the building.
        The method should call the `damage_from_tremor` method to determine the damage the building will suffer from
        the earthquake. It then calls the `get_injured` method of each occupant of the building to determine the
        injuries.

        Parameters
        ----------
        magnitude : the intensity of the earthquake expressed as a magnitude. Expected to be between 5 and 9

        Returns
        -------
        None

        """
        damage = self.damage_from_tremor(magnitude)
        for occupant in self.occupants:
            occupant.get_injured(3 + damage * 3)

    # Create a list object to store the occupants of the building

    def assign_home(self, agent: Citizen) -> None:
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
        citizen : Citizen
        """
        if not self.is_full():
            self.occupants.append(citizen)

    def remove_citizen(self, citizen) -> None:
        """
        Method to remove a citizen from the building.
        """
        self.occupants.remove(citizen)


# Create a hospital class the inherits from the Building class
class Hospital(Building):
    service_area: Set[int] | None  # Set of nodes that the hospital serves
    ambulances: List[Ambulance]  # List of ambulances assigned to service_area of the hospital

    # When hospitals are created, they get assigned a voronoi_cell based on the cells that are closest to them

    def __init__(self, unique_id, model, initial_state=0, initial_capacity=None):
        super().__init__(unique_id, model, initial_state, initial_capacity)
        self.agentFamily = str(type(self).__name__)
        self.service_area = set()
        self.ambulances = []
        self.open_incident_sites = []


    def get_own_voronoi_cell(self):
        self.service_area = self.model.hospital_voronoi[self._pos]


class MobileAgent(MinimalAgent):
    """
    Basic agent capable of traversing the spatial network in the model
    Attributes
    model_parent : the model in which the agent is being initialized
    """

    def __init__(self, unique_id, model: MinimalModel):
        super().__init__(unique_id, model)
        self.current_path = None

    def find_path(self, destination):
        # Use the networkx library to find the shortest route from the position to the destination
        path = nx.shortest_path(self.model.G, source=self.position_node(), target=destination)
        return path

    def _move(self, destination):
        """
        Method to move the agent to a new node in the network. The method should update the agent's `position` attribute
        to the new node.
        # Steps to move agents to a new position
        # Remove the agent from the current position - using the position's remove_agent method
        # Add the agent to the new position - using the position's add_agent method
        # Update the agent's position attribute to the new position

        Parameters
        ----------
        destination : the node to which the agent is moving
        """

        if self.position == destination:
            return
        if self.position is not None:
            del self.position
        self.position = destination

    def step_to_destination(self, destination):
        """Method to move the agent one node towards an arbitrary destination

        Parameters
        ----------
        destination: int, node id of the destination
        Returns
        -------
        """
        if self.position_node() == destination:
            return 'Reached'

        if self.current_path is None or len(self.current_path) == 0 or self.current_path[-1] != destination:
            self.current_path = self.find_path(destination)
            if len(self.current_path) == 0:
                raise ValueError('No path to destination')
            self.current_path.pop(0)
            if len(self.current_path) == 0:
                raise ValueError(f'The impossible has happened in {self.unique_id}')

        next_node = self.current_path.pop(0)
        self._move(next_node)
        return next_node


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
    home: Building

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
        self.agentFamily = str(type(self).__name__)

        self.health = 13
        self.trapped = trapped

        self.home = None
        self.transported = False
        self.is_injured = False

        self.triage_status = None

    def spawn(self, location: int | Agent = None, ) -> None:
        """
        Method to spawn a citizen in the model. The method should assign the citizen to a building and add the citizen
        to the building's list of occupants. It should extend the `spawn` method of the parent class.
        If the location is not specified, the method should call the parent's spawn method.
        If the location is an integer, the method should leave the home attribute unassigned and call the parent
        class's spawn method.
        If the location is a Building agent, the method should assign the citizen to the building using the
        building's resident method.
        Parameters
        ----------
        location : int | Agent
        """
        if location is None:
            super().spawn()
        elif isinstance(location, int):
            super().spawn(location)
        elif isinstance(location, Building):
            self.home = location
            location.assign_home(self)
            super().spawn(location)

    def get_injured(self, severity):
        """reduces the health of the Citizen in response to external events

        When an earthquake occurs, a building collapses, or when there is some other external cause for damage

        Parameters ---------- severity: integer value between 0 and 12 representing the possible total value of
        damage that the citizen can suffer. Determined and passed by the calling event.

        Returns
        -------

        """

        probability_escape = 1.0 - (severity / (2 * 12.0))
        random_number = self.model.random.random()
        if random_number < probability_escape:
            self.is_injured = False
            return

        hurt = self.model.random.randint(1, severity)
        self.health -= hurt
        if self.health < 0:
            self.health = 0
        self.is_injured = True

    counter_health = -1
    ticks_to_next_health = {12: 90,
                            11: 90,
                            10: 60,
                            9: 60,
                            8: 60,
                            7: 30,
                            6: 15,
                            5: 15,
                            4: 15,
                            3: 15,
                            2: 10,
                            1: 5,
                            0: 0,
                            }
    counter_stabilize = 0

    def set_counter_health(self):
        if self.health == 13:
            self.counter_health = -1
            return 'healthy'

        if self.health == 0:
            self.counter_health = -100
            return 'dead'

        self.counter_health = self.ticks_to_next_health[self.health]
        return 'injured'

    def tick_health(self):
        if not self.is_injured:
            return 'not injured'

        if isinstance(self._pos, Hospital):
            return 'in hospital'

        if self.counter_health == -1:
            self.set_counter_health()
            return 'set counter'

        if self.counter_health > 0:
            self.counter_health -= 1

        if self.counter_health == 0:
            self.deteriorate_health()
            self.set_counter_health()

        else:
            return 'no ticks'
        return None

    def deteriorate_health(self):
        #
        """internal process of deterioration over time

        RPM , Mean time to next level in mins
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
        if self.health == 0:
            return None

        if self.health != 13:
            self.health -= 1
        return None

    def make_choice(self):
        """
        Make a choice about what to do based on the current situation
        Possible Choices: Status
        - Do nothing: None
        - Go toward home: Moving, destination = home
        - 1 Go to hospital: Moving, destination = nearest_hospital
        - Go to random location: Moving, destination = random Node int
        - Call for help: Calling
        - Help someone else: Carry -> then Go to Hospital

        Returns
        -------
        choice
        """


class Responder(MobileAgent):
    destination: int
    """Node ID of the position where there agent needs to reach"""

    status: None
    order: str
    """A string representing the Citizen that the agent has to respond to
    or the building that the agent has to respond to"""


    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agentFamily = str(type(self).__name__)

        self.order = None

        self.destination = None


    def get_closest_hospital(self):
        """Find the location of the closest hospital to current location

        Returns
        -------
        int: node id of the closest hospital

        """
        if isinstance(self._pos, int):
            return self.model.G.nodes[self._pos]['nearest_hospital']
        else:
            return "not on a node"

    def step_to_hospital(self):
        """Method to move the ambulance towards the closest hospital

        Parameters
        ----------


        Returns
        -------
        """
        self.destination = self.get_closest_hospital()
        return self.step_to_destination(self.destination)

    def get_order(self):
        """Method to get an order from the dispatcher

        Parameters
        ----------
        self.model.dispatcher: Dispatcher
        """
        self.order = self.model.dispatcher.get_call()
        # Get the first part of the order string which is the type of agent
        # that the responder needs to respond to

        if self.order is None:
            status = 'Idle'
            return None
        else:
            self.destination = self.model.schedule.find_agent_by_id(self.order).position_node()
            self.status = 'Moving'
            return self.status

class Ambulance(Responder):
    occupants = []
    """Occupants other than the patient being carried, if any"""
    patient: None
    """the object representing the patient which is being carried by the responder"""
    status: str
    # Todo: Action decision making
    # Todo: State Management for all agents

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agentFamily = str(type(self).__name__)
        self.occupants = []
        self.patient = None
        self.status = 'Idle'

        self.choice_dict = {
            'Idle': 'get_order',
            'Moving': 'step_to_destination',
            'Reached': 'pick_up_patient',
            'Returning': 'step_to_hospital',
            'Hospital': 'drop_off_patient'
        }

    def make_choice(self):
        """
        Returns
        -------
        choice:
        for Status -> Call action
        """
        if self.status is None:
            raise ValueError('Status is None')

        getattr(self, self.choice_dict[self.status])()
        return self.status

    def step_to_destination(self):
        dest = self.destination
        val = super().step_to_destination(dest)
        if val == 'Reached':
            self.status = 'Reached'
        return self.status

    def pick_up_patient(self):
        """Method to pick up a patient and add them to the ambulance

        Parameters
        ----------
        patient: Citizen

        Returns
        -------
        """
        # Confirm the postion of the ordered patient
        patient = self.model.schedule.find_agent_by_id(self.order)
        if patient is None:
            raise ValueError('Patient is None')
        if patient.position_node() != self.position_node():
            raise ValueError('Patient is not at the same location as the ambulance')

        patient.position = self
        self.patient = patient

        self.status = 'Returning'
        return self.status

    def step_to_hospital(self):
        self.destination = self.get_closest_hospital()
        val = self.step_to_destination()
        if val == 'Reached':
            self.status = 'Hospital'
        return self.status

    def drop_off_patient(self):
        """Method to drop off a patient and remove them from the ambulance
        Must be called when the ambulance is at the hospital
        Parameters
        ----------

        Returns
        -------
        """
        hospital = self.model.G.nodes[self.position_node()]['Hospital'][0]
        patient = self.patient

        if hospital is None:
            raise ValueError('Ambulance is not at a hospital')

        patient.position = hospital
        self.patient = None
        self.status = 'Idle'
        return self.status

    def step(self):
        """Method to step the ambulance through the model

        Parameters
        ----------

        Returns
        -------
        """
        if self.status is None:
            self.status = 'Idle'
        self.make_choice()


class DoctorTeam(Responder):
    # TOdo: Assign the DoctorTeam to an ambulance.
    # TODO; Create a triage function to model the doctors performing triage on the site
    # Todo: Create a stabilising function/status for when the are stabilising a patient
    # Todo: Changing sites with shared decision making of ambulance
    pass
