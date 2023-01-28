from __future__ import annotations

from collections import defaultdict

import types
from typing import Callable, Dict, List, Tuple, Union, Any, Set
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # We ensure that these are not imported during runtime to prevent cyclic
    # dependency.
    from model import MinimalModel
    from agents import MinimalAgent

import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector

from functools import partial
import itertools
from operator import attrgetter

import pandas as pd


class SpatialNetwork(space.NetworkGrid):
    def __init__(self, G) -> None:
        """

        Parameters
        ----------
        G : nx.Graph
        """
        self.G = G

        for node_id in self.G.nodes:
            G.nodes[node_id]["Building"] = list()
            G.nodes[node_id]["Hospital"] = list()
            G.nodes[node_id]["Ambulance"] = list()
            G.nodes[node_id]["DoctorTeam"] = list()
            G.nodes[node_id]["Citizen"] = list()

    def place_agent_node(self, agent, node_id):
        """Place an agent on the given node, and set its pos.
        Args:
            agent: Agent to place
            node_id: Node ID of node to place agent on

        Parameters
        ----------
        agent : MinimalAgent
        """
        # TODO: Can be refactored to use agent.family instead of isinstance
        self.G.nodes[node_id][agent.agentFamily].append(agent)
        """
        if isinstance(agent, Building):
            self.G.nodes[node_id]["Building"].append(agent)
        elif isinstance(agent, Hospital):
            self.G.nodes[node_id]["Hospital"].append(agent)
        elif isinstance(agent, Ambulance):
            self.G.nodes[node_id]["Ambulance"].append(agent)
        elif isinstance(agent, DoctorTeam):
            self.G.nodes[node_id]["DoctorTeam"].append(agent)
        else:
            self.G.nodes[node_id]["Agent"].append(agent)"""

        agent.pos = node_id

    def remove_agent(self, agent: MinimalAgent) -> None:
        """Remove the agent from the network """
        node_id = agent.position_node
        # TODO: Can be refactored to use agent.family instead of isinstance
        self.G.nodes[node_id][agent.agentFamily].remove(agent)

    def get_node_agents(self, node_id):
        raise 'NotImplementedError'
        # return self.G.nodes[node_id]["Agent"]


class StagedAndTypedTime(time.BaseScheduler):
    agents_by_type: defaultdict[str, dict]
    """A dictionary of all the agents organised by type
    Contains a key for each agent type and as a value contains a dictionary of all agents of that type 
    The nested dictionary is keyed by the agent's unique ID and the value is the respective agent object"""

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_type = defaultdict(dict)

    def add(self, agent: MinimalAgent) -> None:
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.

        Parameters
        ----------
        agent : MinimalAgent
        """
        super().add(agent)
        agent_class: str = agent.agentFamily
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent: MinimalAgent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class: str = agent.agentFamily
        del self.agents_by_type[agent_class][agent.unique_id]

    def trigger_agent_action_by_type(self, agent_type: str, agent_action: str, agent_filter=None, **kwargs):
        if agent_filter is None:
            def agent_filter(test_agent):
                return True

        for agent in self.agents_by_type[agent_type].values():
            if agent_filter(agent):
                getattr(agent, agent_action)(**kwargs)

    def find_agent_by_id(self, agent_id):
        agent_type = agent_id.split("_")[0]
        return self.agents_by_type[agent_type][agent_id]


class ExtendedDataCollector(DataCollector):

    def __init__(self, model_reporters=None, agent_reporters=None, tables=None):
        """Instantiate a DataCollector with lists of model and agent reporters by type.

        Args:
            model_reporters: Dictionary of reporter names and attributes/funcs
            agent_reporters: Dictionary of reporter names and attributes/funcs.
            tables: Dictionary of table names to lists of column names.

            Model reporters can take four types of arguments:
            lambda like above:
            {"agent_count": lambda m: m.schedule.get_agent_count() }
            method with @property decorators
            {"agent_count": schedule.get_agent_count()
            class attributes of model
            {"model_attribute": "model_attribute"}
            functions with parameters that have placed in a list
            {"Model_Function":[function, [param_1, param_2]]}

        """
        self.agent_types = agent_reporters.keys()

        self.model_reporters = {}
        self.model_vars = {}

        if model_reporters is not None:
            for name, reporter in model_reporters.items():
                self._new_model_reporter(name, reporter)

        self.agent_reporters = {}
        self._agent_records = {}
        self.agent_attr_index = {}
        self.agent_name_index = {}

        for agent_type in self.agent_types:
            self.agent_reporters[agent_type] = {}
            self._agent_records[agent_type] = {}
            self.agent_name_index[agent_type] = {}
        self.tables = {}

        for agent_type in self.agent_types:
            if agent_reporters[agent_type] is not None:
                for name, reporter in agent_reporters[agent_type].items():
                    self.agent_name_index[agent_type][name] = reporter
                    self._new_agent_reporter(name, reporter, agent_type)

        if tables is not None:
            for name, columns in tables.items():
                self._new_table(name, columns)

    def _new_agent_reporter(self, name, reporter, agent_type):
        """Add a new agent-level reporter to collect.

        Args:
            name: Name of the agent-level variable to collect.
            reporter: Attribute string, or function object that returns the
                      variable when given a model instance.

        """
        if type(reporter) is str:
            attribute_name = reporter
            reporter = partial(self._getattr, reporter)
            reporter.attribute_name = attribute_name
        self.agent_reporters[agent_type][name] = reporter

    def _record_agents(self, model, agent_type):
        """Record agents data in a mapping of functions and agents."""
        rep_funcs = self.agent_reporters[agent_type].values()
        if all([hasattr(rep, "attribute_name") for rep in rep_funcs]):
            prefix = ["model.schedule.steps", "unique_id"]
            attributes = [func.attribute_name for func in rep_funcs]
            self.agent_attr_index[agent_type] = {k: v for v, k in enumerate(prefix + attributes)}
            if len(self.agent_attr_index[agent_type]) == 0:
                raise ValueError("No agent attributes to record.")
            get_reports = attrgetter(*prefix + attributes)

        else:
            def get_reports(agent):
                _prefix = (agent.model.schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in rep_funcs)
                return _prefix + reports

        agent_records = map(get_reports, model.schedule.agents_by_type[agent_type].values())
        return agent_records

    def collect(self, model):
        """Collect all the data for the given model object."""
        if self.model_reporters:

            for var, reporter in self.model_reporters.items():
                # Check if Lambda operator
                if isinstance(reporter, types.LambdaType):
                    self.model_vars[var].append(reporter(model))
                # Check if model attribute
                elif isinstance(reporter, partial):
                    self.model_vars[var].append(reporter(model))
                # Check if function with arguments
                elif isinstance(reporter, list):
                    self.model_vars[var].append(reporter[0](*reporter[1]))
                else:
                    self.model_vars[var].append(self._reporter_decorator(reporter))

        for agent_type in self.agent_types:
            if self.agent_reporters[agent_type]:
                agent_records = self._record_agents(model, agent_type)
                self._agent_records[agent_type][model.schedule.steps] = list(agent_records)

    def get_agent_vars_dataframe(self, agent_type):
        """Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        """
        all_records = itertools.chain.from_iterable(self._agent_records[agent_type].values())
        rep_names = [rep_name for rep_name in self.agent_reporters[agent_type]]

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID"] + rep_names,
        )
        df = df.set_index(["Step", "AgentID"], drop=True)
        return df