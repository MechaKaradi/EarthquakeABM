from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, NetworkModule
from mesa.visualization.UserParam import UserSettableParameter

import math


from .model import MinimalModel
from .agents import MinimalAgent


def network_portrayal(G):
    # The model ensures there is 0 or 1 agent per node

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 3,
            "color": "#CC0000",
        }
        for each in list(G.nodes)
        ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": "#000000"
        }
        for (source, target) in G.edges()
    ]

    return portrayal

model_params = {
    }

grid = NetworkModule(network_portrayal, 500, 500)

# network = NetworkModule(portrayal_method= agent_portrayal)
# library='d3'
server = ModularServer(MinimalModel, [grid], "Rich Model Visualisation",model_params)
server.port = 8521
