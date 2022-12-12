from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from .network_visualiser_v2 import NetworkModule_xy
from mesa.visualization.UserParam import UserSettableParameter

import math


from .model import MinimalModel
from .agents import MinimalAgent


def network_portrayal(G):
    # The model ensures there is 0 or 1 agent per node

    def get_edge_list(G):
        return [
            {
                'source': G.nodes[edge[0]],
                'target': G.nodes[edge[1]],
                'edge_attributes': G.get_edge_data(edge[0], edge[1])
            }
            for edge in G.edges()
        ]

    edge_list = get_edge_list(G)

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "id": n[0],
            "x": n[1][0] -24000,
            "y":n[1][1] -5200,
            "size": 10,
            "color": "#CC0000",
        }
        for n in G.nodes.data('pos')
        ]

    portrayal["edges"] = [
        {
            "sourcex": edge['source']['pos'][0] -24000,
            "sourcey": edge['source']['pos'][1] -5200,
            "targetx": edge['target']['pos'][0] -24000,
            "targety": edge['target']['pos'][1] -5200,
            "width": 5,
            "color": "#000000"
        }
        for edge in edge_list
    ]

    return portrayal

model_params = {
    }

grid = NetworkModule_xy(network_portrayal, 500, 500)

# network = NetworkModule(portrayal_method= agent_portrayal)
# library='d3'
server = ModularServer(MinimalModel, [grid], "Rich Model Visualisation",model_params)
server.port = 8521
