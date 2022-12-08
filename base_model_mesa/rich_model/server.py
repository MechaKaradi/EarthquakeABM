from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, NetworkModule
from mesa.visualization.UserParam import UserSettableParameter

from rich_model.model import MinimalModel 
from rich_model.agents import MinimalAgent


def agent_portrayal(agent):
    if agent is None:
        return
    
    portrayal = {}

    if type(agent) is MinimalAgent:
        portrayal["Shape"] = "circle"
        portrayal["color"] = agent.color
    
    return portrayal

network = NetworkModule(portrayal_method= agent_portrayal)
# library='d3'
server = ModularServer(MinimalModel, [network],"Rich Model Visualisation")
server.port = 8521
