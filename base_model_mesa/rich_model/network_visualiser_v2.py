"""
Network Visualization Module
============

Module for rendering the network, using [d3.js](https://d3js.org/) framework.

"""
from mesa.visualization.ModularVisualization import VisualizationElement, D3_JS_FILE


class NetworkModule_xy(VisualizationElement):
    package_includes = []

    def __init__(
            self,
            portrayal_method,
            canvas_height=500,
            canvas_width=500,
    ):
        self.package_includes = [D3_JS_FILE]
        NetworkModule_xy.local_includes = ["NetworkModule_xy_d3.js"]
        NetworkModule_xy.local_dir = "C:/Users/bonro/OneDrive - Delft University of Technology/COSEM/Q2 2022 2023/SEN1211 Agent-based Modelling/Claxxius/EarthquakeABM-1211/base_model_mesa/rich_model"
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = f"new NetworkModule_xy({self.canvas_width}, {self.canvas_height})"
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        return self.portrayal_method(model.G)
