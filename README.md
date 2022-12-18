# EarthquakeABM-1211
Repository for the development of Earthquake disaster response simulation using ABM and MESA
For SEN1211

## Directory Structure

The folder `base_model_mesa` contains the main utilities and code used to set up the base-line model provided by the course instructors. 
The `make_network_object` notebook imports the list of coordinates in the `csv` files and uses the information to build a NetworkX graph object, and then save that object as a .data file. That .data file is treated as a static input for the creation of the graph object inside the model. 

The `run.py` file can be executed to activate a visualization of the model in tornado server. To understand the visualization, follow mesa docs.

Inside the `rich-model` folder is the main model folder. The `agents.py` file currently contains the minimal agent specification, and any other agents dependent on it. This may need to be refactored into static and mobile agents separately. 
The `model.py` file loads the model and model behaviour. 

The `test_model.ipynb` file can be used to run and interact with the model while the visualization elements of the model have not been fully fleshed out. 

# TO-DO
- Create the init function to assign a building to every node
- Create a function to dynamically allocate capacity to all buildings
- Create a function to make some % of buildings start in a damaged condition.
- Give each buidling a `tremor` function that accepts a `magnitude` as an input and outputs an int [0,2] based on a probability distribution, then adds that number to its status.
- In the Model Step function create an `event` called `earthquake`. Earthquake has a function that assigns it a `magnitude` property. When the earthquake event takes place, the tremor function is called in all buildings. 