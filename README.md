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


# By Claxxius

Agents.py
In the agents.py file, I added the following changes:
New features added
•	Added a new agent class called Citizen which inherits from MinimalAgent and has a health status attribute (healthy or injured).
•	Added a new agent class called Ambulance which inherits from MinimalAgent and can transport one injured Citizen at a time.
•	Citizen class now has a transported attribute that indicates if the citizen is transported or not.
•	Ambulance class has a new method called transport_patient which assigns the patient attribute to the passed citizen and moves the ambulance to the patient's location.
•	Ambulance class has a new method called step which moves the ambulance to the nearest hospital and changes the patient's health status to "treated"
•	Citizen class has a new method called set_health_status which allows to change the health status of the citizen.
Changes in existing classes
•	MinimalAgent class now has a new attribute called transported
•	MinimalAgent class now has a new attribute called is_injured
•	Ambulance class now has a new attribute called patient
How to use the new features
•	To use the new classes and methods, you need to import them first.
•	Once imported, you can create instances of the Citizen and Ambulance classes and add them to the model's schedule.
•	You can use the transport_patient method on the Ambulance class to transport a Citizen object. This method takes a Citizen object as a parameter and assigns it to the patient attribute of the Ambulance object. Additionally, it sets the transported attribute of the Citizen object to True, and moves the Ambulance object to the same location as the Citizen.
•	In the agents.py file, I added the Citizen class and the Ambulance class.
•	In the Citizen class, I added the attributes transported and is_injured. The transported attribute is a boolean that indicates whether the citizen is currently being transported by an ambulance or not. The is_injured attribute is a boolean that indicates whether the citizen has been injured or not.
•	I also added a method called set_health_status that allows to set the health status of the citizen.
•	In the Ambulance class, I added the attribute patient which is a reference to the citizen that the ambulance is currently transporting.
•	I also added a method called transport_patient that allows the ambulance to transport a citizen. This method sets the citizen's transported attribute to True and updates the ambulance's patient attribute to reference the citizen.
•	Finally, I modified the step method of the Ambulance class to move the ambulance to the closest hospital when it has a patient on board and update the health status of the patient to "treated" and sets the patient attribute to None.
In the model.py file, I added the following changes:
New features added
•	Created a new class Ambulance which inherits from the MinimalAgent class.
•	The Ambulance class has an attribute patient which is initially set to None.
•	The Ambulance class has a method transport_patient which takes in a Citizen object as an argument and assigns it to the patient attribute. The transported attribute of the Citizen object is set to True. The ambulance's position is set to the patient's position.
•	The Ambulance class has a step method which, if there is a patient, moves the ambulance to the closest hospital and sets the patient's health_status attribute to "treated" and sets the patient attribute to None.
•	In the Citizen class, I added an attribute transported which is initially set to False.
I also added a way to create instances of the Citizen and Ambulance classes and add them to the model's schedule in the create_agents method in the MinimalModel class.
It will need to import the Citizen class from the agents.py file and create an instance of the Ambulance class and add it to the model's schedule in the main script that runs the model.
Also can use the transport_patient method on the Ambulance class to transport a Citizen object to the closest hospital in the simulation.
