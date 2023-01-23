I have issues with the current implementation of the Building class in the agents.py file. One of the main issues is that the Citizen class is not defined, causing a NameError when the Building class is defined. This issue can be resolved by either defining the Citizen class in the agents.py file before the Building class, or by importing it from another module before it is used in the Building class, i think is Pierre the responsible for the citizens if i am correct.

Additionally, it is necessary to initialize an instance of the MinimalModel class before creating any building agents. This can be done by adding the following code snippet at the end of the model.py file:

model = MinimalModel() create_building = model.create_agents(Building) for i in range(5): create_building(i)

This will create 5 building agents and add them to the model's schedule.

It is also necessary to check the add_citizen method of the Building class, as it seems that it is not working as intended. The function is attempting to add a citizen object to the building, but the citizen class is not defined and this is causing an error.

Lastly, somebody should also check that the spawn method of the building class is working as intended. This method is responsible for placing the building agents in the grid, and it is not clear if it is working correctly.

Once these issues have been resolved, the model should run without errors and create the desired number of building agents.

Finally, in order to run the script in my own computer i change the path in the line 22 of network_visualiser_v2.py . 