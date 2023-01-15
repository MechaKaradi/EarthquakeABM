# Scheduler Logic
This file contains the logic for the scheduler. The scheduler is responsible for controlling the order in which agents take actions in the model. The scheduler is also responsible for calling events that occur at certain times. 

There are 2 approaches to the implementation of the schedule in the model. 
- The first is similar to the built-in method, where the model has a step function which calls the scheduler, a separate definition of the scheduler where the selection and execution of agents is defined, and a separate definition of the events that occur at certain times. The second approach is to have the scheduler and the events in the same definition. 
- The second approach is to have only a step function, and to define the entire logic of the scheduler inside this function.
The pros and cons of this will be evaluated based on the complexity of the order of operations vs the complexity of the internal arrangement of the code. 

Behaviour that are Likely to be impacted by the choice of scheduler implementation:
- Modelling of the network congestion and the impact of the network congestion on the agents' ability to perform actions.
- Whether some agents consistently get to perform actions before others.

## Required Sub-steps in the Scheduler
A generic step of the model should include the following sub-steps:
1. Resolve any external events that will impact the model. e.g. an earthquake occurs, a new agent is added to the model, an agent is removed from the model.
2. Resolve internal model events such as an agent health deteriorating, an agent health improving, a building collapsing etc.
3. Resolve the actions of the agents. This includes the following sub-steps:
   1. The agents evaluate their internal state e.g. "Am I in motion" "Am I injured" "Do I want social attention"
   2. The agents observe their environment "What is the state of the building?" "What is the state of my neighbours?"
   3. The agents decide what actions to take. e.g. "I want to move to a safer location" "I want to help my neighbour" "I want to call for help" "I want to go to the hospital" "I want to make a social call"
   4. The agents initiate the actions. "Make a social call" "Move to a safer location" "Call for help" "Go to the hospital"
   5. The agents are given information based on the results of the action that was performed. 
   6. The agents update their internal state based on the results that they are provided 
4. Resolve the effects of any agent actions on the environment. e.g. "building stabilised", "ambulance dispatched" "Hospital bed occupied" etc.  

## Challenge 1: Modelling congestion
Whether actions resolve successfully depends on counting the number of agents attempting an action. However this invovles first allowing all agents to select an action, then evaluating congestion and then finally passing information back to the agents based on the state of the network.  

## Challenge 2: Agent behaviour 
Remove 
