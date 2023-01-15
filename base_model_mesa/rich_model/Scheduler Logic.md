# Scheduler Approach
This file contains the logic for the scheduler. The scheduler is responsible for controlling the order in which agents take actions in the model. The scheduler is also responsible for calling events that occur at certain times. 

There are 2 approaches to the implementation of the schedule in the model. 
- The first is using the building in staged activation scheduler. This would consolidate and make the order of operations easier to control.  
- The second approach is to create a unique scheduler for each "phase" of the model and then inside the 'tick' of the model, to call each scheduler in turn as the turn order progresses. However, this means that each function would have to be careful to correctly extend the scheduler class.

Need to evaluate which option is best. 

Behaviour that are Likely to be impacted by the choice of scheduler implementation:
- Modelling of the network congestion and the impact of the network congestion on the agents' ability to perform actions.
- Whether some agents consistently get to perform actions before others.

## Challenge 1: Modelling congestion
Whether actions resolve successfully depends on counting the number of agents attempting an action. However this invovles first allowing all agents to select an action, then evaluating congestion and then finally passing information back to the agents based on the state of the network.  

## Challenge 2: Order of execution: Random or consistent
Does the 'queue' in which congestion and agent order resolve remain the same round to round or does it change randomly? 

## Challenge 3: Multi tick actions
Each action that takes more than 1 tick will need to be modelled as a state, or by changing how different schedulers load and discharge their step routines.

# Requirements

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

**Summary:**
1. External Events 
2. Internal Events 
3. Observation
4. Decision 
5. Action 
6. Resolution

## External Events 
1. Tremor (Earthquake) -> Consequence chain
When the initial earthquake or an aftershock 'hits' the event is triggered at the top of the execution chain.

2. 


## Internal Events

## Observation

## Decision

## Action

## Resolution
