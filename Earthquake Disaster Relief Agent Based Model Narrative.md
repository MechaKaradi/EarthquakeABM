---
BC-link-note:: down
---
# Earthquake Disaster Relief Agent Based Model Narrative

The simulation of the earthquake features the following phenomenon: 
1. Earthquake and Aftershocks - Initial earthquake damages buildings - aftershocks are weaker, but cause further damage and disruption
2. Injuries and Illnesses: Citizens are injured and ill due to earthquakes and exposure. Hospitals heal serious injuries. Residences protect citizens from illnesses. Without care, citizens die from illnesses.
3. Trapped: Damaged buildings increase the rate at which citizens die, and prevent them from leaving voluntarily. 
4. Communications: Information between citizens and agencies is the bottleneck to citizens receiving services - if calls do not connect, services are not dispatched
5. Transportation? : - congestion on streets, road damage, or other causes for the ambulances or rescue vehicles to not be able to reach the citizens? 

The following model elements are necessary to implement these phenomena in the model. 

## Buildings 
**[[Buildings]]** are damaged in the initial tremor and the people in the buildings need rescue, and medical care. 
Each node has 1 ‘building’ agent represents 1 or more related structures in that location. All structures of the same type are treated as the same building.
i.e. a row of G+1 townhouses conceptually located on the same node are treated as 1 building for the model. However a hospital located on the same node is treated as a different building for the calculations and behaviour modeling.

### Houses/Residences
A subset of buildings are treated as Residences- which are locations where citizens can ordinarily live and where they have reliable access to food, services etc. Each residence has a ‘Capacity’ - this  represents the number of citizens that can live there. 


### Housing
Hospitals are [[Buildings]] with the ability to heal people. Hospital will dispatch ambulances to collect injured or sick citizens IF there are beds available in the hospital. If no beds are available, no ambulance will be dispatched. A bed is considered occupied while an injured or sick person is in the bed. Citizens who are more healthy than a certain threshold will be discharged. For each hour that a bed is occupied, the hospital will heal the citizen by a certain amount.
> [!question] Stretch Goal
>Once citizens are at a determined ‘wellness level’ they will be released IF housing is available to them. Otherwise, they will be held until either 100% recovered, or unless housing is allocated. 

## Citizens 
The [[occupants/Citizens]] of the damaged buildings need to have alternative accommodation, or risk illness and injury from exposure to elements, lack of sanitation, etc - presumed that food and water are being taken care of. 

Citizens can become injured due to the earthquake or sick due to exposure from being outdoors - illness and injury can be probability functions that very with level of risk.

Citizens attempt to call for help if they are injured or sick - they will call a hospital and attempt to request assistance - if they do not get confirmation they will keep attempting to call hospitals until they receive confirmation. Once they have confirmation, while waiting for an ambulance they will continue attempting to make calls. 

Citizens that are trapped or seriously trapped will not attempt to make calls. However, other citizens on the same node as them will attempt to call for rescue services. 



## Communications: 
All citizens wish to have information about what is going on. To collect this information they will attempt to make phone calls. 

Each healthy citizen that has not performed a phone call within the last hour will have a probability to attempt a call. If they successfully make a call, their desire to connect resets. If they are not successful, their desire grows and the probability increases. 

Injured/sick citizen attempt to call an ambulance to help them. Each tick, the citizen will attempt to call the hospital. 

Each tick, the system has a finite capacity for calls. If the total number of attempted calls in a tick exceeds the total capacity of the system, no calls are completed. 

