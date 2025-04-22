
# Description of the contingencies dictionaries

## Introduction

In this document, it is going to be explained in detail the format and all the 
entries and data stored inside the python dictionaries generated from the results 
of the simulation of contingencies in both Hades and Dynawo simulators. Each of 
the dictionaries will be described separately below.

## Hades contingencies dictionary

This dictionary has its keys represent the number of the contingency, and its 
values contain another dictionary with the data regarding said contingency. e.g.

```
    {contingency_number: {contingency_data}}
```

The dictionary that contains each specific contingency's resulting data has 
the following key values:
* **name**: Contains the name of the specific contingency.
* **type**: Numerical value that represents the type of the specific contingency.
* **affected_elements**: List containing the numbers of the network elements affected 
by the contingency.
* **coefrepquad**: Numerical value representing the generator number where the contingency 
originates from.
* **min_voltages**: List of tuples where each entry contains firstly the number of the 
network bus that, during the execution of this contingency, has achieved its minimum voltage 
value compared to the rest of contingencies, and secondly said minimum voltage value.
* **max_voltages**: List of tuples where each entry contains firstly the number of the 
network bus that, during the execution of this contingency, has achieved its maximum voltage 
value compared to the rest of contingencies, and secondly said maximum voltage value.
* **max_flow**: List of tuples where each entry contains firstly the number of the network
line that, during the execution of this contingency, has achieved its maximum flow 
value compared to the rest of contingencies, and secondly said maximum flow value.
* **status**: Numerical value that represents the final state of the contingency simulation. 
Possible values can be as follows:
  * 0: Simulation final state is 'Convergence'.
  * 1: Simulation final state is 'Divergence'.
  * 2: Simulation final state is 'Generic fail'.
  * 3: Simulation final state is 'No computation'.
  * 4: Simulation has been interrupted before reaching a final state.
  * 5: Simulation final state is 'Non-realistic solution'.
  * 6: Simulation final state is 'Power balance fail'.
  * 7: Simulation time has exceeded the Timeout value.
  * 8: Simulation final state is unknown.
* **cause**: Numerical value that represents the cause of the contingency's simulation final 
status value.
* **n_iter**: Numerical value that represents the number of iterations the NR simulation needed 
in order to reach a final state value.
* **calc_duration**: Numerical value that represents the time needed for the contingency 
simulation to reach a final state value.
* **constr_volt**: List of dictionaries containing the constraint data of network elements, the
voltage of which has surpassed or fallen below the set limits during the contingency simulation.
The dictionary has the following entries:
  * *elem_num*
  * *before*
  * *after*
  * *limit*
  * *elem_name*
  * *thresh_type*
  * *tempo*
* **constr_flow**: List of dictionaries containing the constraint data of network elements, the
flow value of which has surpassed or fallen below the set limits during the contingency simulation.
The dictionary has the following entries:
  * *elem_num*
  * *before*
  * *after*
  * *limit*
  * *elem_name*
  * *tempo*
  * *beforeMW*
  * *afterMW*
  * *sideOr*
* **constr_gen_Q**: List of dictionaries containing the constraint data of network generators, the
reactive power value of which has surpassed or fallen below the set limits during the contingency 
simulation. The dictionary has the following entries:
  * *elem_num*
  * *before*
  * *after*
  * *limit*
  * *elem_name*
  * *typeLim*
  * *type*
* **constr_gen_U**: List of dictionaries containing the constraint data of network generators, the
voltage value of which has surpassed or fallen below the set limits during the contingency 
simulation. The dictionary has the following entries:
  * *elem_num*
  * *before*
  * *after*
  * *limit*
  * *elem_name*
  * *typeLim*
  * *type*
* **coef_report**: List of dictionaries containing the data regarding the reports from the contingency
simulation.
* **res_node**: List of dictionaries containing the number of blocked nodes after PV-PQ cycling.
* **tap_changers**: List of dictionaries containing the tap changers data generated from the 
contingency simulation. *Note: this entry will not be present in the dictionary if the 
'tap_changers' option is not activated (see the 
[Tutorial](Tutorial.md)
document for further information).* The dictionary has the following entries:
  * *quadripole_num*
  * *quadripole_name*
  * *previous_value*
  * *after_value*
  * *diff_value*
  * *stopper*
* **final_score**: Value representing, either the contingency's final state if the calculation
cannot converge, or a numerical value representing its severity.


## Dynawo continegncies dictionary

This dictionary has the contingencies' name as keys, and its values are another 
dictionary that contains the specific contingency data. e.g.

```
    {contingency_name: {contingency_data}}
```

The dictionary that contains each specific contingency's resulting data has 
the following key values:
* **status**: Final state of the contingency after its simulation with Dynawo.
Possible values are: ['CONVERGENCE', 'DIVERGENCE'].
* **constraints**: List of dictionaries containing the data regarding the exceeding of
the minimum or the maximum allowed voltage value of a certain model element, or the minimum or 
maximum reactive power value of a generator. 
The element dictionary data contains the following elements:
  * *modelName*: Name of the affected element.
  * *description*: Description of the event affecting the element.
  * *time*: Simulation time when the described event occurred.
  * *type*: Type of the affected network element.
  * *kind*: Type of the event. Values can be: ['USupUmax', 'UInfUmin'].
  * *limit*: Limit allowed value of the specific network element. This entry does not appear in
  the reactive power of the generators entries.
  * *value*: Value achieved during the simulation in the network element. This entry does not appear in
  the reactive power of the generators entries.
* **tap_changers**: Dictionary containing the resulting data of the tap changers that 
have been activated during the simulation. This dictionary splits the tap changers into 
two types:
  * *phase taps*: Dictionary that has as keys the ID of the transformers which the 
  different phase taps are connected to, and as value another dictionary containing the following 
  phase tap data:
    * *lowTapPosition*
    * *tapPosition*
    * *targetDeadband*
    * *regulationMode*
    * *regulationValue*
    * *regulating*
    * *step*: Array of dictionaries containing the data originated from every step the tap has taken
    during the simulation.
  * *ratio taps*: Dictionary that has as keys the ID of the transformers which the 
  different ratio taps are connected to, and as value another dictionary containing the following 
  ratio tap data (*Note: Not every entry of the ratio taps dictionary will contain all the following
  ratio tap data attributes*):
    * *lowTapPosition*
    * *tapPosition*
    * *targetDeadband*
    * *targetV*
    * *loadTapChangingCapabilities*
    * *regulating*
    * *step*: Array of dictionaries containing the data originated from every step the tap has taken
    during the simulation.
* **tap_diffs**: Dictionary containing the resulting data from the difference in tap changers between 
the specific contingency simulation and the no contingency simulation. This dictionary splits the 
tap changers into two types:
  * *phase taps*: Dictionary that has as keys the ID of the transformers which the 
  different phase taps are connected to and are different between the results of the contingency 
  and the no contingency simulations, and as value another dictionary containing the differences 
  in the following phase tap data:
    * *lowTapPosition*
    * *tapPosition*
    * *targetDeadband*
    * *targetV*
    * *loadTapChangingCapabilities*
    * *regulating*
  * *ratio taps*: Dictionary that has as keys the ID of the transformers which the 
  different ratio taps are connected to and are different between the results of the contingency 
  and the no contingency simulations, and as value another dictionary containing the differences 
  in the following ratio tap data:
    * *lowTapPosition*
    * *tapPosition*
    * *loadTapChangingCapabilities*