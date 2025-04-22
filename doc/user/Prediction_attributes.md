
# Description of the prediction attributes

## Introduction

In this document, it is going to be explained in detail all the attributes used 
in the final score predictions for every contingency, as well as how each of their 
values are calculated.

## List of the prediction attributes

* **volt_min**: Value of the metric used to predict if there will be a difference
between the minimal voltage values achieved in certain network buses during Hades 
and Dynawo simulations for every contingency. It is calculated by obtaining the 
difference between the minimal voltage value of the network elements and their respective
loadflow value, adding up the difference values of all the contingency specific buses
and also the count of contingency specific buses, and lastly multiplying the value 
with the attribute specific weight.


* **volt_max**: Value of the metric used to predict if there will be a difference
between the maximum voltage values achieved in certain network buses during Hades 
and Dynawo simulations for every contingency. It is calculated by obtaining the 
difference between the maximum voltage value of the network elements and their respective
loadflow value, adding up the difference values of all the contingency specific buses
and also the count of contingency specific buses, and lastly multiplying the value 
with the attribute specific weight.


* **iter**: Value of the metric regarding the number of iterations needed to complete the
Hades simulation. It is calculated by multiplying the number of iterations needed with the 
attribute specific weight.


* **poste**: Value of the metric regarding the number of afected elements in a specific contingency
simulation. It is calculated by multiplying the count of all affected network elements with the 
attribute specific weight.


* **constr_gen_Q**: Value of the metric used to predict if there will be a difference in 
the reactive power values achieved in certain generators that have surpassed or 
fallen below the set limits during the simulation, between the Hades and Dynawo simulations 
for every contingency. It is calculated by obtaining the difference between the starting and 
ending reactive power values of the specific generators, multiplying this value with the 
generator voltage level factor, adding up all the difference values and also the count of 
contingency specific generators, and lastly multiplying the value with the attribute specific weight.


* **constr_gen_U**: Value of the metric used to predict if there will be a difference between the 
Hades and Dynawo simulations, in the voltage values achieved in certain generators that have surpassed or 
fallen below the set limits during the simulation for every contingency. It is calculated by obtaining 
the difference between the starting and ending voltage values of the specific generators, multiplying 
this value with the generator voltage level factor, adding up all the difference values and also the 
count of contingency specific generators, and lastly multiplying the value with the attribute specific 
weight.


* **constr_volt**: Value of the metric used to predict if there will be a difference between the 
Hades and Dynawo simulations, in the voltage values achieved in certain network elements that have 
surpassed or fallen below the set limits during the simulation for every contingency. It is calculated 
by adding up all the 'tempo' values of every network element that has breached the set voltage limit values
after having multiplied this 'tempo' value with the specific element voltage level, and lastly multiplying 
the value with the attribute specific weight.


* **constr_flow**: Value of the metric used to predict if there will be a difference between the 
Hades and Dynawo simulations, in the flow values achieved in certain network elements that have 
surpassed or fallen below the set limits during the simulation for every contingency. It is calculated 
by adding up all the 'tempo' values of every network element that has breached the set flow limit values
after having multiplied this 'tempo' value with the specific element voltage level.


* **node**: Value of the metric regarding the number of blocked nodes after PV-PQ cycling in a specific 
contingency simulation. It is calculated by multiplying the count of all blocked nodes with the 
attribute specific weight.


* **tap**: Value of the metric used to predict if there will be a difference between the Hades and 
Dynawo simulations, in the tap changer data generated during the simulation for every contingency. It 
is calculated by adding up all the differences between the starting and final position of the tap 
changers after having multiplied the difference value with the attribute specific weight.
*Note: this attribute will not be used in the final calculation if the 'tap_changers' option is not 
activated (see the [Tutorial](Tutorial.md) document for further information).*


* **flow**: Value of the metric used to predict if there will be a difference between the maximum 
flow values achieved in certain network line during Hades and Dynawo simulations for every contingency. 
It is calculated by adding up all the maximum flow values achieved in specific lines during the 
contingency, and lastly multiplying the value with the attribute specific weight.


* **coef_report**: Value of the metric of the data regarding the reports from the contingency
simulation. It is calculated by multiplying the count of contingency specific report entries with 
the attribute specific weight.