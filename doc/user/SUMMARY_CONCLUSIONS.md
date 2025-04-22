# GENERAL RESULTS

After executing all contingencies, a general data analysis was performed to assess data distribution, accuracy and robustness. Subsequently, we conducted a more detailed examination, focusing on specific contentious cases, from which we derived conclusive insights.
The results of the comparative study revealed significant differences between the obtained outcomes. These differences fall into two broad categories: (a) over-pessimistic results, where contingencies flagged as critical by power-flow were deemed manageable by DynaFlow; and (b) over-optimistic results, where power-flow indicated stability while DynaFlow uncovered potential issues. One can refer to these two as false positives and false negatives, respectively.

## Descriptive Statistics

The first metric that can be observed when analyzing the data globally is the
number of convergences and divergences that occurred throughout the execution.
The following list shows the percentages:
   - Converge only in classic power-flow: 3.69%
   - Converge only in Dynawo: 0.17%
   - Converge in both: 96.13%
   - Diverge in both: 0.01%

As can be observed, almost all contingencies converge in both power flows. Nevertheless, there is an interesting percentage that diverges in Dynawo and converges in the first quick run (static simulation). These cases are intriguing and have been analyzed in more detail. It was determined that a significant portion of non-convergence is due to modeling issues, but the other part is because the classic power-flow was overly optimistic in some contingencies that were genuinely problematic. These cases need to be detected for re-execution by Dynawo. Additionally, there is also a percentage of cases in which the initial run diverged and must be re-executed with the dynamic simulator.

We can also observe that the solution distance metric follows a fairly normal distribution, which will then serve as one of the features to verify the reliability of the designed models. Furthermore, if we examine the evolution of the metric over time, clear patterns emerge, both at the weekly and hourly levels.

Finally, analyzing the results, it would be reasonable to suggest that when defining a number of contingencies to re-run, these could be approximately 10\%
of the total, as the 90th percentile is 8870. Alternatively, a threshold could be directly defined at this value, and all contingencies above it executed
again. This approach would help prevent a conflictive snapshot from causing many contingencies to be poorly simulated with the classic power-flow, which would not be re-simulated with Dynawo.

## Analysis of shap values

### CONSTR_GEN_Q
An accumulation of points near zero is evident, but drawing definitive conclusions is challenging. Values greater than 0 show a subtle tendency to increase the Shap value, indicating an increase in the final target.

### TAP CHANGERS
There is a noticeable inclination for the Shap value to rise as the variable value increases. This suggests that a higher number of taps changed in the powerflow simulation leads to a more distinct second simulation compared to the first.

### CONSTR_VOLT
In this variable, a pattern emerges where, in most cases, a non-zero variable value corresponds to an increased Shap value. Consequently, there is a greater difference between the dynamic simulation and the powerflow.

### N_ITER
No clear trend is observed for this Shap value concerning the final target. However, it is closely linked to the variable CONSTR_GEN_Q, indicating that higher values of CONSTR_GEN_Q lead to more iterations.

### AFFECTED_ELEM
For the number of elements affected by the contingency, a slight increase in the Shap value or target is noticeable as the variable increases, though some points deviate from this general trend.

### COEF_REPORT
While the GBM model identifies it as an important variable, there are no apparent linear patterns to determine its behavior at first glance.

### MAX_FLOW
Contrary to previous observations, higher values of this parameter are generally associated with Shap values equal to 0, indicating no increase in the final target. It correlates with the MAX_VOLT variable—higher values of one correspond to higher values of the other.

### MIN_VOLT
A clear tendency is observed in this variable—values greater than zero lead to an increase in the Shap value as the characteristic value rises. It correlates with the MAX_VOLT variable, with higher values of one corresponding to higher values of the other.

### MAX_VOLT
A clear tendency is observed in this variable—values greater than zero lead to an increase in the Shap value as the characteristic value rises.

### RES_NODE
No clear signal is observed for values larger than zero in this characteristic. However, for values close to 0, the distance between the results of the two simulations tends to decrease.

### CONSTR_FLOW
For this variable, no clear signal is observed for values larger than zero. However, for values close to 0, it does not contribute significantly to the difference between the dynamic simulation and the larger loadflow.
