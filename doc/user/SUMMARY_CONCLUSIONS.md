# ANALYSIS OF RESULTS AND CONCLUSIONS: EBM AND GBR MODELS FOR PREDICTING THE DISCREPANCY BETWEEN STATIC AND DYNAMIC CONTINGENCY SIMULATIONS

## GENERAL INTRODUCTION AND CONTEXT OF THE ANALYSIS

This analysis focuses on the exhaustive evaluation of Simple and Machine Learning (ML) models developed within the "Contingencies Screening" project. The fundamental objective of this project is to efficiently identify those contingencies whose results, obtained through a standard static power flow analysis (Hades), might be inaccurate or insufficient. These "suspicious" contingencies require a more in-depth investigation using more sophisticated power flow simulation tools that incorporate system dynamics and the behavior of controls over time (Dynawo).

The central metric of this study is the "real score" (`REAL_SCORE`). It is crucial to understand that this score **is not a direct indicator of the severity or intrinsic impact of a contingency on the electrical grid.** Instead, the `REAL_SCORE` is a **quantitative measure of the divergence or distance between the results of the static simulation (Hades) and those of the more detailed dynamic simulation (Dynawo)** for the same contingency. Therefore, a high `REAL_SCORE` indicates a significant discrepancy between both solutions, suggesting that the simple static analysis might not be reliable for that particular contingency.

The objective of the models is to predict this `REAL_SCORE` using only variables available from the static simulation (Hades) and contingency characteristics. In this way, the aim is to efficiently identify those contingencies that are likely to show a significant deviation when analyzed with Dynawo, thus optimizing the use of computational and engineering resources.

The global dataset, detailed in `analyze_results_RTE_Global`, covers 1108 snapshots `contg_df.csv` (870,942 valid contingencies). The convergence analysis on this set revealed the following updated rates:
* Convergence only in classic power flow (Hades): 0.211%
* Convergence only in Dynawo: 0.051%
* Convergence in both: 99.736%
* Divergence in both: 0.002%

These figures show a very high convergence rate in both methods (99.736%), which underscores the general robustness of the simulation tools for most cases. Nevertheless, the 0.211% of cases where Hades converges but Dynawo does not (divergence or failure in Dynawo) remains of particular interest. These may represent scenarios where system dynamics, not captured by Hades, lead to stability problems that Dynawo does detect (or attempts to simulate, resulting in divergence). Conversely, the 0.051% where only Dynawo converges indicates situations where Hades fails to find a static solution, while the dynamic simulation succeeds.

It is illustrative to mention some specific observed contingencies:
* The contingency `MANDAL71PEUP5` has shown a tendency to diverge on multiple occasions in simulations with Dynawo. This suggests that this contingency might be associated with complex dynamic phenomena or instabilities that are particularly challenging for the Dynawo engine or that Dynawo correctly identifies as non-convergent from a dynamic perspective.
* On the other hand, contingencies like `.ALGR 7 .LIXH 1`, `.ESC5 7 .FUEN 1`, and `COCHEL71P.AND` have presented divergences several times in Hades simulations. This indicates that the static power flow solver has difficulties with these particular configurations, which could be due to numerical problems, severe boundary conditions, or the inability of the static model to adequately represent the system state under those contingencies. In such cases, the need for a tool like Dynawo becomes even more evident if it manages to converge or provide additional information.

### Distribution of the `REAL_SCORE` and its Relevance for Screening

The analysis of the `REAL_SCORE` (visualized in `analyze_results_RTE_Global`) shows a distribution with a majority of low values. This implies that, for many contingencies, Hades' static solution is reasonably close to Dynawo's dynamic solution. However, the presence of a "long tail" with high `REAL_SCORE` values is the main focus. These contingencies with high discrepancy are those for which the static model is likely a poor representation of the system's real behavior, and therefore, require the more precise analysis of Dynawo, or, at least, should be re-run with it.

### Analysis of `REAL_SCORE` Quantiles

To better understand the distribution of the `REAL_SCORE`, its quantiles are analyzed (as observed in `analyze_results_RTE_Global`). These quantiles reveal how the discrepancy values are distributed:
* For example, the median (0.5 or 50% quantile) of the `REAL_SCORE` is 1480.0, indicating that half of the analyzed contingencies have a discrepancy equal to or lower than this value.
* The 0.75 quantile (75%) is 1860.0, meaning that 75% of contingencies have a `REAL_SCORE` below this threshold.
* However, the upper quantiles show a significant increase: the 0.90 quantile (90%) is 2340.0, the 0.95 quantile (95%) reaches 2800.0.
* This indicates that, although a large proportion of contingencies show moderate discrepancies, a small percentage (the top 10% or 5%) presents considerably higher `REAL_SCORE` values, signaling much more pronounced divergences between static and dynamic simulations. Identifying these contingencies in the upper quantiles is a key objective of the screening.

### Correlation Matrix of Features with `REAL_SCORE`

The correlation matrix (visible in `analyze_results_RTE_Global`) offers a quantitative view of the linear relationships between input features and the `REAL_SCORE`. The most notable correlations with `REAL_SCORE` include:
* **Positive Correlations (examples):**
    * `LOAD_PMAX` (approx. 0.26): Higher total active load in the system correlates with a higher `REAL_SCORE`, suggesting greater discrepancies between Hades and Dynawo under high demand conditions.
    * `GEN_APPARENT_POWER` (approx. 0.22): Similar to `LOAD_PMAX`, higher total apparent generation is also associated with greater discrepancies.
    * `AFFECTED_ELEM` (approx. 0.17): A larger number of elements affected by the contingency tends to correlate with a higher `REAL_SCORE`.
    * `MAX_FLOW` (approx. 0.16): Higher maximum flows in specific lines are also associated with greater discrepancies.
* **Correlations Close to Zero or Negative (examples):**
    * Some features may show very low or slightly negative correlations, indicating a weak or inverse linear relationship with the `REAL_SCORE`. It is important to note that linear correlation does not capture non-linear relationships, which ML models can learn.

These correlations provide an initial indication of which factors might be important for predicting the `REAL_SCORE`, although more complex ML models like EBM and GBR capture more sophisticated interactions.

### Analysis of Contingencies with the Highest Discrepancy in the Global Dataset (Top N with Examples)

By examining the more exhaustive dataset processed by `analyze_results_RTE_Global`, we can identify those contingencies that, across the extensive set of scenarios, consistently show the largest differences between Hades and Dynawo solutions. These represent the cases where the prediction of `REAL_SCORE` is most critical for signaling the potential insufficiency of static analysis.

Below are some examples of contingencies extracted from the top of this ranking (All Time Median Score):
* `MANDAL71MANDA.`: `REAL_SCORE` 34730.0
* `PETRER71PETRE.`: `REAL_SCORE` 24310.0
* `SABINA71SABIN.`: `REAL_SCORE` 20600.0
* `GUADAJ71GUADA.`: `REAL_SCORE` 19860.0

These contingencies, and others with similarly high `REAL_SCORE` values, are prime candidates for detailed dynamic analysis. The specific reasons for such high discrepancies in these particular cases would require an individualized study of each, but they are generally due to a combination of high-impact factors. Some of the reasons for their recurrence may be, for example:
* Contingencies involving the loss of critical elements in heavily loaded areas or with high flows may appear in this category.
* Situations where operational limits are reached and the response of controls (modeled differently in Hades and Dynawo) leads to different final operating points.

Analyzing these contingencies helps to understand the subtleties and specific reasons why the two simulation methodologies can diverge in their final results, even if both reach a stable solution.

## ANALYSIS OF DISCREPANCY PREDICTION MODELS

Two ML models, EBM and GBR, and a simple model, LR, were evaluated to predict the `REAL_SCORE` on a test set (277 snapshots, ~216k contingencies), as detailed in `analyze_results_RTE_EBM` and `analyze_results_RTE_GBR`. The key metric, Mean Absolute Error (MAE), measures the average deviation between the predicted and observed `REAL_SCORE`.

* **Explainable Boosting Machine (EBM)**:
    * MAE (Real score vs Predicted score): 365.93
* **Gradient Boosting Regressor (GBR)**:
    * MAE (Real score vs Predicted score): 343.76
* **Linear Regression (Human Model)**:
    * MAE (Real score vs Predicted score): 390.27

Machine Learning models (GBR and EBM) achieve an MAE in the approximate range of 340-370. Considering that the `REAL_SCORE` can vary significantly, this level of error indicates a useful capability of these models to help distinguish between contingencies with low, medium, and high probability of divergence between Hades and Dynawo simulations. This prediction is valuable for prioritizing the use of dynamic simulations with Dynawo in those cases where they are expected to provide more informational value, i.e., where a high `REAL_SCORE` is anticipated and, therefore, a greater insufficiency of static analysis.

When comparing the predictive performance of the three approaches, the GBR model stands out as the most accurate, closely followed by the EBM. The Linear Regression model, while offering direct interpretability through its coefficients, presents a higher MAE, reflecting a lesser ability to capture the complexity of the relationships that determine the `REAL_SCORE` compared to boosting models.

Focusing on the two best-performing Machine Learning models, although GBR is slightly more accurate than EBM, the difference in their MAEs is relatively small. In this context, the choice of the preferred model should consider other factors besides pure accuracy. **Although the EBM model has a subtly higher MAE than GBR, its main advantage lies in its inherent interpretability.** The EBM is designed to be a "white-box" or "glass-box" model, where the contribution of each feature to the prediction is directly visible and understandable without the need for complex post-hoc techniques (such as those GBR would require for a similar explanation, for example, using SHAP). This ability to offer reasoned explanations behind its predictions is fundamental in critical applications such as power system security analysis, where understanding the "why" of a prediction is as important as the prediction itself. **For this reason, and in line with the project's emphasis on interpretability and the need for models understandable by power system engineers, EBM is the recommended model,** as it offers an excellent balance between robust predictive capability and total transparency.

## DETAILED INTERPRETATION OF THE MODELS

The interpretation of the models is fundamental to understanding which system or contingency characteristics cause Hades and Dynawo solutions to differ more.

### Scatter Plot: Real Score vs. Predicted Score (LR)

The scatter plot presents the `REAL_SCORE` (actual discrepancy) on the Y-axis versus the `PREDICTED_SCORE` (discrepancy predicted by the Linear Regression model) on the X-axis. This graph is generated for contingencies where both Hades and Dynawo simulations converged.

From the visual analysis of the graph, the following observations can be made:
* A general positive trend is observed, indicating that as the `REAL_SCORE` increases, the `PREDICTED_SCORE` also tends to increase. The points are mostly concentrated in the lower-left region of the graph, suggesting that the model tends to correctly predict low discrepancies when they are, in fact, low.
* The cloud of points shows considerable dispersion around the ideal diagonal line (where y=x, which would represent a perfect prediction). This dispersion is noticeably more pronounced compared to that observed in the EBM and GBR models.
* A notable feature is the tendency of the Linear Regression model to underestimate higher `REAL_SCORE` values. For contingencies with high actual discrepancies (high values on the Y-axis), the model's predictions (values on the X-axis) frequently cluster in a range of significantly lower values. This manifests as a large number of points located below the y=x diagonal line, especially in the high `REAL_SCORE` zone, which is relevant but not crucial, as in these situations we are interested in detecting high scores (outliers), with the magnitude of the prediction being less important above a certain threshold.

In summary, while the Linear Regression model shows a basic positive correlation, it suffers from greater dispersion and a tendency to underestimate high discrepancies compared to more complex models like EBM or GBR, as will be seen later.

### Comparison of Histograms: Real Score vs. Predicted Score (LR)
The comparison of the histograms of `REAL_SCORE` and `PREDICTED_SCORE` from the Linear Regression model offers more information about the model's ability to replicate the distribution of discrepancies:

* **Range and Trend:**
    * The `REAL_SCORE` histogram shows a strong right skew, with most values concentrated at the low end (a very high peak near zero) and a long tail extending towards high values, indicating that few contingencies have very large discrepancies.
    * The `PREDICTED_SCORE` histogram from the Linear Regression model also shows a peak at low values and a right skew. However, this peak is noticeably wider and less sharp than that of `REAL_SCORE`. The overall distribution of predictions seems to be more concentrated in a range of low to moderate values.
    * The right tail of the `PREDICTED_SCORE` distribution is significantly less pronounced than that of `REAL_SCORE`. This indicates that the Linear Regression model has difficulty reproducing the frequency and magnitude of the very high discrepancies observed in the real data.

In summary, the Linear Regression model tends to generate predictions with less variability than the actual distribution of discrepancies. While it can capture the general trend that many discrepancies are low, it consistently underestimates the magnitude of the highest `REAL_SCORE` values and fails to replicate the full shape of the distribution, especially its right tail.

### Global Importance of Features (LR) for Predicting Discrepancy

The global interpretation of a Linear Regression model is based directly on its coefficients. Each coefficient represents the change in the `PREDICTED_SCORE` for each unit increase in the corresponding feature, holding all others constant. The sign of the coefficient indicates the direction of this impact (positive or negative), and its absolute magnitude reflects the strength of the feature's linear influence on the discrepancy prediction.

For this specific Linear Regression model, the provided coefficients and intercept are as follows:

* **`INTERCEPTION`**: 658.4221691468623
    * This is the predicted base value for `REAL_SCORE` when all input features are zero (or their reference value if they had been normalized in a particular way before these coefficients were defined).

* **Coefficients of the features (impact on `PREDICTED_SCORE`):**
    Below is a list of features with their respective coefficients, ordered approximately by the absolute magnitude of their impact:

    1.  **`RES_NODE`**: -87.5528375798924
        * Strong negative impact. An increase in the value of this metric (possibly related to residual or problematic nodes in the Hades simulation) is associated with a considerable decrease in the predicted `REAL_SCORE`. This is the feature with the greatest influence in this linear model.
    2.  **`N_ITER`**: 44.57099033305606
        * Strong positive impact. A higher number of iterations (`N_ITER`) required for the convergence of the Hades simulation translates into a significant increase in the predicted `REAL_SCORE`.
    3.  **`CONSTR_VOLT`**: 4.672724978570376
        * Moderate positive impact. More significant voltage problems in elements with constraints (`CONSTR_VOLT`) tend to increase the predicted discrepancy.
    4.  **`AFFECTED_ELEM`**: -4.108740304072612
        * Moderate negative impact. Contrary to what other more complex models might suggest, in this linear model a larger number of elements affected by the contingency (`AFFECTED_ELEM`) is associated with a slight decrease in the predicted `REAL_SCORE`.
    5.  **`CONSTR_GEN_Q`**: 1.8009523189639502
        * Mild positive impact. Constraints on reactive power generation (`CONSTR_GEN_Q`) contribute to a slight increase in the predicted discrepancy.
    6.  **`TAP_CHANGERS`**: 1.5388643917999723
        * Mild positive impact. Greater activity or difference in the position of tap changers (`TAP_CHANGERS`) is associated with a slight increase in the predicted `REAL_SCORE`.
    7.  **`MAX_FLOW`**: 1.290858983927217
        * Mild positive impact. Higher values in the maximum flows metric (`MAX_FLOW`) tend to slightly increase the predicted discrepancy.
    8.  **`CONSTR_FLOW`**: 0.5833716271666991
        * Very mild positive impact. Flow problems in elements with constraints (`CONSTR_FLOW`) have a small positive contribution to the predicted `REAL_SCORE`.
    9.  **`MAX_VOLT`**: 0.4960433568494613
        * Very mild positive impact. Higher values in the maximum voltage metric (`MAX_VOLT`) slightly increase the predicted discrepancy.
    10. **`MIN_VOLT`**: -0.2817229512691065
        * Very mild negative impact. Higher values in this minimum voltage metric are associated with a very small decrease in the predicted `REAL_SCORE`.
    11. **`COEF_REPORT`**: -0.0007074262569540246
        * Practically null impact. The coefficient is very close to zero.
    12. **`CONSTR_GEN_U`**: -3.517186542012496e-13
        * Practically null impact (numerically zero).

### Scatter Plot: Real Score vs. Predicted Score (EBM)

From the visual analysis of the graph, the following observations can be made:
* Ideally, the points should align along the y=x diagonal, indicating a perfect prediction.
* A concentration of points is observed in the lower-left region, meaning the model tends to correctly predict low discrepancies as low.
* There is a general positive correlation; as `REAL_SCORE` increases, `PREDICTED_SCORE` also tends to increase.
* However, there is visible dispersion, especially for higher values of `REAL_SCORE`. In many cases with high `REAL_SCORE`, the EBM model may slightly underestimate the magnitude of the discrepancy (points fall below the y=x diagonal).
* As is normal, some outliers are visible, where the prediction deviates notably from the actual value.

### Comparison of Histograms: Real Score vs. Predicted Score (EBM)

Comparing the histograms of `REAL_SCORE` and `PREDICTED_SCORE` helps to understand if the EBM model captures the general distribution of discrepancies.
* **Range and Trend:**
    * Both distributions cover a similar range of scores.
    * Visually, the EBM model seems to capture the general trend that most scores are low.

### Global Importance of Features (EBM) for Predicting Discrepancy
The main features that, according to the EBM, contribute to a higher `REAL_SCORE` (greater discrepancy) are (see `analyze_results_RTE_EBM`):

Okay, here is the completed section, summarized to approximately 20%, maintaining Markdown format and with a single paragraph for each list item. I have incorporated explanations and possible electrical or load flow-related hypotheses to give them meaning.

Markdown

### Global Importance of Features (EBM) for Predicting Discrepancy

The global explanation of the EBM model reveals the influence of individual features and interactions on the `REAL_SCORE` prediction. The most influential ones according to the provided list and graph visualization are described below:

1.  **`CONSTR_GEN_Q`** (Individual Feature):
    Being the most influential, a possible explanation suggests that scenarios where generators operate near their excitation limits, whose detailed control responses (like limiter actions) are better captured by Dynawo than by Hades, lead to high discrepancy.

2.  **`TAP_CHANGERS`** (Individual Feature):
    This metric, related to tap changer activity, has a non-monotonic impact on `REAL_SCORE`. One hypothesis is that this intermediate activity reflects active and complex voltage regulation scenarios. In these cases, the detailed modeling of tap logic and timing in Dynawo can differ significantly from the simplifications of static flow, which does not capture the sequence or response speed of these control devices.

3.  **`CONSTR_VOLT`** (Individual Feature):
    The contribution of `CONSTR_VOLT` (metric of voltage problems in constraints) to `REAL_SCORE` increases notably with the metric's value. This suggests that voltage stress already visible in Hades' static analysis predisposes to greater discrepancies, as the dynamic responses of loads, generation, and other controls (better modeled in Dynawo) become more determinant and can lead to a different system state.

4.  **`MAX_FLOW`** (Individual Feature):
    `MAX_FLOW` (metric of maximum flows in the network) shows an increasing positive contribution to `REAL_SCORE` as its value increases. This could happen because higher line loadings, and thus smaller operating margins, increase the probability of discrepancies. The system might be closer to thermal or stability limits where dynamic effects (such as protection actions or load responses to voltage/frequency variations) are more differentiating between Dynawo and Hades.

5.  **Interaction: `CONSTR_GEN_Q & TAP_CHANGERS`**:
    This is the first prominent interaction in the list, underscoring that the EBM model considers the combination of these two features an important predictor. Specific combinations of reactive power constraints on generators (`CONSTR_GEN_Q`, especially in its sensitive range) along with tap changer activity (`TAP_CHANGERS`, possibly in its intermediate activity range where they contribute most individually) likely generate high contributions to `REAL_SCORE`. This points to complex voltage regulation scenarios where the temporal response and possible coordination (or lack thereof) between these controls, which Dynawo can capture but Hades cannot, are crucial for the final system state.

6.  **`N_ITER`** (Individual Feature):
    A higher number of iterations in Hades (`N_ITER`) tends to increase the predicted `REAL_SCORE`, indicating that more difficult convergences in static analysis are associated with greater discrepancies. This is theoretically plausible, as a high number of iterations can indicate that the load flow case is ill-conditioned, close to solution limits, or that system non-linearities are particularly pronounced. In such "tense" conditions, it is more likely that dynamic phenomena or detailed control responses, modeled in Dynawo, will lead to a significantly different result than Hades eventually reaches.

7.  **Successive Interactions (Examples: `CONSTR_VOLT & TAP_CHANGERS`, `N_ITER & TAP_CHANGERS`, `N_ITER & CONSTR_GEN_Q COEF_REPORT`, etc.)**:
    The importance of multiple interactions at the TOP of the list highlights EBM's ability to explicitly model how one feature's effect on `REAL_SCORE` is conditioned by others. These interactions capture synergies and non-additive effects; for example, the impact of a high `N_ITER` might be magnified if combined with significant `TAP_CHANGERS` activity or if it coincides with `CONSTR_GEN_Q` in its critical range and a high `COEF_REPORT`. EBM models these complex dependencies that better reflect the interconnectivity of phenomena in a power system, overcoming the limitations of models that only consider individual effects.

8.  **`AFFECTED_ELEM`** (Individual Feature):
    A larger number of `AFFECTED_ELEM` (elements affected by a contingency) tends to increase the predicted `REAL_SCORE`. Although its individual impact might be moderate in this specific EBM model configuration compared to other features or interactions, its contribution is logical: more extensive disturbances involving more network components are inherently more complex and have greater potential to trigger propagated dynamic responses or cascading effects that a static analysis would not capture with the same fidelity as Dynawo, thus generating greater discrepancies.

9.  **`RES_NODE`** (Individual Feature):
    An increase in `RES_NODE` (metric related to problematic or "stuck" nodes in the Hades simulation) raises the contribution to `REAL_SCORE` up to a certain point. This might suggest that difficulties in the static solution, reflected by a higher number of `RES_NODE`, are a good indicator that there will also be significant discrepancies with the dynamic solution.


### Local Explanations (EBM)
In addition to the global explanation, the EBM model is especially powerful for generating local explanations, i.e., for understanding how the `REAL_SCORE` prediction for an individual contingency is reached. The Jupyter notebook `analyze_results_RTE_EBM` contains the specific tools and code for this task. For each contingency, a detailed breakdown can be obtained of how the value of each feature for that particular contingency has contributed (positively or negatively) to the predicted `REAL_SCORE`. This allows for an extremely valuable case-by-case analysis, identifying the exact factors that, according to the model, cause a high (or low) discrepancy to be expected for a given contingency. This functionality is crucial for engineers to trust and act on the model's predictions.

### Scatter Plot: Real Score vs. Predicted Score (GBR)

The scatter plot "Real score vs Predicted score for both status" for the GBR model presents a similar picture to that of the EBM:
* Concentration of points in the lower-left zone (low discrepancies well predicted).
* General positive correlation between `REAL_SCORE` and `PREDICTED_SCORE`.
* Slight dispersion is also observed, and similar to EBM, the GBR model tends to underestimate the highest `REAL_SCORE` values (points below the y=x diagonal).

### Comparison of Histograms: Real Score vs. Predicted Score (GBR)

The comparison of `REAL_SCORE` histograms reveals:
* **Range and Trend:**
    * GBR predictions cover a similar range of scores.
    * The GBR model, like the EBM, seems to capture the general trend of the `REAL_SCORE` distribution. Corresponding boxplots can offer a more precise comparison of medians and dispersions.

### Global Importance of Features (SHAP for GBR) for Predicting Discrepancy

The GBR, interpreted with SHAP (Shapley Additive exPlanations) – a methodology that calculates the contribution of each feature to the specific prediction of an instance, helping to understand how the model arrives at a particular result – corroborates and expands the understanding of factors leading to discrepancies. Although effective, SHAP is a post-hoc interpretability technique, applied after the model (considered more of a "black box" than EBM) has been trained.

The most influential features (`analyze_results_RTE_GBR`) are consistent with EBM. This coherence between models reinforces the idea that these features are robust indicators of conditions where static and dynamic simulations tend to diverge.

The SHAP dependency plots (`analyze_results_RTE_GBR`) illustrate how the GBR model has learned these relationships:

1.  **`CONSTR_GEN_Q`**:
    According to the SHAP dependency plot for `CONSTR_GEN_Q`, this feature shows non-linear behavior. SHAP values are close to zero for most of the `CONSTR_GEN_Q` range. However, for a specific interval of positive `CONSTR_GEN_Q` values (approximately between 0 and 500), SHAP values become significantly positive. This suggests that not just any reactive power constraint is a strong predictor of discrepancy, but rather certain moderate levels of this metric. Electrically, this could indicate situations where generators are close to but have not yet fully saturated their reactive power limits, and the dynamic response of excitation controls or coordination with other voltage regulation devices (which Dynawo models) differs substantially from the assumptions of a static load flow.

2.  **`TAP_CHANGERS`**:
    The SHAP dependency plot provided for `TAP_CHANGERS` in the GBR model shows a linear and distinct relationship between the tap changer activity metric and its impact on `REAL_SCORE` prediction. For low `TAP_CHANGERS` values, SHAP values are low, close to zero, or slightly negative, indicating minimal contribution or even a slight decrease in predicted discrepancy. As `TAP_CHANGERS` activity increases, SHAP values significantly increase. This suggests that moderate or high activity is associated with greater discrepancy between Hades and Dynawo, similar to the explanation obtained by GBR.


3.  **`MAX_FLOW`**:
    The SHAP dependency plot for `MAX_FLOW` (page 14 of `analyze_results_RTE_GBR.pdf`) indicates a general positive and somewhat linear trend: as the `MAX_FLOW` metric (related to maximum flows in the network) increases, the SHAP value also tends to increase. This is consistent with the idea that systems with higher power flows operate with smaller safety margins. Under these more stressed conditions, dynamic effects (such as protection responses, load dynamics, or post-contingency flow redistribution that Dynawo can simulate over time) become more critical and can lead to results that differ from static flow predictions.

4.  **`N_ITER`**:
    For `N_ITER` (number of iterations in Hades), the SHAP dependency plot shows that, in general, a higher number of iterations correlates with higher and positive SHAP values. This suggests that contingencies that are numerically more difficult to solve for static load flow (requiring more iterations) often correspond to electrically tense or ill-conditioned situations. In these cases, it is more likely that the inherent simplifications of static analysis do not adequately capture the system's actual response, which would be reflected by a more complete dynamic simulation like Dynawo's.

5.  **`CONSTR_VOLT`**:
    The SHAP dependency plot for `CONSTR_VOLT` reveals a clear positive trend. As the `CONSTR_VOLT` metric (indicative of the severity of voltage problems in constrained elements) increases, its SHAP value also significantly increases. This indicates that significant voltage violations, already identified in static analysis, are strong predictors of high discrepancy. The reason is that, in the face of severe voltage problems, the response of the system's dynamic controls and the behavior of voltage-dependent loads become crucial, and these are modeled in much greater detail in Dynawo.

6.  **`COEF_REPORT`**:
    For `COEF_REPORT`, representing a metric from reports generated by the Hades simulation, the SHAP dependency plot does not show a simple and unequivocal relationship with the predicted `REAL_SCORE`, leading to the conclusion that "there are no clear conclusions" about its isolated impact.

7.  **`RES_NODE`**:
     The SHAP dependency for `RES_NODE`, which measures problematic nodes in Hades, indicates that an increase in this metric is generally associated with more negative SHAP values.


8.  **`AFFECTED_ELEM`**:
   In the GBR's SHAP analysis for `AFFECTED_ELEM`, two clear groups of differential impact on the predicted `REAL_SCORE` are observed. This relevance of `AFFECTED_ELEM` conceptually aligns with that given by the LR.

9.  **`CONSTR_FLOW`**:
    The SHAP dependency plot for `CONSTR_FLOW` shows a general positive trend, but it is less defined and exhibits notable point dispersion, especially for low and intermediate values of the feature. Therefore, it is considered that it "does not provide much clear signal" across its entire range, although very high values of `CONSTR_FLOW` do tend to be associated with more positive SHAP values.


10. **`MAX_VOLT`** AND **MAX VOLT**:
    The SHAP dependency for `MAX_VOLT` and for `MIN_VOLT` shows that high values of this metric (indicating significant overvoltages or undervoltages with respect to a reference) are strongly associated with high positive SHAP values. Overvoltages or undervoltages often require the intervention of dynamic controls whose detailed modeling in Dynawo can lead to important differences from the final state predicted by Hades.


11. **`CONSTR_GEN_U`**:
    This feature does not provide a significant signal for predicting `REAL_SCORE` in this GBR model, implying that the model has not found a strong or consistent relationship between this metric and the discrepancy between simulators.


## GENERAL CONCLUSIONS AND PERSPECTIVES ON THE MODELS

1.  **Validation of the Screening Methodology**: EBM and GBR models have proven to be effective tools for predicting `REAL_SCORE`, i.e., the expected magnitude of divergence between static (Hades) and dynamic (Dynawo) simulation solutions. This allows for intelligent prioritization, focusing more costly dynamic simulation resources on contingencies where static analysis is predicted to be less reliable. The Human Model (Linear Regression), while providing a transparent framework for screening and being inherently interpretable, has shown lower predictive accuracy (MAE of 390.27) compared to EBM (365.93) and GBR (343.76), which might make it less effective for fine-grained prioritization of the most critical contingencies, though it remains useful as a simple baseline model.

2.  **Identification of Conditions Leading to Discrepancies**: EBM and GBR Machine Learning models consistently identify a coherent set of features (such as high levels of load/generation, the extent of the disturbance, and proximity to voltage/flow operational limits) as predictors of a higher `REAL_SCORE`. This suggests that under these conditions, the simplifications of static analysis are more prone to be insufficient. The Human Model (Linear Regression), depending on the specific configuration of its coefficients, may highlight different factors or capture their effects in a strictly linear manner, offering a complementary but potentially less nuanced perspective of these complex conditions.

3.  **Deepening the Understanding of Static Analysis Limitations**:
    * The **EBM** offers high transparency, graphically showing how each individual factor and second-order interactions contribute to the discrepancy prediction.
    * The **GBR with SHAP** quantifies the contribution of each feature to specific predictions and allows visualization of complex relationships, helping to understand under what particular circumstances divergence is expected to be greater.
    * **Linear Regression (Human Model)** is interpretable through its coefficients, which offer a direct view of the linear weight assigned to each feature. However, its ability to model complex non-linear relationships and subtle interactions between features is limited compared to EBM or GBR, which may restrict the depth of understanding of the causes of more complex discrepancies that do not follow a strictly additive or linear pattern.

4.  **Operational and Planning Implications for the Use of Simulation Tools**:
    * **Prioritization of Detailed Studies**: The primary use of predictive models is the informed selection of contingencies requiring dynamic analysis with Dynawo. Given their higher accuracy, EBM and GBR models are more suitable for this fine-grained prioritization task than the Linear Regression model.
    * **Improved Confidence in Results**: For contingencies with a low predicted `REAL_SCORE` by EBM or GBR models, greater confidence can be placed in the likelihood that Hades' static analysis results are adequate. This confidence might be lower if based solely on the Linear Regression model's predictions due to its higher MAE.
    * **Resource Efficiency**: By avoiding unnecessary Dynawo simulations (for cases where a low `REAL_SCORE` is consistently predicted by the more accurate models), computational time and engineering effort are saved.

5.  **Synergy between Predictive Models and Physical Analysis of Power Systems**: These predictive models (including the Human Model and ML models) do not replace detailed physical analysis but make it more efficient. They act as an early warning system for "static model reliability." Contingencies with a high predicted `REAL_SCORE` should be analyzed with Dynawo to understand the precise nature of the dynamic response and its implications for system security. The fact that Hades and Dynawo differ (high `REAL_SCORE`) does not automatically imply that the contingency is "worse" in Dynawo; it could also be that Hades is excessively pessimistic in some cases. A high `REAL_SCORE` simply says: "Investigate further with the more precise tool because the simple one is likely not sufficient here."

In summary, this study demonstrates the value of predictive modeling approaches for guiding power system security analysis. Interpretable Machine Learning, especially the EBM model, stands out as a particularly sophisticated tool that balances accuracy and explainability. The Human Model (Linear Regression), although exhibiting lower predictive accuracy in this context, serves as a valuable benchmark due to its simplicity and direct transparency, and can offer a first approximation of the influence of certain variables. By predicting the discrepancy between different modeling depths, these tools collectively enable a more efficient and targeted use of advanced simulations, ultimately improving the quality and reliability of network security studies.
