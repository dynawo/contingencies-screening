# Instructions for running the analysis notebooks

## Prepare the environment

0. Activate the virtual environment with the command:  
   `source /path/to/the/virtual/environment/bin/activate`

1. [ONLY OLDER VERSIONS, NOT NEEDED ANYMORE] Configure the Jupyter kernel with
   the command:  
   `python -m ipykernel install --user --name=myenv`

2. Start the Jupyter lab server:  
   `jupyter lab`

3. Open your browser with the specified URL (this may have been already launched
   automatically for you), and open one of the notebooks located under
   `src/dynawo_contingencies_screening/notebook_analysis`. It can be either
   `analyze_results_RTE.ipynb` or `analyze_individual_contg_RTE.ipynb`.
  
4. [ONLY OLDER VERSIONS, NOT NEEDED ANYMORE] Once this is done, you should
   select the ipykernel that you created on step 1 following this image:
   [KERNEL](https://i.stack.imgur.com/TF5uW.png)


## Execute analyze_results_RTE

For this notebook, all you need is a successful execution of the code pipeline,
which should include having also replayed with DynaFlow a sufficient number of
screened snapshots. You only have to define the path to the output folder of the
mentioned execution, and make sure to run all the notebook cells in the
specified order.


## Execute analyze_individual_contg_RTE.ipynb

To be able to run this notebook, you need to manually prepare the individual
case in question. To do this, you should take a specific snapshot and select,
using the contingency definition list within the DynaFlow (`contng.json`) and Hades
(`donneesEntreeHADES2_ANALYSE_SECU.xml`) files, a single contingency (the same in
both files!). Once this is done, both cases should be manually executed with
their respective launchers. The final folder structure should resemble the
following:

```
recollement-auto-20230615-0500-replay
├── dynawo
│   ├── config.json
│   ├── contng.json
│   ├── Dynaflow_Output
...
│   │   ├── outputs
│   │   │    ├── finalState
│   │   │    │   ├── outputIIDM.xml
│   │   │    │   └── outputState.dmp
...
│   └── recollement-auto-20230615-0500-enrichi.xiidm
└── hades
    ├── donneesEntreeHADES2_ANALYSE_SECU.xml
...
    └── output.xml
```

To run the notebook, all you have to do is define the path to output execution
files (Hades --> `output.xml`, DynaFlow --> `outputs/finalState/outputIIDM.xml`)
and execute all the notebook cells in order.

