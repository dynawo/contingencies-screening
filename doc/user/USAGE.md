
# Overview

As explained in the project's [README](README.md), this software has two modes
of usage:

  a) The _screening_ mode, where for each contingency calculated by Hades 2
  Security Analysis, we produce a score that predicts how different the results
  are expected to be, compared to DynaFlow's.

  b) The _model-training_ mode, in which we run both Hades and DynaFlow on a
  large set of snapshots and contingency lists to obtain predictive models via
  various supervised learning techniques (linear regression and gradient
  boosting).

# Instructions for running the screening pipeline

## Preparing the input

The input directory can be freely structured, as long as each snapshot is
contained in its own directory, containing both the Hades and the DynaFlow
input. This is a typical example:

```
data/
└── 2023
    └── 01
        └── 02
            ├── unique_snapshot_name_1
            │   ├── donneesEntreeHADES2_ANALYSE_SECU.xml
            │   └── recollement-auto-20230102-0000-enrichi.xiidm
            ...
            └── unique_snapshot_name_N
                ├── donneesEntreeHADES2_ANALYSE_SECU.xml
                └── recollement-auto-20230102-xxxx-enrichi.xiidm
```

Under the main directory, these folders represent the years, months, and days of
the snapshots, and there can be as many as desired at each level. It is not
strictly necessary to organize the folders using these names (in fact, any
directory tree will be traversed), but it is useful for keeping things
organized. What is very important is that each snapshot has its own unique
folder, and to name the files according to the patterns
`donneesEntreeHADES2*.xml` (for Hades 2) and `*.*iidm` (for DynaFlow).

In addition, one would typically provide a specific database of dynamic models
to override Dynawo's default ones (see option `-b`). When providing this, the
database should be structured as follows:

```
dyndb_folder_name
├── assembling.xml
└── setting.xml
```

In this case, the folder name and its location are flexible, but these two
file names (`assembling.xml` and `setting.xml`) must be adhered to precisely.

## Running the pipeline

Make sure to have activated the Python virtual environment in which you have
installed the software: `source /path/to/the/virtual/environment/bin/activate`

The pipeline is run using the command `run_contingencies_screening`:

```
 Usage: run_contingencies_screening [OPTIONS] INPUT_DIR OUTPUT_DIR                                                                                                                      
                                    HADES_LAUNCHER                                                                                                                                      
                                                                                                                                                                                        
 Main execution pipeline for contingency screening using Hades and optionally Dynawo. Processes time-structured directories (YEAR/MONTH/DAY/HOUR).                                      
                                                                                                                                                                                        
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input_dir           DIRECTORY  Path to the folder containing the case files (e.g., .../YEAR/) [default: None] [required]                                                        │
│ *    output_dir          DIRECTORY  Path to the base output folder [default: None] [required]                                                                                        │
│ *    hades_launcher      PATH       Define the Hades launcher (path to executable or command name) [default: None] [required]                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --tap-changers        -t                 Run the simulations with activated tap changers                                                                                             │
│ --replay-hades-obo    -a                 Replay the most interesting contingencies with Hades one by one                                                                             │
│ --replay-dynawo       -d      PATH       Replay the most interesting contingencies with Dynawo. Provide the Dynaflow launcher path. [default: None]                                  │
│ --n-replay            -n      INTEGER    Number of most interesting contingencies to replay (default: 25, use -1 for all) [default: 25]                                              │
│ --score-type          -s      INTEGER    Type of scoring for ranking (1 = human made, 2 = machine learning) [default: 1]                                                             │
│ --dynamic-database    -b      DIRECTORY  Path to a standalone dynamic database folder for Dynawo [default: None]                                                                     │
│ --multithreading      -m                 Enable multithreading for processing time directories in parallel                                                                           │
│ --calc-contingencies  -c                 WARNING: It does not accept compressed results. Assume input dir structure contains pre-calculated contingencies (expects HADES/DYNAWO      │
│                                          subfolders in time dirs)                                                                                                                    │
│ --compress-results    -z                 Clean intermediate files and compress results for each time directory                                                                       │
│ --model-path          -l      DIRECTORY  Manually define the path to the ML model path for predictions [default: None]                                                               │
│ --install-completion                     Install completion for the current shell.                                                                                                   │
│ --show-completion                        Show completion for the current shell, to copy it or customize the installation.                                                            │
│ --help                                   Show this message and exit.                                                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Let us now see these options in detail.


Let us now examine these command-line options in detail:

### `input_dir` (required)

This argument specifies the directory containing the snapshot files. You can provide either an absolute path (the full path from the root of your file system) or a relative path (a path relative to your current working directory). For details on how the input files should be organized within folders and their naming conventions, please refer to the "Preparing the input" section above.

### `output_dir` (required)

This argument defines the directory where the results of the screening process will be stored. Similar to `input_dir`, you can use either an absolute or a relative path. The directory structure created in the output directory will mirror the organization of the `input_dir`.

The software offers the capability to merge results with a previously used output folder. If you provide such a folder:

* If the snapshots being processed in the current run are different from those in the previous execution, the software will intelligently merge the folder structures without overwriting any existing data.
* If the current run includes input cases that were already present in the reused output folder, the software will prompt you on a case-by-case basis, asking whether you want to delete the existing data or stop the execution to avoid accidental overwriting.

### `hades_launcher` (required)

This argument specifies the path to the Hades executable. You can either provide the absolute path to the executable file or ensure that the directory containing the Hades executable is included in your system's `$PATH` environment variable.

### `-t`, `--tap_changers`

This option, when enabled, activates the tap changers during the contingency calculation performed by the Hades 2 Security Analysis.

### `-a`, `--replay_hades_obo`

Enabling this option triggers a replay of the top `N_REPLAY` most interesting contingencies using the Hades 2 simulator, one contingency at a time ("one by one"). This feature was implemented to investigate potential discrepancies that can arise between a full Security Analysis session in Hades 2 and running individual contingency cases as standard power flow calculations. When activated, the software prepares individual Hades input files for the top N ranked contingencies (based on the model's predictions) and executes them separately to analyze any variations in the results compared to the Security Analysis session.

### `-d`, `--replay_dynawo`

This option enables the replay of the top `N_REPLAY` most interesting contingencies using the DynaFlow simulator. When used in conjunction with the `-n` option, the software identifies the top N most divergent contingencies (based on the predictive model) and runs them using the DynaFlow Launcher. This allows for the calculation of their actual score, which represents the true "distance" between the Hades and DynaFlow results. These actual scores are then added to the results dataframe, providing a means to evaluate the accuracy of the predictive model. You must provide the absolute path to the Dynawo launcher executable or ensure that it is included in your system's `$PATH`.

### `-n N_REPLAY`, `--n_replay N_REPLAY` (default value: 25)

This option sets the number of top N contingencies that will be replayed using the `-a` (for Hades) and `-d` (for DynaFlow) options.

**IMPORTANT:** When generating a dataset for training new models, you should use the "magic" value **-1** for `N_REPLAY`. This instructs the software to run **all** contingencies under both DynaFlow and Hades. The resulting dataframe will include the actual score (in the last column), which serves as the target variable for the model training process. This output dataframe is then used as the input for the training script (described in the "Train the new model" section).

### `-s SCORE_TYPE`, `--score_type SCORE_TYPE` (default value: 1)

This option defines the type of scoring mechanism used for ranking the contingencies. The possible values are:

* `1`: Uses a human-explicable scoring method.
* `2`: Uses a machine learning-based scoring method.

### `-b DYNAMIC_DATABASE`, `--dynamic_database DYNAMIC_DATABASE`

This option allows you to specify the path to a custom Dynawo model database that will be used for the DynaFlow simulations. The directory provided with this option **must** contain two files named `assembling.xml` and `setting.xml`.

### `-m`, `--multithreading`

Enabling this option activates multithreaded execution for the Hades 2 simulations, potentially reducing the overall processing time.

### `-c`, `--calc_contingencies`

This option enables the software to use input files where the contingency calculations have already been performed in a previous execution. You need to provide the path to the output directory of that previous run.

**WARNING:** This option is designed to work with uncompressed results and expects a specific directory structure within the time-structured input directories, containing `HADES` and `DYNAWO` subfolders. It will not function correctly with compressed results.

### `-z`, `--compress_results`

When this option is enabled, the software will automatically clean up any unnecessary intermediate data generated during the DynaFlow simulations and then compress the final output folder into a `tar.gz` archive.

### `-l MODEL_PATH`, `--model_path MODEL_PATH`

This option allows you to manually specify the path to a particular machine learning model file that you want the software to use for making predictions. This overrides the default model selection logic.

### `--install-completion`

This option, when used, attempts to install shell completion for the `run_contingencies_screening` command for your current shell environment, potentially making command-line usage more convenient.

### `--show-completion`

This option displays the shell completion script for the `run_contingencies_screening` command for your current shell. You can then copy this output and manually configure shell completion if the `--install-completion` option does not work automatically.

### `--help`

This standard option displays a help message outlining the usage of the `run_contingencies_screening` command and its available arguments and options, and then exits.


# Instructions for running the model-training command

Models can degrade in performance, in what is called model drift / data
drift. For instance, the network scenarios may start to change and become more
stressed in some areas and less in others, compared to the ones used in the
previous train cycle (this would be an example of data drift). Or, changes in
either Hades or DynaFlow Launcher (or Dynawo models) could result in significant new
differences in what they calculate (this would be an example of model drift).
Therefore it may be necessary to re-train the predictive models once in a while.

With the current software, this is done in two stages: first, one must produce a
new dataset of "ground truth" data, that is, a large data set of Hades and
DynaFlow executions with their actual scoring (their distance). Second, one must
perform the model training (regression, etc.) on that dataset.


## Preparing a new train dataset

Before running the train script, one should prepare (calculate) the new training
dataset. This consists in running all contingencies (for all available
snapshots), both in Hades2 SA and in DynaFlow. For example, three months of data
took about 2 months to calculate on a powerful server!

The command to launch this is actually the same as the one used for running the
screening pipeline, except that one should use the options **`-d -n -1`**:

```
run_contingencies_screening -t -d dynawo_launcher -n -1 -b ../dyndb input_dir output_dir
hades_launcher
```

This will execute, both with Hades and Dynawo, _all_ the contingencies
configured in the input snapshots.


## Train the new model

The train script will produce the new models (LR, GBM and EBM). These consist
in one file each, one in “pickle” format (GBM and EBM) and one in readable JSON format
(LR).

Run the train process by launching the following command line:

```
train_test_loadflow_results ./massive_execution_output_folder ./folder_where_to_save_models
```

where `massive_execution_output_folder` is the output folder from the previous
execution.


## Using the newly trained models for screening

**TODO:** currently, the newly created models should be copied to the Python source,
following this manual procedure. This should be managed through configurable
user directories instead.

  1. Rename the `GBR_model.pkl` or the `EBM_model.pkl` file to `ML_taps.pkl` or `ML_no_taps.pkl` and
     then copy it to the package folder
     `src/dynawo_contingencies_screening/analyze_loadflow/`, inside the Python
     sources.

  2. Perform the same action for the `LR_median.json` or `LR_mean.json` file, renaming it to
     `LR_taps.json` or `LR_no_taps.json`.

  3. Reinstall the Python package as in the INSTALLATION instructions.

