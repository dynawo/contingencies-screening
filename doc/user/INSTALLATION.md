
# Instructions for installing the Contingency Screening software

This software is fully written in Python, designed to be installed as a standard Python
package using `pip` under a Python virtual environment.  However, it also needs Hades 2
(RTE's static powerflow) and DynaFlow launcher.

The steps to install the whole system, in outline form:

  * Install Hades 2 and DynaFlow launcher
  
  * Install the Operating System requirements (a basic Python system)
  
  * Create a Python virtual environment
  
  * Install the package under the Python virtual environment


We now explain each step in more detail.


## Install Hades 2 and DynaFlow Launcher

You will need recent versions of both Hades 2 and DynaFlow Launcher.  During the
development of this project we used the following versions:

  * **DynaFlow Launcher 1.7.0 (rev:HEAD-0e19760)**  

  * **Hades 2 V6.9.1.8**  

You can either have their respective launchers accessible through your $PATH, or provide
the full path when you use the screening/training pipeline, it's your choice (although
having them set up in your $PATH will make for much shorter command-line invocation).


## Install the Operating System requirements

We only need a basic Python system (version 3.12 and higher), so that we can later create
the Python virtual environment under a non-root account.

If you are on Debian, Ubuntu, or any other Debian derivative, this means making sure
that you have packages python3-minimal, python3-venv, and python3-pip:

```
apt install python3-minimal python3-venv python3-pip
```

Verify that you have a Python version 3.12 or higher, by running:
```
python --version
```


## Setting up the virtual environment

The following steps guide you through creating and activating a Python virtual environment for this project:

  * Create a venv:  
    ```
	python3 -m venv /path/to/new/virtual/environment
    ```
  * Activate this venv: 
    ```
	source /path/to/new/virtual/environment/bin/activate
    ```
    Note how the console prompt changes: you should now see the name of your virtual
    environment in parentheses, before your usual command line prompt.

  * Upgrade pip (always do this before installing anything in your venv):  
    ```
	pip install --upgrade pip
	```

Other aspects to consider:
  - To deactivate the virtual environment: run `deactivate`
  - To remove the virtual environment: `rm -rf /path/to/new/virtual/environment`


## Install the package under the Python virtual environment

The following steps detail how to install the Contingency Screening package within your activated virtual environment:

  0. You first need to install pip & friends in your venv:

	pip install --upgrade pip wheel setuptools build

(Note 1: it's important to upgrade `pip`, since the version installed in the Operating System is likely to be too old.)  
(Note 2: with more recent versions of `build`, it is redundant to explicitly require the installation of `wheel` and `setuptools`.)

  1. Clone the repo and move to contingencies-screening branch:
 
 	git clone https://github.com/dynawo/contingencies-screening/

  2. Build the Python package (from the main directory of the package):
	 
	python -m build

  3. Install the package:
  
	pip install dist/contingencies_screening-0.1.0-py3-none-any.whl

Now everything should be ready to run the software in the active virtual environment.
Test that the installation was successful by invoking the command
`run_contingencies_screening` with the help option.  You should obtain the following
output:

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

## Installation using uv (Optional - Faster Alternative)

Assuming you have `uv` installed globally (`pip install --upgrade uv`), and your project root contains `pyproject.toml` and `uv.lock`, you can set up and activate the virtual environment as follows:

1.  **Initialize the virtual environment using `uv`:**

    Navigate to the root directory of your project (the one containing `pyproject.toml` and `uv.lock`) in your terminal and run:

    ```bash
    uv venv
    ```

    This command will create a virtual environment based on the dependencies specified in your project files (typically `.venv`).

2.  **Activate the `uv`-created virtual environment:**

    The activation command is similar to the standard `venv`, but you'll point to the `bin/activate` script within the `uv`-created environment (usually `.venv`):

    ```bash
    source .venv/bin/activate
    ```

    After running this command, your terminal prompt should be prefixed with the name of your virtual environment in parentheses (e.g., `(.venv) user@host$`), indicating that the environment is active.

Once the virtual environment is activated using these `uv` commands, you can proceed with building and installing the package as described in the previous sections (using either `pip` or `uv`). Remember that `uv` can also be used for the build and install steps for potentially faster performance:

```bash
uv sync