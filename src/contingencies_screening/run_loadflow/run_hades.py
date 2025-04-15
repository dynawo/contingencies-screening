import os
import shutil
import subprocess
from pathlib import Path

from lxml import etree

from contingencies_screening.commons import manage_files
from contingencies_screening.config import settings


def activate_tap_changers(hades_file: Path, activate_taps: bool, multithreading: bool) -> None:
    """
    Modifies the Hades input file to activate/deactivate tap changers and set the number of threads.

    Args:
        hades_file: Path to the Hades input file.
        activate_taps: Boolean indicating whether to activate tap changers.
        multithreading: Boolean indicating whether to enable multithreading.
    """

    activate_taps_str = "true" if activate_taps else "false"
    hades_tree = manage_files.parse_xml_file(str(hades_file))
    root = hades_tree.getroot()
    ns = etree.QName(root).namespace

    for paramHades in root.iter(f"{{{ns}}}paramHades"):
        paramHades.set("regChrg", activate_taps_str)
        paramHades.set("nbThreads", str(settings.N_THREADS_LAUNCHER) if multithreading else "1")

    hades_tree.write(
        str(hades_file),
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
        pretty_print=True,
    )


def run_hades(
    hades_input_file: Path,
    hades_output_file: Path,
    hades_launcher: Path,
    tap_changers: bool,
    multithreading: bool,
    calc_contingencies: bool,
) -> int:
    """
    Runs the Hades simulation.

    Args:
        hades_input_file: Path to the Hades input file.
        hades_output_file: Path to the Hades output file.
        hades_launcher: Path to the Hades launcher executable.
        tap_changers: Boolean indicating whether tap changers are activated.
        multithreading: Boolean indicating whether multithreading is enabled.
        calc_contingencies: Boolean indicating if contingencies are calculated.

    Returns:
        0 on success, 1 on failure.
    """

    output_folder = hades_output_file.parent

    if not calc_contingencies:
        if output_folder != hades_input_file.parent:
            shutil.copy(hades_input_file, output_folder)

        # Activate tap changers if needed
        activate_tap_changers(output_folder / hades_input_file.name, tap_changers, multithreading)

        # Run the simulation on the specified hades launcher
        try:
            subprocess.run(
                "cd "
                + str(output_folder)
                + " && "
                + str(hades_launcher)
                + " "
                + str(output_folder / hades_input_file.name)
                + " "
                + str(hades_output_file),
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Hades execution failed: {e.stderr}")
            return 1

        return 0

    else:
        if output_folder != hades_input_file.parent:
            try:
                os.system(
                    "ln -sf " + str(hades_input_file.parent) + "/* " + str(output_folder) + "/"
                )  # Create symlink
            except OSError as e:
                # Handle cases where the link already exists or other errors
                print(f"Error creating symlink: {e}")
                shutil.copytree(
                    hades_input_file.parent, output_folder, dirs_exist_ok=True
                )  # Fallback to copytree

        return 0
