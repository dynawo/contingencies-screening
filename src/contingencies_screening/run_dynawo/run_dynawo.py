import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

from lxml import etree

from contingencies_screening.commons import manage_files


def run_dynaflow(input_dir: Path, output_dir: Path, dynawo_launcher: Path) -> None:
    """Runs Dynawo simulation from the specified input directory."""

    # Copy input files to the output directory
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

    # Run Dynawo
    try:
        subprocess.run(
            str(dynawo_launcher) + " jobs " + str(output_dir) + "/JOB.xml ",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Dynawo execution failed: {e.stderr}")
        raise


def check_calculated_contg(input_dir: Path, matching_contng_dict: Dict[str, Any]) -> bool:
    """
    Checks if all required contingencies are already calculated.

    Args:
        input_dir: Path to the directory with Dynawo results.
        matching_contng_dict: Dictionary of contingencies to be simulated.

    Returns:
        True if Dynawo needs to be re-run, False otherwise.
    """

    parsed_file = manage_files.parse_xml_file(input_dir / "aggregatedResults.xml")
    root = parsed_file.getroot()
    ns = etree.QName(root).namespace

    required_contgs = {contg["id"] for contg in matching_contng_dict["contingencies"]}
    executed_contgs = {contg.attrib["id"] for contg in root.iter(f"{{{ns}}}scenarioResults")}

    return not required_contgs.issubset(executed_contgs)


def run_dynaflow_SA(
    input_dir: Path,
    output_dir: Path,
    dynawo_launcher: Path,
    config_file: Path,
    contng_file: Path,
    calc_contingencies: bool,
    matching_contng_dict: Dict[str, Any],
) -> int:
    """
    Runs Dynawo SA simulation or reuses existing results if applicable.

    Args:
        input_dir: Path to the directory with input files.
        output_dir: Path to the output directory.
        dynawo_launcher: Path to the Dynawo launcher executable.
        config_file: Path to the Dynawo configuration file.
        contng_file: Path to the contingencies file.
        calc_contingencies: Flag to indicate whether to calculate contingencies or reuse.
        matching_contng_dict: Dictionary of contingencies to be simulated.

    Returns:
        0 on success, 1 on failure.
    """

    replay_dynawo = True
    if calc_contingencies:
        replay_dynawo = check_calculated_contg(input_dir, matching_contng_dict)
        if not replay_dynawo:
            os.system(
                "find "
                + str(input_dir)
                + " -mindepth 1 -maxdepth 1 -exec ln -sf '{}' "
                + str(output_dir)
                + "/ \;"
            )

    if replay_dynawo:
        iidm_file = next(input_dir.glob("DYN*.xml"))  # Get the first matching file
        shutil.copy2(iidm_file, output_dir)

        try:
            subprocess.run(
                str(dynawo_launcher)
                + " launch --network "
                + str(output_dir / iidm_file.name)
                + " --config "
                + str(config_file)
                + " --contingencies "
                + str(contng_file)
                + " --nsa",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Dynawo SA execution failed: {e.stderr}")
            return 1

    return 0
