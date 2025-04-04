from pathlib import Path
import shutil
from typing import Tuple, Optional
from lxml import etree as ET


def check_basecase_dir(input_dir: Path) -> bool:
    """
    Verifies the directory structure in the input directory.

    Args:
        input_dir: The base directory.

    Returns:
        True if the structure is valid, False otherwise.
    """

    def is_valid_snapshot_name(name: str) -> bool:
        """Validates the snapshot directory name format."""
        if not name.startswith("snapshot_") or len(name) != 22:  # snapshot_YYYYMMDD_HHMM
            return False
        parts = name[9:].split("_")
        if len(parts) != 2 or len(parts[0]) != 8 or len(parts[1]) != 4:
            return False
        return all(part.isdigit() for part in parts)

    input_dir = Path(input_dir)

    for year_dir in input_dir.iterdir():
        if year_dir.is_dir():
            if year_dir.name != "dyn_db" and (
                not year_dir.name.isdigit() or len(year_dir.name) != 4
            ):
                print(
                    f"Error: Year directory '{year_dir.name}' is invalid. It should have 4 digits (YYYY)."
                )
                return False
        elif year_dir.suffix != ".json":
            print(f"Error: '{year_dir}' is not a directory.")
            return False

        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not (
                month_dir.name.isdigit() and len(month_dir.name) == 2
            ):
                print(
                    f"Error: Month directory '{month_dir.name}' is invalid. It should have 2 digits (MM)."
                )
                return False

            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir() or not (day_dir.name.isdigit() and len(day_dir.name) == 2):
                    print(
                        f"Error: Day directory '{day_dir.name}' is invalid. It should have 2 digits (DD)."
                    )
                    return False

                for snapshot_dir in day_dir.iterdir():
                    if not snapshot_dir.is_dir() or not is_valid_snapshot_name(snapshot_dir.name):
                        print(
                            f"Error: Snapshot directory '{snapshot_dir.name}' is invalid. It should have the format 'snapshot_YYYYMMDD_HHMM'."
                        )
                        return False

    return True


def create_basecases_from_RTE(
    hades_contingencies_path: Path,
    hades_snapshots_path: Path,
    dynawo_contingencies_path: Path,
    dynawo_snapshots_path: Path,
    dynawo_config_path: Path,
    dynawo_assembling_path: Path,
    dynawo_setting_path: Path,
    output_path: Path,
) -> None:
    """
    Creates basecases by processing Hades and Dynawo snapshots.

    Args:
        hades_contingencies_path: Path to Hades contingencies XML.
        hades_snapshots_path: Path to Hades snapshot XMLs.
        dynawo_contingencies_path: Path to Dynawo contingencies JSON.
        dynawo_snapshots_path: Path to Dynawo snapshot XMLs.
        dynawo_config_path: Path to Dynawo config JSON.
        dynawo_assembling_path: Path to Dynawo assembling XML.
        dynawo_setting_path: Path to Dynawo setting XML.
        output_path: Path to output directory.
    """
    params = {
        "hades_contingencies_path": Path(hades_contingencies_path),
        "hades_snapshots_path": Path(hades_snapshots_path),
        "dynawo_snapshots_path": Path(dynawo_snapshots_path),
        "dynawo_config_path": Path(dynawo_config_path),
        "dynawo_contingencies_path": Path(dynawo_contingencies_path),
        "dynawo_assembling_path": Path(dynawo_assembling_path),
        "dynawo_setting_path": Path(dynawo_setting_path),
        "output_path": Path(output_path),
    }

    process_snapshots(params, "hades")
    process_snapshots(params, "dynawo")


def process_snapshots(params: dict, system: str) -> None:
    """
    Processes snapshots for a given system (Hades or Dynawo).

    Args:
        params: Dictionary of paths.
        system: "hades" or "dynawo".
    """
    snapshots_path = params[f"{system}_snapshots_path"]
    output_path = params["output_path"]

    n_files = str(len(list(snapshots_path.glob("*.xml"))))
    i_file = 0

    for file in snapshots_path.glob("*.xml"):
        print(f"Processing file {file}. {i_file}/{n_files}")
        i_file += 1
        year, month, day, hour, minute = extract_date_time(file.name)
        output_folder_path = create_directory_structure(
            output_path / "data", year, month, day, hour, minute
        )

        if system == "hades":
            copy_file(
                file, output_folder_path, f"HDS_{output_folder_path.name}.xml"
            )  # Changed filename
        elif system == "dynawo":
            copy_dynawo_snapshot(file, output_folder_path)
            copy_file(params["dynawo_config_path"], output_path)
            copy_file(params["dynawo_contingencies_path"], output_path)
            copy_file(
                params["dynawo_assembling_path"],
                output_path / "dyndb",
                "assembling.xml",
            )
            copy_file(params["dynawo_setting_path"], output_path / "dyndb", "setting.xml")


def copy_dynawo_snapshot(file: Path, output_path: Path) -> None:
    """Copies a Dynawo snapshot file."""
    output_filename = f"DYN_{output_path.name}.xml"
    shutil.copy(file, output_path / output_filename)


def copy_file(file_path: Path, output_path: Path, file_name: Optional[str] = None) -> None:
    """Copies a file to the output directory."""
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, output_path / (file_name or file_path.name))


def overwrite_paramHades(root1: ET.Element, root2: ET.Element) -> None:
    """Overwrites paramHades tag in root2 with the one from root1."""
    param_hades1 = root1.find(".//paramHades")
    param_hades2 = root2.find(".//paramHades")
    if param_hades1 is not None and param_hades2 is not None:
        parent2 = next((parent for parent in root2.iter() if param_hades2 in list(parent)), None)
        if parent2 is not None:
            parent2.remove(param_hades2)
            parent2.append(param_hades1)


def extract_date_time(filename: str) -> Tuple[str, str, str, str, str]:
    """Extracts date and time from filename."""
    parts = filename.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Filename '{filename}' does not contain expected date and time components."
        )

    date_str = parts[1]
    time_str = parts[2].split(".")[0]

    if len(date_str) != 8 or len(time_str) != 6:
        raise ValueError(f"Date or time string in filename '{filename}' has incorrect length.")

    return (
        date_str[:4],
        date_str[4:6],
        date_str[6:],
        time_str[:2],
        time_str[2:4],
    )


def create_directory_structure(
    output_dir: Path, year: str, month: str, day: str, hour: str, minute: str
) -> Path:
    """Creates the directory structure for the output file."""
    output_path = output_dir / year / month / day / f"snapshot_{year}{month}{day}_{hour}{minute}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def remove_namespace(elem: ET.Element) -> None:
    """Removes namespace prefixes from XML tags."""
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def add_variants_to_entreesHades(root1: ET.Element, root2: ET.Element) -> None:
    """Adds variante tags from root1 to entreesHades in root2."""
    entrees_hades2 = root2.find(".//entreesHades")
    if entrees_hades2 is not None:
        quadripoles = {qp.get("nom"): qp for qp in root2.findall(".//quadripole")}
        i = 0
        for variante in root1.findall(".//variante"):
            nom_variant = variante.get("nom")
            quadripole = quadripoles.get(nom_variant)

            if quadripole and quadripole.get("nor") != "-1" and quadripole.get("nex") != "-1":
                entrees_hades2.insert(i, variante)
                i += 1
            else:
                print(
                    f"{nom_variant} quadripole disconnected or not found. Not adding the variante {variante.get('num')}."
                )


def save_combined_xml(tree: ET.ElementTree, output_path: Path) -> None:
    """Saves the combined XML tree to the specified output path."""
    output_filename = f"HDS_{output_path.name}.xml"
    tree.write(str(output_path / output_filename), encoding="utf-8", xml_declaration=True)


def merge_xml_files(file1: Path, file2: Path, output_dir: Path) -> None:
    """Merges two XML files and saves the result."""
    try:
        tree1 = ET.parse(str(file1))
        root1 = tree1.getroot()
        tree2 = ET.parse(str(file2))
        root2 = tree2.getroot()

        remove_namespace(root1)
        remove_namespace(root2)

        overwrite_paramHades(root1, root2)
        add_variants_to_entreesHades(root1, root2)

        year, month, day, hour, minute = extract_date_time(file2.name)
        output_path = create_directory_structure(
            output_dir / "data", year, month, day, hour, minute
        )

        save_combined_xml(tree2, output_path)

    except ET.XMLSyntaxError as e:
        print(f"Error parsing XML file: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        base_dir = Path("/home/guiu/Projects/CONT_SCR_CRV_REC/Data")
        create_basecases_from_RTE(
            base_dir / "Hades" / "contingencies.xml",
            base_dir / "Hades" / "adn",
            base_dir / "Dynawo" / "contingencies.json",
            base_dir / "Dynawo" / "iidm",
            base_dir / "Dynawo" / "DFLConfig.json",
            base_dir
            / "Dynawo"
            / "interfaceDynamo_v1.7.0_powsybl"
            / "share"
            / "assembling_dynaflow_StanWay.xml",
            base_dir
            / "Dynawo"
            / "interfaceDynamo_v1.7.0_powsybl"
            / "share"
            / "setting_dynaflow_StanWay.xml",
            base_dir / "New_Data_DCS_Prepared",
        )
        check_basecase_dir(base_dir / "New_Data_DCS_Prepared")
    except Exception as e:
        print(f"An error occurred during main execution: {e}")
