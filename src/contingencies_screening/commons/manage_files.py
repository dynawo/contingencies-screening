from lxml import etree
import os
import shutil
from pathlib import Path


def parse_xml_file(xml_file: str) -> etree._ElementTree:
    """Parses an XML file."""
    parser = etree.XMLParser()
    try:
        parsed_xml = etree.parse(xml_file, parser)
        return parsed_xml
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid XML syntax in file: {xml_file}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"XML file not found: {xml_file}")


def dir_exists(output_dir: Path, input_dir: Path) -> None:
    """Checks if the output directory exists and handles user confirmation for removal."""
    if output_dir.exists():
        remove_dir = input(
            f"The output directory exists {output_dir}, do you want to remove it? [y/N] "
        )
        if remove_dir.lower() == "y":
            if output_dir == input_dir or output_dir in input_dir.parents:
                raise ValueError(
                    "Error: specified input directory is the same or a subdirectory "
                    "of the specified output directory."
                )
            shutil.rmtree(output_dir)
        else:
            raise SystemExit("User chose not to remove output directory.")


def clean_data(
    dynawo_output_folder: Path,
    sorted_loadflow_score_list: list,
    number_pos_replay: int,
) -> None:
    """Cleans unnecessary data from Dynawo results."""
    replay_contgs = [elem_list[1]["name"] for elem_list in sorted_loadflow_score_list]
    if number_pos_replay != -1:
        replay_contgs = replay_contgs[:number_pos_replay]

    retain_files = ["aggregatedResults.xml"]
    dyn_xml_files = list(dynawo_output_folder.glob("DYN*.xml"))
    if dyn_xml_files:
        retain_files.append(dyn_xml_files[0].name)

    retain_folders = ["outputs", "constraints", "timeLine"]
    retain_folders_contg = replay_contgs

    for elem in dynawo_output_folder.iterdir():
        try:
            if elem.is_file():
                if elem.name not in retain_files:
                    if elem.is_symlink():
                        elem.unlink()
                    else:
                        elem.unlink(missing_ok=True)
            elif elem.is_dir():
                if elem.name not in retain_folders and elem.name not in retain_folders_contg:
                    shutil.rmtree(elem, ignore_errors=True)
                elif elem.name in retain_folders_contg:
                    for elem_contg_dir in elem.iterdir():
                        if elem_contg_dir.is_file():
                            elem_contg_dir.unlink(missing_ok=True)
                        elif elem_contg_dir.is_dir() and elem_contg_dir.name != "outputs":
                            shutil.rmtree(elem_contg_dir, ignore_errors=True)
                        elif elem_contg_dir.is_dir() and elem_contg_dir.name == "outputs":
                            for output_dir in elem_contg_dir.iterdir():
                                if output_dir.is_file():
                                    output_dir.unlink(missing_ok=True)
                                elif output_dir.is_dir() and output_dir.name != "finalState":
                                    shutil.rmtree(output_dir, ignore_errors=True)
        except OSError as e:
            print(f"Error processing {elem}: {e}")


def compress_results(path: Path) -> None:
    """Compresses a folder with tar.gz."""
    try:
        os.system(
            "cd "
            + str(path.parent)
            + " && tar zcf "
            + str(path.name)
            + ".tar.gz "
            + str(path.name)
            + " && rm -rf "
            + str(path.name)
        )
    except OSError as e:
        print(f"Error compressing results at {path}: {e}")
