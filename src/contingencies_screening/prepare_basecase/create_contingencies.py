import json
from lxml import etree
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from contingencies_screening.commons import manage_files
from contingencies_screening.config import settings


def generate_branch_contingency(
    root: etree._Element, element_name: str, disconnection_mode: str
) -> None:
    """Generates a contingency by disconnecting a branch."""

    reseau = root.find("./reseau", root.nsmap)
    donneesGroupes = reseau.find("./donneesQuadripoles", root.nsmap)

    hades_branch = next(
        (
            g
            for g in donneesGroupes.iterfind("./quadripole", root.nsmap)
            if g.get("nom") == element_name
        ),
        None,
    )

    if hades_branch is None:
        raise ValueError(f"Error: Branch with name '{element_name}' does not exist")

    if disconnection_mode == "FROM":
        hades_branch.set("nor", "-1")
    elif disconnection_mode == "TO":
        hades_branch.set("nex", "-1")
    elif disconnection_mode == "BOTH":
        hades_branch.set("nex", "-1")
        hades_branch.set("nor", "-1")
    else:
        raise ValueError(f"Error: Wrong disconnection mode specified: {disconnection_mode}")


def generate_generator_contingency(root: etree._Element, element_name: str) -> None:
    """Generates a contingency by disconnecting a generator."""

    reseau = root.find("./reseau", root.nsmap)
    donneesGroupes = reseau.find("./donneesGroupes", root.nsmap)

    hades_gen = next(
        (
            g
            for g in donneesGroupes.iterfind("./groupe", root.nsmap)
            if g.get("nom") == element_name
        ),
        None,
    )

    if hades_gen is None:
        raise ValueError(f"Error: Generator with name '{element_name}' does not exist")

    hades_gen.set("noeud", "-1")


def generate_load_contingency(root: etree._Element, element_name: str) -> None:
    """Generates a contingency by disconnecting a load."""

    reseau = root.find("./reseau", root.nsmap)
    donneesConsos = reseau.find("./donneesConsos", root.nsmap)

    hades_load = next(
        (g for g in donneesConsos.iterfind("./conso", root.nsmap) if g.get("nom") == element_name),
        None,
    )

    if hades_load is None:
        raise ValueError(f"Error: Load with name '{element_name}' does not exist")

    hades_load.set("noeud", "-1")


def generate_shunt_contingency(root: etree._Element, element_name: str) -> None:
    """Generates a contingency by disconnecting a shunt."""

    reseau = root.find("./reseau", root.nsmap)
    donneesShunts = reseau.find("./donneesShunts", root.nsmap)

    hades_shunt = next(
        (g for g in donneesShunts.iterfind("./shunt", root.nsmap) if g.get("nom") == element_name),
        None,
    )

    if hades_shunt is None:
        raise ValueError(f"Error: Shunt with name '{element_name}' does not exist")

    hades_shunt.set("noeud", "-1")


def clean_contingencies(
    parsed_input_xml: etree._ElementTree, root: etree._Element, ns: str
) -> None:
    """Clears contingencies from a parsed XML tree."""

    for variante in root.iter(f"{{{ns}}}variante"):
        variante.getparent().remove(variante)


def generate_contingency(
    hades_original_file_parsed: etree._ElementTree,
    hades_contingency_file: str,
    contingency_element_name: str,
    contingency_element_type: int,
    disconnection_mode: str,
) -> None:
    """Generates a Hades contingency and saves the modified XML."""

    root = hades_original_file_parsed.getroot()

    contingency_actions = {
        1: generate_branch_contingency,
        2: generate_generator_contingency,
        3: generate_load_contingency,
        4: generate_shunt_contingency,
    }

    action = contingency_actions.get(contingency_element_type)
    if action:
        action(root, contingency_element_name, disconnection_mode)
    else:
        raise ValueError(
            f"Error: Invalid contingency element type provided: {contingency_element_type}"
        )

    etree.indent(hades_original_file_parsed)
    hades_original_file_parsed.write(
        hades_contingency_file,
        pretty_print=True,
        xml_declaration='<?xml version="1.0" encoding="ISO-8859-1"?>',
        encoding="ISO-8859-1",
        standalone=False,
    )


def get_types_cont(hades_input_file: str) -> Dict[str, int]:
    """Gets a dictionary of Hades contingency types."""

    parsed_hades = manage_files.parse_xml_file(hades_input_file)
    root = parsed_hades.getroot()
    ns = etree.QName(root).namespace

    dict_types_cont: Dict[str, int] = {}
    for entry in root.iter(f"{{{ns}}}quadripole"):
        dict_types_cont[entry.attrib["nom"]] = 1

    for entry in root.iter(f"{{{ns}}}groupe"):
        dict_types_cont[entry.attrib["nom"]] = 2

    for entry in root.iter(f"{{{ns}}}conso"):
        dict_types_cont[entry.attrib["nom"]] = 3

    for entry in root.iter(f"{{{ns}}}shunt"):
        dict_types_cont[entry.attrib["nom"]] = 4

    return dict_types_cont


def create_hades_contingency_n_1(
    hades_input_file: str,
    hades_input_file_parsed: etree._ElementTree,
    hades_output_folder: Path,
    replay_cont: str,
    dict_types_cont: Dict[str, int],
) -> Optional[Path]:
    """Creates a Hades N-1 contingency file."""

    cont_type = dict_types_cont.get(replay_cont)
    if cont_type is None:
        print(f"Contingency {replay_cont} not found in Hades model")
        return None

    cont_output_dir = hades_output_folder / replay_cont.replace(" ", "_")
    cont_output_dir.mkdir(parents=True, exist_ok=True)

    disconnection_mode = "BOTH"

    generate_contingency(
        hades_input_file_parsed,
        str(cont_output_dir / Path(hades_input_file).name),
        replay_cont,
        cont_type,
        disconnection_mode,
    )

    return cont_output_dir


def get_dynawo_types_cont(dynawo_input_file: str) -> Dict[str, int]:
    """Gets a dictionary of Dynawo contingency types."""

    parsed_iidm = manage_files.parse_xml_file(dynawo_input_file)
    root = parsed_iidm.getroot()
    ns = etree.QName(root).namespace

    dict_types_cont: Dict[str, int] = {}
    for entry in root.iter(f"{{{ns}}}line", f"{{{ns}}}twoWindingsTransformer"):
        if entry.get("bus1") is not None and entry.get("bus2") is not None:
            dict_types_cont[entry.attrib["id"]] = 1

    for entry in root.iter(f"{{{ns}}}generator"):
        if entry.get("bus") is not None:
            dict_types_cont[entry.attrib["id"]] = 2

    for entry in root.iter(f"{{{ns}}}load"):
        if entry.get("bus") is not None:
            dict_types_cont[entry.attrib["id"]] = 3

    for entry in root.iter(f"{{{ns}}}shunt"):
        if entry.get("bus") is not None:
            dict_types_cont[entry.attrib["id"]] = 4

    return dict_types_cont


def get_endbus(
    root: etree._Element, branch: etree._Element, branch_type: str, side: str
) -> Optional[str]:
    """Gets the end bus of a branch in Dynawo."""

    ns = etree.QName(root).namespace
    end_bus = branch.get(f"bus{side}") or branch.get(f"connectableBus{side}")

    if end_bus is None:
        pnode = root if branch_type == "Line" else branch.getparent()
        for vl in pnode.iter(f"{{{ns}}}voltageLevel"):
            if vl.get("id") == branch.get(f"voltageLevelId{side}"):
                topo = vl.find(f"{{{ns}}}nodeBreakerTopology")
                if topo is not None:
                    end_bus = next(
                        (
                            node.get("id")
                            for node in topo
                            if etree.QName(node).localname == "busbarSection"
                            and node.get("v") is not None
                        ),
                        None,
                    )
                break

    return end_bus


def create_dynawo_SA(
    dynawo_output_folder: Path,
    replay_contgs: List[str],
    dict_types_cont: Dict[str, int],
    dynamic_database: Optional[Path],
    matched_branches: Dict[str, str],
    matched_generators: List[str],
    matched_loads: List[str],
    matched_shunts: List[str],
    multithreading: bool,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Creates Dynawo SA input files."""

    n_threads = settings.N_THREADS_LAUNCHER if multithreading else 1

    config_dict: Dict[str, Any] = {
        "dfl-config": {
            "OutputDir": str(dynawo_output_folder),
            "ChosenOutputs": ["STEADYSTATE", "LOSTEQ", "TIMELINE", "CONSTRAINTS"],
            "sa": {"NumberOfThreads": n_threads},
        }
    }
    contng_dict: Dict[str, Any] = {
        "version": "1.0",
        "name": "list",
        "contingencies": [],
    }

    if dynamic_database:
        setting_xml = next(dynamic_database.glob("*setting*.xml"), None)
        assembling_xml = next(dynamic_database.glob("*assembling*.xml"), None)

        if setting_xml and assembling_xml:
            config_dict["dfl-config"].update(
                {
                    "SettingPath": str(setting_xml),
                    "AssemblingPath": str(assembling_xml),
                }
            )
        else:
            raise FileNotFoundError(
                "Error: Setting or assembling XML files not found in dynamic database directory."
            )

    dynawo_output_folder.mkdir(parents=True, exist_ok=True)

    for replay_cont in replay_contgs:
        cont_type = dict_types_cont.get(replay_cont)
        if cont_type:
            element_data = {
                1: {
                    "type": "LINE"
                    if matched_branches.get(replay_cont) == "Line"
                    else "TWO_WINDINGS_TRANSFORMER",
                    "matched_in": matched_branches,
                },
                2: {"type": "GENERATOR", "matched_in": matched_generators},
                3: {"type": "LOAD", "matched_in": matched_loads},
                4: {"type": "SHUNT_COMPENSATOR", "matched_in": matched_shunts},
            }.get(cont_type)

            if element_data:
                if element_data["matched_in"] and replay_cont not in element_data["matched_in"]:
                    print(f"Contingency {replay_cont} not matched in Dynawo.")
                    continue
                contng_dict["contingencies"].append(
                    {
                        "id": replay_cont,
                        "elements": [{"id": replay_cont, "type": element_data["type"]}],
                    }
                )
        else:
            print(f"Contingency {replay_cont} not found in Dynawo model.")

    config_path = dynawo_output_folder / "config.json"
    contng_path = dynawo_output_folder / "contng.json"

    with open(config_path, "w") as outfile:
        json.dump(config_dict, outfile, indent=2)

    with open(contng_path, "w") as outfile:
        json.dump(contng_dict, outfile, indent=2)

    return config_path, contng_path, contng_dict
