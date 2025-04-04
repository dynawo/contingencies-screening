from lxml import etree
from contingencies_screening.commons import manage_files
from typing import Dict, List, Any, Tuple


def get_elements_dict(
    parsed_hades_input_file: etree._ElementTree,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Extracts element data from a parsed Hades input XML file.

    Args:
        parsed_hades_input_file: Parsed XML tree.

    Returns:
        A dictionary containing element data, organized by element type.
    """
    root = parsed_hades_input_file.getroot()
    ns = etree.QName(root).namespace

    elements_dict: Dict[str, Dict[int, Dict[str, Any]]] = {}

    poste_dict: Dict[int, Dict[str, Any]] = {
        int(poste.attrib["num"]): {
            "nom": poste.attrib["nom"],
            "unom": poste.attrib["unom"],
            "volt_level": int(poste.attrib["nivTension"]),
        }
        for poste in root.iter(f"{{{ns}}}poste")
    }

    noeud_dict: Dict[int, Dict[str, Any]] = {
        int(noeud.attrib["num"]): {
            "nom": noeud.attrib["nom"],
            "poste": noeud.attrib["poste"],
            "vmax": noeud.attrib["vmax"],
            "vmin": noeud.attrib["vmin"],
            "volt_level": poste_dict.get(int(noeud.attrib["poste"]), {}).get("volt_level"),
        }
        for noeud in root.iter(f"{{{ns}}}noeud")
    }

    groupe_dict: Dict[int, Dict[str, Any]] = {
        int(groupe.attrib["num"]): {
            "nom": groupe.attrib["nom"],
            "poste": groupe.attrib["poste"],
            "noeud": groupe.attrib["noeud"],
            "pmax": groupe.attrib["pmax"],
            "pmin": groupe.attrib["pmin"],
            "volt_level": poste_dict.get(int(groupe.attrib["poste"]), {}).get("volt_level"),
        }
        for groupe in root.iter(f"{{{ns}}}groupe")
    }

    conso_dict: Dict[int, Dict[str, Any]] = {
        int(conso.attrib["num"]): {
            "nom": conso.attrib["nom"],
            "poste": conso.attrib["poste"],
            "noeud": conso.attrib["noeud"],
            "volt_level": poste_dict.get(int(conso.attrib["poste"]), {}).get("volt_level"),
        }
        for conso in root.iter(f"{{{ns}}}conso")
    }

    shunt_dict: Dict[int, Dict[str, Any]] = {
        int(shunt.attrib["num"]): {
            "nom": shunt.attrib["nom"],
            "poste": shunt.attrib["poste"],
            "noeud": shunt.attrib["noeud"],
            "volt_level": poste_dict.get(int(shunt.attrib["poste"]), {}).get("volt_level"),
        }
        for shunt in root.iter(f"{{{ns}}}shunt")
    }

    quadripole_dict: Dict[int, Dict[str, Any]] = {
        int(quadripole.attrib["num"]): {
            "nom": quadripole.attrib["nom"],
            "absnor": quadripole.attrib["absnor"],
            "absnex": quadripole.attrib["absnex"],
            "nor": quadripole.attrib["nor"],
            "nex": quadripole.attrib["nex"],
            "postor": quadripole.attrib["postor"],
            "postex": quadripole.attrib["postex"],
            "resistance": quadripole.attrib["resistance"],
            "reactance": quadripole.attrib["reactance"],
            "volt_level": poste_dict.get(int(quadripole.attrib["postor"]), {}).get("volt_level"),
        }
        for quadripole in root.iter(f"{{{ns}}}quadripole")
    }

    elements_dict["poste"] = poste_dict
    elements_dict["noeud"] = noeud_dict
    elements_dict["groupe"] = groupe_dict
    elements_dict["conso"] = conso_dict
    elements_dict["shunt"] = shunt_dict
    elements_dict["quadripole"] = quadripole_dict

    return elements_dict


def get_contingencies_dict(
    parsed_hades_input_file: etree._ElementTree,
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts contingency data from a parsed Hades input XML file.

    Args:
        parsed_hades_input_file: Parsed XML tree.

    Returns:
        A dictionary containing contingency data.
    """
    root = parsed_hades_input_file.getroot()
    ns = etree.QName(root).namespace

    contingencies_dict: Dict[str, Dict[str, Any]] = {}

    for variante in root.iter(f"{{{ns}}}variante"):
        affected_elements_list: List[str] = (
            [elem.text for elem in variante.iter(f"{{{ns}}}quadex")]
            + [elem.text for elem in variante.iter(f"{{{ns}}}quador")]
            + [ouvrage.attrib["num"] for ouvrage in variante.iter(f"{{{ns}}}ouvrage")]
            + [vscor.text for vscor in variante.iter(f"{{{ns}}}vscor")]
        )

        contingencies_dict[variante.attrib["num"]] = {
            "name": variante.attrib["nom"],
            "type": int(variante.attrib["type"]),
            "affected_elements": affected_elements_list,
            "coefrepquad": int(variante.attrib["coefrepquad"]),
        }

    return contingencies_dict


def get_max_min_voltages(
    root: etree._Element, ns: str, contingencies_list: List[str]
) -> Tuple[Dict[str, List[List[Any]]], Dict[str, List[List[Any]]], Dict[int, int]]:
    """
    Extracts maximum and minimum voltage data from a parsed Hades output XML.

    Args:
        root: Root element of the parsed XML.
        ns: Namespace of the XML.
        contingencies_list: List of contingency identifiers.

    Returns:
        Tuple containing dictionaries for minimum voltages, maximum voltages, and poste-node voltage mapping.
    """
    max_voltages_dict: Dict[str, List[List[Any]]] = {key: [] for key in contingencies_list}
    min_voltages_dict: Dict[str, List[List[Any]]] = {key: [] for key in contingencies_list}
    poste_node_volt_dict: Dict[int, int] = {}

    for entry in root.iter(f"{{{ns}}}posteSurv"):
        poste_node_volt_dict[int(entry.attrib["poste"])] = int(entry.attrib["noeud"])
        vmin = float(entry.attrib["vmin"])
        vmax = float(entry.attrib["vmax"])
        if vmin != vmax:
            min_voltages_dict[entry.attrib["varianteVmin"]].append(
                [int(entry.attrib["poste"]), vmin]
            )
            max_voltages_dict[entry.attrib["varianteVmax"]].append(
                [int(entry.attrib["poste"]), vmax]
            )

    return min_voltages_dict, max_voltages_dict, poste_node_volt_dict


def get_poste_node_voltages(
    root: etree._Element,
    ns: str,
    elements_dict: Dict[str, Dict[int, Any]],
    poste_node_volt_dict: Dict[int, int],
) -> Dict[str, Dict[int, Any]]:
    """
    Adds voltage data to the elements dictionary.

    Args:
        root: Root element of the parsed XML.
        ns: Namespace of the XML.
        elements_dict: Dictionary containing element data.
        poste_node_volt_dict: Dictionary mapping postes to nodes.

    Returns:
        Updated elements dictionary.
    """
    for noeud in root.iter(f"{{{ns}}}noeud"):
        variable = noeud.find(f"{{{ns}}}variables")
        if variable is not None:
            poste_num = elements_dict["noeud"][int(noeud.attrib["num"])]["poste"]
            elements_dict["noeud"][int(noeud.attrib["num"])]["volt"] = (
                float(variable.attrib["v"])
                * float(elements_dict["poste"][int(poste_num)]["unom"])
                / 100
            )

    for poste, node in poste_node_volt_dict.items():
        elements_dict["poste"][poste]["volt"] = elements_dict["noeud"][node]["volt"]

    return elements_dict


def get_line_flows(
    root: etree._Element, ns: str, contingencies_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, List[List[Any]]]:
    """
    Extracts line flow data from a parsed Hades output XML.

    Args:
        root: Root element of the parsed XML.
        ns: Namespace of the XML.
        contingencies_dict: Dictionary containing contingency data.

    Returns:
        Dictionary containing line flow data.
    """
    invert_dict: Dict[int, str] = {
        contingencies_dict[contg]["coefrepquad"]: contg for contg in contingencies_dict.keys()
    }

    if len(invert_dict) != len(contingencies_dict):
        raise ValueError("Ill-defined coefrepquad contingencies")

    line_flows_dict: Dict[str, List[List[Any]]] = {key: [] for key in contingencies_dict.keys()}

    for entry in root.iter(f"{{{ns}}}resChargeMax"):
        line_flows_dict[invert_dict[int(entry.attrib["quadripole"])]].append(
            [int(entry.attrib["numOuvrSurv"]), float(entry.attrib["chargeMax"])]
        )

    return line_flows_dict


def get_fault_data(
    root: etree._Element, ns: str, contingencies_list: List[str]
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, float],
    Dict[str, Dict[str, List[Dict[str, Any]]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
]:
    """
    Extracts fault data from a parsed Hades output XML.

    Args:
        root: Root element of the parsed XML.
        ns: Namespace of the XML.
        contingencies_list: List of contingency identifiers.

    Returns:
        Tuple containing dictionaries for fault data.
    """
    status_dict: Dict[str, int] = {}
    cause_dict: Dict[str, int] = {}
    iter_number_dict: Dict[str, int] = {}
    calc_duration_dict: Dict[str, float] = {}
    constraint_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "contrTransit": {key: [] for key in contingencies_list},
        "contrTension": {key: [] for key in contingencies_list},
        "contrGroupe": {key: [] for key in contingencies_list},
    }
    coef_report_dict: Dict[str, List[Dict[str, Any]]] = {key: [] for key in contingencies_list}
    res_node_dict: Dict[str, List[Dict[str, Any]]] = {key: [] for key in contingencies_list}
    tap_changers_dict: Dict[str, List[Dict[str, Any]]] = {key: [] for key in contingencies_list}

    groupe_name: Dict[int, str] = {
        int(groupe.attrib["num"]): groupe.attrib["nom"] for groupe in root.iter(f"{{{ns}}}groupe")
    }
    quadripole_name: Dict[int, str] = {
        int(quadripole.attrib["num"]): quadripole.attrib["nom"]
        for quadripole in root.iter(f"{{{ns}}}quadripole")
    }
    node_name: Dict[int, str] = {
        int(node.attrib["num"]): node.attrib["nom"] for node in root.iter(f"{{{ns}}}noeud")
    }

    for contingency in root.iter(f"{{{ns}}}defaut"):
        contingency_number = contingency.attrib["num"]
        load_flow_branch = contingency.find(f"{{{ns}}}resLF")
        if load_flow_branch is not None:
            status_dict[contingency_number] = int(load_flow_branch.attrib["statut"])
            cause_dict[contingency_number] = int(load_flow_branch.attrib["cause"])
            iter_number_dict[contingency_number] = int(load_flow_branch.attrib["nbIter"])
            calc_duration_dict[contingency_number] = round(
                float(load_flow_branch.attrib["dureeCalcul"]), 5
            )

            for subelement in load_flow_branch:
                if subelement.tag in [
                    f"{{{ns}}}contrTransit",
                    f"{{{ns}}}contrTension",
                    f"{{{ns}}}contrGroupe",
                ]:
                    constraint_entry: Dict[str, Any] = {
                        "elem_num": int(subelement.attrib["ouvrage"]),
                        "before": subelement.attrib["avant"],
                        "after": subelement.attrib["apres"],
                        "limit": subelement.attrib["limite"],
                    }

                    if subelement.tag == f"{{{ns}}}contrGroupe":
                        constraint_entry["elem_name"] = groupe_name.get(
                            constraint_entry["elem_num"]
                        )
                        constraint_entry["typeLim"] = int(subelement.attrib["typeLim"])
                        constraint_entry["type"] = subelement.attrib["type"]
                        constraint_dict["contrGroupe"][contingency_number].append(constraint_entry)
                    elif subelement.tag == f"{{{ns}}}contrTransit":
                        constraint_entry["elem_name"] = quadripole_name.get(
                            constraint_entry["elem_num"]
                        )
                        constraint_entry["tempo"] = subelement.attrib["tempo"]
                        constraint_entry["beforeMW"] = subelement.attrib["avantMW"]
                        constraint_entry["afterMW"] = subelement.attrib["apresMW"]
                        constraint_entry["sideOr"] = subelement.attrib["coteOr"]
                        constraint_dict["contrTransit"][contingency_number].append(
                            constraint_entry
                        )
                    else:
                        constraint_entry["elem_name"] = node_name.get(constraint_entry["elem_num"])
                        constraint_entry["threshType"] = subelement.attrib["typeSeuil"]
                        constraint_entry["tempo"] = subelement.attrib["tempo"]
                        constraint_dict["contrTension"][contingency_number].append(
                            constraint_entry
                        )
                elif subelement.tag == f"{{{ns}}}coefReport":
                    report_entry: Dict[str, Any] = {
                        "num": subelement.attrib["num"],
                        "coefAmpere": subelement.attrib["coefAmpere"],
                        "coefMW": subelement.attrib["coefMW"],
                        "transitActN": subelement.attrib["transitActN"],
                        "transitAct": subelement.attrib["transitAct"],
                        "intensityN": subelement.attrib["intensiteN"],
                        "intensity": subelement.attrib["intensite"],
                        "charge": subelement.attrib["charge"],
                        "threshold": subelement.attrib["seuil"],
                        "sideOr": subelement.attrib["coteOr"],
                    }
                    coef_report_dict[contingency_number].append(report_entry)
                elif subelement.tag == f"{{{ns}}}resnoeud":
                    node_entry = {"quadripole_num": subelement.attrib["numOuvrSurv"]}
                    res_node_dict[contingency_number].append(node_entry)
                elif subelement.tag == f"{{{ns}}}resregleur":
                    tap_entry: Dict[str, Any] = {}
                    gen_name: str = next(
                        (
                            gen.attrib["nom"]
                            for gen in root.iter(f"{{{ns}}}quadripole")
                            if gen.attrib["num"] == subelement.attrib["numOuvrSurv"]
                        ),
                        "",
                    )

                    tap_entry["quadripole_num"] = subelement.attrib["numOuvrSurv"]
                    tap_entry["quadripole_name"] = gen_name
                    tap_entry["previous_value"] = subelement.attrib["priseDeb"]
                    tap_entry["after_value"] = subelement.attrib["priseFin"]
                    tap_entry["diff_value"] = int(subelement.attrib["priseFin"]) - int(
                        subelement.attrib["priseDeb"]
                    )
                    tap_entry["stopper"] = subelement.attrib["butee"]

                    tap_changers_dict[contingency_number].append(tap_entry)

    return (
        status_dict,
        cause_dict,
        iter_number_dict,
        calc_duration_dict,
        constraint_dict,
        coef_report_dict,
        res_node_dict,
        tap_changers_dict,
    )


def collect_hades_results(
    elements_dict: Dict[str, Dict[int, Any]],
    contingencies_dict: Dict[str, Dict[str, Any]],
    parsed_hades_output_file: etree._ElementTree,
    tap_changers: bool,
) -> Tuple[Dict[str, Dict[int, Any]], Dict[str, Dict[str, Any]], int]:
    """
    Collects results from a parsed Hades output XML file.

    Args:
        elements_dict: Dictionary containing element data.
        contingencies_dict: Dictionary containing contingency data.
        parsed_hades_output_file: Parsed Hades output XML tree.
        tap_changers: Boolean indicating whether to include tap changer data.

    Returns:
        Tuple containing updated elements and contingencies dictionaries, and an error code.
    """
    root = parsed_hades_output_file.getroot()
    ns = etree.QName(root).namespace

    if not list(root.iter(f"{{{ns}}}defaut")):
        print("No contingencies executed")
        return elements_dict, contingencies_dict, 1

    min_voltages_dict, max_voltages_dict, poste_node_volt_dict = get_max_min_voltages(
        root, ns, list(contingencies_dict.keys())
    )

    elements_dict = get_poste_node_voltages(root, ns, elements_dict, poste_node_volt_dict)

    line_flows_dict = get_line_flows(root, ns, contingencies_dict)

    (
        status_dict,
        cause_dict,
        iter_number_dict,
        calc_duration_dict,
        constraint_dict,
        coef_report_dict,
        res_node_dict,
        tap_changers_dict,
    ) = get_fault_data(root, ns, list(contingencies_dict.keys()))

    for key in contingencies_dict.keys():
        contingencies_dict[key]["min_voltages"] = min_voltages_dict[key]
        contingencies_dict[key]["max_voltages"] = max_voltages_dict[key]
        contingencies_dict[key]["max_flow"] = line_flows_dict[key]
        contingencies_dict[key]["status"] = status_dict[key]
        contingencies_dict[key]["cause"] = cause_dict[key]
        contingencies_dict[key]["n_iter"] = iter_number_dict[key]
        contingencies_dict[key]["calc_duration"] = calc_duration_dict[key]
        contingencies_dict[key]["constr_volt"] = constraint_dict["contrTension"][key]
        contingencies_dict[key]["constr_flow"] = constraint_dict["contrTransit"][key]

        constr_gen_Q = [
            constr_i
            for constr_i in constraint_dict["contrGroupe"][key]
            if constr_i["typeLim"] in [0, 1]
        ]
        constr_gen_U = [
            constr_i
            for constr_i in constraint_dict["contrGroupe"][key]
            if constr_i["typeLim"] in [2, 3]
        ]
        contingencies_dict[key]["constr_gen_Q"] = constr_gen_Q
        contingencies_dict[key]["constr_gen_U"] = constr_gen_U
        contingencies_dict[key]["coef_report"] = coef_report_dict[key]
        contingencies_dict[key]["res_node"] = res_node_dict[key]

        if tap_changers:
            contingencies_dict[key]["tap_changers"] = tap_changers_dict[key]

    return elements_dict, contingencies_dict, 0


def get_dynawo_contingencies(
    dynawo_xml_root: etree._Element, ns: str, contg_set: set
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts Dynawo contingency data from a parsed XML root.

    Args:
        dynawo_xml_root: Root element of the Dynawo XML.
        ns: Namespace of the XML.
        contg_set: Set of contingency identifiers.

    Returns:
        Dictionary containing Dynawo contingency data.
    """
    contingencies_dict: Dict[str, Dict[str, Any]] = {
        contg.attrib["id"]: {"status": contg.attrib["status"], "constraints": []}
        for contg in dynawo_xml_root.iter(f"{{{ns}}}scenarioResults")
        if contg.attrib["id"] in contg_set
    }

    return contingencies_dict


def get_dynawo_timeline_constraints(
    root: etree._Element, ns: str, dwo_constraint_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extracts timeline constraint data from a parsed Dynawo timeline XML.

    Args:
        root: Root element of the timeline XML.
        ns: Namespace of the XML.
        dwo_constraint_list: List to store timeline constraints.

    Returns:
        List of timeline constraint dictionaries.
    """
    timeline = list(reversed(root.findall(f"{{{ns}}}event")))
    checked_models: set = set()

    for entry in timeline:
        if entry.attrib["modelName"] in checked_models:
            continue
        checked_models.add(entry.attrib["modelName"])

        if entry.attrib["message"] in [
            "Generator : minimum reactive power limit reached",
            "Generator : maximum reactive power limit reached",
        ]:
            limit_constr: Dict[str, Any] = {"modelName": entry.attrib["modelName"]}
            limit_constr["description"] = (
                "min Q limit reached"
                if "minimum" in entry.attrib["message"]
                else "max Q limit reached"
            )
            limit_constr["kind"] = (
                "QInfQMin" if "minimum" in entry.attrib["message"] else "QSupQMax"
            )
            limit_constr["time"] = entry.attrib["time"]
            limit_constr["type"] = "Generator"

            dwo_constraint_list.append(limit_constr)

    return dwo_constraint_list


def get_dynawo_contingency_data(
    dynawo_contingencies_dict: Dict[str, Dict[str, Any]],
    dynawo_nocontg_tap_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dynawo_output_folder: str,
) -> None:
    """
    Extracts contingency data from Dynawo output files.

    Args:
        dynawo_contingencies_dict: Dictionary containing Dynawo contingency data.
        dynawo_nocontg_tap_dict: Dictionary containing tap data from the no-contingency case.
        dynawo_output_folder: Path to the Dynawo output folder.
    """
    for contg in dynawo_contingencies_dict.keys():
        if dynawo_contingencies_dict[contg]["status"] == "CONVERGENCE":
            constraints_file = dynawo_output_folder / "constraints" / f"constraints_{contg}.xml"
            parsed_constraints_file = manage_files.parse_xml_file(constraints_file)
            root = parsed_constraints_file.getroot()
            ns = etree.QName(root).namespace

            dynawo_contingencies_dict[contg]["constraints"].extend(
                [entry.attrib for entry in root.iter(f"{{{ns}}}constraint")]
            )

            timeline_file = dynawo_output_folder / "timeLine" / f"timeline_{contg}.xml"
            parsed_timeline_file = manage_files.parse_xml_file(timeline_file)
            root = parsed_timeline_file.getroot()
            ns = etree.QName(root).namespace

            dynawo_contingencies_dict[contg]["constraints"] = get_dynawo_timeline_constraints(
                root, ns, dynawo_contingencies_dict[contg]["constraints"]
            )

            contg_output_file = (
                dynawo_output_folder / contg / "outputs" / "finalState" / "outputIIDM.xml"
            )
            parsed_output_file = manage_files.parse_xml_file(contg_output_file)
            root = parsed_output_file.getroot()
            ns = etree.QName(root).namespace

            dynawo_contingencies_dict[contg]["tap_changers"] = get_dynawo_tap_data(root, ns)
            get_dynawo_tap_diffs(dynawo_contingencies_dict, dynawo_nocontg_tap_dict, contg)


def get_dynawo_tap_data(
    output_file_root: etree._Element, ns: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Extracts tap data from a parsed Dynawo output XML.

    Args:
        output_file_root: Root element of the Dynawo output XML.
        ns: Namespace of the XML.

    Returns:
        Dictionary containing tap data.
    """
    dynawo_taps_dict: Dict[str, Dict[str, Dict[str, Any]]] = {
        "phase_taps": {},
        "ratio_taps": {},
    }

    for phase_tap in output_file_root.iter(f"{{{ns}}}phaseTapChanger"):
        phase_tap_dict: Dict[str, Any] = dict(phase_tap.attrib)
        phase_tap_transformer_id = phase_tap.getparent().attrib["id"]
        dynawo_taps_dict["phase_taps"][phase_tap_transformer_id] = phase_tap_dict

    for ratio_tap in output_file_root.iter(f"{{{ns}}}ratioTapChanger"):
        ratio_tap_dict: Dict[str, Any] = dict(ratio_tap.attrib)
        ratio_tap_transformer_id = ratio_tap.getparent().attrib["id"]
        dynawo_taps_dict["ratio_taps"][ratio_tap_transformer_id] = ratio_tap_dict

    return dynawo_taps_dict


def get_dynawo_tap_diffs(
    dynawo_contingencies_dict: Dict[str, Dict[str, Any]],
    dynawo_nocontg_tap_dict: Dict[str, Dict[str, Dict[str, Any]]],
    contingency_name: str,
) -> None:
    """
    Calculates differences in tap positions between contingency and no-contingency cases.

    Args:
        dynawo_contingencies_dict: Dictionary containing Dynawo contingency data.
        dynawo_nocontg_tap_dict: Dictionary containing tap data from the no-contingency case.
        contingency_name: Name of the contingency.
    """
    tap_diff_dict: Dict[str, Dict[str, int]] = {"phase_taps": {}, "ratio_taps": {}}

    for phase_tap_id, contg_phase_tap in dynawo_contingencies_dict[contingency_name][
        "tap_changers"
    ]["phase_taps"].items():
        if phase_tap_id in dynawo_nocontg_tap_dict["phase_taps"]:
            nocontg_phase_tap = dynawo_nocontg_tap_dict["phase_taps"][phase_tap_id]
            phase_tap_diff = int(contg_phase_tap["tapPosition"]) - int(
                nocontg_phase_tap["tapPosition"]
            )
            if phase_tap_diff != 0:
                tap_diff_dict["phase_taps"][phase_tap_id] = phase_tap_diff

    for ratio_tap_id, contg_ratio_tap in dynawo_contingencies_dict[contingency_name][
        "tap_changers"
    ]["ratio_taps"].items():
        if ratio_tap_id in dynawo_nocontg_tap_dict["ratio_taps"]:
            nocontg_ratio_tap = dynawo_nocontg_tap_dict["ratio_taps"][ratio_tap_id]
            ratio_tap_diff = int(contg_ratio_tap["tapPosition"]) - int(
                nocontg_ratio_tap["tapPosition"]
            )
            if ratio_tap_diff != 0:
                tap_diff_dict["ratio_taps"][ratio_tap_id] = ratio_tap_diff

    dynawo_contingencies_dict[contingency_name]["tap_diffs"] = tap_diff_dict


def collect_dynawo_results(
    parsed_output_xml: etree._ElementTree,
    parsed_aggregated_xml: etree._ElementTree,
    dynawo_output_dir: str,
    contg_set: set,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Collects Dynawo results from parsed XML files.

    Args:
        parsed_output_xml: Parsed output XML tree.
        parsed_aggregated_xml: Parsed aggregated XML tree.
        dynawo_output_dir: Path to the Dynawo output directory.
        contg_set: Set of contingency identifiers.

    Returns:
        Tuple containing Dynawo contingency data and no-contingency tap data.
    """
    root = parsed_output_xml.getroot()
    ns = etree.QName(root).namespace
    dynawo_nocont_tap_dict = get_dynawo_tap_data(root, ns)

    root = parsed_aggregated_xml.getroot()
    ns = etree.QName(root).namespace
    dynawo_contingencies_dict = get_dynawo_contingencies(root, ns, contg_set)

    get_dynawo_contingency_data(
        dynawo_contingencies_dict, dynawo_nocont_tap_dict, dynawo_output_dir
    )

    return dynawo_contingencies_dict, dynawo_nocont_tap_dict
