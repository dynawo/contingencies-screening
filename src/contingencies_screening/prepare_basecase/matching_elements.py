from lxml import etree
from typing import Dict, List, Tuple, Set
from contingencies_screening.commons import manage_files
from contingencies_screening.prepare_basecase import create_contingencies


def get_dynawo_branches(dynawo_iidm_root: etree._Element, ns: str) -> Dict[str, str]:
    """Gets Dynawo branches as a dictionary."""

    dynawo_branches: Dict[str, str] = {}

    for dynawo_branch in dynawo_iidm_root.iter(f"{{{ns}}}line", f"{{{ns}}}twoWindingsTransformer"):
        if (
            dynawo_branch.get("p1") is None
            or dynawo_branch.get("q1") is None
            or dynawo_branch.get("p2") is None
            or dynawo_branch.get("q2") is None
            or float(dynawo_branch.get("p1", 0.0)) == 0.0
            and float(dynawo_branch.get("q1", 0.0)) == 0.0
            or float(dynawo_branch.get("p2", 0.0)) == 0.0
            and float(dynawo_branch.get("q2", 0.0)) == 0.0
        ):
            continue

        branch_name = dynawo_branch.get("id")
        branch_tag = etree.QName(dynawo_branch).localname

        branch_type = (
            "Line"
            if branch_tag == "line"
            else "PhaseShitfer"
            if dynawo_branch.find(f"{{{ns}}}phaseTapChanger") is not None
            else "Transformer"
        )

        bus_from = create_contingencies.get_endbus(
            dynawo_iidm_root, dynawo_branch, branch_type, "1"
        )
        bus_to = create_contingencies.get_endbus(dynawo_iidm_root, dynawo_branch, branch_type, "2")

        if bus_from is None or bus_to is None:
            continue

        dynawo_branches[branch_name] = branch_type

    return dynawo_branches


def get_hades_branches(hades_root: etree._Element) -> Set[str]:
    """Gets Hades branches as a set."""

    hades_branches: Set[str] = set()

    reseau = hades_root.find("./reseau", hades_root.nsmap)
    donneesQuadripoles = reseau.find("./donneesQuadripoles", hades_root.nsmap)

    hades_branches.update(
        branch.get("nom")
        for branch in donneesQuadripoles.iterfind("./quadripole", hades_root.nsmap)
    )

    return hades_branches


def extract_matching_branches(
    hades_root: etree._Element, dynawo_iidm_root: etree._Element
) -> Dict[str, str]:
    """Extracts matching branches between Hades and Dynawo."""

    ns = etree.QName(dynawo_iidm_root).namespace
    dynawo_branches = get_dynawo_branches(dynawo_iidm_root, ns)
    hades_branches = get_hades_branches(hades_root)

    return {
        branch_name: branch_type
        for branch_name, branch_type in dynawo_branches.items()
        if branch_name in hades_branches
    }


def get_dynawo_generators(dynawo_iidm_root: etree._Element, ns: str) -> List[str]:
    """Gets Dynawo generators as a list."""

    dynawo_generators: List[str] = []

    for gen in dynawo_iidm_root.iter(f"{{{ns}}}generator"):
        if gen.get("p") is None or (gen.get("q") is None and gen.get("targetQ") is None):
            continue

        P_val = float(gen.get("p", 0.0))
        Q_val = float(gen.get("q", gen.get("targetQ", 0.0)))

        if P_val == 0.0 and Q_val == 0.0:
            continue
        if gen.get("bus") is None:
            continue

        dynawo_generators.append(gen.get("id"))

    return dynawo_generators


def get_hades_generators(hades_root: etree._Element) -> Set[str]:
    """Gets Hades generators as a set."""

    hades_generators: Set[str] = set()

    reseau = hades_root.find("./reseau", hades_root.nsmap)
    donneesGroupes = reseau.find("./donneesGroupes", hades_root.nsmap)

    hades_generators.update(
        gen.get("nom")
        for gen in donneesGroupes.iterfind("./groupe", hades_root.nsmap)
        if gen.get("noeud") != "-1"
    )

    return hades_generators


def extract_matching_generators(
    hades_root: etree._Element, dynawo_iidm_root: etree._Element
) -> List[str]:
    """Extracts matching generators between Hades and Dynawo."""

    ns = etree.QName(dynawo_iidm_root).namespace
    dynawo_generators = get_dynawo_generators(dynawo_iidm_root, ns)
    hades_generators = get_hades_generators(hades_root)

    return [gen_name for gen_name in dynawo_generators if gen_name in hades_generators]


def get_dynawo_loads(dynawo_iidm_root: etree._Element) -> List[str]:
    """Gets Dynawo loads as a list."""

    dynawo_loads: List[str] = []

    ns = etree.QName(dynawo_iidm_root).namespace
    for load in dynawo_iidm_root.iter(f"{{{ns}}}load"):
        if load.getparent().get("topologyKind") == "BUS_BREAKER" and load.get("bus") is not None:
            dynawo_loads.append(load.get("id"))

    return dynawo_loads


def get_hades_loads(hades_root: etree._Element) -> Set[str]:
    """Gets Hades loads as a set."""

    hades_loads: Set[str] = set()

    reseau = hades_root.find("./reseau", hades_root.nsmap)
    donneesConsos = reseau.find("./donneesConsos", hades_root.nsmap)

    hades_loads.update(
        load.get("nom")
        for load in donneesConsos.iterfind("./conso", hades_root.nsmap)
        if load.get("noeud") != "-1"
    )

    return hades_loads


def extract_matching_loads(
    hades_root: etree._Element, dynawo_iidm_root: etree._Element
) -> List[str]:
    """Extracts matching loads between Hades and Dynawo."""

    dynawo_loads = get_dynawo_loads(dynawo_iidm_root)
    hades_loads = get_hades_loads(hades_root)

    return [load_name for load_name in dynawo_loads if load_name in hades_loads]


def get_dynawo_shunts(dynawo_iidm_root: etree._Element, ns: str) -> List[str]:
    """Gets Dynawo shunts as a list."""

    dynawo_shunts: List[str] = []

    ns = etree.QName(dynawo_iidm_root).namespace
    dynawo_shunts.extend(
        shunt.get("id")
        for shunt in dynawo_iidm_root.iter(f"{{{ns}}}shunt")
        if shunt.get("bus") is not None
    )

    return dynawo_shunts


def get_hades_shunts(hades_root: etree._Element) -> Set[str]:
    """Gets Hades shunts as a set."""

    hades_shunts: Set[str] = set()

    reseau = hades_root.find("./reseau", hades_root.nsmap)
    donneesShunts = reseau.find("./donneesShunts", hades_root.nsmap)

    hades_shunts.update(
        shunt.get("nom")
        for shunt in donneesShunts.iterfind("./shunt", hades_root.nsmap)
        if shunt.get("noeud") != "-1"
    )

    return hades_shunts


def extract_matching_shunts(
    hades_root: etree._Element, dynawo_iidm_root: etree._Element
) -> List[str]:
    """Extracts matching shunts between Hades and Dynawo."""

    ns = etree.QName(dynawo_iidm_root).namespace
    dynawo_shunts = get_dynawo_shunts(dynawo_iidm_root, ns)
    hades_shunts = get_hades_shunts(hades_root)

    return [shunt_name for shunt_name in dynawo_shunts if shunt_name in hades_shunts]


def matching_elements(
    hades_input_file: str, iidm_file_path: str
) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
    """Gets all matching elements between Hades and Dynawo."""

    hades_input_file_parsed = manage_files.parse_xml_file(hades_input_file)
    hades_root = hades_input_file_parsed.getroot()
    dynawo_iidm_file_parsed = manage_files.parse_xml_file(iidm_file_path)
    dynawo_iidm_root = dynawo_iidm_file_parsed.getroot()

    matched_branches = extract_matching_branches(hades_root, dynawo_iidm_root)
    matched_generators = extract_matching_generators(hades_root, dynawo_iidm_root)
    matched_loads = extract_matching_loads(hades_root, dynawo_iidm_root)
    matched_shunts = extract_matching_shunts(hades_root, dynawo_iidm_root)

    return matched_branches, matched_generators, matched_loads, matched_shunts
