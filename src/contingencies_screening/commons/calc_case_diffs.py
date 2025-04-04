import numpy as np
import math
from typing import List, Dict, Any, Tuple


def get_hades_id(case: str, sorted_loadflow_score_list: List[Tuple[int, Dict[str, Any]]]) -> int:
    """Gets the Hades contingency ID."""
    for i, (_, contg_data) in enumerate(sorted_loadflow_score_list):
        if case == contg_data["name"]:
            return i
    raise ValueError("Error, Hades index not found")


def compare_status(hades_status: int, dynawo_status: str) -> str:
    """Compares the final state of the two loadflows."""
    if hades_status == 0 and dynawo_status == "CONVERGENCE":
        return "BOTH"
    elif hades_status != 0 and dynawo_status == "CONVERGENCE":
        return "DWO"
    elif hades_status == 0 and dynawo_status == "DIVERGENCE":
        return "HDS"
    else:
        return "NONE"


def match_3_dictionaries(keys1: set, keys2: set, keys3: set) -> Tuple[set, set, set, set]:
    """Finds matching and non-matching keys in three sets."""
    matching_keys = keys1.intersection(keys2).union(keys1.intersection(keys3))
    keys1_not_matching = keys1.difference(matching_keys)
    keys2_not_matching = keys2.difference(matching_keys)
    keys3_not_matching = keys3.difference(matching_keys)
    return matching_keys, keys1_not_matching, keys2_not_matching, keys3_not_matching


def get_tap_score_diff(
    tap1_diff: int, tap2_diff: int, lim1: bool, lim2: bool, block1: bool, block2: bool
) -> float:
    """Calculates the difference between two taps."""
    diff_score = 0
    weight_diff = 10

    if (
        (tap1_diff < 0 and tap2_diff > 0)
        or (tap1_diff > 0 and tap2_diff < 0)
        or (tap1_diff < 0 and tap2_diff == 0)
        or (tap1_diff == 0 and tap2_diff < 0)
        or (tap1_diff > 0 and tap2_diff == 0)
        or (tap1_diff == 0 and tap2_diff > 0)
    ):
        diff_score += (abs(tap1_diff) + abs(tap2_diff)) * 2
    elif (tap1_diff < 0 and tap2_diff < 0) or (tap1_diff > 0 and tap2_diff > 0):
        diff_score += abs(abs(tap1_diff) - abs(tap2_diff))
    elif tap1_diff != 0 or tap2_diff != 0:
        print("Warning, tap diff case not contemplated.")

    if (lim1 and not lim2) or (not lim1 and lim2):
        diff_score += 1
    elif lim1 and lim2:
        if (tap1_diff < 0 and tap2_diff > 0) or (tap1_diff > 0 and tap2_diff < 0):
            diff_score += 3
        elif (tap1_diff < 0 and tap2_diff < 0) or (tap1_diff > 0 and tap2_diff > 0):
            diff_score += 0.5
        elif tap1_diff != 0 or tap2_diff != 0:
            print("Warning, tap diff lim case not contemplated.")
    elif lim1 or lim2:
        pass  # Both are False, no action needed

    if (block1 and not block2) or (not block1 and block2):
        diff_score += 5
    elif block1 and block2:
        diff_score += 1
    elif block1 or block2:
        pass  # Both are False, no action needed
    elif block1 != block2:
        print("Warning, tap diff block case not contemplated.")

    return diff_score * weight_diff


def compare_taps(hades_taps: List[Dict[str, Any]], dwo_taps: Dict[str, Dict[str, Any]]) -> float:
    """Calculates differences between tap states."""

    final_tap_score = 0

    set_phase_taps = set(dwo_taps["phase_taps"].keys())
    set_ratio_taps = set(dwo_taps["ratio_taps"].keys())
    hades_tap_names = set(hds_tap["quadripole_name"] for hds_tap in hades_taps)

    (
        matching_keys,
        keys_hades_not_matching,
        keys_phase_not_matching,
        keys_ratio_not_matching,
    ) = match_3_dictionaries(hades_tap_names, set_phase_taps, set_ratio_taps)

    for matching_key in matching_keys:
        dwo_diff = dwo_taps.get("phase_taps", {}).get(matching_key) or dwo_taps.get(
            "ratio_taps", {}
        ).get(matching_key)
        hds_tap = next(
            (
                hds_tap_ent
                for hds_tap_ent in hades_taps
                if hds_tap_ent["quadripole_name"] == matching_key
            ),
            {},
        )

        hds_diff = hds_tap.get("diff_value", 0)
        lim_hds = hds_tap.get("stopper") in (2, 1)
        block_hds = hds_tap.get("stopper") == 3

        final_tap_score += get_tap_score_diff(
            hds_diff, dwo_diff.get("tapPosition", 0), lim_hds, False, block_hds, False
        )

    for hades_key in keys_hades_not_matching:
        hds_tap = next(
            (
                hds_tap_ent
                for hds_tap_ent in hades_taps
                if hds_tap_ent["quadripole_name"] == hades_key
            ),
            {},
        )

        hds_diff = hds_tap.get("diff_value", 0)
        lim_hds = hds_tap.get("stopper") in (2, 1)
        block_hds = hds_tap.get("stopper") == 3

        final_tap_score += get_tap_score_diff(hds_diff, 0, lim_hds, False, block_hds, False)

    for dwo_key in keys_phase_not_matching:
        dwo_diff = dwo_taps["phase_taps"][dwo_key]
        final_tap_score += get_tap_score_diff(
            0, dwo_diff.get("tapPosition", 0), False, False, False, False
        )

    for dwo_key in keys_ratio_not_matching:
        dwo_diff = dwo_taps["ratio_taps"][dwo_key]
        final_tap_score += get_tap_score_diff(
            0, dwo_diff.get("tapPosition", 0), False, False, False, False
        )

    return final_tap_score


def calc_volt_constr(
    matched_volt_constr: Dict[str, List[List[Dict[str, Any]]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    """Calculates the difference for voltage constraints."""
    diff_score_volt = 0

    for case_diffs_list in matched_volt_constr.values():
        for case_diffs in case_diffs_list:
            hds_type = int(case_diffs[0].get("threshType", -1))
            dwo_kind = case_diffs[1].get("kind", "")
            if hds_type == 1 and dwo_kind == "UInfUmin":
                diff_score_volt += 1
            elif hds_type == 1 and dwo_kind == "USupUmax":
                diff_score_volt += 5
            elif hds_type == 0 and dwo_kind == "UInfUmin":
                diff_score_volt += 5
            elif hds_type == 0 and dwo_kind == "USupUmax":
                diff_score_volt += 1
            elif hds_type != -1 and dwo_kind:
                print("Volt constraint type not matched.")

    diff_score_volt += 3 * (len(unique_constr_hds) + len(unique_constr_dwo))

    return diff_score_volt


def calc_flow_constr(
    matched_flow_constr: Dict[str, List[List[Dict[str, Any]]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    """Calculates the difference for flow constraints."""
    diff_score_flow = 0

    for case_diffs_list in matched_flow_constr.values():
        for case_diffs in case_diffs_list:
            dwo_kind = case_diffs[1].get("kind", "")
            if dwo_kind in ("PATL", "OverloadUp"):
                diff_score_flow += 3
            elif dwo_kind == "OverloadOpen":
                diff_score_flow += 10
            elif dwo_kind:
                print("Flow constraint type not matched.")

    diff_score_flow += 3 * (len(unique_constr_hds) + len(unique_constr_dwo))

    return diff_score_flow


def calc_gen_Q_constr(
    matched_gen_Q_constr: Dict[str, List[List[Dict[str, Any]]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    """Calculates the difference for generator Q constraints."""
    diff_score_gen_Q = 0

    for case_diffs_list in matched_gen_Q_constr.values():
        for case_diffs in case_diffs_list:
            hds_type = int(case_diffs[0].get("typeLim", -1))
            dwo_kind = case_diffs[1].get("kind", "")
            if hds_type == 1 and dwo_kind == "QInfQMin":
                diff_score_gen_Q += 1
            elif hds_type == 1 and dwo_kind == "QSupQMax":
                diff_score_gen_Q += 5
            elif hds_type == 0 and dwo_kind == "QInfQMin":
                diff_score_gen_Q += 5
            elif hds_type == 0 and dwo_kind == "QSupQMax":
                diff_score_gen_Q += 1
            elif hds_type != -1 and dwo_kind:
                print("Gen_Q constraint type not matched.")

    diff_score_gen_Q += 3 * (len(unique_constr_hds) + len(unique_constr_dwo))

    return diff_score_gen_Q


def calc_gen_U_constr(
    matched_gen_U_constr: Dict[str, List[List[Dict[str, Any]]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    """Calculates the difference for generator U constraints."""
    # TODO: Implement it
    return 0.0


def match_constraints(
    hades_constr: List[Dict[str, Any]], dwo_constraints: List[Dict[str, Any]]
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Finds matching constraints between Hades and Dynawo."""
    matched_constr: Dict[str, List[List[Dict[str, Any]]]] = {}

    for constr_hds in hades_constr:
        for constr_dwo in dwo_constraints:
            if constr_hds.get("elem_name") == constr_dwo.get("modelName"):
                matched_constr.setdefault(constr_hds["elem_name"], []).append(
                    [constr_hds, constr_dwo]
                )

    return matched_constr


def get_unmatched_constr(
    hades_constr: List[Dict[str, Any]],
    dwo_constraints: List[Dict[str, Any]],
    matched_constr: Dict[str, List[List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Gets constraints not shared between Hades and Dynawo."""
    unique_constr_hds = [
        constr_hds
        for constr_hds in hades_constr
        if constr_hds.get("elem_name") not in matched_constr
    ]
    unique_constr_dwo = [
        constr_dwo
        for constr_dwo in dwo_constraints
        if constr_dwo.get("modelName") not in matched_constr
    ]

    return unique_constr_hds, unique_constr_dwo


def compare_constraints(
    hades_constr_volt: List[Dict[str, Any]],
    hades_constr_flow: List[Dict[str, Any]],
    hades_constr_gen_Q: List[Dict[str, Any]],
    hades_constr_gen_U: List[Dict[str, Any]],
    dwo_constraints: List[Dict[str, Any]],
) -> float:
    """Calculates the difference between constraints in Dynawo and Hades."""

    dwo_volt_constraints = [
        constraint
        for constraint in dwo_constraints
        if constraint.get("kind") in ("UInfUmin", "USupUmax")
    ]
    dwo_gen_Q_constraints = [
        constraint
        for constraint in dwo_constraints
        if constraint.get("kind") in ("QInfQMin", "QSupQMax")
    ]
    dwo_gen_U_constraints = [
        constraint for constraint in dwo_constraints if constraint.get("kind") == "Pending"
    ]
    dwo_flow_constraints = [
        constraint
        for constraint in dwo_constraints
        if constraint.get("kind") in ("PATL", "OverloadOpen", "OverloadUp")
    ]

    matched_volt_constr = match_constraints(hades_constr_volt, dwo_volt_constraints)
    matched_flow_constr = match_constraints(hades_constr_flow, dwo_flow_constraints)
    matched_gen_Q_constr = match_constraints(hades_constr_gen_Q, dwo_gen_Q_constraints)
    matched_gen_U_constr = match_constraints(hades_constr_gen_U, dwo_gen_U_constraints)

    unique_constr_volt_hds, unique_constr_volt_dwo = get_unmatched_constr(
        hades_constr_volt, dwo_volt_constraints, matched_volt_constr
    )
    unique_constr_flow_hds, unique_constr_flow_dwo = get_unmatched_constr(
        hades_constr_flow, dwo_flow_constraints, matched_flow_constr
    )
    unique_constr_gen_Q_hds, unique_constr_gen_Q_dwo = get_unmatched_constr(
        hades_constr_gen_Q, dwo_gen_Q_constraints, matched_gen_Q_constr
    )
    unique_constr_gen_U_hds, unique_constr_gen_U_dwo = get_unmatched_constr(
        hades_constr_gen_U, dwo_gen_U_constraints, matched_gen_U_constr
    )

    weight_diff_volt = 20
    weight_diff_flow = 20
    weight_diff_gen_Q = 20
    weight_diff_gen_U = 20

    diff_score_volt = calc_volt_constr(
        matched_volt_constr, unique_constr_volt_hds, unique_constr_volt_dwo
    )
    diff_score_flow = calc_flow_constr(
        matched_flow_constr, unique_constr_flow_hds, unique_constr_flow_dwo
    )
    diff_score_gen_Q = calc_gen_Q_constr(
        matched_gen_Q_constr, unique_constr_gen_Q_hds, unique_constr_gen_Q_dwo
    )
    diff_score_gen_U = calc_gen_U_constr(
        matched_gen_U_constr, unique_constr_gen_U_hds, unique_constr_gen_U_dwo
    )

    return (
        diff_score_volt * weight_diff_volt
        + diff_score_flow * weight_diff_flow
        + diff_score_gen_Q * weight_diff_gen_Q
        + diff_score_gen_U * weight_diff_gen_U
    )


def calculate_diffs_hades_dynawo(
    hades_info: Dict[str, Any], dwo_info: Dict[str, Any]
) -> List[Any]:
    """Calculates the differences between Hades and Dynawo results."""

    dict_diffs: Dict[str, Any] = {}

    status_diff = compare_status(hades_info.get("status", -1), dwo_info.get("status", ""))

    if status_diff in ("DWO", "HDS"):
        dict_diffs["conv_status"] = status_diff
        dict_diffs["diff_value"] = 100000
    elif status_diff == "NONE":
        dict_diffs["conv_status"] = status_diff
        dict_diffs["diff_value"] = 50000
    else:
        dict_diffs["conv_status"] = status_diff

        taps_diff = 0
        if hades_info.get("tap_changers") and dwo_info.get("status") == "CONVERGENCE":
            taps_diff = compare_taps(hades_info["tap_changers"], dwo_info.get("tap_diffs", {}))

        constraints_diffs = compare_constraints(
            hades_info.get("constr_volt", []),
            hades_info.get("constr_flow", []),
            hades_info.get("constr_gen_Q", []),
            hades_info.get("constr_gen_U", []),
            dwo_info.get("constraints", []),
        )

        dict_diffs["diff_value"] = abs(taps_diff) + abs(constraints_diffs)

    return [
        hades_info.get("name", ""),
        dict_diffs["conv_status"],
        dict_diffs["diff_value"],
    ]


def calc_rmse(df_contg: Any) -> float:
    """Calculates the RMSE between predicted and real scores."""
    df_both = df_contg.loc[df_contg["STATUS"] == "BOTH"]
    if df_both.empty:
        return float("inf")  # Or another suitable value to indicate no valid data

    mse = np.mean(
        np.square(df_both["REAL_SCORE"].astype(float) - df_both["PREDICTED_SCORE"].astype(float))
    )
    rmse = math.sqrt(mse)
    return rmse
