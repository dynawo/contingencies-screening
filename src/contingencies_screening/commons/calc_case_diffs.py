from typing import Any, Dict, List, Set, Tuple
import numpy as np
import math


def get_hades_id(case: str, sorted_loadflow_score_list: List[Tuple[float, Dict[str, Any]]]) -> int:
    # Get the contingency id of Hades
    for i in range(len(sorted_loadflow_score_list)):
        if case == sorted_loadflow_score_list[i][1]["name"]:
            return i
    exit("Error, hades index not found")


def compare_status(hades_status: int, dynawo_status: str) -> str:
    # Compare the final state of the two loadflows
    if hades_status == 0 and dynawo_status == "CONVERGENCE":
        return "BOTH"
    elif hades_status != 0 and dynawo_status == "CONVERGENCE":
        return "DWO"
    elif hades_status == 0 and dynawo_status == "DIVERGENCE":
        return "HDS"
    else:
        return "NONE"


def match_3_dictionaries(
    keys1: Set[Any], keys2: Set[Any], keys3: Set[Any]
) -> Tuple[Set[Any], Set[Any], Set[Any], Set[Any]]:
    # Keys that match in all three dictionaries
    matching_keys2 = keys1.intersection(keys2)
    matching_keys3 = keys1.intersection(keys3)

    matching_keys = matching_keys2.union(matching_keys3)

    # Keys that don't match in each dictionary
    keys1_not_matching = keys1.difference(matching_keys)
    keys2_not_matching = keys2.difference(matching_keys)
    keys3_not_matching = keys3.difference(matching_keys)

    return matching_keys, keys1_not_matching, keys2_not_matching, keys3_not_matching


def get_tap_score_diff(
    tap1_diff: float,
    tap2_diff: float,
    lim1: bool,
    lim2: bool,
    block1: bool,
    block2: bool,
) -> float:
    # Calculate the difference between two taps
    diff_score = 0
    weight_diff = 10

    # Get a score for tap diffs movement
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

    elif tap1_diff == 0 and tap2_diff == 0:
        diff_score += 0

    else:
        print("Warning, tap diff case not contemplated.")

    # Add to the movement difference score if the taps have been saturated or not
    if (lim1 and not lim2) or (not lim1 and lim2):
        diff_score += 1

    elif lim1 and lim2:
        if (tap1_diff < 0 and tap2_diff > 0) or (tap1_diff > 0 and tap2_diff < 0):
            diff_score += 3
        elif (tap1_diff < 0 and tap2_diff < 0) or (tap1_diff > 0 and tap2_diff > 0):
            diff_score += 0.5
        else:
            print("Warning, tap diff lim case not contemplated.")

    elif not lim1 and not lim2:
        diff_score += 0

    else:
        print("Warning, tap diff lim case not contemplated.")

    # Add to movement difference score whether or not taps have been blocked
    if (block1 and not block2) or (not block1 and block2):
        diff_score += 5

    elif block1 and block2:
        diff_score += 1

    elif not block1 and not block2:
        diff_score += 0

    else:
        print("Warning, tap diff block case not contemplated.")

    return diff_score * weight_diff


def compare_taps(hades_taps: List[Dict[str, Any]], dwo_taps: Dict[str, Dict[str, float]]) -> float:
    # Calculate the differences between the final states of all the taps of the two loadflows

    final_tap_score = 0

    set_phase_taps = set(dwo_taps["phase_taps"].keys())
    set_ratio_taps = set(dwo_taps["ratio_taps"].keys())
    # Match the different taps of the two snapshots
    (
        matching_keys,
        keys_hades_not_matching,
        keys_phase_not_matching,
        keys_ratio_not_matching,
    ) = match_3_dictionaries(
        set([hds_tap["quadripole_name"] for hds_tap in hades_taps]),
        set_phase_taps,
        set_ratio_taps,
    )

    # Calculate scores for matched taps
    for matching_key in matching_keys:
        dwo_diff: float
        if matching_key in dwo_taps["phase_taps"]:
            dwo_diff = dwo_taps["phase_taps"][matching_key]
        elif matching_key in dwo_taps["ratio_taps"]:
            dwo_diff = dwo_taps["ratio_taps"][matching_key]
        else:
            continue  # Should not happen based on matching_keys

        hds_tap: Dict[str, Any] = {}
        for hds_tap_ent in hades_taps:
            if hds_tap_ent["quadripole_name"] == matching_key:
                hds_tap = hds_tap_ent
                break

        hds_diff = hds_tap["diff_value"]
        lim_hds = False
        block_hds = False
        if int(hds_tap["stopper"]) == 2 or int(hds_tap["stopper"]) == 1:
            lim_hds = True
        elif int(hds_tap["stopper"]) == 3:
            block_hds = True

        # TODO: get block and lim dwo
        final_tap_score += get_tap_score_diff(hds_diff, dwo_diff, lim_hds, False, block_hds, False)

    # Calculate scores for taps found only in Hades
    for hades_key in keys_hades_not_matching:
        hds_tap: Dict[str, Any] = {}
        for hds_tap_ent in hades_taps:
            if hds_tap_ent["quadripole_name"] == hades_key:
                hds_tap = hds_tap_ent
                break

        hds_diff = hds_tap["diff_value"]
        lim_hds = False
        block_hds = False
        if int(hds_tap["stopper"]) == 2 or int(hds_tap["stopper"]) == 1:
            lim_hds = True
        elif int(hds_tap["stopper"]) == 3:
            block_hds = True

        final_tap_score += get_tap_score_diff(hds_diff, 0, lim_hds, False, block_hds, False)

    # Calculate the scores for the taps that are only found in Dynawo
    for dwo_key in keys_phase_not_matching:
        dwo_diff = dwo_taps["phase_taps"][dwo_key]

        # TODO: get block and lim dwo
        final_tap_score += get_tap_score_diff(0, dwo_diff, False, False, False, False)

    for dwo_key in keys_ratio_not_matching:
        dwo_diff = dwo_taps["ratio_taps"][dwo_key]

        # TODO: get block and lim dwo
        final_tap_score += get_tap_score_diff(0, dwo_diff, False, False, False, False)

    return final_tap_score


def calc_volt_constr(
    matched_volt_constr: Dict[str, List[Dict[str, Any]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    # Calculate the REAL difference between the final values of the loadflows of the corresponding
    # variable
    diff_score_volt = 0

    # TODO: Check double constraints
    for case_diffs_list in matched_volt_constr.values():
        for case_diffs in case_diffs_list:
            if int(case_diffs[0]["threshType"]) == 1 and case_diffs[1]["kind"] == "UInfUmin":
                diff_score_volt += 1
            elif int(case_diffs[0]["threshType"]) == 1 and case_diffs[1]["kind"] == "USupUmax":
                diff_score_volt += 5
            elif int(case_diffs[0]["threshType"]) == 0 and case_diffs[1]["kind"] == "UInfUmin":
                diff_score_volt += 5
            elif int(case_diffs[0]["threshType"]) == 0 and case_diffs[1]["kind"] == "USupUmax":
                diff_score_volt += 1
            else:
                print("Volt constraint type not matched.")

    for case_diffs in unique_constr_hds:
        diff_score_volt += 3
    for case_diffs in unique_constr_dwo:
        diff_score_volt += 3

    return diff_score_volt


def calc_flow_constr(
    matched_flow_constr: Dict[str, List[Dict[str, Any]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    # Calculate the REAL difference between the final values of the loadflows of the corresponding
    # variable
    diff_score_flow = 0

    # TODO: Check double constraints
    for case_diffs_list in matched_flow_constr.values():
        for case_diffs in case_diffs_list:
            if case_diffs[1]["kind"] == "PATL" or case_diffs[1]["kind"] == "OverloadUp":
                diff_score_flow += 3
            elif case_diffs[1]["kind"] == "OverloadOpen":
                diff_score_flow += 10
            else:
                print("flow constraint type not matched.")

    for case_diffs in unique_constr_hds:
        diff_score_flow += 3
    for case_diffs in unique_constr_dwo:
        diff_score_flow += 3

    return diff_score_flow


def calc_gen_Q_constr(
    matched_gen_Q_constr: Dict[str, List[Dict[str, Any]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    # Calculate the REAL difference between the final values of the loadflows of the corresponding
    # variable
    diff_score_gen_Q = 0

    # TODO: Check double constraints
    for case_diffs_list in matched_gen_Q_constr.values():
        for case_diffs in case_diffs_list:
            if int(case_diffs[0]["typeLim"]) == 1 and case_diffs[1]["kind"] == "QInfQMin":
                diff_score_gen_Q += 1
            elif int(case_diffs[0]["typeLim"]) == 1 and case_diffs[1]["kind"] == "QSupQMax":
                diff_score_gen_Q += 5
            elif int(case_diffs[0]["typeLim"]) == 0 and case_diffs[1]["kind"] == "QInfQMin":
                diff_score_gen_Q += 5
            elif int(case_diffs[0]["typeLim"]) == 0 and case_diffs[1]["kind"] == "QSupQMax":
                diff_score_gen_Q += 1
            else:
                print("gen_Q constraint type not matched.")

    for case_diffs in unique_constr_hds:
        diff_score_gen_Q += 3
    for case_diffs in unique_constr_dwo:
        diff_score_gen_Q += 3

    return diff_score_gen_Q


def calc_gen_U_constr(
    matched_gen_U_constr: Dict[str, List[Dict[str, Any]]],
    unique_constr_hds: List[Dict[str, Any]],
    unique_constr_dwo: List[Dict[str, Any]],
) -> float:
    # Calculate the REAL difference between the final values of the loadflows of the corresponding
    # variable
    diff_score_gen_U = 0
    # TODO: Implement it
    return diff_score_gen_U


def match_constraints(
    hades_constr: List[Dict[str, Any]], dwo_constraints: List[Dict[str, Any]]
) -> Dict[str, List[List[Dict[str, Any]]]]:
    # Get the constraints shared between Hades and Dynawo
    matched_constr: Dict[str, List[List[Dict[str, Any]]]] = {}

    for constr_hds in hades_constr:
        for constr_dwo in dwo_constraints:
            if constr_hds["elem_name"] == constr_dwo["modelName"]:
                if constr_hds["elem_name"] in matched_constr:
                    matched_constr[constr_hds["elem_name"]].append([constr_hds, constr_dwo])
                else:
                    matched_constr[constr_hds["elem_name"]] = [[constr_hds, constr_dwo]]

    return matched_constr


def get_unmatched_constr(
    hades_constr: List[Dict[str, Any]],
    dwo_constraints: List[Dict[str, Any]],
    matched_constr: Dict[str, List[List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Get the constraints not shared between Hades and Dynawo

    unique_constr_hds: List[Dict[str, Any]] = []
    unique_constr_dwo: List[Dict[str, Any]] = []

    for constr_hds in hades_constr:
        if constr_hds["elem_name"] not in matched_constr:
            unique_constr_hds.append(constr_hds)

    for constr_dwo in dwo_constraints:
        if constr_dwo["modelName"] not in matched_constr:
            unique_constr_dwo.append(constr_dwo)

    return unique_constr_hds, unique_constr_dwo


def compare_constraints(
    hades_constr_volt: List[Dict[str, Any]],
    hades_constr_flow: List[Dict[str, Any]],
    hades_constr_gen_Q: List[Dict[str, Any]],
    hades_constr_gen_U: List[Dict[str, Any]],
    dwo_constraints: List[Dict[str, Any]],
) -> float:
    # Calculate the final REAL value of the differences between the constraints obtained in Dynawo
    # and in Hades
    # Separe dynawo constraints
    dwo_volt_constraints: List[Dict[str, Any]] = []
    dwo_gen_Q_constraints: List[Dict[str, Any]] = []
    dwo_gen_U_constraints: List[Dict[str, Any]] = []
    dwo_flow_constraints: List[Dict[str, Any]] = []
    dwo_non_defined_constraints: List[Dict[str, Any]] = []

    weight_diff_volt = 20
    weight_diff_flow = 20
    weight_diff_gen_Q = 20
    weight_diff_gen_U = 20

    for constraint in dwo_constraints:
        if constraint["kind"] == "UInfUmin" or constraint["kind"] == "USupUmax":
            dwo_volt_constraints.append(constraint)
        elif (
            constraint["kind"] == "PATL"
            or constraint["kind"] == "OverloadOpen"
            or constraint["kind"] == "OverloadUp"
        ):
            dwo_flow_constraints.append(constraint)
        elif constraint["kind"] == "QInfQMin" or constraint["kind"] == "QSupQMax":
            dwo_gen_Q_constraints.append(constraint)
        # TODO: Change this condition
        elif constraint["kind"] == "Pending":
            dwo_gen_U_constraints.append(constraint)
        else:
            dwo_non_defined_constraints.append(constraint)

    # Compare volt constraints
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
        (diff_score_volt * weight_diff_volt)
        + (diff_score_flow * weight_diff_flow)
        + (diff_score_gen_Q * weight_diff_gen_Q)
        + (diff_score_gen_U * weight_diff_gen_U)
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
    # Calculation of the RMSE between the prediction and the real score
    df_contg = df_contg.loc[df_contg["STATUS"] == "BOTH"]

    mse = np.square(
        np.subtract(
            df_contg["REAL_SCORE"].astype("float"), df_contg["PREDICTED_SCORE"].astype("float")
        )
    ).mean()
    rmse = math.sqrt(mse)
    return rmse
