import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


def calc_diff_volt(
    contingency_values: List[List[Union[int, float]]],
    loadflow_values: Dict[int, Dict[str, float]],
) -> float:
    """Calculates the total voltage difference between contingency values and load flow values."""
    return sum(
        abs(poste_v[1] - loadflow_values[poste_v[0]]["volt"]) for poste_v in contingency_values
    ) + len(contingency_values)


def calc_diff_max_flow(list_values: List[List[float]]) -> float:
    """Calculates the total maximum flow difference."""
    return sum(abs(max_flow[1] / 10) for max_flow in list_values) + len(list_values)


def calc_constr_gen_Q(
    contingency_values: List[Dict[str, Any]], elem_dict: Dict[int, Dict[str, Any]]
) -> float:
    """Calculates the constraint score for reactive power generation (Q)."""
    return sum(
        abs(float(constr["after"]) - float(constr["before"]))
        * (1 + elem_dict[constr["elem_num"]]["volt_level"] / 10)
        for constr in contingency_values
    ) + len(contingency_values)


def calc_constr_gen_U(
    contingency_values: List[Dict[str, Any]], elem_dict: Dict[int, Dict[str, Any]]
) -> float:
    """Calculates the constraint score for active power generation (U)."""
    return sum(
        abs(float(constr["after"]) - float(constr["before"]))
        * (1 + elem_dict[constr["elem_num"]]["volt_level"] / 10)
        for constr in contingency_values
    ) + len(contingency_values)


def calc_constr_volt(
    contingency_values: List[Dict[str, Any]], elem_dict: Dict[int, Dict[str, Any]]
) -> float:
    """Calculates the value of the voltage constraint."""
    final_value = 0
    for volt_constr in contingency_values:
        tempo = int(volt_constr["tempo"])
        value = 5 if tempo in (99999, 9999) else min((1 / tempo) * 10000, 100)
        final_value += value * (1 + elem_dict[volt_constr["elem_num"]]["volt_level"] / 10)
    return final_value


def calc_constr_flow(
    contingency_values: List[Dict[str, Any]], elem_dict: Dict[int, Dict[str, Any]]
) -> float:
    """Calculates the value of the flow constraint."""
    final_value = 0
    for flow_constr in contingency_values:
        tempo = int(flow_constr["tempo"])
        value = 5 if tempo in (99999, 9999) else min((1 / tempo) * 10000, 100)
        final_value += value * (1 + elem_dict[flow_constr["elem_num"]]["volt_level"] / 10)
    return final_value


STD_TAP_VALUE = 20


def analyze_loadflow_results_continuous(
    contingencies_dict: Dict[str, Dict[str, Any]],
    elements_dict: Dict[str, Dict[int, Dict[str, Any]]],
    tap_changers: bool,
    model_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Predicts the difference between Hades and Dynawo load flow calculation results."""

    print(
        "\nWARNING: Remember that if you have selected the human analysis option, you must provide the path of a LR in JSON format that matches (has been trained) the option selected on the taps (activated or not activated).\n"
    )

    if model_path is None:
        model_path = Path(os.path.dirname(os.path.realpath(__file__))) / (
            "LR_taps.json" if tap_changers else "LR_no_taps.json"
        )

    with open(model_path) as f:
        data = json.load(f)

    w_volt_min = data["MIN_VOLT"]
    w_volt_max = data["MAX_VOLT"]
    w_iter = data["N_ITER"]
    w_poste = data["AFFECTED_ELEM"]
    w_constr_gen_Q = data["CONSTR_GEN_Q"]
    w_constr_gen_U = data["CONSTR_GEN_U"]
    w_constr_volt = data["CONSTR_VOLT"]
    w_constr_flow = data["CONSTR_FLOW"]
    w_node = data["RES_NODE"]
    w_tap = data.get("TAP_CHANGERS", 0) if tap_changers else 0
    w_flow = data["MAX_FLOW"]
    w_coefreport = data["COEF_REPORT"]
    independent_term = data["INTERCEPTION"]

    for key, contingency in contingencies_dict.items():
        if contingency["status"] == 0:
            diff_min_voltages = calc_diff_volt(contingency["min_voltages"], elements_dict["poste"])
            diff_max_voltages = calc_diff_volt(contingency["max_voltages"], elements_dict["poste"])
            diff_max_flows = calc_diff_max_flow(contingency["max_flow"])
            value_constr_gen_Q = calc_constr_gen_Q(
                contingency["constr_gen_Q"], elements_dict["groupe"]
            )
            value_constr_gen_U = calc_constr_gen_U(
                contingency["constr_gen_U"], elements_dict["groupe"]
            )
            value_constr_volt = calc_constr_volt(
                contingency["constr_volt"], elements_dict["noeud"]
            )
            value_constr_flow = calc_constr_flow(
                contingency["constr_flow"], elements_dict["quadripole"]
            )

            total_tap_value = sum(
                abs(tap["diff_value"]) * w_tap
                if int(tap["stopper"]) == 0
                else STD_TAP_VALUE * w_tap
                for tap in contingency.get("tap_changers", [])
            )

            contingency["final_score"] = round(
                (
                    (diff_min_voltages * w_volt_min + diff_max_voltages * w_volt_max)
                    + contingency["n_iter"] * w_iter
                    + len(contingency["affected_elements"]) * w_poste
                    + value_constr_gen_Q * w_constr_gen_Q
                    + value_constr_gen_U * w_constr_gen_U
                    + value_constr_volt * w_constr_volt
                    + value_constr_flow * w_constr_flow
                    + len(contingency["res_node"]) * w_node
                    + diff_max_flows * w_flow
                    + len(contingency["coef_report"]) * w_coefreport
                    + total_tap_value
                    + independent_term
                ),
                4,
            )
        else:
            status_map = {
                1: "Divergence",
                2: "Generic fail",
                3: "No computation",
                4: "Interrupted",
                5: "No output",
                6: "Nonrealistic solution",
                7: "Power balance fail",
                8: "Timeout",
            }
            contingency["final_score"] = status_map.get(
                contingency["status"], "Final state unknown"
            )

    return contingencies_dict