import os
import copy
import multiprocessing
import pandas as pd
from pathlib import Path
from lxml import etree
from typing import Optional

import typer

# Assuming these modules exist in the specified structure
from contingencies_screening.run_loadflow import run_hades
from contingencies_screening.analyze_loadflow import (
    extract_results_data,
    human_analysis,
    machine_learning_analysis,
)
from contingencies_screening.prepare_basecase import (
    create_contingencies,
    matching_elements,
)
from contingencies_screening.run_dynawo import run_dynawo
from contingencies_screening.commons import manage_files, calc_case_diffs
from contingencies_screening.config import settings

app = typer.Typer(help="Contingencies Screening Tool")


def solve_launcher(launcher: Path) -> Path:
    """Resolves the launcher path."""
    return launcher.resolve() if launcher.is_file() else launcher


def run_hades_contingencies_code(
    hades_input_folder: Path,
    hades_output_folder: Path,
    hades_launcher: Path,
    tap_changers: bool,
    multithreading: bool,
    calc_contingencies: bool,
) -> tuple[Path, Path, int]:
    """Executes Hades for contingencies."""
    hades_input_list = list(hades_input_folder.glob("HDS*.xml"))
    if not hades_input_list:
        print(f"Hades input file not found in: {hades_input_folder}")
        return Path(), Path(), 1

    hades_input_file = hades_input_list[0]
    hades_output_file = hades_output_folder / "hadesOut.xml"
    hades_output_folder.mkdir(parents=True, exist_ok=True)

    status = run_hades.run_hades(
        hades_input_file,
        hades_output_file,
        str(hades_launcher),
        tap_changers,
        multithreading,
        calc_contingencies,
    )

    return hades_input_file, hades_output_file, status


def sort_ranking(elem: tuple) -> float:
    """Sorting criterion for contingency ranking."""
    return (
        elem[1].get("final_score", -float("inf"))
        if isinstance(elem[1].get("final_score"), (int, float))
        else -float("inf")
    )


def create_contingencies_ranking_code(
    hades_input_file: Path,
    hades_output_file: Path,
    output_dir_path: Path,
    score_type: int,
    tap_changers: bool,
    model_path: Optional[Path],
) -> tuple[list, int]:
    """Creates the contingency ranking."""
    parsed_hades_input_file = manage_files.parse_xml_file(hades_input_file)
    parsed_hades_output_file = manage_files.parse_xml_file(hades_output_file)

    if not parsed_hades_input_file or not parsed_hades_output_file:
        print(f"Error parsing Hades files: {hades_input_file}, {hades_output_file}")
        return [], 1

    hades_elements_dict = extract_results_data.get_elements_dict(parsed_hades_input_file)
    hades_contingencies_dict = extract_results_data.get_contingencies_dict(parsed_hades_input_file)

    (
        hades_elements_dict,
        hades_contingencies_dict,
        status,
    ) = extract_results_data.collect_hades_results(
        hades_elements_dict,
        hades_contingencies_dict,
        parsed_hades_output_file,
        tap_changers,
    )

    if status == 1:
        print("Error collecting Hades results.")
        return [], 1

    try:
        if score_type == 1:
            hades_contingencies_dict = human_analysis.analyze_loadflow_results_continuous(
                hades_contingencies_dict,
                hades_elements_dict,
                tap_changers,
                model_path,
            )
        elif score_type == 2:
            hades_contingencies_dict = machine_learning_analysis.analyze_loadflow_results(
                hades_contingencies_dict,
                hades_elements_dict,
                tap_changers,
                model_path,
            )
        else:
            print(f"Invalid score_type: {score_type}. Choose 1 or 2.")
            return [], 1
    except Exception as e:
        print(f"Error during result analysis (score_type={score_type}): {e}")
        return [], 1

    try:
        df_temp, error_contg = machine_learning_analysis.convert_dict_to_df(
            hades_contingencies_dict, hades_elements_dict, tap_changers, True
        )
        df_temp.to_csv(output_dir_path / "contg_df.csv", sep=";")
        if error_contg:
            print(
                "Warning: Errors encountered while converting to DataFrame for contingencies:"
                f" {error_contg}"
            )
    except Exception as e:
        print(f"Error converting results to DataFrame or saving CSV: {e}")

    sorted_items = sorted(hades_contingencies_dict.items(), key=sort_ranking, reverse=True)
    return sorted_items, 0


def prepare_hades_contingencies(
    sorted_loadflow_score_list: list,
    hades_input_file: Path,
    hades_output_folder: Path,
    number_pos_replay: int,
) -> list[Path]:
    """Prepares contingencies for Hades replay."""
    replay_contgs = [
        [elem_list[1]["name"], elem_list[1]["type"]]
        for elem_list in sorted_loadflow_score_list
        if "name" in elem_list[1] and "type" in elem_list[1]
    ]

    if number_pos_replay != -1:
        replay_contgs = replay_contgs[:number_pos_replay]

    hades_output_list = []
    dict_types_cont = create_contingencies.get_types_cont(hades_input_file)
    hades_input_file_parsed = manage_files.parse_xml_file(hades_input_file)

    if not hades_input_file_parsed:
        print(f"Error parsing Hades input file for replay preparation: {hades_input_file}")
        return []

    root = hades_input_file_parsed.getroot()
    namespace = (
        etree.QName(root).namespace
        if root is not None and hasattr(root, "tag") and isinstance(root.tag, str)
        else ""
    )
    create_contingencies.clean_contingencies(hades_input_file_parsed, root, namespace)

    for replay_cont in replay_contgs:
        if replay_cont[1] == 0:
            hades_input_file_parsed_copy = copy.deepcopy(hades_input_file_parsed)
            hades_output_file = create_contingencies.create_hades_contingency_n_1(
                hades_input_file,
                hades_input_file_parsed_copy,
                hades_output_folder,
                replay_cont[0],
                dict_types_cont,
            )
            if isinstance(hades_output_file, Path):
                hades_output_list.append(hades_output_file)
            elif hades_output_file == -1:
                print(f"Warning: Failed to create Hades N-1 contingency file for {replay_cont[0]}")
        else:
            print(
                "Skipping replay for contingency"
                f" '{replay_cont[0]}' (type {replay_cont[1]}): This program"
                " currently only supports replaying N-1 contingencies (type 0)."
            )
    return hades_output_list


def prepare_dynawo_SA(
    hades_input_file: Path,
    sorted_loadflow_score_list: list,
    dynawo_input_folder: Path,
    dynawo_output_folder: Path,
    number_pos_replay: int,
    dynamic_database: Optional[Path],
    multithreading: bool,
) -> tuple[Path, Path, dict, int]:
    """Prepares Dynawo SA."""
    replay_contgs = [
        elem_list[1]["name"] for elem_list in sorted_loadflow_score_list if "name" in elem_list[1]
    ]
    if number_pos_replay != -1:
        replay_contgs = replay_contgs[:number_pos_replay]

    iidm_list = list(dynawo_input_folder.glob("DYN*.xml"))
    if not iidm_list:
        print(f"Dynawo input file (DYN*.xml) not found in: {dynawo_input_folder}")
        return Path(), Path(), {}, 1

    iidm_file = iidm_list[0]

    try:
        (
            matched_branches,
            matched_generators,
            matched_loads,
            matched_shunts,
        ) = matching_elements.matching_elements(hades_input_file, iidm_file)
    except Exception as e:
        print(
            "Error during element matching between Hades"
            f" ({hades_input_file.name}) and Dynawo ({iidm_file.name}): {e}"
        )
        return Path(), Path(), {}, 1

    dict_types_cont = create_contingencies.get_dynawo_types_cont(iidm_file)
    dynawo_output_folder.mkdir(parents=True, exist_ok=True)

    try:
        (
            config_file,
            contng_file,
            contng_dict,
        ) = create_contingencies.create_dynawo_SA(
            dynawo_output_folder,
            replay_contgs,
            dict_types_cont,
            dynamic_database,
            matched_branches,
            matched_generators,
            matched_loads,
            matched_shunts,
            multithreading,
        )
    except Exception as e:
        print(f"Error creating Dynawo SA configuration/contingency files: {e}")
        return Path(), Path(), {}, 1

    if not config_file or not contng_file:
        print("Failed to create necessary Dynawo config or contingency files.")
        return Path(), Path(), {}, 1

    return config_file, contng_file, contng_dict, 0


def run_dynawo_contingencies_SA_code(
    input_dir: Path,
    output_dir: Path,
    dynawo_launcher: Path,
    config_file: Path,
    contng_file: Path,
    calc_contingencies: bool,
    matching_contng_dict: dict,
) -> int:
    """Executes Dynawo for contingencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = run_dynawo.run_dynaflow_SA(
        input_dir,
        output_dir,
        str(dynawo_launcher),
        config_file,
        contng_file,
        calc_contingencies,
        matching_contng_dict,
    )
    return status


def extract_dynawo_results(
    dynawo_output_folder: Path, sorted_loadflow_score_list: list, number_pos_replay: int
) -> dict:
    """Extracts Dynawo results."""
    dynawo_output_file = dynawo_output_folder / "outputs" / "finalState" / "outputIIDM.xml"
    dynawo_aggregated_xml = dynawo_output_folder / "aggregatedResults.xml"

    if not dynawo_output_file.is_file() or not dynawo_aggregated_xml.is_file():
        print(f"Dynawo output files not found: {dynawo_output_file}, {dynawo_aggregated_xml}")
        return {}

    parsed_output_file = manage_files.parse_xml_file(dynawo_output_file)
    parsed_aggregated_file = manage_files.parse_xml_file(dynawo_aggregated_xml)

    if not parsed_output_file or not parsed_aggregated_file:
        print("Error parsing Dynawo output or aggregated XML files.")
        return {}

    replay_contgs = [
        elem_list[1]["name"] for elem_list in sorted_loadflow_score_list if "name" in elem_list[1]
    ]
    if number_pos_replay != -1:
        replay_contgs = replay_contgs[:number_pos_replay]

    contg_set = set(replay_contgs)

    try:
        dynawo_contingency_data, _ = extract_results_data.collect_dynawo_results(
            parsed_output_file, parsed_aggregated_file, dynawo_output_folder, contg_set
        )
    except Exception as e:
        print(f"Error collecting Dynawo results: {e}")
        return {}

    return dynawo_contingency_data


def calculate_case_differences(
    sorted_loadflow_score_list: list,
    dynawo_contingency_data: dict,
    matching_contingencies_dict: dict,
) -> pd.DataFrame:
    """Calculates differences between Hades and Dynawo cases."""
    dict_diffs = {}
    if (
        not isinstance(matching_contingencies_dict, dict)
        or "contingencies" not in matching_contingencies_dict
    ):
        print("Error: Invalid format for matching_contingencies_dict")
        return pd.DataFrame()

    for case in matching_contingencies_dict.get("contingencies", []):
        if not isinstance(case, dict) or "id" not in case:
            print(f"Warning: Skipping invalid case entry in matching_contingencies_dict: {case}")
            continue

        case_id = case["id"]
        # Find corresponding Hades entry by contingency name (ID)
        hades_entry = None
        for idx, hades_data in enumerate(sorted_loadflow_score_list):
            # hades_data is a tuple (original_index, dict_data)
            if (
                isinstance(hades_data, tuple)
                and len(hades_data) > 1
                and isinstance(hades_data[1], dict)
            ):
                if hades_data[1].get("name") == case_id:
                    hades_entry = hades_data  # Keep the tuple (idx, dict)
                    break
            else:
                print(
                    f"Warning: Unexpected item format in sorted_loadflow_score_list: {hades_data}"
                )

        if not hades_entry:
            print(
                "Warning: Case '{case_id}' found in matching dict but not in Hades results list."
            )
            continue

        if case_id in dynawo_contingency_data:
            try:
                diff_result = calc_case_diffs.calculate_diffs_hades_dynawo(
                    hades_entry[1], dynawo_contingency_data[case_id]
                )
                if isinstance(diff_result, (list, tuple)) and len(diff_result) == 3:
                    dict_diffs[case_id] = diff_result
                else:
                    print(
                        "Warning: Unexpected result format from"
                        f" calculate_diffs_hades_dynawo for {case_id}"
                    )
            except Exception as e:
                print(f"Error calculating differences for case '{case_id}': {e}")

        else:
            print("WARNING: Case '{case_id}' not found in Dynawo execution results.")
            dict_diffs[case_id] = [case_id, "DYN_MISSING", None]

    if not dict_diffs:
        print("No differences could be calculated.")
        return pd.DataFrame()

    try:
        df = pd.DataFrame.from_dict(
            dict_diffs, orient="index", columns=["NAME", "STATUS", "REAL_SCORE"]
        )
        df.index.name = "CASE_ID"
        return df
    except Exception as e:
        print(f"Error creating DataFrame from differences dictionary: {e}")
        return pd.DataFrame()


def display_results_table(
    output_dir: Path, sorted_loadflow_score_list: list, tap_changers: bool
) -> None:
    """Displays and saves the Hades contingency ranking table."""
    if not sorted_loadflow_score_list:
        print("No loadflow results to display.")
        return

    headers_common = [
        "POS",
        "NUM",
        "NAME",
        "AFFECTED_ELEM",
        "STATUS",
        "MIN_VOLT",
        "MAX_VOLT",
        "N_ITER",
        "CONSTR_GEN_Q",
        "CONSTR_GEN_U",
        "CONSTR_VOLT",
        "CONSTR_FLOW",
        "COEF_REPORT",
        "RES_NODE",
    ]
    formats_common = [
        "<7",
        "<7",
        "<14",
        "<14",
        "<7",
        "<10",
        "<10",
        "<10",
        "<13",
        "<13",
        "<13",
        "<13",
        "<12",
        "<10",
    ]

    if tap_changers:
        headers = headers_common + ["TAP_CHANGERS", "FINAL_SCORE"]
        formats = formats_common + ["<14", "<14"]
    else:
        headers = headers_common + ["FINAL_SCORE"]
        formats = formats_common + ["<14"]

    header_format_string = " ".join([f"{{{i}:{fmt}}}" for i, fmt in enumerate(formats)]) + "\n"
    row_format_string = " ".join([f"{{{i}:{fmt}}}" for i, fmt in enumerate(formats)]) + "\n"

    str_table = header_format_string.format(*headers)

    for i_count, elem_tuple in enumerate(sorted_loadflow_score_list, 1):
        # elem_tuple should be (original_index, data_dict)
        if not (
            isinstance(elem_tuple, tuple)
            and len(elem_tuple) > 1
            and isinstance(elem_tuple[1], dict)
        ):
            print(
                "Warning: Skipping malformed entry in sorted_loadflow_score_list at"
                f" position {i_count}"
            )
            continue

        original_index, elem_data = elem_tuple

        # Helper to safely get list lengths or 0
        def safe_len(key):
            val = elem_data.get(key)
            return len(val) if isinstance(val, list) else 0

        # Helper to safely get values or a default string
        def safe_get(key, default="N/A"):
            return str(elem_data.get(key, default))  # Ensure string conversion

        row_data_common = [
            i_count,
            original_index,  # The original index or identifier
            safe_get("name", "Unknown"),
            safe_len("affected_elements"),
            safe_get("status"),
            safe_len("min_voltages"),
            safe_len("max_voltages"),
            safe_get("n_iter"),
            safe_len("constr_gen_Q"),
            safe_len("constr_gen_U"),
            safe_len("constr_volt"),
            safe_len("constr_flow"),
            safe_len("coef_report"),
            safe_len("res_node"),
        ]

        final_score = elem_data.get("final_score", "N/A")
        # Format score nicely if it's numeric
        try:
            if isinstance(final_score, (int, float)):
                final_score_str = f"{final_score:.4f}"  # Example formatting
            else:
                final_score_str = str(final_score)
        except TypeError:
            final_score_str = str(final_score)

        if tap_changers:
            row_data = row_data_common + [
                safe_len("tap_changers"),
                final_score_str,
            ]
        else:
            row_data = row_data_common + [final_score_str]

        try:
            str_table += row_format_string.format(*row_data)
        except (IndexError, ValueError, TypeError) as e:
            print(f"Warning: Could not format row for contingency '{safe_get('name')}': {e}")
            print(f"Row data: {row_data}")

    print("\nRANKING OF HADES CONTINGENCIES WITH SUMMARY SA INFORMATION\n")
    print(str_table)

    # Save the results table in a txt
    results_file = output_dir / "hades_ranking_table.txt"
    try:
        with open(results_file, "w") as text_file:
            text_file.write("RANKING OF HADES CONTINGENCIES WITH SUMMARY SA INFORMATION\n\n")
            text_file.write(str_table)
        print(f"Results table saved to: {results_file}")
    except IOError as e:
        print(f"Error writing results table to {results_file}: {e}")


def run_contingencies_screening_thread_loop(
    time_dir: Path,
    input_dir_path: Path,
    output_dir_path: Path,
    hades_launcher_solved: Path,
    dynawo_launcher_solved: Optional[Path],
    tap_changers: bool,
    replay_dynawo: Optional[Path],
    n_replay: int,
    score_type: int,
    dynamic_database: Optional[Path],
    multithreading: bool,
    calc_contingencies: bool,
    compress_results: bool,
    model_path: Optional[Path],
    replay_hades_obo: bool,
) -> None:
    """Processes a single time directory (snapshot)."""
    if not time_dir.is_dir():
        print(f"Skipping non-directory item: {time_dir.name}")
        return

    print("\n################################################################")
    print(f"Running the {time_dir.name} case")
    print("################################################################\n")

    relative_path = time_dir.relative_to(input_dir_path)
    output_dir_final_path = output_dir_path / relative_path
    output_hades_path = output_dir_final_path / settings.HADES_FOLDER
    output_dynawo_path = output_dir_final_path / settings.DYNAWO_FOLDER

    manage_files.dir_exists(output_dir_final_path, input_dir_path / relative_path)
    output_dir_final_path.mkdir(parents=True, exist_ok=True)

    hades_input_dir_for_run: Path
    potential_hades_sub = time_dir / settings.HADES_FOLDER
    potential_dynawo_sub = time_dir / settings.DYNAWO_FOLDER

    has_subfolders = potential_hades_sub.is_dir() and potential_dynawo_sub.is_dir()

    if calc_contingencies:
        hades_input_dir_for_run = potential_hades_sub
        if not hades_input_dir_for_run.is_dir():
            print(
                "Error: --calc-contingencies enabled, but input folder"
                f" {hades_input_dir_for_run} not found."
            )
            return
    else:
        hades_input_dir_for_run = potential_hades_sub if has_subfolders else time_dir

    print(f"Using Hades input from: {hades_input_dir_for_run}")
    print(f"Using Hades output to: {output_hades_path}")

    hades_input_file, hades_output_file, status = run_hades_contingencies_code(
        hades_input_dir_for_run,
        output_hades_path,
        hades_launcher_solved,
        tap_changers,
        multithreading,
        calc_contingencies,
    )

    if status == 1:
        print(f"Hades execution failed for {time_dir.name}. Stopping.")
        return
    if not hades_input_file or not hades_output_file:
        print(f"Hades execution did not return valid file paths for {time_dir.name}. Stopping.")
        return

    print("Analyzing Hades results and ranking contingencies...")
    sorted_loadflow_score_list, status = create_contingencies_ranking_code(
        hades_input_file,
        hades_output_file,
        output_dir_final_path,
        score_type,
        tap_changers,
        model_path,
    )

    if status == 1:
        print(f"Hades results analysis failed for {time_dir.name}. Stopping.")
        return
    if not sorted_loadflow_score_list:
        print(f"No contingencies ranked for {time_dir.name}. Check analysis.")

    display_results_table(output_dir_final_path, sorted_loadflow_score_list, tap_changers)

    if replay_dynawo is not None:
        print("\n--- Preparing and Running Dynawo Replay ---")
        dynawo_input_dir_for_run: Path
        dynawo_input_dir_for_run = (
            potential_dynawo_sub
            if calc_contingencies
            else (potential_dynawo_sub if has_subfolders else time_dir)
        )

        print(f"Using Dynawo input from: {dynawo_input_dir_for_run}")
        print(f"Using Dynawo output to: {output_dynawo_path}")

        config_file, contng_file, matching_contng_dict, status = prepare_dynawo_SA(
            hades_input_file,
            sorted_loadflow_score_list,
            dynawo_input_dir_for_run,
            output_dynawo_path,
            n_replay,
            dynamic_database,
            multithreading,
        )

        if status == 1:
            print(f"Dynawo preparation failed for {time_dir.name}. Stopping Dynawo replay.")
            return

        print("Running Dynawo Security Analysis...")
        status = run_dynawo_contingencies_SA_code(
            dynawo_input_dir_for_run,
            output_dynawo_path,
            dynawo_launcher_solved,
            config_file,
            contng_file,
            calc_contingencies,
            matching_contng_dict,
        )

        if status != 1:
            print("Dynawo execution completed.")
            print("Extracting Dynawo results...")
            dynawo_contingency_data = extract_dynawo_results(
                output_dynawo_path, sorted_loadflow_score_list, n_replay
            )

            if not dynawo_contingency_data:
                print("Could not extract Dynawo contingency data.")
            else:
                print("Calculating Hades vs Dynawo differences...")
                df_diffs = calculate_case_differences(
                    sorted_loadflow_score_list,
                    dynawo_contingency_data,
                    matching_contng_dict,
                )

                if not df_diffs.empty:
                    contg_df_path = output_dir_final_path / "contg_df.csv"

                    if contg_df_path.is_file():
                        try:
                            df_contg = pd.read_csv(contg_df_path, sep=";", index_col="NUM")
                            if "NAME" in df_contg.columns and "NAME" in df_diffs.columns:
                                df_contg = pd.merge(df_contg, df_diffs, how="left", on="NAME")
                                if "REAL_SCORE" in df_contg.columns:
                                    df_contg = df_contg.sort_values(
                                        "REAL_SCORE",
                                        ascending=False,
                                        na_position="last",
                                    )

                                df_for_rmse = df_contg.dropna(
                                    subset=["PREDICTED_SCORE", "REAL_SCORE"]
                                )
                                target_rmse_df = (
                                    df_for_rmse if n_replay == -1 else df_for_rmse.head(n_replay)
                                )

                                if not target_rmse_df.empty:
                                    rmse = calc_case_diffs.calc_rmse(target_rmse_df)
                                    print(
                                        "\nRMSE"
                                        f" ({'all contingencies' if n_replay == -1 else f'top {n_replay} contingencies'})"
                                        " (predicted vs real diff score, excluding"
                                        f" divergences/missing):\n{rmse}"
                                    )
                                else:
                                    print("Not enough valid data points to calculate RMSE.")

                                df_contg.to_csv(contg_df_path, index=False, sep=";")
                                print(f"Updated contingency DataFrame saved to {contg_df_path}")
                            else:
                                print(
                                    "Warning: 'NAME' column missing in contg_df.csv"
                                    " or diffs DataFrame. Cannot merge."
                                )
                                df_diffs.to_csv(output_dir_final_path / "diffs_df.csv", sep=";")
                        except Exception as e:
                            print(f"Error merging/saving differences to CSV {contg_df_path}: {e}")
                    else:
                        print(
                            "Warning: Base contingency DataFrame {contg_df_path} not"
                            " found. Saving differences separately."
                        )
                        df_diffs.to_csv(output_dir_final_path / "diffs_df.csv", sep=";")

        if compress_results:
            print("Cleaning/Compressing Dynawo results...")
            try:
                manage_files.clean_data(output_dynawo_path, sorted_loadflow_score_list, n_replay)
            except Exception as e:
                print(f"Error during Dynawo results cleanup: {e}")

    if replay_hades_obo:
        print("\n--- Preparing and Running Hades One-by-One Replay ---")
        replay_hades_paths = prepare_hades_contingencies(
            sorted_loadflow_score_list,
            hades_input_file,
            output_hades_path,
            n_replay,
        )

        print(f"Replaying {len(replay_hades_paths)} contingencies with Hades one-by-one...")
        for replay_hades_case_folder in replay_hades_paths:
            if not replay_hades_case_folder.is_dir():
                print(
                    f"Warning: Expected Hades replay folder not found: {replay_hades_case_folder}"
                )
                continue

            print(f"Running Hades for: {replay_hades_case_folder.name}")
            (
                _h_input_file,
                _h_output_file,
                status,
            ) = run_hades_contingencies_code(
                replay_hades_case_folder,
                replay_hades_case_folder,
                hades_launcher_solved,
                tap_changers,
                False,
                False,
            )
            if status == 1:
                print(f"Hades one-by-one replay failed for {replay_hades_case_folder.name}")

    if compress_results:
        print(f"Compressing final results for {time_dir.name}...")
        try:
            manage_files.compress_results(output_dir_final_path)
        except Exception as e:
            print(f"Error during final compression for {output_dir_final_path}: {e}")

    print(f"\nFinished processing {time_dir.name}")


# --- Typer Command Definition ---
@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Path to the folder containing the case files (e.g., .../YEAR/)",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Argument(
        ..., help="Path to the base output folder", file_okay=False, writable=True
    ),
    hades_launcher: Path = typer.Argument(
        ..., help="Define the Hades launcher (path to executable or command name)"
    ),
    tap_changers: bool = typer.Option(
        False,
        "--tap-changers",
        "-t",
        help="Run the simulations with activated tap changers",
    ),
    replay_hades_obo: bool = typer.Option(
        False,
        "--replay-hades-obo",
        "-a",
        help="Replay the most interesting contingencies with Hades one by one",
    ),
    replay_dynawo: Optional[Path] = typer.Option(
        None,
        "--replay-dynawo",
        "-d",
        help="Replay the most interesting contingencies with Dynawo. Provide the Dynaflow launcher path.",
    ),
    n_replay: int = typer.Option(
        settings.REPLAY_NUM,
        "--n-replay",
        "-n",
        help=f"Number of most interesting contingencies to replay (default: {settings.REPLAY_NUM}, use -1 for all)",
    ),
    score_type: int = typer.Option(
        settings.DEFAULT_SCORE,
        "--score-type",
        "-s",
        help="Type of scoring for ranking (1 = human made, 2 = machine learning)",
    ),
    dynamic_database: Optional[Path] = typer.Option(
        None,
        "--dynamic-database",
        "-b",
        help="Path to a standalone dynamic database folder for Dynawo",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    multithreading: bool = typer.Option(
        False,
        "--multithreading",
        "-m",
        help="Enable multithreading for processing time directories in parallel",
    ),
    calc_contingencies: bool = typer.Option(
        False,
        "--calc-contingencies",
        "-c",
        help="WARNING: It does not accept compressed results. Assume input dir structure contains pre-calculated contingencies (expects HADES/DYNAWO subfolders in time dirs)",
    ),
    compress_results: bool = typer.Option(
        False,
        "--compress-results",
        "-z",
        help="Clean intermediate files and compress results for each time directory",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        "-l",
        help="Manually define the path to the ML model path for predictions",
        exists=True,
        file_okay=False,
        writable=True,
    ),
):
    """
    Main execution pipeline for contingency screening using Hades and optionally Dynawo.
    Processes time-structured directories (YEAR/MONTH/DAY/HOUR).
    """
    input_dir_path = input_dir.resolve()  # Get absolute path
    output_dir_path = output_dir.resolve()

    # Ensure output directory exists or can be created
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir_path}: {e}")
        raise typer.Exit(code=1)

    print("--- Initial Configuration ---")
    print(f"Input Directory: {input_dir_path}")
    print(f"Output Directory: {output_dir_path}")
    print(f"Hades Launcher: {hades_launcher}")
    print(f"Tap Changers Activated: {tap_changers}")
    print(f"Replay Hades OBO: {replay_hades_obo}")
    print(f"Replay Dynawo Launcher: {replay_dynawo if replay_dynawo else 'Not Enabled'}")
    print(f"Number of Contingencies to Replay: {'All' if n_replay == -1 else n_replay}")
    print(
        f"Score Type: {score_type} ({'Human' if score_type == 1 else 'ML' if score_type == 2 else 'Unknown'})"
    )
    print(f"Dynamic Database: {dynamic_database if dynamic_database else 'Not Specified'}")
    print(f"Multithreading Enabled: {multithreading}")
    print(f"Using Pre-calculated Contingencies Structure: {calc_contingencies}")
    print(f"Compress Results: {compress_results}")
    print(f"ML Model Path: {model_path if model_path else 'Default/Not Specified'}")
    print("-----------------------------")

    # Check if specified launchers are files or resolve them
    # solve_launcher now returns Path, convert to string if commands expect strings
    hades_launcher_solved = solve_launcher(hades_launcher)
    dynawo_launcher_solved = solve_launcher(replay_dynawo) if replay_dynawo else None

    # Iterate through the different directories
    # Assuming structure YEAR -> MONTH -> DAY -> TIME_DIR
    processed_count = 0
    try:
        for year_dir in input_dir_path.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue

                    time_dirs = [
                        d for d in day_dir.iterdir() if d.is_dir()
                    ]  # Get list of time directories
                    if not time_dirs:
                        continue

                    print(f"\nProcessing Day: {day_dir.name}")

                    if multithreading:
                        # Prepare arguments for each time directory
                        arguments_list = []
                        for time_dir in time_dirs:
                            arguments_list.append(
                                (
                                    time_dir,
                                    input_dir_path,  # Pass base input path
                                    output_dir_path,  # Pass base output path
                                    hades_launcher_solved,
                                    dynawo_launcher_solved,
                                    # Pass all options needed by the loop function
                                    tap_changers,
                                    replay_dynawo,  # Pass the launcher path or None
                                    n_replay,
                                    score_type,
                                    dynamic_database,  # Pass Path or None
                                    multithreading,  # May not be needed inside if only used here
                                    calc_contingencies,
                                    compress_results,
                                    model_path,  # Pass Path or None
                                    replay_hades_obo,  # Pass flag
                                )
                            )

                        # Determine number of processes
                        num_processes = min(
                            settings.N_THREADS_SNAPSHOT,
                            len(arguments_list),
                            os.cpu_count() or 1,
                        )
                        print(f"Using multiprocessing with {num_processes} processes...")

                        with multiprocessing.Pool(processes=num_processes) as pool:
                            # Use starmap to call the function with tuple arguments
                            pool.starmap(run_contingencies_screening_thread_loop, arguments_list)
                        processed_count += len(arguments_list)

                    else:  # Run sequentially
                        print("Running sequentially...")
                        for time_dir in time_dirs:
                            run_contingencies_screening_thread_loop(
                                time_dir,
                                input_dir_path,
                                output_dir_path,
                                hades_launcher_solved,
                                dynawo_launcher_solved,
                                # Pass all options
                                tap_changers,
                                replay_dynawo,
                                n_replay,
                                score_type,
                                dynamic_database,
                                multithreading,
                                calc_contingencies,
                                compress_results,
                                model_path,
                                replay_hades_obo,
                            )
                            processed_count += 1

    except Exception as e:
        print(f"\nAn error occurred during directory traversal or processing: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)

    print("\n--- Processing Complete ---")
    print(f"Processed {processed_count} time directories.")
    print(f"Output generated in: {output_dir_path}")
