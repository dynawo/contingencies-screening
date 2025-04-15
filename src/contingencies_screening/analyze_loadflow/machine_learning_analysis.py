import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import typer
from numpy import mean, std
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.model_selection import cross_val_score, KFold
from interpret.glassbox import ExplainableBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from contingencies_screening.analyze_loadflow import human_analysis
from contingencies_screening.commons import manage_files

app = typer.Typer(help="Train and evaluate ML models for contingency screening results.")


def convert_dict_to_df(
    contingencies_dict: Dict[Any, Dict[str, Any]],
    elements_dict: Dict[str, Any],
    tap_changers: bool,
    predicted_score: bool = False,
) -> Tuple[pd.DataFrame, Dict[Any, str]]:
    """
    Converts and computes result dictionaries into a usable DataFrame.

    Args:
        contingencies_dict (Dict[Any, Dict[str, Any]]): Dictionary containing contingency data.
        elements_dict (Dict[str, Any]): Dictionary containing element data.
        tap_changers (bool): Flag indicating if tap changers are considered.
        predicted_score (bool, optional): Flag indicating if predicted scores should be included. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, Dict[Any, str]]: A tuple containing the processed DataFrame and a dictionary of errors.
    """
    contingencies_df_data: Dict[str, List[Any]] = {
        "NUM": [],
        "NAME": [],
        "MIN_VOLT": [],
        "MAX_VOLT": [],
        "MAX_FLOW": [],
        "N_ITER": [],
        "AFFECTED_ELEM": [],
        "CONSTR_GEN_Q": [],
        "CONSTR_GEN_U": [],
        "CONSTR_VOLT": [],
        "CONSTR_FLOW": [],
        "RES_NODE": [],
        "COEF_REPORT": [],
    }

    if tap_changers:
        contingencies_df_data["TAP_CHANGERS"] = []

    if predicted_score:
        contingencies_df_data["PREDICTED_SCORE"] = []

    error_contg: Dict[Any, str] = {}

    for key, cont_data in contingencies_dict.items():
        try:
            poste_values = elements_dict.get("poste", {})
            groupe_values = elements_dict.get("groupe", {})
            noeud_values = elements_dict.get("noeud", {})
            quadripole_values = elements_dict.get("quadripole", {})

            value_min_voltages = human_analysis.calc_diff_volt(
                cont_data.get("min_voltages", []), poste_values
            )
            value_max_voltages = human_analysis.calc_diff_volt(
                cont_data.get("max_voltages", []), poste_values
            )
            value_max_flows = human_analysis.calc_diff_max_flow(cont_data.get("max_flow", []))
            value_constr_gen_Q = human_analysis.calc_constr_gen_Q(
                cont_data.get("constr_gen_Q", []), groupe_values
            )
            value_constr_gen_U = human_analysis.calc_constr_gen_U(
                cont_data.get("constr_gen_U", []), groupe_values
            )
            value_constr_volt = human_analysis.calc_constr_volt(
                cont_data.get("constr_volt", []), noeud_values
            )
            value_constr_flow = human_analysis.calc_constr_flow(
                cont_data.get("constr_flow", []), quadripole_values
            )
            value_n_iter = cont_data.get("n_iter", 0)
            value_affected_elem = len(cont_data.get("affected_elements", []))
            value_constr_res_node = len(cont_data.get("res_node", []))
            value_coef_report = len(cont_data.get("coef_report", []))

            value_tap_changer = (
                sum(
                    abs(float(tap.get("diff_value", 0)))
                    if tap.get("stopper") is not None and int(tap.get("stopper")) == 0
                    else human_analysis.STD_TAP_VALUE
                    for tap in cont_data.get("tap_changers", [])
                )
                if tap_changers
                else 0.0
            )

            contingencies_df_data["NUM"].append(key)
            contingencies_df_data["NAME"].append(cont_data.get("name", f"Unknown_{key}"))
            contingencies_df_data["MIN_VOLT"].append(value_min_voltages)
            contingencies_df_data["MAX_VOLT"].append(value_max_voltages)
            contingencies_df_data["MAX_FLOW"].append(value_max_flows)
            contingencies_df_data["N_ITER"].append(value_n_iter)
            contingencies_df_data["AFFECTED_ELEM"].append(value_affected_elem)
            contingencies_df_data["CONSTR_GEN_Q"].append(value_constr_gen_Q)
            contingencies_df_data["CONSTR_GEN_U"].append(value_constr_gen_U)
            contingencies_df_data["CONSTR_VOLT"].append(value_constr_volt)
            contingencies_df_data["CONSTR_FLOW"].append(value_constr_flow)
            contingencies_df_data["RES_NODE"].append(value_constr_res_node)
            contingencies_df_data["COEF_REPORT"].append(value_coef_report)
            if tap_changers:
                contingencies_df_data["TAP_CHANGERS"].append(value_tap_changer)
            if predicted_score:
                contingencies_df_data["PREDICTED_SCORE"].append(cont_data.get("final_score", None))

            status = cont_data.get("status", -1)
            if status != 0:
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
                error_contg[key] = status_map.get(status, f"Unknown Error (Status {status})")

        except Exception as e:
            print(f"Error processing contingency {key}: {e}")
            error_contg[key] = f"Processing Error: {e}"

    try:
        df = pd.DataFrame(contingencies_df_data)
        if "NUM" in df.columns:
            df = df.set_index("NUM")
    except ValueError as ve:
        print(f"Error creating DataFrame: {ve}")
        df = pd.DataFrame()

    return df, error_contg


def predict_scores(contingencies_df: pd.DataFrame, model_filename: Path) -> Dict[str, float]:
    """
    Predicts scores using a pre-loaded ML model.

    Args:
        contingencies_df (pd.DataFrame): DataFrame containing contingency data.
        model_filename (Path): Path to the pickled ML model file.

    Returns:
        Dict[str, float]: Dictionary of contingency numbers and their predicted scores.
    """
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_filename}")
        return {}
    except Exception as e:
        print(f"Error loading model from {model_filename}: {e}")
        return {}

    contg_scores: Dict[str, float] = {}
    try:
        predict_df = contingencies_df.drop("NAME", axis=1, errors="ignore")
        for i in predict_df.index:
            prediction = model.predict(predict_df.loc[[i]])
            contg_scores[str(i)] = float(prediction[0])
    except KeyError as ke:
        print(f"Error during prediction preparation: {ke}")
        return {}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}

    return contg_scores


def analyze_loadflow_results(
    contingencies_dict: Dict[Any, Dict[str, Any]],
    elements_dict: Dict[str, Any],
    tap_changers: bool,
    model_path: Path = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Predicts the difference score using an ML model.

    Args:
        contingencies_dict (Dict[Any, Dict[str, Any]]): Dictionary containing contingency data.
        elements_dict (Dict[str, Any]): Dictionary containing element data.
        tap_changers (bool): Flag indicating if tap changers were considered in the load flow analysis.
        model_path (Path, optional): Path to the ML model file. Defaults to None.

    Returns:
        Dict[Any, Dict[str, Any]]: Updated contingency dictionary with predicted scores.
    """
    print("Converting Hades results to DataFrame for ML prediction...")
    contingencies_df, error_contg = convert_dict_to_df(
        contingencies_dict, elements_dict, tap_changers
    )

    if contingencies_df.empty and not error_contg:
        print("Warning: DataFrame conversion resulted in an empty DataFrame.")
        return contingencies_dict

    if model_path is None:
        script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        model_filename = "ML_taps.pkl" if tap_changers else "ML_no_taps.pkl"
        model_path = script_dir / model_filename
        print(f"Model path not provided, using default: {model_path}")
    elif not model_path.is_file():
        print(f"Error: Provided model path is not a valid file: {model_path}")
        for key in contingencies_dict:
            if key not in error_contg:
                contingencies_dict[key]["final_score"] = "Model File Error"
        return contingencies_dict

    print(f"Predicting scores using model: {model_path}")
    print(
        "\nNOTE: Ensure the selected ML model (.pkl) matches the tap changer setting (--tap-changers) used during analysis.\n"
    )

    contg_scores = predict_scores(contingencies_df, model_path)

    print("Updating contingency scores...")
    updated_count = 0
    prediction_error_count = 0
    for key in contingencies_dict.keys():
        if key in error_contg:
            contingencies_dict[key]["final_score"] = error_contg[key]
        elif str(key) in contg_scores:
            contingencies_dict[key]["final_score"] = contg_scores[str(key)]
            updated_count += 1
        else:
            contingencies_dict[key]["final_score"] = "Prediction Failed"
            prediction_error_count += 1

    print(f"Updated {updated_count} contingencies with predicted scores.")
    if prediction_error_count > 0:
        print(f"Warning: Failed to get prediction for {prediction_error_count} contingencies.")
    if error_contg:
        print(f"Marked {len(error_contg)} contingencies with error status (e.g., Divergence).")

    return contingencies_dict


def normalize_LR(X: pd.DataFrame, coefs: List[float]) -> List[float]:
    """
    Normalizes Linear Regression coefficients by the mean of their respective features.

    Args:
        X (pd.DataFrame): DataFrame of features.
        coefs (List[float]): List of Linear Regression coefficients.

    Returns:
        List[float]: List of normalized coefficients. Returns an empty list if there's a mismatch
                     in the number of columns and coefficients or if a feature mean cannot be calculated.
    """
    coefs_norm = []
    cols = list(X.columns)

    if len(cols) != len(coefs):
        print(
            f"Warning: Mismatch between number of columns ({len(cols)}) and coefficients ({len(coefs)}) in normalize_LR."
        )
        return []

    for i in range(len(cols)):
        try:
            feature_mean = X[cols[i]].mean()
            coefs_norm.append(feature_mean * coefs[i])
        except (TypeError, KeyError) as e:
            print(
                f"Warning: Could not calculate mean for feature '{cols[i]}' or multiply coefficient: {e}"
            )
            coefs_norm.append(0.0)

    return coefs_norm


def load_df(path: Path) -> pd.DataFrame:
    """
    Loads and concatenates contingency DataFrames from CSV files within a nested directory structure.

    Args:
        path (Path): Path to the base folder containing the training data.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing all loaded contingency data.
                      Returns an empty DataFrame if no valid files are found or loaded.
    """
    all_dfs: List[pd.DataFrame] = []
    file_count = 0

    print(f"Loading data from base path: {path}")
    target_filename = "*contg_df.csv"
    csv_files = list(path.rglob(f"**/{target_filename}"))

    if not csv_files:
        print(f"Warning: No '{target_filename}' files found recursively under {path}")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} '{target_filename}' files to load.")

    for csv_file in csv_files:
        if csv_file.is_file():
            try:
                df_new = pd.read_csv(csv_file, sep=";")
                all_dfs.append(df_new)
                file_count += 1
                if file_count % 50 == 0:
                    print(f"Loaded {file_count} files...")
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty file: {csv_file}")
            except Exception as e:
                print(f"Warning: Failed to load or process file {csv_file}: {e}")

    if not all_dfs:
        print("No valid DataFrames were loaded.")
        return pd.DataFrame()

    print(f"Concatenating {len(all_dfs)} DataFrames...")
    try:
        df_contg = pd.concat(all_dfs, axis=0, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame()

    df_contg = df_contg.drop("NUM", axis=1, errors="ignore")

    print(f"Loaded DataFrame shape before dropna: {df_contg.shape}")
    df_contg = df_contg.dropna()
    print(f"Final DataFrame shape after dropna: {df_contg.shape}")

    return df_contg


def tune_model(
    model,
    parameters: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    n_iterations: int = 50,
    n_jobs: int = -1,
    random_state: int = 0,
    output_dir: Path = None,
    model_name: str = "BayesSearchCV_results",
):
    """
    Tunes a given model using BayesSearchCV and saves the results.

    Args:
        model: The scikit-learn estimator to tune.
        parameters (Dict[str, Any]): The parameter search space.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        cv_splits (int, optional): Number of cross-validation folds. Defaults to 5.
        n_iterations (int, optional): Number of optimization iterations. Defaults to 50.
        n_jobs (int, optional): Number of jobs to run in parallel (-1 means all processors). Defaults to 8.
        random_state (int, optional): Random state for reproducibility. Defaults to 0.
        output_dir (Path, optional): Directory to save the results CSV. Defaults to None.
        model_name (str, optional): Base name for the output CSV file. Defaults to "BayesSearchCV_results".
    """
    print(f"\n--- Tuning {type(model).__name__} using BayesSearchCV ---")
    clf = BayesSearchCV(
        estimator=model,
        search_spaces=parameters,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        verbose=5,
        n_jobs=n_jobs,
        random_state=random_state,
        n_iter=n_iterations,
    )
    clf.fit(X, y)

    print("\nBayesSearchCV Best Parameters:")
    print(clf.best_params_)

    print("\nBayesSearchCV Cross-Validation Results:")
    print(clf.cv_results_)

    if output_dir:
        results_df = pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
        output_csv_path = output_dir / f"{model_name}_{type(model).__name__}.csv"
        try:
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nBayesSearchCV results saved to: {output_csv_path}")
        except OSError as e:
            print(f"Error saving BayesSearchCV results to CSV: {e}")

    return clf.best_params_


def load_tuned_parameters(file_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Loads tuned parameters from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing tuned parameters.

    Returns:
        Optional[Dict[str, Dict[str, Any]]]: A dictionary where keys are model names
                                             and values are dictionaries of their best parameters.
                                             Returns None if the file is not found or loading fails.
    """
    try:
        with open(file_path, "r") as f:
            tuned_params = json.load(f)
        print(f"Loaded tuned parameters from: {file_path}")
        return tuned_params
    except FileNotFoundError:
        print(f"Warning: Tuned parameters file not found at: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from: {file_path}")
        return None
    except Exception as e:
        print(f"Warning: An error occurred while loading tuned parameters: {e}")
        return None


@app.command()
def main(
    output_path: Path = typer.Argument(
        ...,
        help="Path to the base folder containing the training dataframe CSV files.",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
    model_path: Path = typer.Argument(
        ...,
        help="Path to the directory where trained models will be saved.",
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    tune: bool = typer.Option(
        False, "--tune", help="Enable hyperparameter tuning using BayesSearchCV."
    ),
    tuned_params_path: Optional[Path] = typer.Option(
        None, "--tuned-params", help="Path to a JSON file containing tuned hyperparameters."
    ),
):
    """Trains and evaluates ML models on contingency screening data."""
    pd.options.mode.chained_assignment = None

    print("--- Starting Model Training and Evaluation ---")
    print(f"Loading data from: {output_path}")
    print(f"Saving models to: {model_path}")

    try:
        manage_files.dir_exists(model_path, output_path)
        model_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create model output directory {model_path}: {e}")
        raise typer.Exit(code=1)

    contingencies_df = load_df(output_path)

    if contingencies_df.empty:
        print("Error: No data loaded. Exiting.")
        raise typer.Exit(code=1)

    print("Preprocessing data...")
    contingencies_df = contingencies_df.drop("PREDICTED_SCORE", axis=1, errors="ignore")

    if "STATUS" in contingencies_df.columns:
        print(f"Filtering DataFrame by STATUS == 'BOTH'. Initial rows: {len(contingencies_df)}")
        contingencies_df = contingencies_df.loc[contingencies_df["STATUS"] == "BOTH"]
        contingencies_df = contingencies_df.drop("STATUS", axis=1, errors="ignore")
        print(f"Rows after filtering: {len(contingencies_df)}")
    else:
        print("Warning: 'STATUS' column not found, cannot filter by it.")

    initial_rows = len(contingencies_df)
    contingencies_df = contingencies_df.dropna()
    if len(contingencies_df) < initial_rows:
        print(f"Dropped {initial_rows - len(contingencies_df)} rows containing NaN values.")

    if contingencies_df.empty:
        print("Error: No data remaining after preprocessing. Exiting.")
        raise typer.Exit(code=1)

    print("Shuffling data...")
    contingencies_df = contingencies_df.sample(frac=1, random_state=42)

    if "REAL_SCORE" not in contingencies_df.columns:
        print("Error: Target column 'REAL_SCORE' not found in the data. Exiting.")
        raise typer.Exit(code=1)

    y = contingencies_df.pop("REAL_SCORE")
    X = contingencies_df.drop("NAME", axis=1, errors="ignore")

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features columns: {list(X.columns)}")

    tuned_parameters = None
    if tuned_params_path:
        tuned_parameters = load_tuned_parameters(tuned_params_path)

    print("Defining models...")

    gbr_params = {
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
        "max_depth": Integer(3, 10),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 5),
        "subsample": Real(0.7, 1.0),
    }
    model_GBR = GradientBoostingRegressor(random_state=42)

    ebm_params = {
        "learning_rate": Real(00.001, 0.1, prior="log-uniform"),
        "max_bins": Integer(8, 64),
        "interactions": Integer(0, 5),
        "outer_bags": Integer(8, 16),
        "smoothing_rounds": Integer(100, 500),
    }
    model_EBM = ExplainableBoostingRegressor(random_state=42)

    if tune:
        best_gbr_params = tune_model(
            GradientBoostingRegressor(random_state=42),
            gbr_params,
            X,
            y,
            output_dir=model_path,
            model_name="GBR",
        )
        best_ebm_params = tune_model(
            ExplainableBoostingRegressor(random_state=42),
            ebm_params,
            X,
            y,
            output_dir=model_path,
            model_name="EBM",
        )
        tuned_parameters_to_save = {
            "GradientBoostingRegressor": best_gbr_params,
            "ExplainableBoostingRegressor": best_ebm_params,
        }
        tuned_params_output_path = model_path / "tuned_hyperparameters.json"
        try:
            with open(tuned_params_output_path, "w") as f:
                json.dump(tuned_parameters_to_save, f, indent=4)
            print(f"\nTuned hyperparameters saved to: {tuned_params_output_path}")
        except OSError as e:
            print(f"Error saving tuned hyperparameters: {e}")
    elif tuned_parameters:
        print("\nUsing loaded tuned hyperparameters...")
        if "GradientBoostingRegressor" in tuned_parameters:
            model_GBR = GradientBoostingRegressor(
                **tuned_parameters["GradientBoostingRegressor"], random_state=42
            )
            print(f"GBR parameters: {tuned_parameters['GradientBoostingRegressor']}")
        if "ExplainableBoostingRegressor" in tuned_parameters:
            model_EBM = ExplainableBoostingRegressor(
                **tuned_parameters["ExplainableBoostingRegressor"], random_state=42
            )
            print(f"EBM parameters: {tuned_parameters['ExplainableBoostingRegressor']}")

    print("Performing 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Evaluating Gradient Boosting Regressor...")
    try:
        n_scores_GBR = cross_val_score(
            model_GBR,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            verbose=1,
        )
        print("MAE GBR: %.3f (+/- %.3f)" % (mean(-n_scores_GBR), std(-n_scores_GBR)))
    except Exception as e:
        print(f"Error during GBR cross-validation: {e}")

    print("Evaluating Explainable Boosting Machine...")
    try:
        n_scores_EBM = cross_val_score(
            model_EBM,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            verbose=1,
        )
        print("MAE EBM: %.3f (+/- %.3f)" % (mean(-n_scores_EBM), std(-n_scores_EBM)))
    except Exception as e:
        print(f"Error during EBM cross-validation: {e}")

    print("Evaluating Linear Regression (Mean)...")
    model_LR_Mean = LinearRegression()
    try:
        n_scores_LR_Mean = cross_val_score(
            model_LR_Mean,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            verbose=1,
        )
        print("MAE LR Mean: %.3f (+/- %.3f)" % (mean(-n_scores_LR_Mean), std(-n_scores_LR_Mean)))
    except Exception as e:
        print(f"Error during LR Mean cross-validation: {e}")

    print("Evaluating Theil-Sen Regressor (Median)...")
    model_LR_Median = TheilSenRegressor(random_state=42)
    try:
        n_scores_LR_Median = cross_val_score(
            model_LR_Median,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            verbose=1,
        )
        print(
            "MAE LR Median: %.3f (+/- %.3f)"
            % (mean(-n_scores_LR_Median), std(-n_scores_LR_Median))
        )
    except Exception as e:
        print(f"Error during LR Median cross-validation: {e}")

    print("\nTraining final models on the whole dataset...")
    try:
        model_GBR.fit(X, y)
        gbr_model_file = model_path / "GBR_model.pkl"
        with open(gbr_model_file, "wb") as f:
            pickle.dump(model_GBR, f)
        print(f"GBR model saved to: {gbr_model_file}")
    except Exception as e:
        print(f"Error training or saving GBR model: {e}")

    try:
        model_EBM.fit(X, y)
        ebm_model_file = model_path / "EBM_model.pkl"
        with open(ebm_model_file, "wb") as f:
            pickle.dump(model_EBM, f)
        print(f"EBM model saved to: {ebm_model_file}")
    except Exception as e:
        print(f"Error training or saving EBM model: {e}")

    try:
        model_LR_Mean.fit(X, y)
        cols = list(X.columns)
        coefs = list(model_LR_Mean.coef_)
        intercept = model_LR_Mean.intercept_
        print("\n--- LR Mean Weights ---")
        print(f"INTERCEPT: {intercept:.4f}")
        lr_mean_weights = {"INTERCEPTION": intercept}
        for i in range(len(cols)):
            print(f"{cols[i]}: {coefs[i]:.4f}")
            lr_mean_weights[cols[i]] = coefs[i]

        coefs_norm = normalize_LR(X, coefs)
        print("\nLR Mean Weights (Normalized by Feature Mean):")
        print(f"INTERCEPT: {intercept:.4f}")
        if len(cols) == len(coefs_norm):
            for i in range(len(cols)):
                print(f"{cols[i]}: {coefs_norm[i]:.4f}")
        else:
            print("Could not display normalized weights due to error during calculation.")

        lr_mean_file = model_path / "LR_model_Mean.json"
        with open(lr_mean_file, "w") as outfile:
            json.dump(lr_mean_weights, outfile, indent=4)
        print(f"LR Mean weights saved to: {lr_mean_file}")

    except Exception as e:
        print(f"Error training, analyzing, or saving LR Mean model: {e}")

    try:
        model_LR_Median.fit(X, y)
        cols = list(X.columns)
        coefs = list(model_LR_Median.coef_)
        intercept = model_LR_Median.intercept_
        print("\n--- LR Median (Theil-Sen) Weights ---")
        print(f"INTERCEPT: {intercept:.4f}")
        lr_median_weights = {"INTERCEPTION": intercept}
        for i in range(len(cols)):
            print(f"{cols[i]}: {coefs[i]:.4f}")
            lr_median_weights[cols[i]] = coefs[i]

        coefs_norm = normalize_LR(X, coefs)
        print("\nLR Median Weights (Normalized by Feature Mean):")
        print(f"INTERCEPT: {intercept:.4f}")
        if len(cols) == len(coefs_norm):
            for i in range(len(cols)):
                print(f"{cols[i]}: {coefs_norm[i]:.4f}")
        else:
            print("Could not display normalized weights due to error during calculation.")

        lr_median_file = model_path / "LR_model_Median.json"
        with open(lr_median_file, "w") as outfile:
            json.dump(lr_median_weights, outfile, indent=4)
        print(f"LR Median weights saved to: {lr_median_file}")

    except Exception as e:
        print(f"Error training, analyzing, or saving LR Median model: {e}")

    print("\n--- Model Training and Evaluation Complete ---")
