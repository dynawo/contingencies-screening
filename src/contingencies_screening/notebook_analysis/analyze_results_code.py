from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


def load_df(path: str) -> pd.DataFrame:
    """
    Loads and concatenates contingency DataFrames from a nested directory structure.

    Args:
        path: The base path to the directory containing the data.

    Returns:
        A pandas DataFrame containing the concatenated data.
    """

    def parse_time_from_dir_name(dir_name: str) -> Optional[datetime]:
        """Parses a datetime object from a directory name."""
        time_str = dir_name.replace("recollement-auto-", "").replace("-enrichi", "")
        try:
            return pd.to_datetime(time_str + "00", format="%Y%m%d-%H%M%S")
        except ValueError:
            return None

    path_obj = Path(path)
    all_dfs: List[pd.DataFrame] = []

    for year_dir in (d for d in path_obj.iterdir() if d.is_dir()):
        for month_dir in (d for d in year_dir.iterdir() if d.is_dir()):
            for day_dir in (d for d in month_dir.iterdir() if d.is_dir()):
                for time_dir in (d for d in day_dir.iterdir() if d.is_dir()):
                    csv_path = time_dir / "contg_df.csv"
                    if csv_path.is_file():
                        df_new = pd.read_csv(csv_path, sep=";", index_col="NAME")
                        dt_obj = parse_time_from_dir_name(time_dir.stem)
                        if dt_obj:
                            df_new["DATE"] = dt_obj
                            all_dfs.append(df_new)
                        else:
                            print(f"Warning: Could not parse date from {time_dir.stem}")

    if not all_dfs:
        return pd.DataFrame()  # Return empty DataFrame if no data

    df_contg = pd.concat(all_dfs, axis=0, ignore_index=False).dropna()
    return df_contg


def all_time_top(df_filtered: pd.DataFrame, top_n: int = 10) -> None:
    """
    Calculates and prints the top N contingencies based on the median REAL_SCORE.

    Args:
        df_filtered: The input DataFrame.
        top_n: The number of top contingencies to print.
    """

    def calculate_median_score(scores: List[Any]) -> float:
        """Calculates the median of a list of scores."""
        return statistics.median(scores) if scores else float("inf")

    dict_cont: Dict[str, List[float]] = {}
    for index, row in df_filtered.itertuples():
        dict_cont.setdefault(index, []).append(row.REAL_SCORE)

    sorted_cont = sorted(
        dict_cont.items(),
        key=lambda item: calculate_median_score(item[1]),
        reverse=True,
    )

    print("\nTop", top_n, "Contingencies (All Time):")
    for j, (name, median_score) in enumerate(sorted_cont[:top_n], start=1):
        print(f"{j}: {name} - {median_score:.4f}")  # Formatted output


def week_day_top(df_filtered: pd.DataFrame, top_n: int = 14) -> None:
    """
    Calculates and prints the top N contingencies for each day of the week.

    Args:
        df_filtered: The input DataFrame.
        top_n: The number of top contingencies to print per day.
    """

    df_filtered["W_DAY"] = df_filtered["DATE"].dt.weekday
    list_dicts: List[List[str]] = []

    for i in range(7):
        df_day = df_filtered[df_filtered["W_DAY"] == i]
        dict_cont: Dict[str, List[float]] = {}
        for index, row in df_day.itertuples():
            dict_cont.setdefault(index, []).append(row.REAL_SCORE)
        sorted_cont = sorted(
            dict_cont.items(), key=lambda item: statistics.median(item[1]), reverse=True
        )
        list_dicts.append([name for name, _ in sorted_cont[:top_n]])

    print("\nTop Contingencies by Day of Week:")
    for i, day_names in enumerate(list_dicts):
        print(f"\nDay {i}:")
        for pos_i in day_names:
            changes = ""
            if pos_i not in list_dicts[(i + 1) % 7]:
                changes += "-"
            if pos_i not in list_dicts[(i - 1) % 7]:
                changes += "+"
            print(f"{changes or '  '} {pos_i}")


def month_top(df_filtered: pd.DataFrame, months: List[int] = [1, 2, 6], top_n: int = 14) -> None:
    """
    Calculates and prints the top N contingencies for specified months.

    Args:
        df_filtered: The input DataFrame.
        months: The list of months to analyze (1-12).
        top_n: The number of top contingencies to print per month.
    """

    df_filtered["MONTH"] = df_filtered["DATE"].dt.month
    list_dicts: List[List[str]] = []

    for i in months:
        df_month = df_filtered[df_filtered["MONTH"] == i]
        dict_cont: Dict[str, List[float]] = {}
        for index, row in df_month.itertuples():
            dict_cont.setdefault(index, []).append(row.REAL_SCORE)
        sorted_cont = sorted(
            dict_cont.items(), key=lambda item: statistics.median(item[1]), reverse=True
        )
        list_dicts.append([name for name, _ in sorted_cont[:top_n]])

    print("\nTop Contingencies by Month:")
    for i, month_names in enumerate(list_dicts):
        print(f"\nMonth {months[i]}:")
        for pos_i in month_names:
            changes = ""
            if pos_i not in list_dicts[(i + 1) % len(list_dicts)]:
                changes += "-"
            if pos_i not in list_dicts[(i - 1) % len(list_dicts)]:
                changes += "+"
            print(f"{changes or '  '} {pos_i}")


def hour_top(df_filtered: pd.DataFrame, top_n: int = 14) -> None:
    """
    Calculates and prints the top N contingencies for each hour of the day.

    Args:
        df_filtered: The input DataFrame.
        top_n: The number of top contingencies to print per hour.
    """

    df_filtered["HOUR"] = df_filtered["DATE"].dt.hour
    list_dicts: List[List[str]] = []

    for i in range(24):
        df_hour = df_filtered[df_filtered["HOUR"] == i]
        dict_cont: Dict[str, List[float]] = {}
        for index, row in df_hour.itertuples():
            dict_cont.setdefault(index, []).append(row.REAL_SCORE)
        sorted_cont = sorted(
            dict_cont.items(), key=lambda item: statistics.median(item[1]), reverse=True
        )
        list_dicts.append([name for name, _ in sorted_cont[:top_n]])

    print("\nTop Contingencies by Hour:")
    for i, hour_names in enumerate(list_dicts):
        print(f"\nHour {i}:")
        for pos_i in hour_names:
            changes = ""
            if pos_i not in list_dicts[(i + 1) % 24]:
                changes += "-"
            if pos_i not in list_dicts[(i - 1) % 24]:
                changes += "+"
            print(f"{changes or '  '} {pos_i}")


def hour_boxplot(df_contg: pd.DataFrame, str_score: str) -> None:
    """
    Creates a boxplot of scores by hour for a specific day.

    Args:
        df_contg: The input DataFrame.
        str_score: The column name for the score.
    """

    start_date = datetime(2024, 12, 2)
    end_date = datetime(2024, 12, 3)

    df_filtered = df_contg.sort_values(by="DATE", ascending=True)
    df_filtered = df_filtered[
        (df_filtered["DATE"] > start_date) & (df_filtered["DATE"] < end_date)
    ]
    df_filtered = df_filtered[df_filtered["STATUS"] == "BOTH"]

    plt.figure(figsize=(12, 6))  # Adjust figure size
    sns.boxplot(
        x=df_filtered["DATE"].dt.strftime("%H:00"),
        y=pd.to_numeric(df_filtered[str_score]),
    )
    plt.xlabel("Hour of Day")
    plt.ylabel(str_score)
    plt.title(f"Boxplot of {str_score} by Hour")
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.ylim(2000, 14000)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()


def day_boxplot(df_contg: pd.DataFrame, str_score: str) -> None:
    """
    Creates a boxplot of scores by day for a specific month.

    Args:
        df_contg: The input DataFrame.
        str_score: The column name for the score.
    """

    start_date = datetime(2024, 12, 1)
    end_date = datetime(2025, 1, 1)

    df_filtered = df_contg.sort_values(by="DATE", ascending=True)
    df_filtered = df_filtered[
        (df_filtered["DATE"] > start_date) & (df_filtered["DATE"] < end_date)
    ]
    df_filtered = df_filtered[df_filtered["STATUS"] == "BOTH"]

    plt.figure(figsize=(14, 6))
    sns.boxplot(x=df_filtered["DATE"].dt.date, y=pd.to_numeric(df_filtered[str_score]))
    plt.xlabel("Day of Month")
    plt.ylabel(str_score)
    plt.title(f"Boxplot of {str_score} by Day")
    plt.xticks(rotation=90)
    plt.ylim(2000, 14000)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def score_histogram(df_contg: pd.DataFrame, column_name: str) -> None:
    """
    Creates a histogram of scores.

    Args:
        df_contg: The input DataFrame.
        column_name: The column name for the score.
    """

    def convertable_to_float(string: Any) -> float:
        """Converts a value to float, returns a large value on failure."""
        try:
            return float(string)
        except ValueError:
            return float("inf")

    df_filtered = df_contg.copy()
    df_filtered[column_name] = df_filtered[column_name].apply(convertable_to_float)
    df_filtered = df_filtered[df_filtered[column_name] < 20000]

    plt.figure(figsize=(8, 5))
    plt.hist(
        df_filtered[column_name],
        bins=50,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2,
    )
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
