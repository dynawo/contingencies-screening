from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statistics
from typing import Dict, List


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
    csv_files = list(Path(path).rglob(f"**/{target_filename}"))

    if not csv_files:
        print(f"Warning: No '{target_filename}' files found recursively under {path}")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} '{target_filename}' files to load.")

    for csv_file in csv_files:
        if csv_file.is_file():
            try:
                df = pd.read_csv(csv_file, sep=";", index_col="NAME")
                time_str = csv_file.stem.replace("snapshot_", "").replace("_contg_df", "")
                if len(time_str) >= 12:
                    date_time_str = time_str[:8] + "_" + time_str[9:13] + "00"
                    df["DATE"] = pd.to_datetime(date_time_str, format="%Y%m%d_%H%M%S")
                else:
                    print(f"Warning: Could not parse date from filename: {csv_file.stem}")
                    df["DATE"] = pd.NaT
                all_dfs.append(df)
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
        df_contg = pd.concat(all_dfs, axis=0, ignore_index=False).dropna(subset=["DATE"])
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame()

    df_contg = df_contg.drop("NUM", axis=1, errors="ignore")

    print(f"Loaded DataFrame shape before dropna: {df_contg.shape}")
    df_contg = df_contg.dropna()
    print(f"Final DataFrame shape after dropna: {df_contg.shape}")

    return df_contg


def all_time_top(df_filtered: pd.DataFrame) -> None:
    """
    Calculates and prints the top 10 entries based on the median 'REAL_SCORE'.

    Args:
        df_filtered: The input DataFrame containing a 'REAL_SCORE' column and an index.
    """
    median_scores: Dict[str, float] = (
        df_filtered.groupby(level=0)["REAL_SCORE"]
        .apply(list)
        .apply(statistics.median)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    print("\nTop 10 Entries (All Time Median Score):\n")
    for i, (name, score) in enumerate(median_scores.items(), 1):
        print(f"{i}: {name} - {score:.2f}")


def week_day_top(df_filtered: pd.DataFrame) -> None:
    """
    Calculates and compares the top 14 entries based on the median 'REAL_SCORE' for each day of the week.

    Args:
        df_filtered: The input DataFrame containing a 'DATE' column and a 'REAL_SCORE' column as well as an index.
    """
    df_filtered["W_DAY"] = df_filtered["DATE"].dt.weekday
    daily_top_lists: List[List[str]] = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day_index in range(7):
        daily_df = df_filtered[df_filtered["W_DAY"] == day_index]
        median_scores = (
            daily_df.groupby(level=0)["REAL_SCORE"]
            .apply(list)
            .apply(statistics.median)
            .sort_values(ascending=False)
            .head(14)
            .index.tolist()
        )
        daily_top_lists.append(median_scores)

    print("\nTop Entries by Day of the Week (Median Score):\n")
    for i, top_list in enumerate(daily_top_lists):
        print(f"\n{day_names[i]}:\n")
        for item in top_list:
            indicator = "  "
            if i > 0 and item not in daily_top_lists[i - 1]:
                indicator = " +"
            if i < 6 and item not in daily_top_lists[i + 1]:
                indicator = "- "
            if (
                i > 0
                and i < 6
                and item not in daily_top_lists[i - 1]
                and item not in daily_top_lists[i + 1]
            ):
                indicator = "-+"
            elif i == 0 and item not in daily_top_lists[6] and item not in daily_top_lists[1]:
                indicator = "-+"
            elif i == 6 and item not in daily_top_lists[5] and item not in daily_top_lists[0]:
                indicator = "-+"
            print(f"{indicator}{item}")


def month_top(df_filtered: pd.DataFrame) -> None:
    """
    Calculates and compares the top 14 entries based on the median 'REAL_SCORE' for specific months (Jan, Feb, Jun).

    Args:
        df_filtered: The input DataFrame containing a 'DATE' column and a 'REAL_SCORE' column as well as an index.
    """
    df_filtered["MONTH"] = df_filtered["DATE"].dt.month
    month_indices = [6, 9, 12, 3]
    monthly_top_lists: List[List[str]] = []
    month_names = ["June", "September", "December", "March"]

    for month_index in month_indices:
        monthly_df = df_filtered[df_filtered["MONTH"] == month_index]
        median_scores = (
            monthly_df.groupby(level=0)["REAL_SCORE"]
            .apply(list)
            .apply(statistics.median)
            .sort_values(ascending=False)
            .head(14)
            .index.tolist()
        )
        monthly_top_lists.append(median_scores)

    print("\nTop Entries by Month (January, February, June - Median Score):\n")
    for i, top_list in enumerate(monthly_top_lists):
        print(f"\n{month_names[i]}:\n")
        for item in top_list:
            indicator = "  "
            if i > 0 and item not in monthly_top_lists[i - 1]:
                indicator = " +"
            if i < len(monthly_top_lists) - 1 and item not in monthly_top_lists[i + 1]:
                indicator = "- "
            if (
                i > 0
                and i < len(monthly_top_lists) - 1
                and item not in monthly_top_lists[i - 1]
                and item not in monthly_top_lists[i + 1]
            ):
                indicator = "-+"
            elif i == 0 and item not in monthly_top_lists[-1] and item not in monthly_top_lists[1]:
                indicator = "-+"
            elif (
                i == len(monthly_top_lists) - 1
                and item not in monthly_top_lists[-2]
                and item not in monthly_top_lists[0]
            ):
                indicator = "-+"
            print(f"{indicator}{item}")


def hour_top(df_filtered: pd.DataFrame) -> None:
    """
    Calculates and compares the top 14 entries based on the median 'REAL_SCORE' for each hour of the day.

    Args:
        df_filtered: The input DataFrame containing a 'DATE' column and a 'REAL_SCORE' column as well as an index.
    """
    df_filtered["HOUR"] = df_filtered["DATE"].dt.hour
    hourly_top_lists: List[List[str]] = []

    for hour_index in range(24):
        hourly_df = df_filtered[df_filtered["HOUR"] == hour_index]
        median_scores = (
            hourly_df.groupby(level=0)["REAL_SCORE"]
            .apply(list)
            .apply(statistics.median)
            .sort_values(ascending=False)
            .head(14)
            .index.tolist()
        )
        hourly_top_lists.append(median_scores)

    print("\nTop Entries by Hour of the Day (Median Score):\n")
    for i, top_list in enumerate(hourly_top_lists):
        print(f"\nHour {i:02d}:\n")
        for item in top_list:
            indicator = "  "
            if i > 0 and item not in hourly_top_lists[i - 1]:
                indicator = " +"
            if i < 23 and item not in hourly_top_lists[i + 1]:
                indicator = "- "
            if (
                i > 0
                and i < 23
                and item not in hourly_top_lists[i - 1]
                and item not in hourly_top_lists[i + 1]
            ):
                indicator = "-+"
            elif i == 0 and item not in hourly_top_lists[23] and item not in hourly_top_lists[1]:
                indicator = "-+"
            elif i == 23 and item not in hourly_top_lists[22] and item not in hourly_top_lists[0]:
                indicator = "-+"
            print(f"{indicator}{item}")


def hour_boxplot(df_contg: pd.DataFrame, str_score: str) -> None:
    """
    Creates a boxplot of a specified score for a specific date.

    Args:
        df_contg: The input DataFrame containing a 'DATE' column and the score column.
        str_score: The name of the column to plot.
    """
    try:
        start_date = datetime(2024, 12, 21, 0, 0, 0)
        end_date = datetime(2024, 12, 21, 23, 59, 59)
        df_filtered = df_contg[
            (df_contg["DATE"] >= start_date)
            & (df_contg["DATE"] <= end_date)
            & (df_contg["STATUS"] == "BOTH")
        ].copy().sort_values(by="DATE")

        if not df_filtered.empty:
            plt.figure(figsize=(12, 6))
            ax = plt.axes()
            ax.set_facecolor("white")
            sns.boxplot(
                x=df_filtered["DATE"].dt.strftime("%H:%M"),
                y=pd.to_numeric(df_filtered[str_score]),
            ).set(xlabel="Time of Day", ylabel=str_score)

            plt.xticks(rotation=45, ha="right")
            lower_limit = df_filtered[str_score].quantile(0.05)
            upper_limit = df_filtered[str_score].quantile(0.95)
            if not pd.isna(lower_limit) and not pd.isna(upper_limit):
                plt.ylim(lower_limit, upper_limit)
            plt.grid(color="grey", linewidth=0.5)
            plt.title(f"Boxplot of {str_score}")
            plt.tight_layout()
            plt.show()
        else:
            print("No data available for the specified date and status in hour_boxplot.")
    except KeyError as e:
        print(f"Error in hour_boxplot: Column not found: {e}")
    except ValueError as e:
        print(f"Error in hour_boxplot: {e}")


def day_boxplot(df_contg: pd.DataFrame, str_score: str) -> None:
    """
    Creates a boxplot of a specified score for the month.

    Args:
        df_contg: The input DataFrame containing a 'DATE' column and the score column.
        str_score: The name of the column to plot.
    """
    try:
        start_date = datetime(2024, 12, 1, 0, 0, 0)
        end_date = datetime(2024, 12, 31, 23, 59, 59)
        df_filtered = df_contg[
            (df_contg["DATE"] >= start_date)
            & (df_contg["DATE"] <= end_date)
            & (df_contg["STATUS"] == "BOTH")
        ].copy()
        df_filtered.loc[:, "DATE_ONLY"] = df_filtered["DATE"].dt.date

        if not df_filtered.empty:
            plt.figure(figsize=(14, 6))
            sns.set(rc={"figure.figsize": (14, 6)})
            ax = plt.axes()
            ax.set_facecolor("white")
            sns.boxplot(
                x="DATE_ONLY", y=pd.to_numeric(df_filtered[str_score]), data=df_filtered
            ).set(xlabel="Date", ylabel=str_score)
            plt.xticks(rotation=45, ha="right")
            lower_limit = df_filtered[str_score].quantile(0.05)
            upper_limit = df_filtered[str_score].quantile(0.95)
            if not pd.isna(lower_limit) and not pd.isna(upper_limit):
                plt.ylim(lower_limit, upper_limit)
            plt.grid(color="grey", linewidth=0.5)
            plt.title(f"Boxplot of {str_score}")
            plt.tight_layout()
            plt.show()
        else:
            print("No data available  and status in day_boxplot.")
    except KeyError as e:
        print(f"Error in day_boxplot: Column not found: {e}")
    except ValueError as e:
        print(f"Error in day_boxplot: {e}")


def score_histogram(df_contg: pd.DataFrame, column_name: str) -> None:
    """
    Creates a histogram of the specified score column, filtering out values above 20000 and non-numeric entries.

    Args:
        df_contg: The input DataFrame containing the score column.
        column_name: The name of the column to plot.
    """
    try:
        df_filtered = df_contg.copy()
        df_filtered.loc[:, column_name] = pd.to_numeric(
            df_filtered[column_name], errors="coerce"
        ).fillna(100000)
        df_filtered = df_filtered[df_filtered[column_name] < 20000]
        column_data = df_filtered[column_name]

        if not column_data.empty:
            plt.figure(figsize=(10, 6))
            ax = plt.axes()
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color("black")
            plt.hist(column_data, bins=50, edgecolor="black")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {column_name} (Values < 20000)")
            plt.grid(color="grey", linewidth=0.5, axis="y", alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No valid data to plot for {column_name} in score_histogram.")
    except KeyError as e:
        print(f"Error in score_histogram: Column not found: {e}")


def real_vs_predicted_score(df_filtered, mae):
    """
    Generates a scatter plot of predicted vs real score for data
    where the 'STATUS' column is 'BOTH', and displays the Mean Absolute Error (MAE).

    Args:
        df (pd.DataFrame): The DataFrame containing the 'STATUS',
                           'REAL_SCORE', and 'PREDICTED_SCORE' columns.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(df_filtered["REAL_SCORE"], df_filtered["PREDICTED_SCORE"], alpha=0.6)
    plt.xlabel("Real Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted Score vs. Real Score (STATUS = BOTH)")
    plt.grid(True)
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.text(0.05, 0.95, f"MAE = {mae:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    """
    Generates and displays a correlation matrix heatmap for the input DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which to calculate the correlation matrix.
    """
    correlation_matrix = df.corr(numeric_only=True)  # Calculate the correlation matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
