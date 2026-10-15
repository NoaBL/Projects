# ====================================================================================================
# functions.py
# Helper Functions and Configuration
# This file contains utility functions for data preprocessing, analysis, and logging.
# It also defines the structure of sections and subsections for the dataset.
# Note: This file is meant to be imported into the main notebook and used as a library of functions.
# =====================================================================================================    
# libraries used by functions.py
from datetime import datetime
import re
from pathlib import Path
from IPython.display import display
from scipy.stats import entropy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure


# ==================================================================
# Logging function
# ==================================================================
# This function records all analysis decisions and prints to screen
# Every preprocessing step should call this.
# ============================================================
def write_log(message: str, log_path):
    """
    Write timestamped message to log file AND print to console.

    Parameters
    ----------
    message : str
        Description of decision, preprocessing step, or action.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    # Write to log file
    with open(log_path, "a", encoding="utf-8-sig") as f:
        f.write(full_message + "\n")

    # Print to console
    print(full_message)


# ============================================================
# Plot / Output Helpers
# ============================================================
# Function: Save plot to outputs directory with logging
def save_plot(fig, filename: str, output_dir, log_path, dpi=300):
    """
    Save matplotlib figure to outputs directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        Name without extension
    dpi : int
        Resolution for saved image
    """

    save_path = output_dir / f"{filename}.png"

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    write_log(f"Plot saved: {filename}.png", log_path)
    print(f"Saved plot: {save_path}")


# Function: PCA visualization
def plot_pca(
    dataframe,
    feature_cols,
    title="PCA plot",
    color_col=None,
    figsize=(8, 6)
):

    # Validate columns
    missing_cols = [
        col for col in feature_cols
        if col not in dataframe.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing PCA columns: {missing_cols}"
        )

    # Create numeric dataframe
    pca_df = dataframe[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove incomplete rows
    pca_df = pca_df.dropna()

    if len(pca_df) == 0:
        raise ValueError(
            "No complete rows available for PCA."
        )

    # Standardize features
    # IMPORTANT: PCA is scale-sensitive
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        pca_df
    )

    # Run PCA
    pca = PCA(n_components=2)

    pca_components = pca.fit_transform(
        scaled_data
    )

    # Create PCA plotting dataframe
    plot_df = pd.DataFrame({
        "PC1": pca_components[:, 0],
        "PC2": pca_components[:, 1]
    })

    # Optional coloring
    if color_col is not None:

        plot_df[color_col] = (
            dataframe.loc[
                pca_df.index,
                color_col
            ].values
        )

    # Explained variance
    explained_var = (
        pca.explained_variance_ratio_ * 100
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    if color_col is None:

        ax.scatter(
            plot_df["PC1"],
            plot_df["PC2"],
            alpha=0.6
        )

    else:

        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            hue=color_col,
            alpha=0.7,
            ax=ax
        )

    # Labels
    ax.set_xlabel(
        f"PC1 ({explained_var[0]:.1f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({explained_var[1]:.1f}% variance)"
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.show()

    plt.close()

    # Return PCA objects
    return {
        "pca_df": plot_df,
        "pca_model": pca,
        "explained_variance": explained_var
    }


# ============================================================
# Dictionary & Label Management
# (subsection/section lookups and label updates)
# ============================================================

# Function: Get subsection columns that still exist in df
def get_subsection(subsection_code, dataframe, subsection_dict):
    """
    Returns only columns from the subsection that still exist
    in the provided dataframe.
    """

    if subsection_code not in subsection_dict:
        raise ValueError(f"Subsection '{subsection_code}' not found.")

    return [c for c in subsection_dict[subsection_code] if c in dataframe.columns]


# Function: Get dataframe slice of a subsection
def get_subsection_df(subsection_code, dataframe, subsection_dict):

    cols = get_subsection(subsection_code, dataframe, subsection_dict)

    return dataframe[cols]


# Function: Get section columns that still exist in df
def get_section(section_code, dataframe, section_dict):
    """
    Returns only columns from the section that still exist
    in the provided dataframe.
    """

    if section_code not in section_dict:
        raise ValueError(f"Section '{section_code}' not found.")

    return [c for c in section_dict[section_code] if c in dataframe.columns]


def get_section_df(section_code, dataframe, section_dict):

    cols = get_section(section_code, dataframe, section_dict)

    return dataframe[cols]



# Function: Get section/subsection label for a column
def get_column_label(column_name, subsection_dict, section_dict):
    """
    Return the section and subsection assignment for a given column.

    Parameters
    ----------
    column_name : str
        Column name to look up.

    Returns
    -------
    dict
        Dictionary with:
        - column_name
        - subsection
        - section
    """

    # Search subsection dictionary first (most direct source)
    for subsection, cols in subsection_dict.items():
        if column_name in cols:

            # Find corresponding section
            for section, section_cols in section_dict.items():
                if column_name in section_cols:
                    return {
                        "column_name": column_name,
                        "subsection": subsection,
                        "section": section
                    }

    raise ValueError(f"Column '{column_name}' not found in dictionary.")


# Function: Register / update dictionary label for a column
def update_column_label(
    column_name,
    subsection,
    section,
    subsection_dict,
    section_dict,
    subsection_column_names_scales,
    log_path,
    subsection_table_name="subsection_column_names_scales"
):
    """
    Register or update a column label across all dictionary objects:
    1. subsection_dict
    2. section_dict
    3. subsection_column_names_scales

    Parameters
    ----------
    column_name : str
        Column to register / update.

    subsection : str
        Subsection label to assign.

    section : str
        Section label to assign.

    subsection_table_name : str, default="subsection_column_names_scales"
        Name of the long-format dictionary table variable in globals().

    Notes
    -----
    - If column already exists under another label, it is removed first.
    - Ensures the column appears exactly once across subsection_dict.
    - Updates section_dict consistently.
    - Updates subsection_column_names_scales consistently.
    """

    # --------------------------------------------------------
    # Step 1 — Remove existing assignment from subsection_dict
    # --------------------------------------------------------
    for sub, cols in subsection_dict.items():
        if column_name in cols:
            subsection_dict[sub] = [c for c in cols if c != column_name]

    # --------------------------------------------------------
    # Step 2 — Remove existing assignment from section_dict
    # --------------------------------------------------------
    for sec, cols in section_dict.items():
        if column_name in cols:
            section_dict[sec] = [c for c in cols if c != column_name]

    # --------------------------------------------------------
    # Step 3 — Add new assignment to subsection_dict
    # --------------------------------------------------------
    if subsection not in subsection_dict:
        subsection_dict[subsection] = []

    if column_name not in subsection_dict[subsection]:
        subsection_dict[subsection].append(column_name)

    # --------------------------------------------------------
    # Step 4 — Add new assignment to section_dict
    # --------------------------------------------------------
    if section not in section_dict:
        section_dict[section] = []

    if column_name not in section_dict[section]:
        section_dict[section].append(column_name)

    # --------------------------------------------------------
    # Step 5 — Update long-format dictionary table
    # --------------------------------------------------------
    subsection_column_names_scales = subsection_column_names_scales[
        subsection_column_names_scales["column_name"] != column_name
    ].copy()

    n_cols = len(subsection_dict[subsection])

    new_row = pd.DataFrame([{
        "subsection": subsection,
        "section": section,
        "n_columns": n_cols,
        "column_name": column_name
    }])

    subsection_column_names_scales = pd.concat(
        [subsection_column_names_scales, new_row],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Step 6 — Deduplicate (safety)
    # --------------------------------------------------------
    for sub in subsection_dict:
        subsection_dict[sub] = list(dict.fromkeys(subsection_dict[sub]))

    for sec in section_dict:
        section_dict[sec] = list(dict.fromkeys(section_dict[sec]))

    subsection_column_names_scales = subsection_column_names_scales.drop_duplicates(
        subset=["column_name"],
        keep="last"
    ).reset_index(drop=True)

    write_log(
        f"Dictionary label updated: {column_name} -> "
        f"subsection='{subsection}', section='{section}'",
        log_path
    )


# ============================================================
# Dataframe Utilities
# ============================================================

# Function: Define columns that should be used for integrity checks
def get_valid_analysis_columns(df, analysis_columns):
    return [col for col in analysis_columns if col in df.columns]


# =================================
# TEXT → NUMERIC PARSING FRAMEWORK
# (time parsing helpers)
# =================================

# TIME IN MINUTES
# Function: extract hour values from text (e.g. "שעתיים" → 2, "חצי שעה" → 0.5)
def extract_hour_values(text):

    values = []

    # numeric values
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    values.extend([float(n) for n in nums])

    # IMPORTANT: order matters (avoid double counting)
    mapping = [
        ("שעתיים", 2),
        ("כשעתיים", 2),
        ("חצי שעה", 0.5),
        ("כחצי שעה", 0.5),
        ("שעה", 1),
    ]

    for key, val in mapping:
        if key in text:
            values.append(val)

    return values

# Core parsers
def parse_time_to_minutes(val, config, idx=None, change_log=None):
    # this parser assumes that the small numbers represent hours, for example 4 will be parsed as 240 (minutes)

    if pd.isna(val):
        return val

    original = str(val).strip().lower()

    # -----------------------------------
    # 1. Pure numeric - SMALL NUMBERS ARE ASSUMED HOURS UNLESS SPECIFIED OTHERWISE! - changed! to default minutes
    # 1. Pure numeric - SMALL NUMBERS ARE ASSUMED MINUTES UNLESS SPECIFIED OTHERWISE!
    # -----------------------------------
    
    if re.fullmatch(r"\d+(\.\d+)?", original):
        return round(float(original))

    # -----------------------------------
    # 2. Minutes
    # -----------------------------------
    if "דק" in original or "דקות" in original:
        num = re.search(r"\d+(\.\d+)?", original)
        if num:
            new_val = round(float(num.group()))
            if change_log is not None:
                change_log.append((idx, original, new_val, "parsed"))
            return new_val

    # -----------------------------------
    # 3. Special explicit Hebrew cases
    # -----------------------------------
    if "חצי שעה" in original or "כחצי שעה" in original:
        if change_log is not None:
            change_log.append((idx, original, 30, "parsed"))
        return 30

    # -----------------------------------
    # 4. Numeric ranges (0-15 → 8)
    # -----------------------------------
    if config.get("allow_ranges", False):
        match = re.search(r"(\d+)\s*-\s*(\d+)", original)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            new_val = round((low + high) / 2)
            if change_log is not None:
                change_log.append((idx, original, new_val, "parsed_range"))
            return new_val

    
    # -----------------------------------
    # 5. Hours expressions
    # -----------------------------------
    #if "שעה" in original or "שעות" in original:
    hour_values = extract_hour_values(original)

    if len(hour_values) > 0:

        hour_values = extract_hour_values(original)

        if len(hour_values) == 1:
            new_val = round(hour_values[0] * 60)
            if change_log is not None:
                change_log.append((idx, original, new_val, "parsed"))
            return new_val

        elif len(hour_values) >= 2:
            new_val = round((sum(hour_values) / len(hour_values)) * 60)
            if change_log is not None:
                change_log.append((idx, original, new_val, "parsed_range"))
            return new_val



    # -----------------------------------
    # 6. Zero-like expressions
    # -----------------------------------
    if config.get("allow_zero_words", False):
        zero_terms = ["לא", "no", "כלל לא", "כמעט אפס"]
        if any(term in original for term in zero_terms):
            if change_log is not None:
                change_log.append((idx, original, 0, "parsed"))
            return 0

    # -----------------------------------
    # 7. Known phrases
    # -----------------------------------
    if "less than an hour" in original:
        return 30

    if "לפחות 120" in original:
        return 120

    if "מעל 3 שעות" in original:
        return 180

    # -----------------------------------
    # 8. Fallback
    # -----------------------------------
    #return val
    return np.nan

# TIME IN YEARS
def parse_time_to_years(val, config, idx=None, change_log=None):

    if pd.isna(val):
        return val

    original = str(val).strip().lower()

    # -----------------------------------
    # 1. Pure numeric → assume YEARS
    # -----------------------------------
    if re.fullmatch(r"\d+(\.\d+)?", original):

        num = float(original)

        # floor to nearest year (2.5 → 2)
        new_val = int(np.floor(num))

        if change_log is not None and new_val != num:
            change_log.append((idx, original, new_val, "parsed_floor"))

        return new_val

    # -----------------------------------
    # 2. Months (English + Hebrew)
    # -----------------------------------
    if any(term in original for term in ["month", "months", "חודש", "חודשים"]):

        num = re.search(r"\d+(\.\d+)?", original)

        if num:
            months = float(num.group())
            years = int(np.floor(months / 12))

            if change_log is not None:
                change_log.append((idx, original, years, "parsed_months_to_years"))

            return years

    # -----------------------------------
    # 3. Mixed expressions (years + months)
    # Example: "1 year 3 months" / "שנה ו-3 חודשים"
    # -----------------------------------
    year_match = re.search(r"(\d+(\.\d+)?)\s*(year|years|שנה|שנים)", original)
    month_match = re.search(r"(\d+(\.\d+)?)\s*(month|months|חודש|חודשים)", original)

    if year_match or month_match:

        years = 0
        months = 0

        if year_match:
            years = float(year_match.group(1))

        if month_match:
            months = float(month_match.group(1))

        total_years = years + (months / 12)
        new_val = int(np.floor(total_years))

        if change_log is not None:
            change_log.append((idx, original, new_val, "parsed_mixed"))

        return new_val

    # -----------------------------------
    # 4. Numeric ranges (optional)
    # Example: "2-3" → 2
    # -----------------------------------
    if config.get("allow_ranges", False):

        match = re.search(r"(\d+)\s*-\s*(\d+)", original)

        if match:
            low = float(match.group(1))
            high = float(match.group(2))

            # conservative → take lower bound (age not yet reached)
            new_val = int(np.floor(low))

            if change_log is not None:
                change_log.append((idx, original, new_val, "parsed_range_floor"))

            return new_val

    # -----------------------------------
    # 5. Hebrew words (basic support)
    # -----------------------------------
    if "חצי שנה" in original:
        if change_log is not None:
            change_log.append((idx, original, 0, "parsed_half_year"))
        return 0

    # -----------------------------------
    # 6. Fallback
    # -----------------------------------
    #return val
    # if non of the above - return nan and will handle the original (problematic) later
    return np.nan

# TIME IN WEEKS
def parse_time_to_weeks(val, config, idx=None, change_log=None):

    if pd.isna(val):
        return val

    original = str(val).strip().lower()

    # -----------------------------------
    # 1. Pure numeric (e.g. 41, 41.5)
    # -----------------------------------
    if re.fullmatch(r"\d+(\.\d+)?", original):

        num = float(original)
        new_val = int(np.floor(num))

        if change_log is not None and new_val != num:
            change_log.append((idx, original, new_val, "parsed_floor"))

        return new_val

    # -----------------------------------
    # 2. Patterns like "41+3", "41 + 3"
    # -----------------------------------
    plus_match = re.search(r"(\d+)\s*\+\s*\d+", original)

    if plus_match:
        week = int(plus_match.group(1))

        if change_log is not None:
            change_log.append((idx, original, week, "parsed_plus_days"))

        return week

    # -----------------------------------
    # 3. Hebrew "פלוס"
    # Example: "שבוע 39 פלוס 2"
    # -----------------------------------
    plus_hebrew_match = re.search(r"(\d+)\s*פלוס\s*\d+", original)

    if plus_hebrew_match:
        week = int(plus_hebrew_match.group(1))

        if change_log is not None:
            change_log.append((idx, original, week, "parsed_plus_days_he"))

        return week

    # -----------------------------------
    # 4. General week expressions
    # "week 40", "40 weeks", "שבוע 40"
    # -----------------------------------
    week_match = re.search(
        r"(\d+(\.\d+)?)\s*(week|weeks|שבוע|שבועות)?",
        original
    )

    if week_match:
        num = float(week_match.group(1))
        new_val = int(np.floor(num))

        if change_log is not None:
            change_log.append((idx, original, new_val, "parsed_week"))

        return new_val

    # -----------------------------------
    # 5. Ranges (optional)
    # Example: "38-39" → 38
    # -----------------------------------
    if config.get("allow_ranges", False):

        match = re.search(r"(\d+)\s*-\s*(\d+)", original)

        if match:
            low = int(match.group(1))

            if change_log is not None:
                change_log.append((idx, original, low, "parsed_range_floor"))

            return low

    # -----------------------------------
    # 6. Fallback
    # -----------------------------------
    return np.nan

# Apply parser to a column with change logging
def apply_parser(df, column, parser_func, config):

    change_log = []

    clean_col = f"{column}_clean"
    raw_col = f"{column}_raw"

    df[raw_col] = df[column]

    df[clean_col] = [
        parser_func(val, config, idx=i, change_log=change_log)
        for i, val in df[column].items()
    ]

    change_df = pd.DataFrame(
        change_log,
        columns=["index", "original", "new", "change_type"]
    )

    if "record_id" in df.columns:
        change_df["record_id"] = change_df["index"].map(df["record_id"])

    return df, change_df

# Function: Enforce output dtype on *_clean column after parsing
def enforce_output_dtype(df, column, config):

    clean_col = f"{column}_clean"
    dtype = config.get("output_dtype", None)

    if dtype is None:
        return df  # no enforcement

    # Convert to numeric safely
    df[clean_col] = pd.to_numeric(df[clean_col], errors="coerce")

    # Apply dtype
    if dtype == "Int64":
        df[clean_col] = np.floor(df[clean_col]).astype("Int64")

    elif dtype == "float":
        df[clean_col] = df[clean_col].astype(float)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return df

# Function: Run parser with manual review step
def run_parser_pre_manual(df, column, parser_func, config, output_dir, log_path):

    write_log(f"Running parser for column: {column}", log_path)

    change_log = []

    raw_col = f"{column}_raw"
    clean_col = f"{column}_clean"

    # Preserve raw
    df[raw_col] = df[column]

    # Apply parser
    df[clean_col] = [
        parser_func(val, config, idx=i, change_log=change_log)
        for i, val in df[column].items()
    ]

    # Build auto change log
    change_df = pd.DataFrame(
        change_log,
        columns=["index", "original", "new", "change_type"]
    )

    if "record_id" in df.columns:
        change_df["record_id"] = change_df["index"].map(df["record_id"])

    # Export review file
    review_df = df[[
        "record_id",
        raw_col,
        clean_col
    ]].copy()

    review_df["final_value"] = review_df[clean_col]

    review_path = output_dir / f"{column}_manual_review.csv"

    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    write_log(f"Manual review file exported: {review_path}", log_path)

    print("STOP! go over the file in:", review_path) 
    print("See if you want to make manual adjustments. If so, make them, save and exit and come back here to run the next block")

    return df, change_df, review_path


# Function: apply manual corrections
def apply_manual_corrections(df, column, change_df, review_path, output_dir, config, log_path):


    write_log(f"Applying manual corrections for column: {column}", log_path)

    raw_col = f"{column}_raw"
    clean_col = f"{column}_clean"

    # ------------------------------------------------------------
    # Load edited file
    # ------------------------------------------------------------
    review_df = pd.read_csv(review_path, encoding="utf-8-sig")

    manual_changes = []

    # ------------------------------------------------------------
    # Apply row-wise logic (same as your working version)
    # ------------------------------------------------------------
    for _, row in review_df.iterrows():

        rid = row["record_id"]
        raw = row[raw_col]
        parsed = row[clean_col]
        final = row["final_value"]

        # normalize numeric
        parsed_num = pd.to_numeric(parsed, errors="coerce")
        final_num = pd.to_numeric(final, errors="coerce")

        changed = False

        # -----------------------------------
        # Case 1 — both text → compare RAW vs FINAL
        # -----------------------------------
        if pd.isna(parsed_num) and pd.isna(final_num):

            if str(raw).strip() != str(final).strip():
                changed = True

        # -----------------------------------
        # Case 2 — one numeric, one not
        # -----------------------------------
        elif pd.isna(parsed_num) or pd.isna(final_num):
            changed = True

        # -----------------------------------
        # Case 3 — both numeric
        # -----------------------------------
        else:
            if parsed_num != final_num:
                changed = True

        # -----------------------------------
        # Apply change
        # -----------------------------------
        if changed:

            manual_changes.append({
                "record_id": rid,
                "original": raw,
                "parsed_value": parsed,
                "final_value": final,
                "change_type": "manual"
            })

            '''df.loc[df["record_id"] == rid, clean_col] = final'''
            # Preserve blank manual values as missing
            if pd.isna(final) or str(final).strip() == "":
                final = np.nan

            df.loc[df["record_id"] == rid, clean_col] = final


    # ------------------------------------------------------------
    # Combine logs
    # ------------------------------------------------------------
    auto_log = change_df.copy()
    manual_log = pd.DataFrame(manual_changes)

    full_log = pd.concat([auto_log, manual_log], ignore_index=True)

    # ------------------------------------------------------------
    # Clean and format log
    # ------------------------------------------------------------
    if "index" in full_log.columns:
        full_log = full_log.drop(columns=["index"])

    if "record_id" not in full_log.columns:
        raise ValueError("record_id column missing from change log")

    full_log["record_id"] = pd.to_numeric(full_log["record_id"], errors="coerce")

    cols = ["record_id"] + [c for c in full_log.columns if c != "record_id"]
    full_log = full_log[cols]

    full_log = full_log.sort_values(by="record_id").reset_index(drop=True)

    # ------------------------------------------------------------
    # Enforce final dtype
    # ------------------------------------------------------------
    df = enforce_output_dtype(df, column, config)
    
    write_log(f"{column}_clean dtype enforced as {config.get('output_dtype')}", log_path)

    # ------------------------------------------------------------
    # Drop *_raw column (clean final structure)
    # ------------------------------------------------------------
    if raw_col in df.columns:
        df = df.drop(columns=[raw_col])
    
        write_log(f"{raw_col} dropped from final dataframe (original column retained)", log_path)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    log_path = output_dir / f"{column}_full_change_log.csv"

    full_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    write_log(f"Total automatic changes: {len(auto_log)}", log_path)
    write_log(f"Total manual changes: {len(manual_log)}", log_path)
    write_log(f"Full change log saved: {log_path}", log_path)

    return df, full_log


# Function: Convert feature values from one scale to another
def convert_feature_scale(
    feature_name,
    current_scale,
    new_scale,
    dataframe,
    log_path
):
    """
    Convert a feature from one numeric scale to another in-place.

    Example
    -------
    current_scale = [1, 5]
    new_scale     = [0, 4]

    Conversion:
    1 -> 0
    2 -> 1
    3 -> 2
    4 -> 3
    5 -> 4

    Parameters
    ----------
    feature_name : str
        Column to convert.

    current_scale : list or tuple of length 2
        Current [min, max] scale of the feature.

    new_scale : list or tuple of length 2
        Desired [min, max] scale to convert into.

    dataframe : pd.DataFrame, default=df
        DataFrame containing the feature.

    Notes
    -----
    - Conversion is done in-place (original column is overwritten).
    - Missing values remain missing.
    - Values outside the declared current scale are set to NaN.
    - Transformation is linear and preserves ordering.

    Returns
    -------
    str
        Name of updated feature (same as input feature_name).
    """

    # --------------------------------------------------------
    # Step 1 — Validate inputs
    # --------------------------------------------------------
    if feature_name not in dataframe.columns:
        raise ValueError(f"Column '{feature_name}' not found.")

    if len(current_scale) != 2 or len(new_scale) != 2:
        raise ValueError("current_scale and new_scale must each have exactly 2 values.")

    old_min, old_max = current_scale
    new_min, new_max = new_scale

    if old_max == old_min:
        raise ValueError("Invalid current_scale: min and max cannot be equal.")

    write_log(
        f"Converting scale for '{feature_name}' "
        f"from [{old_min}, {old_max}] to [{new_min}, {new_max}]", 
        log_path
    )

    # --------------------------------------------------------
    # Step 2 — Coerce to numeric
    # --------------------------------------------------------
    # Non-numeric values become NaN and remain missing
    values = pd.to_numeric(dataframe[feature_name], errors="coerce")

    # --------------------------------------------------------
    # Step 3 — Identify invalid values outside declared scale
    # --------------------------------------------------------
    invalid_mask = values.notna() & (
        (values < old_min) | (values > old_max)
    )

    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        write_log(
            f"{feature_name}: found {n_invalid} values outside declared "
            f"current scale [{old_min}, {old_max}] -> set to NaN",
            log_path
        )

    # Values outside declared current scale are invalid for conversion
    values.loc[invalid_mask] = np.nan

    # --------------------------------------------------------
    # Step 4 — Apply linear scale conversion
    # --------------------------------------------------------
    # Formula:
    converted = new_min + ((values - old_min) / (old_max - old_min)) * (new_max - new_min)

    # --------------------------------------------------------
    # Step 5 — Overwrite original feature in-place
    # --------------------------------------------------------
    dataframe[feature_name] = converted

    write_log(
        f"Scale conversion complete for '{feature_name}' "
        f"(values overwritten in-place)", log_path
    )

    return feature_name

def parse_change_direction(media_2wk, media_reserv, media_2wk_clean, media_reserv_clean):

    # Numeric comparison available
    if pd.notna(media_2wk_clean) and pd.notna(media_reserv_clean):

        if media_reserv_clean > media_2wk_clean:
            return 1

        if media_reserv_clean < media_2wk_clean:
            return -1

        return 0

    # Text comparison
    text = str(media_reserv).strip().lower()

    more_terms = [
        "יותר",
        "קצת יותר",
        "הרבה יותר",
        "ראיתי הרבה יותר והתעדכנתי"
    ]

    less_terms = [
        "פחות",
        "קצת פחות",
        "הרבה פחות"
    ]

    same_terms = [
        "גם ככה",
        "אותו דבר",
        "ללא שינוי",
        "כרגיל"
    ]

    if any(term in text for term in more_terms):
        return 1

    if any(term in text for term in less_terms):
        return -1

    if any(term in text for term in same_terms):
        return 0

    return np.nan

# Function: Reverse feature scale
def reverse_scale(
    feature_name,
    scale_min,
    scale_max,
    dataframe,
    log_path,
    suffix="_rev"
):
    """
    Reverse the scale of a feature.

    Example
    -------
    On a 1–5 scale:
    1 -> 5
    2 -> 4
    3 -> 3
    4 -> 2
    5 -> 1

    Parameters
    ----------
    feature_name : str
        Source feature to reverse.

    scale_min : int or float
        Minimum scale value.

    scale_max : int or float
        Maximum scale value.

    suffix : str, default="_rev"
        Suffix for reversed feature.

    dataframe : pd.DataFrame, default=df
        DataFrame to operate on.

    Process
    -------
    1. Coerce feature to numeric.
    2. Reverse scale using:
       reversed = scale_max + scale_min - value
    3. Store result in new feature.
    4. Log everything.

    Returns
    -------
    str
        Name of reversed feature.
    """

    if feature_name not in dataframe.columns:
        raise ValueError(f"Column '{feature_name}' not found.")

    reversed_feature = f"{feature_name}{suffix}"

    values = pd.to_numeric(dataframe[feature_name], errors="coerce")

    dataframe[reversed_feature] = scale_max + scale_min - values

    write_log(
        f"Created reversed feature '{reversed_feature}' "
        f"from '{feature_name}' using scale [{scale_min}, {scale_max}]", 
        log_path
    )

    return reversed_feature


# Function: bulk scale conversion; Apply convert_feature_scale to all
# features in a subsection, writing *_scalefix copies
def convert_section_subsection_scale(
    section,
    subsection,
    current_scale,
    new_scale,
    dataframe,
    log_path,
    subsection_dict,
    section_dict,
    subsection_column_names_scales,
    suffix="_scalefix"
):
    """
    Apply the same scale conversion to all features belonging to a
    specific dictionary-defined subsection.

    Instead of overwriting the original features, this function creates
    converted copies named: original_name + suffix
    (example: children_2 -> children_2_scalefix)

    The new derived features inherit the same subsection / section labels
    as their source features via `update_column_label()`.

    Parameters
    ----------
    section : str
        Expected section label for validation.

    subsection : str
        Subsection label to target.

    current_scale : list or tuple of length 2
        Current [min, max] scale shared by all items.

    new_scale : list or tuple of length 2
        Desired [min, max] scale to convert into.

    dataframe : pd.DataFrame, default=df
        DataFrame containing the actual item values.

    suffix : str, default="_scalefix"
        Suffix appended to converted feature names.

    Notes
    -----
    - Uses subsection_dict / get_subsection() as the source of truth.
    - Original features are preserved.
    - Converted values are written to new *_scalefix columns.
    - Missing values are preserved.
    - Values outside the declared current scale are set to NaN
      by `convert_feature_scale()`.
    - New derived columns inherit dictionary labels from source columns.

    Returns
    -------
    list[str]
        List of newly created converted features.
    """

    write_log(
        f"Bulk scale conversion | section='{section}' | subsection='{subsection}' "
        f"| from {current_scale} to {new_scale}",
        log_path
    )

    # --------------------------------------------------------
    # Step 1 — Validate subsection exists in dictionary
    # --------------------------------------------------------
    if subsection not in subsection_dict:
        raise ValueError(f"Subsection '{subsection}' not found in subsection_dict.")

    # --------------------------------------------------------
    # Step 2 — Validate subsection belongs to requested section
    # --------------------------------------------------------
    subsection_features = [
        c for c in get_subsection(subsection, dataframe, subsection_dict)
        if not c.endswith(suffix)
    ]

    if len(subsection_features) == 0:
        raise ValueError(
            f"No dataframe columns found for subsection '{subsection}'."
        )

    inferred_section = get_column_label(subsection_features[0],subsection_dict=subsection_dict, section_dict=section_dict)["section"]

    if inferred_section != section:
        raise ValueError(
            f"Subsection '{subsection}' belongs to section '{inferred_section}', "
            f"not '{section}'."
        )

    target_features = subsection_features

    write_log(
        f"Features to convert ({len(target_features)}): "
        + ", ".join(target_features),
        log_path
    )

    # --------------------------------------------------------
    # Step 3 — Convert each feature into a new *_scalefix column
    # --------------------------------------------------------
    converted_features = []

    for feature in target_features:

        new_feature = f"{feature}{suffix}"

        # Copy original values into new feature first
        dataframe[new_feature] = dataframe[feature]

        # Convert scale in-place on the copied feature
        convert_feature_scale(
            feature_name=new_feature,
            current_scale=current_scale,
            new_scale=new_scale,
            dataframe=dataframe,
            log_path=log_path
        )

        # Inherit dictionary labels from source feature
        source_label = get_column_label(feature, subsection_dict=subsection_dict, section_dict=section_dict)

        update_column_label(
            column_name=new_feature,
            subsection=source_label["subsection"],
            section=source_label["section"],
            subsection_dict=subsection_dict,
            section_dict=section_dict,
            log_path=log_path,
            subsection_column_names_scales=subsection_column_names_scales
        )

        converted_features.append(new_feature)

    # --------------------------------------------------------
    # Step 4 — Final logging
    # --------------------------------------------------------
    write_log(
        f"Bulk scale conversion complete | created {len(converted_features)} converted features.", log_path)

    return converted_features

# Function: Create binary categorization from score cutoff
def create_score_categorization(
    dataframe,
    score_col,
    cutoff,
    log_path,
    output_dir,
    comparison="ge",
    plot=False,
    output_col=None
):

    # Validate score column exists
    if score_col not in dataframe.columns:

        raise ValueError(
            f"Column not found: {score_col}"
        )

    # Validate comparison mode
    valid_comparisons = ["ge", "le"]

    if comparison not in valid_comparisons:

        raise ValueError(
            f"comparison must be one of: "
            f"{valid_comparisons}"
        )

    # Create output column name
    # Use default if not explicitly provided
    if output_col is None:

        output_col = (
            f"{score_col}_categorization"
        )

    # Convert score column to numeric
    # Invalid values become NaN and remain missing
    score_series = pd.to_numeric(
        dataframe[score_col],
        errors="coerce"
    )

    # Initialize output column as nullable integer
    # IMPORTANT: Keeps missing values as <NA> instead of converting to 0/1
    dataframe[output_col] = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="Int64"
    )

    # Identify non-missing values
    valid_mask = score_series.notna()

    # Apply cutoff logic
    if comparison == "ge":

        dataframe.loc[
            valid_mask,
            output_col
        ] = (
            score_series[valid_mask] >= cutoff
        ).astype("Int64")

        comparison_text = ">="

    elif comparison == "le":

        dataframe.loc[
            valid_mask,
            output_col
        ] = (
            score_series[valid_mask] <= cutoff
        ).astype("Int64")

        comparison_text = "<="

    # Logging
    write_log(
        f"Created binary categorization: "
        f"{output_col}",
        log_path
    )

    write_log(
        f"Comparison rule: "
        f"{score_col} {comparison_text} {cutoff}", 
        log_path
    )

    n_positive = (
        dataframe[output_col] == 1
    ).sum()

    n_negative = (
        dataframe[output_col] == 0
    ).sum()

    n_missing = (
        dataframe[output_col].isna()
    ).sum()

    write_log(f"Positive cases: {n_positive}", log_path)
    write_log(f"Negative cases: {n_negative}", log_path)
    write_log(f"Missing cases: {n_missing}", log_path)

    # OPTIONAL PLOTS
    if plot:

        write_log(
            f"Creating categorization plots for "
            f"{score_col}", 
            log_path
        )

        # Create plotting dataframe
        plot_df = pd.DataFrame({
            "score": score_series,
            "category": dataframe[output_col]
        }).dropna(subset=["score"])

        # FIGURE 1 — Score distribution + cutoff
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            plot_df["score"],
            bins=30,
            kde=True,
            stat="density",
            ax=ax
        )

        # Add cutoff line
        ax.axvline(
            cutoff,
            linestyle=":",
            linewidth=2,
            color="red"
        )

        # Add summary statistics
        stats_text = (
            f"N = {len(plot_df)}\n"
            f"Mean = {plot_df['score'].mean():.2f}\n"
            f"Median = {plot_df['score'].median():.2f}\n"
            f"SD = {plot_df['score'].std():.2f}\n"
            f"Cutoff = {comparison_text} {cutoff}"
        )

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.85
            )
        )

        # Labels
        ax.set_title(
            f"Distribution of {score_col}"
        )

        ax.set_xlabel(score_col)
        ax.set_ylabel("Density")

        save_plot(
            fig,
            f"{output_col}_distribution_cutoff", 
            output_dir=output_dir, 
            log_path=log_path
        )

        plt.show()
        plt.close()

        # FIGURE 2 — Category distribution
        category_counts = (
            plot_df["category"]
            .value_counts()
            .sort_index()
        )

        category_percent = (
            category_counts
            / category_counts.sum()
        ) * 100

        fig, ax = plt.subplots(figsize=(6, 5))

        bars = ax.bar(
            category_percent.index.astype(str),
            category_percent.values
        )

        # Add % and N above bars
        for bar, pct, n in zip(
            bars,
            category_percent.values,
            category_counts.values
        ):

            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{pct:.1f}%, N={n}",
                ha="center",
                va="bottom"
            )

        # Labels
        ax.set_title(
            f"{output_col} distribution"
        )

        ax.set_xlabel(output_col)

        ax.set_ylabel(
            "Participants (%)"
        )

        plt.xticks(rotation=0)

        save_plot(
            fig,
            f"{output_col}_distribution", 
            output_dir=output_dir, 
            log_path=log_path
        )

        plt.show()
        plt.close()

        write_log(
            f"Plots completed for {score_col}",
            log_path
        )

    return dataframe


# Function: Audit and optionally remove features.
# Returns modified COPY of df. Original df remains untouched unless reassigned by user.
def audit_and_remove_features(
    dataframe,
    columns,
    removal_reason,
    log_path,
    removed_feature_values_df,
    removal_log_df,
    feature_removal_audit_path,
    override_main_df=False,
    removal_stage="feature_engineering"
):


    # ========================================================
    # CREATE WORKING COPY
    # IMPORTANT:
    # Original dataframe remains untouched
    # unless user explicitly reassigns output
    # ========================================================
    working_df = dataframe.copy()

    # STANDARDIZE INPUT
   
    # Standardize columns input
    # Supports:
    # - single string
    # - flat list
    # - nested lists
    
    # Single column
    if isinstance(columns, str):    
        columns = [columns]
        
    # Flatten nested lists
    # Example: [["a", "b"], ["c"]] -> ["a", "b", "c"]
    flattened_columns = []
    
    for item in columns:    
        if isinstance(item, (list, tuple, set)):    
            flattened_columns.extend(item)    
        else:    
            flattened_columns.append(item)
    
    # Remove duplicates while preserving order
    columns = list(dict.fromkeys(flattened_columns))

    # VALIDATE COLUMNS
    existing_cols = [
        col for col in columns
        if col in working_df.columns
    ]

    missing_cols = [
        col for col in columns
        if col not in working_df.columns
    ]

    # Log missing columns
    if missing_cols:
        write_log("Columns not found in dataframe:", log_path)
        write_log(", ".join(missing_cols), log_path)

    # Stop if nothing valid remains
    if len(existing_cols) == 0:
        write_log("No valid columns available for audit.", log_path)
        return (working_df, removed_feature_values_df, removal_log_df)

    # CREATE VALUES AUDIT TABLE
    # Validate record_id exists
    if "record_id" not in working_df.columns:
        raise ValueError("record_id column not found in dataframe.")

    # Keep record_id + removed columns
    values_df = working_df[["record_id"] + existing_cols].copy()

    # Append to global audit dataframe
    removed_feature_values_df = pd.concat([removed_feature_values_df,values_df],axis=1
    )

    # Remove duplicate record_id columns
    removed_feature_values_df = (removed_feature_values_df.loc[:,~removed_feature_values_df.columns.duplicated()])

    # CREATE REMOVAL LOG TABLE
    log_rows = []

    for col in existing_cols:

        log_rows.append({
            "feature_name": col,
            "removal_reason": removal_reason,
            "removal_stage": removal_stage,
            "removed_from_returned_df": override_main_df,
            "dtype": str(working_df[col].dtype),
            "n_non_missing": (
                working_df[col]
                .notna()
                .sum()
            ),
            "timestamp": pd.Timestamp.now()
        })

    log_df = pd.DataFrame(log_rows)

    # Append to global removal log
    removal_log_df = pd.concat(
        [removal_log_df,log_df],
        ignore_index=True)

    # OPTIONAL FEATURE REMOVAL
    # IMPORTANT: Removal occurs ONLY on working_df

    if override_main_df:
        write_log("Removing features from returned dataframe copy.", log_path)
        working_df = working_df.drop(columns=existing_cols)

    else:
        write_log("Features retained in returned dataframe copy.", log_path)

    # EXPORT AUDIT WORKBOOK
    write_log("Exporting feature removal audit workbook.", log_path)

    with pd.ExcelWriter(
        feature_removal_audit_path,
        engine="openpyxl"
    ) as writer:

        removed_feature_values_df.to_excel(
            writer,
            sheet_name="removed_feature_values",
            index=False)

        removal_log_df.to_excel(
            writer,
            sheet_name="removal_log",
            index=False)

    # LOGGING
    write_log(f"Audited {len(existing_cols)} features.", log_path)
    write_log("Features:", log_path)
    write_log(", ".join(existing_cols), log_path)
    write_log(f"Removal reason: {removal_reason}", log_path)
    write_log(f"Removal stage: {removal_stage}", log_path)
    write_log(f"override_main_df = {override_main_df}", log_path)
    write_log("IMPORTANT:", log_path)
    write_log("Original dataframe was NOT modified in-place.", log_path)
    write_log("Only the returned dataframe copy was altered.", log_path)
    write_log(f"Audit workbook updated:", log_path)

    write_log(str(feature_removal_audit_path), log_path)

    # RETURN MODIFIED COPY
    return (working_df, removed_feature_values_df, removal_log_df)


# ============================================================
# QC parameter functions 
# ============================================================
# Function: compute entropy of a row (for categorical features)
def compute_entropy(row):
    counts = row.dropna().value_counts(normalize=True)

    if len(counts) == 0:
        return np.nan

    return entropy(counts.to_numpy(dtype=float))

# Function: compute longest string length of a row (for categorical features)
def longest_string(row):

    values = row.dropna().values

    if len(values) == 0:
        return np.nan

    max_len = 1
    current_len = 1

    for i in range(1, len(values)):

        if values[i] == values[i-1]:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1

    return max_len

# Function: Analyze score inputs before score construction
def analyze_score_inputs(
    sections_df,
    section,
    subsection,
    dataframe,
    log_path, 
    output_dir,  
    plot=True,
    subgroup_cols=None,
    export_prefix=None
):

    write_log(f"Analyzing score inputs | section='{section}' | subsection='{subsection}'", log_path)

    # Define prefix ONCE
    # Always derive prefix from section + subsection unless explicitly overridden
    prefix = f"{section}_{subsection}" if export_prefix is None else export_prefix

    # Step 1 — Select relevant score components
    score_info = sections_df.loc[
        (sections_df["section"] == section) &
        (sections_df["subsection"] == subsection) &
        (
            sections_df["score_component"]
            .fillna(False)
            .astype(str)
            .str.upper()
            .eq("TRUE")
        )
    ].copy()

    if score_info.empty:
        raise ValueError(
            f"No score_component=TRUE items found for "
            f"section='{section}', subsection='{subsection}'."
        )

    items = score_info["column_name"].tolist()

    write_log(f"Score components identified: {len(items)}", log_path)
    write_log("Components: " + ", ".join(items), log_path)

    # Step 2 — Validate items exist
    missing_items = [col for col in items if col not in dataframe.columns]

    if missing_items:
        raise ValueError(f"Missing columns in df: {missing_items}")

    score_df = dataframe[items].copy()
    numeric_score_df = score_df.apply(pd.to_numeric, errors="coerce")

    # Step 3 — Structural diagnostics
    write_log("Running structural diagnostics.", log_path)

    # --- duplicates ---
    duplicate_mask = numeric_score_df.T.duplicated()
    duplicate_df = pd.DataFrame({
        "item": numeric_score_df.columns,
        "is_duplicate": duplicate_mask.values
    })

    display(duplicate_df)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        duplicate_df["is_duplicate"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Duplicate Items | {section} / {subsection}")
        save_plot(fig, f"{prefix}_duplicates", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

    # --- correlation ---
    corr_matrix = numeric_score_df.corr()
    display(corr_matrix)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_matrix, aspect="auto")

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.columns)

        ax.set_title(f"Correlation Matrix | {section} / {subsection}")
        fig.colorbar(im, ax=ax)

        save_plot(fig, f"{prefix}_correlation_matrix", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

    # --- variance ---
    variance_series = numeric_score_df.var()
    display(variance_series)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        variance_series.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Variance per Item | {section} / {subsection}")
        save_plot(fig, f"{prefix}_variance", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

    # Step 4 — Missingness
    pct_missing_per_item = (numeric_score_df.isna().mean() * 100).sort_values(ascending=False)

    # Step 5 — Participant completeness
    participant_completeness = numeric_score_df.notna().mean(axis=1) * 100

    # Step 6 — Item-total correlations
    total_score = numeric_score_df.sum(axis=1, skipna=True)

    item_total_corr = {}
    for item in items:
        rest_score = total_score - numeric_score_df[item]
        item_total_corr[item] = numeric_score_df[item].corr(rest_score)

    item_total_corr = pd.Series(item_total_corr).sort_values()

    # Step 7 — Cronbach's alpha
    complete_df = numeric_score_df.dropna()

    k = complete_df.shape[1]

    if k <= 1 or complete_df.shape[0] < 2:
        cronbach_alpha = np.nan
    else:
        item_variances = complete_df.var(axis=0, ddof=1)
        total_score = complete_df.sum(axis=1)
        total_variance = total_score.var(ddof=1)

        if total_variance == 0 or pd.isna(total_variance):
            cronbach_alpha = np.nan
        else:
            cronbach_alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    write_log(f"Cronbach's alpha: {cronbach_alpha:.4f}", log_path)

    # Step 8 — Plots (existing diagnostics)
    if plot:

        fig, ax = plt.subplots(figsize=(6, 4))
        pct_missing_per_item.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Missingness | {section} / {subsection}")
        save_plot(fig, f"{prefix}_missingness", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        participant_completeness.plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Completeness | {section} / {subsection}")
        save_plot(fig, f"{prefix}_completeness", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        item_total_corr.plot(kind="barh", ax=ax)
        ax.set_title(f"Item-total correlation | {section} / {subsection}")
        save_plot(fig, f"{prefix}_item_total_corr", output_dir=output_dir, log_path=log_path)
        plt.show()
        plt.close()

    return {
        "items": items,
        "pct_missing_per_item": pct_missing_per_item,
        "participant_completeness": participant_completeness,
        "item_total_correlations": item_total_corr,
        "cronbach_alpha": cronbach_alpha
    }

# Function: Create composite score
def create_composite_score(
    sections_df,
    section,
    subsection,
    dataframe,
    log_path,
    output_dir,
    score_column_name=None,
    score_method="mean",   # "mean" or "sum"
    min_valid_fraction=0.7,
    plot=True
):
    """
    Create a composite score from score_component=TRUE items
    defined in sections_df.

    Parameters
    ----------
    sections_df : pd.DataFrame
        Metadata table containing score definitions.

    section : str
        Section name.

    subsection : str
        Subsection name.

    dataframe : pd.DataFrame
        Source dataframe containing participant responses.

    score_column_name : str or None
        Optional explicit score column name.
        Default: "<subsection>_score"

    score_method : str
        Score aggregation method:
        - "mean"
        - "sum"

    min_valid_fraction : float
        Minimum fraction of valid items required to compute score.

    plot : bool
        Whether to generate and save visualizations.

    Returns
    -------
    pd.DataFrame
        Input dataframe with new score column added.
    """

    write_log(
        f"Creating composite score | "
        f"section='{section}' | subsection='{subsection}'",
        log_path
    )

    # Define output score column name
    score_col = (
        f"{subsection}_score"
        if score_column_name is None
        else score_column_name
    )

    write_log(f"Output score column: {score_col}", log_path)

    # Select score components from sections_df
    score_info = sections_df.loc[
        (sections_df["section"] == section) &
        (sections_df["subsection"] == subsection) &
        (
            sections_df["score_component"]
            .fillna(False)
            .astype(str)
            .str.upper()
            .eq("TRUE")
        )
    ].copy()

    if score_info.empty:
        raise ValueError(
            f"No score_component=TRUE items found for "
            f"section='{section}', subsection='{subsection}'."
        )

    items = score_info["column_name"].tolist()

    write_log(f"Score components identified: {len(items)}", log_path)
    write_log("Components: " + ", ".join(items), log_path)

    # Validate all items exist in dataframe
    missing_items = [col for col in items if col not in dataframe.columns]

    if missing_items:
        raise ValueError(
            f"These score components are missing from df: {missing_items}"
        )

    # Create numeric score dataframe
    score_df = dataframe[items].copy()
    numeric_score_df = score_df.apply(pd.to_numeric, errors="coerce")

    # Compute participant completeness
    valid_counts = numeric_score_df.notna().sum(axis=1)

    min_items_required = int(np.ceil(len(items) * min_valid_fraction))

    write_log(
        f"Minimum valid items required: "
        f"{min_items_required}/{len(items)}",
        log_path
    )

    # Create composite score
    if score_method == "mean":

        composite_score = numeric_score_df.mean(axis=1)

    elif score_method == "sum":

        composite_score = numeric_score_df.sum(axis=1)

    else:
        raise ValueError(
            f"Unsupported score_method='{score_method}'. "
            f"Use 'mean' or 'sum'."
        )

    # Apply completeness threshold
    composite_score.loc[
        valid_counts < min_items_required
    ] = np.nan

    # Add score to dataframe
    dataframe[score_col] = composite_score

    write_log(
        f"Composite score created successfully: '{score_col}'",
        log_path
    )

    write_log(
        f"Participants with valid scores: "
        f"{dataframe[score_col].notna().sum()}",
        log_path
    )

    # Plot diagnostics
    if plot:

        # Score distribution
        fig, ax = plt.subplots(figsize=(7, 5))

        dataframe[score_col].plot(
            kind="hist",
            bins=20,
            ax=ax
        )

        ax.set_title(
            f"Score Distribution | {section} / {subsection}"
        )

        ax.set_xlabel(score_col)
        ax.set_ylabel("Participants")

        save_plot(
            fig,
            f"{section}_{subsection}_score_distribution",
            output_dir=output_dir,
            log_path=log_path
        )

        plt.show()
        plt.close()

        # Mean component scores
        # Force x-axis to use full raw score scale
        component_means = numeric_score_df.mean().sort_values()

        # Infer raw score range directly from observed data
        raw_score_min = numeric_score_df.min().min()
        raw_score_max = numeric_score_df.max().max()

        fig, ax = plt.subplots(figsize=(7, 5))

        component_means.plot(
            kind="barh",
            ax=ax
        )

        # Force axis to span full possible score range
        ax.set_xlim(raw_score_min, raw_score_max)

        ax.set_title(
            f"Mean Component Scores | {section} / {subsection}"
        )

        ax.set_xlabel("Mean score")
        ax.set_ylabel("Component")

        save_plot(
            fig,
            f"{section}_{subsection}_component_means",
            output_dir=output_dir,
            log_path=log_path
        )

        plt.show()
        plt.close()

    # Return updated dataframe
    return dataframe


# ============================================================
# Function: Create *_any and *_total summary features
# from binary indicator columns
# ============================================================
def create_binary_summary_features(
    dataframe,
    columns,
    log_path,
    output_dir,
    plot=False
):

    # Validate input columns
    missing_cols = [
        col for col in columns
        if col not in dataframe.columns
    ]

    if missing_cols:

        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    # Infer shared prefix
    # Example: psych_hx___1 -> psych_hx
    prefix = columns[0].split("___")[0]

    # Convert selected columns to numeric
    binary_df = dataframe[
        columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Create *_total
    # Counts number of positive indicators
    total_col = f"{prefix}_total"

    dataframe[total_col] = (
        binary_df
        .fillna(0)
        .sum(axis=1)
        .astype("Int64")
    )

    # Create *_any
    # 1 if any indicator is positive, 0 if all are negative, <NA> if all missing
    any_col = f"{prefix}_any"

    dataframe[any_col] = (
        dataframe[total_col] > 0
    ).astype("Int64")

    # Logging
    write_log(
        f"Created summary features:",
        log_path
    )

    write_log(f"- {any_col}", log_path)
    write_log(f"- {total_col}", log_path)

    n_positive = (
        dataframe[any_col] == 1
    ).sum()

    n_negative = (
        dataframe[any_col] == 0
    ).sum()

    write_log(
        f"{any_col} positive participants: "
        f"{n_positive}",
        log_path
    )

    write_log(
        f"{any_col} negative participants: "
        f"{n_negative}",
        log_path
    )

    write_log(
        f"Mean {total_col}: "
        f"{dataframe[total_col].mean():.2f}",
        log_path
    )

    # OPTIONAL VISUALIZATIONS
    if plot:

        write_log(
            f"Creating plots for "
            f"{prefix} summary features.",
            log_path
        )

        # FIGURE 1 — Distribution of *_total
        total_plot_df = (
            dataframe[total_col]
            .dropna()
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            total_plot_df,
            bins=range(
                int(total_plot_df.min()),
                int(total_plot_df.max()) + 2
            ),
            discrete=True,
            ax=ax
        )

        # Add summary statistics
        stats_text = (
            f"N = {len(total_plot_df)}\n"
            f"Mean = {total_plot_df.mean():.2f}\n"
            f"Median = {total_plot_df.median():.2f}\n"
            f"SD = {total_plot_df.std():.2f}\n"
            f"Min = {total_plot_df.min()}\n"
            f"Max = {total_plot_df.max()}"
        )

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.85
            )
        )

        # Labels
        ax.set_title(
            f"Distribution of {total_col}"
        )

        ax.set_xlabel(
            "Number of positive indicators"
        )

        ax.set_ylabel(
            "Participants"
        )

        save_plot(
            fig,
            f"{total_col}_distribution",
            output_dir=output_dir,
            log_path=log_path
        )

        plt.show()
        plt.close()

        # FIGURE 2 — Distribution of *_any
        any_counts = (
            dataframe[any_col]
            .value_counts()
            .sort_index()
        )

        any_percent = (
            any_counts
            / any_counts.sum()
        ) * 100

        fig, ax = plt.subplots(figsize=(6, 5))

        bars = ax.bar(
            ["No", "Yes"],
            any_percent.values
        )

        # Add labels above bars
        for bar, pct, n in zip(
            bars,
            any_percent.values,
            any_counts.values
        ):

            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{pct:.1f}%, N={n}",
                ha="center",
                va="bottom"
            )

        # Labels
        ax.set_title(
            f"Distribution of {any_col}"
        )

        ax.set_xlabel(
            any_col
        )

        ax.set_ylabel(
            "Participants (%)"
        )

        save_plot(
            fig,
            f"{any_col}_distribution",
            output_dir=output_dir,
            log_path=log_path
        )

        plt.show()
        plt.close()

        write_log(
            f"Plots completed for {prefix}",
            log_path
        )

    return dataframe


