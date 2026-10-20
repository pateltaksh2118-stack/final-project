"""
Titanic Survival Analysis  (Exploratory Data Analysis)
======================================================
Final Project - Python Data Analysis

Goal: understand which factors influenced passenger survival on the Titanic
and present the findings as clear visualisations.

Libraries: Pandas (cleaning / manipulation), NumPy, Matplotlib & Seaborn (plots)

Usage
-----
    python titanic_analysis.py                    # uses data/titanic.csv, or downloads it
    python titanic_analysis.py --data train.csv   # use your own Kaggle train.csv
    python titanic_analysis.py --show             # also open plot windows
"""
import argparse
import os
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DATA_URLS = [
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
]

SURVIVAL_PALETTE = {"Died": "#c0504d", "Survived": "#2a7f62"}


def download_dataset(dest):
    """Try each public URL until one works. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for url in DATA_URLS:
        try:
            print(f"Downloading dataset from {url} ...")
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 1000:
                return True
        except Exception as exc:  
            print(f"  failed: {exc}")
    return False


def load_data(path=None):
    """Load the Titanic CSV (downloading it if needed) and return a DataFrame."""
    path = path or DEFAULT_DATA
    if not os.path.isfile(path):
        if path != DEFAULT_DATA or not download_dataset(path):
            sys.exit(
                f"\nCould not find or download the dataset ({path}).\n"
                "Download 'train.csv' from https://www.kaggle.com/c/titanic/data,\n"
                "save it as data/titanic.csv (or pass --data <file>) and run again."
            )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {path}")
    return df


def standardize_columns(df):
    """Support both the Kaggle schema and the seaborn schema."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sex": "sex", "pclass": "pclass", "sibsp": "sibsp"})
    required = ["survived", "pclass", "sex", "age", "fare"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing required column(s): {', '.join(missing)}")
    return df


def clean_data(df):
    """Handle missing values, fix types and create analysis features."""
    df = standardize_columns(df)
    report = {"rows_before": len(df), "missing_before": df.isna().sum()[lambda s: s > 0].to_dict()}

    df = df.drop_duplicates()
    df["sex"] = df["sex"].str.lower().str.strip()

    df["age_was_missing"] = df["age"].isna()
    df["age"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df.groupby("pclass")["fare"].transform("median"))

    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
        df["embarked"] = df["embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}) \
            .fillna(df["embarked"])
        
    for cabin_col in ("cabin", "deck"):
        if cabin_col in df.columns:
            df["has_cabin"] = df[cabin_col].notna()
            break

    df["survival"] = df["survived"].map({0: "Died", 1: "Survived"})
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 12, 18, 35, 60, 120],
        labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle-aged (36-60)", "Senior (60+)"],
    )
    if {"sibsp", "parch"} <= set(df.columns):
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        df["is_alone"] = df["family_size"] == 1
        df["family_group"] = pd.cut(df["family_size"], bins=[0, 1, 4, 20],
                                    labels=["Alone", "Small (2-4)", "Large (5+)"])
    df["fare_band"] = pd.qcut(df["fare"], 4, labels=["Low", "Medium", "High", "Very high"], duplicates="drop")
    if "name" in df.columns:
        title = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        df["title"] = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        common = ["Mr", "Mrs", "Miss", "Master"]
        df["title"] = df["title"].where(df["title"].isin(common), "Other")

    report["rows_after"] = len(df)
    return df.reset_index(drop=True), report


def rate(df, by):
    """Survival rate (%) and passenger count for each value of `by`."""
    g = df.groupby(by, observed=True)["survived"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(1)
    return g.rename(columns={"mean": "survival_rate_%", "count": "passengers"})


def build_summary(df, report):
    """Return the text report (also saved to output/summary.txt)."""
    lines = []
    add = lines.append
    overall = df["survived"].mean() * 100
    add("=" * 60)
    add("TITANIC SURVIVAL ANALYSIS - SUMMARY")
    add("=" * 60)
    add(f"Passengers analysed : {len(df)}")
    add(f"Overall survival    : {overall:.1f}%  ({int(df['survived'].sum())} survived, "
        f"{int((1 - df['survived']).sum())} died)")
    add(f"Missing values fixed: {report['missing_before']}")

    sections = [("Passenger class", "pclass"), ("Gender", "sex"), ("Age group", "age_group")]
    if "family_group" in df.columns:
        sections.append(("Family size", "family_group"))
    if "embarked" in df.columns:
        sections.append(("Port of embarkation", "embarked"))
    sections.append(("Fare band", "fare_band"))
    if "title" in df.columns:
        sections.append(("Title", "title"))
    for label, col in sections:
        add(f"\nSurvival by {label}:")
        add(rate(df, col).to_string())

    add("\nSurvival by Class x Gender (%):")
    add((df.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean") * 100).round(1).to_string())

    add("\nAverage age / fare (survivors vs. non-survivors):")
    add(df.groupby("survival")[["age", "fare"]].mean().round(2).to_string())

    num_cols = [c for c in ("pclass", "age", "fare", "family_size") if c in df.columns]
    corr = {c: float(np.corrcoef(df[c].to_numpy(float), df["survived"].to_numpy(float))[0, 1]) for c in num_cols}
    add("\nCorrelation with survival (Pearson r):")
    for c, v in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        add(f"  {c:<12} {v:+.3f}")

    sx = rate(df, "sex")["survival_rate_%"]
    pc = rate(df, "pclass")["survival_rate_%"]
    ag = rate(df, "age_group")["survival_rate_%"]
    add("\nKEY FINDINGS")
    if {"female", "male"} <= set(sx.index):
        add(f"  - Women survived at {sx['female']:.0f}% vs. men at {sx['male']:.0f}%.")
    add(f"  - 1st class survival was {pc.get(1, float('nan')):.0f}% vs. {pc.get(3, float('nan')):.0f}% in 3rd class.")
    add(f"  - Best-surviving age group: {ag.idxmax()} ({ag.max():.0f}%); "
        f"worst: {ag.idxmin()} ({ag.min():.0f}%).")
    add("=" * 60)
    return "\n".join(lines)


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)


def _label_bars(ax, fmt="{:.0f}%"):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) for v in c.datavalues], padding=2, fontsize=9)


def plot_overview(df):
    """Survival count + survival rate by class and gender."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    counts = df["survival"].value_counts().reindex(["Died", "Survived"])
    axes[0].bar(counts.index, counts.values, color=[SURVIVAL_PALETTE[k] for k in counts.index])
    axes[0].bar_label(axes[0].containers[0], padding=2)
    axes[0].set(title="Overall outcome", ylabel="Passengers")

    for ax, col, title in ((axes[1], "pclass", "Survival rate by class"),
                           (axes[2], "sex", "Survival rate by gender")):
        r = rate(df, col).reset_index()
        sns.barplot(data=r, x=col, y="survival_rate_%", color="#2a6f97", ax=ax)
        _label_bars(ax)
        ax.set(title=title, xlabel=col.capitalize(), ylabel="Survival rate (%)", ylim=(0, 100))
    _save(fig, "01_overview.png")
    return fig


def plot_class_gender(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    r = (df.groupby(["pclass", "sex"])["survived"].mean() * 100).reset_index()
    sns.barplot(data=r, x="pclass", y="survived", hue="sex",
                palette={"female": "#d1495b", "male": "#30638e"}, ax=ax)
    _label_bars(ax)
    ax.set(title="Survival rate by class and gender", xlabel="Passenger class",
           ylabel="Survival rate (%)", ylim=(0, 105))
    ax.legend(title="Gender")
    _save(fig, "02_class_gender.png")
    return fig


def plot_age(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="age", hue="survival", multiple="stack", bins=25,
                 palette=SURVIVAL_PALETTE, ax=axes[0])
    axes[0].set(title="Age distribution by outcome", xlabel="Age", ylabel="Passengers")
    r = rate(df, "age_group").reset_index()
    sns.barplot(data=r, x="age_group", y="survival_rate_%", color="#2a6f97", ax=axes[1])
    _label_bars(axes[1])
    axes[1].set(title="Survival rate by age group", xlabel="", ylabel="Survival rate (%)", ylim=(0, 100))
    axes[1].tick_params(axis="x", rotation=20)
    _save(fig, "03_age.png")
    return fig


def plot_fare(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survival", palette=SURVIVAL_PALETTE,
                showfliers=False, ax=axes[0])
    axes[0].set(title="Fare by class and outcome (outliers hidden)", xlabel="Passenger class", ylabel="Fare")
    r = rate(df, "fare_band").reset_index()
    sns.barplot(data=r, x="fare_band", y="survival_rate_%", color="#2a6f97", ax=axes[1])
    _label_bars(axes[1])
    axes[1].set(title="Survival rate by fare quartile", xlabel="Fare band", ylabel="Survival rate (%)", ylim=(0, 100))
    _save(fig, "04_fare.png")
    return fig


def plot_family_and_port(df):
    panels = []
    if "family_group" in df.columns:
        panels.append(("family_group", "Survival rate by family size", "Family size"))
    if "embarked" in df.columns:
        panels.append(("embarked", "Survival rate by port of embarkation", "Port"))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
    for ax, (col, title, xl) in zip(axes[0], panels):
        r = rate(df, col).reset_index()
        sns.barplot(data=r, x=col, y="survival_rate_%", color="#2a6f97", ax=ax)
        _label_bars(ax)
        ax.set(title=title, xlabel=xl, ylabel="Survival rate (%)", ylim=(0, 100))
    _save(fig, "05_family_port.png")
    return fig


def plot_heatmaps(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cols = [c for c in ("survived", "pclass", "age", "fare", "family_size") if c in df.columns]
    data = df[cols].copy()
    data["is_female"] = (df["sex"] == "female").astype(int)
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=axes[0])
    axes[0].set_title("Correlation matrix")
    pivot = df.pivot_table(index="age_group", columns="pclass", values="survived",
                           aggfunc="mean", observed=True) * 100
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGn", vmin=0, vmax=100, ax=axes[1],
                cbar_kws={"label": "Survival rate (%)"})
    axes[1].set(title="Survival rate (%) by age group and class", xlabel="Passenger class", ylabel="")
    _save(fig, "06_heatmaps.png")
    return fig


def plot_title(df):
    if "title" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    r = rate(df, "title").reset_index()
    sns.barplot(data=r, x="title", y="survival_rate_%", color="#2a6f97", ax=ax)
    _label_bars(ax)
    ax.set(title="Survival rate by title (from passenger name)", xlabel="Title",
           ylabel="Survival rate (%)", ylim=(0, 100))
    _save(fig, "07_title.png")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Titanic survival EDA")
    parser.add_argument("--data", help="path to Titanic CSV (default: data/titanic.csv, auto-download)")
    parser.add_argument("--show", action="store_true", help="open plot windows as well as saving PNGs")
    args = parser.parse_args()

    if args.show:
        plt.switch_backend("TkAgg")
    sns.set_theme(style="whitegrid", context="notebook")

    df, report = clean_data(load_data(args.data))
    summary = build_summary(df, report)
    print("\n" + summary)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    df.to_csv(os.path.join(OUTPUT_DIR, "titanic_cleaned.csv"), index=False)

    for plot in (plot_overview, plot_class_gender, plot_age, plot_fare,
                 plot_family_and_port, plot_heatmaps, plot_title):
        plot(df)
    print(f"\nCharts, summary.txt and titanic_cleaned.csv saved in: {OUTPUT_DIR}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
