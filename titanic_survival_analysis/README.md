# Titanic Survival Analysis (Exploratory Data Analysis)

**Final Project - Python Data Analysis** (Project option 3)

Exploratory analysis of the Titanic passenger dataset to understand which factors influenced
survival - class, gender, age, fare, family size and port of embarkation - presented through
Matplotlib & Seaborn visualisations.

## Project structure
```
titanic_survival_analysis/
├── titanic_analysis.py    # main script (load -> clean -> analyse -> plot)
├── make_test_data.py      # builds SYNTHETIC data for offline testing only
├── data/                  # titanic.csv goes here (auto-downloaded if missing)
├── output/                # generated charts + summary.txt + titanic_cleaned.csv
├── requirements.txt
└── README.md
```

## Setup & run
```bash
python -m pip install -r requirements.txt
python titanic_analysis.py                    # auto-downloads the dataset on first run
python titanic_analysis.py --data train.csv   # or use a local Kaggle train.csv
python titanic_analysis.py --show             # also open plot windows
```
Windows tip: if `pip` is blocked, use `python -m pip ...`.

## Dataset
Titanic passenger data (891 passengers in the Kaggle `train.csv`), from Kaggle -
https://www.kaggle.com/c/titanic/data. The script also accepts the seaborn copy of the dataset
(lower-case columns), which is what it downloads automatically.

## What the script does
1. **Load** the CSV (download if absent) and validate required columns.
2. **Clean**: drop duplicates; fill missing `Age` with the median of the passenger's class + gender;
   fill missing `Fare`/`Embarked`; reduce `Cabin` (~77% missing) to a `has_cabin` flag.
3. **Feature engineering**: `age_group`, `family_size`, `family_group`, `is_alone`, `fare_band`
   (quartiles) and passenger `title` (extracted from name, when available).
4. **Analyse**: survival rate by every factor, class x gender table, mean age/fare by outcome,
   Pearson correlation with survival (NumPy), and auto-generated key findings.
5. **Visualise** (saved in `output/`):

| File | Shows |
|---|---|
| `01_overview.png` | overall outcome, survival by class and by gender |
| `02_class_gender.png` | survival by class split by gender |
| `03_age.png` | age distribution by outcome, survival by age group |
| `04_fare.png` | fare by class/outcome, survival by fare quartile |
| `05_family_port.png` | survival by family size and port of embarkation |
| `06_heatmaps.png` | correlation matrix and survival rate by age group x class |
| `07_title.png` | survival by title (Kaggle schema only) |

`summary.txt` holds all numeric results and the key findings; the findings are computed from
the data, so they always match the dataset you run on.

## Assumptions
- Missing ages are imputed by class + gender median (ages are strongly related to both).
- `Cabin` is too sparse to impute, so only whether a cabin was recorded is used.
- Passengers 12 or younger are treated as children; age bands: 0-12, 13-18, 19-35, 36-60, 60+.
- Fare bands are quartiles of the cleaned `Fare` column.
- Correlation is not causation: results describe associations in this dataset.

## Libraries
Pandas, NumPy, Matplotlib, Seaborn.
