## Penguins Mini-Project

#%%
# Imports

import os
import pandas as pd
import matplotlib.pyplot as plt

#%%

# Set file paths

RAW_DATA_PATH = "data/raw/penguins_size.csv"
CLEAN_DATA_PATH = "data/processed/penguins_clean.csv"
FIGURES_DIR = "figures"
OUTPUTS_DIR = "outputs"

os.makedirs("data/processed", exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

#%% 
# Check working directory and files

print("Current working directory:")
print(os.getcwd())

print("\nFiles/folders here:")
print(os.listdir())

print("\nFiles in data/raw:")
print(os.listdir("data/raw"))

#%%
# Load raw data

df = pd.read_csv(RAW_DATA_PATH)

# %%
# Initial inspection before cleaning
print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values recognized by pandas:")
print(df.isna().sum())

#%%
# Check for any placeholder missing values

missing_tokens = [".", "", "NA", "N/A", "na", "n/a", "None", "null", "NULL", "?"]

token_check = df.astype("string").apply(
    lambda col: col.str.strip().isin(missing_tokens)
)

print("Possible placeholder missing values by column:")
print(token_check.sum())

print("\nRows containing possible placeholder missing values:")
print(df[token_check.any(axis=1)])

#%%
# Initial cleaning 

# Clean placeholder missing values

df = df.mask(token_check, pd.NA)

print("Missing values after replacing placeholder values:")
print(df.isna().sum())

#%%
# Keep the columns needed for research question
# keeping sex introduces more missing values 
# And interpretation issues
# I did not to include it to keep this in scope

model_columns = [
    "species",
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

df_clean = df.dropna(subset=model_columns).copy()

print("Original dataset shape:")
print(df.shape)

print("\nCleaned dataset shape:")
print(df_clean.shape)

print("\nMissing values in modeling columns after cleaning:")
print(df_clean[model_columns].isna().sum())


#%%
# Save cleaned dataset

df_clean.to_csv(CLEAN_DATA_PATH, index=False)

print("Cleaned dataset saved to:")
print(CLEAN_DATA_PATH)

## EDA
#%%
# Species counts

species_counts = df_clean["species"].value_counts()
species_percentages = df_clean["species"].value_counts(normalize=True) * 100

print("Species counts:")
print(species_counts)

print("\nSpecies percentages:")
print(species_percentages.round(2))

#%%
# Don't use, include the next graph
# # Bar chart of species counts

species_counts.plot(kind="bar")

plt.title("Number of Observations by Penguin Species")
plt.xlabel("Species")
plt.ylabel("Number of Observations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Species count and percentage summary

species_summary = pd.DataFrame({
    "count": species_counts,
    "percentage": species_percentages.round(2)
})

print(species_summary)

#%%
# Bar chart of species percentages

species_summary["percentage"].plot(kind="bar")

plt.title("Percentage of Observations by Penguin Species")
plt.xlabel("Species")
plt.ylabel("Percentage of Observations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
# Species by island

species_by_island = pd.crosstab(df_clean["island"], df_clean["species"])

print("Species counts by island:")
print(species_by_island)

# %%
# Species by island as percentages within each island
# on what island, what % belong to each species 

species_by_island_pct = pd.crosstab(
    df_clean["island"],
    df_clean["species"],
    normalize="index"
) * 100

print("Species percentages within each island:")
print(species_by_island_pct.round(2))

#%%
# Bar chart species counts by island
# Not for report, transparency

species_by_island.plot(kind="bar")

plt.title("Penguin Species Counts by Island")
plt.xlabel("Island")
plt.ylabel("Number of Observations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
# Stacked bar chart species %'s by island

species_by_island_pct.plot(kind="bar", stacked=True)

plt.title("Species Proportions by Island")
plt.xlabel("Island")
plt.ylabel("Percentage of Observations")
plt.xticks(rotation=0)
plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#%%
# Summary Stats: measurements by species

measurement_columns = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

species_measurement_summary = df_clean.groupby("species")[measurement_columns].mean().round(2)

print("Average measurements by species:")
print(species_measurement_summary)

#%%
# Detailed summary stats

species_measurement_detail = df_clean.groupby("species")[measurement_columns].agg(
    ["mean", "median", "std", "min", "max"]
).round(2)

print("Detailed measurement summary by species:")
print(species_measurement_detail)

#%%
# Boxplot: measurements by species

measurement_labels = {
    "culmen_length_mm": "Culmen Length (mm)",
    "culmen_depth_mm": "Culmen Depth (mm)",
    "flipper_length_mm": "Flipper Length (mm)",
    "body_mass_g": "Body Mass (g)"
}

for col, label in measurement_labels.items():
    fig, ax = plt.subplots(figsize=(7, 5))

    df_clean.boxplot(
        column=col,
        by="species",
        ax=ax
    )

    ax.set_title(f"{label} by Species")
    ax.set_xlabel("Species")
    ax.set_ylabel(label)

    plt.suptitle("")
    plt.tight_layout()
    plt.show() 

#%%
# Scatterplot: culmen length vs. culmen depth by species

fig, ax = plt.subplots(figsize=(8, 6))

for species, group in df_clean.groupby("species"):
    ax.scatter(
        group["culmen_length_mm"],
        group["culmen_depth_mm"],
        label=species,
        alpha=0.75
    )

ax.set_title("Culmen Length vs. Culmen Depth by Species")
ax.set_xlabel("Culmen Length (mm)")
ax.set_ylabel("Culmen Depth (mm)")
ax.legend(title="Species")
plt.tight_layout()
plt.show()

# %%
# Scatterplot: flipper length vs. body mass 

fig, ax = plt.subplots(figsize=(8, 6))

for species, group in df_clean.groupby("species"):
    ax.scatter(
        group["flipper_length_mm"],
        group["body_mass_g"],
        label=species,
        alpha=0.75
    )

ax.set_title("Flipper Length vs. Body Mass by Species")
ax.set_xlabel("Flipper Length (mm)")
ax.set_ylabel("Body Mass (g)")
ax.legend(title="Species")
plt.tight_layout()
plt.show()

# %%
# Correlation matrix: numeric measurements

numeric_columns = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

correlation_matrix = df_clean[numeric_columns].corr().round(2)

print("Correlation matrix for numeric measurements:")
print(correlation_matrix)

# %%
# Matrix heatmap

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(correlation_matrix)

ax.set_xticks(range(len(numeric_columns)))
ax.set_yticks(range(len(numeric_columns)))
ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
ax.set_yticklabels(numeric_columns)

for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        ax.text(j, i, correlation_matrix.iloc[i, j], ha="center", va="center")

ax.set_title("Correlation Matrix of Penguin Measurements")
plt.colorbar(im)
plt.tight_layout()
plt.show()

# %%
# Target and predictor variables
# x = the variables used to predict species
# y = species labels I'm trying to predict

target = "species"

measurement_features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

features_with_island = measurement_features + ["island"]

X = df_clean[features_with_island]
y = df_clean[target]

print("Feature columns:")
print(X.columns)

print("\nTarget variable:")
print(y.name)

print("\nFeature dataset shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

# %%
# Split data into training and test sets
# 80% of data for training, 20% held for testing
# split is reproducible
# stratify is used to keep species proportions similiar 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training feature shape:")
print(X_train.shape)

print("\nTest feature shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(2))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).round(2))

# %%
# Multinomial logistic regression: measurements only

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model 1 only using measurements 
X_train_measurements = X_train[measurement_features]
X_test_measurements = X_test[measurement_features]

# Modeling pipeline
model_measurements = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000))
])

# Fit model with training data
model_measurements.fit(X_train_measurements, y_train)

# Make predictions on test data
y_pred_measurements = model_measurements.predict(X_test_measurements)

# Evaluate model
accuracy_measurements = accuracy_score(y_test, y_pred_measurements)

print("Model 1: Measurements Only")
print("Accuracy:")
print(round(accuracy_measurements, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_measurements))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_measurements))
# Results are surprising, with model accuracy of 1.0
# All 3 species were classified with 100% using 69 test obsv
# Verify there is no leakage or mistake in setup

# %%
# Check feature columns

print("Measurement features used in Model 1:")
print(X_train_measurements.columns)

print("\nFirst five rows of training features:")
print(X_train_measurements.head())

#%%
# actual vs predicted species

prediction_check = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred_measurements
})

print(prediction_check.head(20))

print("\nNumber of incorrect predictions:")
print((prediction_check["actual"] != prediction_check["predicted"]).sum())

# %%
# Cross-validation check: Model 1

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model_measurements,
    df_clean[measurement_features],
    y,
    cv=cv,
    scoring="accuracy"
)

print("Cross-validation accuracy scores:")
print(cv_scores.round(4))

print("\nMean cross-validation accuracy:")
print(round(cv_scores.mean(), 4))

print("\nStandard deviation:")
print(round(cv_scores.std(), 4))

# Perfect test result was not due to one lucky split
# Model performs very well across multiple splits
# Has very little variation 

# %%
# Model 2: measurements + island

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Feature groups
numeric_features = measurement_features
categorical_features = ["island"]

# Preprocessing:
# scale numeric features 
# one-hot encode island
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

# Model pipeline
model_with_island = Pipeline([
    ("preprocessor", preprocessor),
    ("log_reg", LogisticRegression(max_iter=1000))
])

# Fit model with training data
model_with_island.fit(X_train, y_train)

# Predict on test data
y_pred_with_island = model_with_island.predict(X_test)

# Evaluate model
accuracy_with_island = accuracy_score(y_test, y_pred_with_island)

print("Model 2: Measurements + Island")
print("Accuracy:")
print(round(accuracy_with_island, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_with_island))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_with_island))

# %%
# Cross-validation: Model 1 vs Model 2

# Build cross-validation check for Model 2 
# Reuse cross-validation check from Model 1

cv_scores_with_island = cross_val_score(
    model_with_island,
    df_clean[features_with_island],
    y,
    cv=cv,
    scoring="accuracy"
)

# Compare

print("Model 1: Measurements Only")
print("CV scores:", cv_scores.round(4))
print("Mean CV accuracy:", round(cv_scores.mean(), 4))
print("CV standard deviation:", round(cv_scores.std(), 4))

print("\nModel 2: Measurements + Island")
print("CV scores:", cv_scores_with_island.round(4))
print("Mean CV accuracy:", round(cv_scores_with_island.mean(), 4))
print("CV standard deviation:", round(cv_scores_with_island.std(), 4))

# %%
# Summary model comparison table
model_comparison = pd.DataFrame({
    "Model": [
        "Model 1: Measurements Only",
        "Model 2: Measurements + Island"
    ],
    "Mean CV Accuracy": [
        cv_scores.mean(),
        cv_scores_with_island.mean()
    ],
    "CV Standard Deviation": [
        cv_scores.std(),
        cv_scores_with_island.std()
    ]
})

model_comparison["Mean CV Accuracy"] = model_comparison["Mean CV Accuracy"].round(4)
model_comparison["CV Standard Deviation"] = model_comparison["CV Standard Deviation"].round(4)

print(model_comparison)

# %%
# Export model summary comparison

html_path = os.path.join(OUTPUTS_DIR, "model_comparison_summary.html")

html_table = model_comparison.to_html(index=False, border=0)

html_page = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Comparison Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 80%;
        }}
        th, td {{
            border: 1px solid #cccccc;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        caption {{
            caption-side: top;
            font-weight: bold;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h2>Model Comparison Summary</h2>
    {html_table}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as file:
    file.write(html_page)

print("HTML table saved to:")
print(html_path)
# %%
