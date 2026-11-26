# Source: notebook cell 41 — Advanced ensemble training
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 16 — ADVANCED MACHINE LEARNING & ENSEMBLE
# NBA DATA SCIENCE ANALYTICS
# MODEL VERSION 3.0
# ============================================================

import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from IPython.display import display

# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

if "BASE_DIR" not in globals():
    raise RuntimeError("Esegui prima STEP 0.")

DATA_PATH = os.path.join(
    BASE_DIR,
    "NBA_Advanced_ML_Features.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "NBA_Advanced_Feature_List.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "nba_player_models_v3.joblib"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "NBA_Advanced_Model_Accuracy.xlsx"
)

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "NBA_Advanced_Test_Predictions.csv"
)

RANDOM_STATE = 42

TARGETS = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "3PM": "fg3m",
    "PRA": "PRA"
}

for path in [DATA_PATH, FEATURE_PATH]:

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

START_TIME = time.time()

# ============================================================
# 2. CARICAMENTO DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    dtype={
        "game_id": str,
        "player_id": str,
        "team_id": str,
        "opponent_id": str
    }
)

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)

feature_cols = config["features"]

df["game_date"] = pd.to_datetime(
    df["game_date"],
    utc=True,
    errors="coerce",
    format="mixed"
)

required = (
    feature_cols
    + list(TARGETS.values())
    + ["game_date", "game_id", "player_id"]
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Colonne mancanti: {missing}"
    )

for c in feature_cols + list(TARGETS.values()):

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "game_date",
        "game_id",
        "player_id",
        *TARGETS.values()
    ]
).drop_duplicates(
    ["game_id", "player_id"]
).sort_values(
    "game_date"
).reset_index(drop=True)

print("Dataset:", len(df))
print("Feature iniziali:", len(feature_cols))
print("Giocatori:", df["player_id"].nunique())

# ============================================================
# 3. TRAIN / VALIDATION / TEST
# SUDDIVISIONE CRONOLOGICA PER GIORNATA NBA
# ============================================================

df["NBA_Day"] = (
    df["game_date"]
    .dt.tz_convert("America/New_York")
    .dt.strftime("%Y-%m-%d")
)

days = sorted(
    df["NBA_Day"].unique()
)

if len(days) < 30:
    raise ValueError(
        "Storico insufficiente per il backtest."
    )

train_end = int(len(days) * 0.70)
val_end = int(len(days) * 0.85)

train_days = days[:train_end]

val_days = days[
    train_end:val_end
]

test_days = days[val_end:]

train = df[
    df["NBA_Day"].isin(train_days)
].copy()

val = df[
    df["NBA_Day"].isin(val_days)
].copy()

test = df[
    df["NBA_Day"].isin(test_days)
].copy()

if min(len(train), len(val), len(test)) == 0:
    raise ValueError("Split temporale non valido.")

# Rimuoviamo feature completamente vuote
# utilizzando esclusivamente il training set.

feature_cols = [
    c for c in feature_cols
    if train[c].notna().any()
]

# Evitiamo che le statistiche target
# siano incluse direttamente tra le feature.

leakage = set(feature_cols).intersection(
    TARGETS.values()
)

if leakage:
    raise ValueError(
        f"Possibile data leakage: {leakage}"
    )

X_train = train[feature_cols]
X_val = val[feature_cols]
X_test = test[feature_cols]

print("\nTEMPORAL SPLIT")

print("Training:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nTraining:", train_days[0], train_days[-1])
print("Validation:", val_days[0], val_days[-1])
print("Test:", test_days[0], test_days[-1])

print("Feature utilizzate:", len(feature_cols))

# ============================================================
# 4. DEFINIZIONE MODELLI
# ============================================================

def build_models():

    return {

        "Ridge": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=8,
                    max_features=0.7,
                    random_state=RANDOM_STATE,
                    n_jobs=2
                )
            )
        ]),

        "Extra Trees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=14,
                    min_samples_leaf=8,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=2
                )
            )
        ]),

        "XGBoost": XGBRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=2
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=7,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=-1
        )
    }

# ============================================================
# 5. METRICHE DI VALUTAZIONE
# ============================================================

def evaluate(y, prediction):

    y = np.asarray(y, dtype=float)

    prediction = np.maximum(
        np.asarray(prediction, dtype=float),
        0
    )

    error = prediction - y

    return {
        "MAE": mean_absolute_error(
            y, prediction
        ),
        "RMSE": np.sqrt(
            mean_squared_error(y, prediction)
        ),
        "Bias": float(error.mean()),
        "Median_Error": float(
            np.median(np.abs(error))
        ),
        "R2": r2_score(y, prediction),
        "Campione": len(y)
    }

# ============================================================
# 6. BASELINE
# ============================================================

def baseline_predict(data, target, fallback):

    col = f"{target}_avg10"

    if col not in data.columns:

        raise ValueError(
            f"Baseline mancante: {col}"
        )

    return np.maximum(
        data[col]
        .fillna(fallback)
        .to_numpy(dtype=float),
        0
    )

# ============================================================
# 7. TRAINING E VALIDATION
# ============================================================

validation_results = []
test_results = []
selected_models = {}
all_predictions = []
importance_rows = []

for target_name, target_col in TARGETS.items():

    print("\n" + "=" * 55)
    print("TRAINING TARGET:", target_name)
    print("=" * 55)

    y_train = train[target_col]
    y_val = val[target_col]
    y_test = test[target_col]

    fallback = float(y_train.mean())

    candidates = build_models()

    fitted_models = {
        "Last 10 Baseline": None
    }

    val_predictions = {}
    val_scores = {}

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_val = baseline_predict(
        val,
        target_col,
        fallback
    )

    val_predictions[
        "Last 10 Baseline"
    ] = baseline_val

    score = evaluate(
        y_val,
        baseline_val
    )

    val_scores[
        "Last 10 Baseline"
    ] = score["MAE"]

    validation_results.append({
        "Target": target_name,
        "Model": "Last 10 Baseline",
        **score
    })

    print(
        "Baseline MAE:",
        round(score["MAE"], 4)
    )

    # --------------------------------------------------------
    # ADDESTRAMENTO MODELLI
    # --------------------------------------------------------

    for name, model in candidates.items():

        model_start = time.time()

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        prediction = np.maximum(
            model.predict(X_val),
            0
        )

        score = evaluate(
            y_val,
            prediction
        )

        fitted_models[name] = model

        val_predictions[name] = prediction

        val_scores[name] = score["MAE"]

        validation_results.append({
            "Target": target_name,
            "Model": name,
            **score
        })

        print(
            f"{name} MAE:",
            round(score["MAE"], 4)
        )

        print(
            "Tempo:",
            round(time.time() - model_start, 1),
            "secondi"
        )

    # ========================================================
    # 8. WEIGHTED ENSEMBLE
    # ========================================================

    ranked = sorted(
        val_scores,
        key=val_scores.get
    )

    top_models = ranked[:3]

    inverse_errors = np.array([
        1 / max(val_scores[name], 1e-8)
        for name in top_models
    ])

    weights = (
        inverse_errors
        / inverse_errors.sum()
    )

    ensemble_val = np.zeros(
        len(val)
    )

    for name, weight in zip(
        top_models,
        weights
    ):

        ensemble_val += (
            weight
            * val_predictions[name]
        )

    ensemble_score = evaluate(
        y_val,
        ensemble_val
    )

    val_scores[
        "Weighted Ensemble"
    ] = ensemble_score["MAE"]

    validation_results.append({
        "Target": target_name,
        "Model": "Weighted Ensemble",
        **ensemble_score
    })

    print(
        "\nEnsemble MAE:",
        round(
            ensemble_score["MAE"],
            4
        )
    )

    # ========================================================
    # 9. SELEZIONE MODELLO
    # SOLO SULLA VALIDATION
    # ========================================================

    winner = min(
        val_scores,
        key=val_scores.get
    )

    print("\nMODELLO SELEZIONATO:", winner)

    # ========================================================
    # 10. REFIT TRAIN + VALIDATION
    # ========================================================

    development = pd.concat(
        [train, val],
        ignore_index=True
    )

    X_dev = development[feature_cols]
    y_dev = development[target_col]

    final_fallback = float(y_dev.mean())

    if winner == "Weighted Ensemble":

        members = top_models

        final_weights = [
            float(w) for w in weights
        ]

    else:

        members = [winner]
        final_weights = [1.0]

    final_models = {}

    test_prediction = np.zeros(
        len(test)
    )

    for name, weight in zip(
        members,
        final_weights
    ):

        if name == "Last 10 Baseline":

            prediction = baseline_predict(
                test,
                target_col,
                final_fallback
            )

            final_models[name] = None

        else:

            model = build_models()[name]

            model.fit(
                X_dev,
                y_dev
            )

            prediction = np.maximum(
                model.predict(X_test),
                0
            )

            final_models[name] = model

        test_prediction += (
            weight * prediction
        )

    # ========================================================
    # 11. TEST FINALE OUT-OF-SAMPLE
    # ========================================================

    test_score = evaluate(
        y_test,
        test_prediction
    )

    baseline_test = baseline_predict(
        test,
        target_col,
        final_fallback
    )

    baseline_score = evaluate(
        y_test,
        baseline_test
    )

    improvement = 100 * (
        baseline_score["MAE"]
        - test_score["MAE"]
    ) / max(
        baseline_score["MAE"],
        1e-8
    )

    test_results.append({
        "Target": target_name,
        "Selected_Model": winner,
        **test_score,
        "Baseline_MAE": baseline_score["MAE"],
        "Improvement_pct": improvement
    })

    selected_models[target_name] = {
        "name": winner,
        "members": members,
        "weights": final_weights,
        "models": final_models,
        "target_column": target_col,
        "baseline_column": f"{target_col}_avg10",
        "fallback": final_fallback
    }

    # ========================================================
    # 12. SALVATAGGIO PREVISIONI TEST
    # ========================================================

    pred_df = test[[
        "game_id",
        "game_date",
        "player_id",
        "player_name",
        "team_name"
    ]].copy()

    pred_df["Target"] = target_name

    pred_df["Actual"] = (
        y_test.to_numpy()
    )

    pred_df["Forecast"] = (
        test_prediction
    )

    pred_df["Baseline"] = (
        baseline_test
    )

    pred_df["Error"] = (
        pred_df["Forecast"]
        - pred_df["Actual"]
    )

    pred_df["Abs_Error"] = (
        pred_df["Error"].abs()
    )

    pred_df["Model"] = winner

    all_predictions.append(pred_df)

    # ========================================================
    # 13. FEATURE IMPORTANCE
    # ========================================================

    for name, model in final_models.items():

        if model is None:
            continue

        estimator = model

        if isinstance(model, Pipeline):

            estimator = model.named_steps[
                "model"
            ]

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importance = (
                estimator.feature_importances_
            )

            total = np.sum(importance)

            if total > 0:
                importance = importance / total

            if len(importance) != len(feature_cols):
                continue

            for feature, value in zip(
                feature_cols,
                importance
            ):

                importance_rows.append({
                    "Target": target_name,
                    "Model": name,
                    "Feature": feature,
                    "Importance": float(value)
                })

    print("\nTEST MAE:", round(test_score["MAE"], 4))

    print(
        "Miglioramento vs baseline:",
        round(improvement, 2),
        "%"
    )

# ============================================================
# 14. CREAZIONE REPORT
# ============================================================

validation_df = pd.DataFrame(
    validation_results
).sort_values(
    ["Target", "MAE"]
)

test_df = pd.DataFrame(
    test_results
)

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

importance_df = pd.DataFrame(
    importance_rows,
    columns=[
        "Target",
        "Model",
        "Feature",
        "Importance"
    ]
)

if not importance_df.empty:

    importance_df = importance_df.sort_values(
        ["Target", "Model", "Importance"],
        ascending=[True, True, False]
    )

# ============================================================
# 15. PERFORMANCE PER GIOCATORE
# ============================================================

player_accuracy = predictions.groupby(
    [
        "player_id",
        "player_name",
        "Target"
    ],
    as_index=False
).agg(
    Partite=("game_id", "nunique"),
    MAE=("Abs_Error", "mean"),
    Bias=("Error", "mean"),
    Actual_Mean=("Actual", "mean"),
    Forecast_Mean=("Forecast", "mean")
)

player_accuracy = player_accuracy[
    player_accuracy["Partite"] >= 5
].sort_values(
    ["Target", "MAE"]
)

# ============================================================
# 16. ENSEMBLE WEIGHTS
# ============================================================

weight_rows = []

for target, info in selected_models.items():

    for name, weight in zip(
        info["members"],
        info["weights"]
    ):

        weight_rows.append({
            "Target": target,
            "Selected_Model": info["name"],
            "Member": name,
            "Weight": weight
        })

weights_df = pd.DataFrame(
    weight_rows
)

# ============================================================
# 17. SALVATAGGIO MODELLO V3
# ============================================================

package_v3 = {
    "model_version": "3.0",
    "models": selected_models,
    "features": feature_cols,
    "targets": TARGETS,
    "minimum_history": config.get(
        "minimum_history", 20
    ),
    "training_cutoff": (
        development["game_date"]
        .max()
        .isoformat()
    ),
    "train_period": [
        train_days[0],
        train_days[-1]
    ],
    "validation_period": [
        val_days[0],
        val_days[-1]
    ],
    "test_period": [
        test_days[0],
        test_days[-1]
    ],
    "validation": (
        validation_df.to_dict("records")
    ),
    "test": test_df.to_dict("records"),
    "created_at": (
        pd.Timestamp.now(tz="UTC")
        .isoformat()
    )
}

joblib.dump(
    package_v3,
    MODEL_PATH + ".tmp",
    compress=3
)

os.replace(
    MODEL_PATH + ".tmp",
    MODEL_PATH
)

# ============================================================
# 18. ESPORTAZIONE CSV ED EXCEL
# ============================================================

predictions["game_date"] = (
    predictions["game_date"].astype(str)
)

predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig"
)

split_report = pd.DataFrame([
    {
        "Dataset": "Training",
        "Rows": len(train),
        "Start": train_days[0],
        "End": train_days[-1]
    },
    {
        "Dataset": "Validation",
        "Rows": len(val),
        "Start": val_days[0],
        "End": val_days[-1]
    },
    {
        "Dataset": "Test",
        "Rows": len(test),
        "Start": test_days[0],
        "End": test_days[-1]
    }
])

with pd.ExcelWriter(
    REPORT_PATH,
    engine="openpyxl"
) as writer:

    sheets = {
        "Validation": validation_df,
        "Test": test_df,
        "Predictions": predictions,
        "Player Accuracy": player_accuracy,
        "Feature Importance": importance_df,
        "Ensemble Weights": weights_df,
        "Temporal Split": split_report
    }

    for name, data in sheets.items():

        data.to_excel(
            writer,
            sheet_name=name,
            index=False
        )

# ============================================================
# 19. VISUALIZZAZIONE
# ============================================================

print("\nVALIDATION RESULTS")

display(
    validation_df.round(4)
)

print("\nFINAL TEST RESULTS")

display(
    test_df.round(4)
)

fig = px.bar(
    validation_df,
    x="Model",
    y="MAE",
    color="Model",
    facet_col="Target",
    facet_col_wrap=2,
    title="NBA V3 — Model Validation",
    template="plotly_dark"
)

fig.update_yaxes(matches=None)

fig.update_layout(
    height=1000,
    showlegend=False
)

fig.show()

# Baseline vs modello selezionato

fig = px.bar(
    test_df,
    x="Target",
    y=["Baseline_MAE", "MAE"],
    barmode="group",
    title="Baseline vs Advanced ML",
    template="plotly_dark"
)

fig.show()

# Miglioramento percentuale

fig = px.bar(
    test_df,
    x="Target",
    y="Improvement_pct",
    color="Improvement_pct",
    title="Model Improvement vs Baseline (%)",
    template="plotly_dark"
)

fig.add_hline(y=0)

fig.show()

# Feature importance per punti

if not importance_df.empty:

    pts_importance = importance_df[
        importance_df["Target"] == "PTS"
    ].copy()

    if not pts_importance.empty:

        pts_importance = (
            pts_importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .groupby("Model")
            .head(15)
        )

        fig = px.bar(
            pts_importance,
            x="Importance",
            y="Feature",
            color="Model",
            orientation="h",
            barmode="group",
            title="PTS — Feature Importance",
            template="plotly_dark"
        )

        fig.update_layout(height=700)

        fig.show()

# ============================================================
# 20. RIEPILOGO FINALE
# ============================================================

elapsed = (
    time.time() - START_TIME
) / 60

print("\n" + "=" * 55)
print("STEP 16 COMPLETATO!")
print("=" * 55)

print("Versione modello: 3.0")
print("Feature:", len(feature_cols))
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))

print(
    "Tempo totale:",
    round(elapsed, 2),
    "minuti"
)

print("\nMODELLI SELEZIONATI")

for target, info in selected_models.items():

    print(
        target,
        "->",
        info["name"]
    )

print("\nFILE SALVATI")

print(MODEL_PATH)
print(REPORT_PATH)
print(PREDICTIONS_PATH)

print("\nPRONTO PER STEP 17!")
