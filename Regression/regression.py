# ==========================================
# 1. Imports
# ==========================================

import pandas as pd
import numpy as np

import cleaning

from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 2. Model Settings
# ==========================================

RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "min_samples_leaf": 2
}


# ==========================================
# 3. Preprocessing
# ==========================================

def build_preprocessor(X_train):

    # Identify numerical and categorical columns
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Handle missing numerical values
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # Handle missing categorical values and encode them
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Apply the correct preprocessing to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


# ==========================================
# 4. Build Random Forest Model
# ==========================================

def build_model(X_train):

    # Prepare the preprocessing steps
    preprocessor = build_preprocessor(X_train)

    # Combine preprocessing and Random Forest
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**MODEL_PARAMS))
        ]
    )

    return model


# ==========================================
# 5. Train and Evaluate One Experiment
# ==========================================

def train_and_evaluate(
    train_df,
    val_df,
    variant_name,
    experiment_name
):

    # Prepare training data
    X_train, y_train, weights = cleaning.prepare_model_data(
        train_df,
        use_weights=(experiment_name == "one_row_weighted")
    )

    # Prepare validation data
    X_val, y_val, _ = cleaning.prepare_model_data(
        val_df,
        use_weights=False
    )

    # Build the model using training features
    model = build_model(X_train)

    # Train with sample weights only when available
    if weights is not None:

        model.fit(
            X_train,
            y_train,
            regressor__sample_weight=weights
        )

    else:

        model.fit(
            X_train,
            y_train
        )

    # Predict ADR for validation rows
    predictions = model.predict(X_val)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    r2 = r2_score(y_val, predictions)

    # Store experiment results
    result = {
        "Variant": variant_name,
        "Duplicate Experiment": experiment_name,
        "Training Rows": len(X_train),
        "Validation Rows": len(X_val),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return result


# ==========================================
# 6. Run All Experiments
# ==========================================

def run_all_experiments():

    # Load the original dataset
    raw = cleaning.load_raw(cleaning.DATA_PATH)

    # Create groups before splitting
    raw = cleaning.add_group_id(raw)

    # Split data using GroupID
    train_raw, val_raw = cleaning.group_train_val_split(raw)

    # Clean training data and learn preprocessing parameters
    base_train, train_variants, params, train_log = cleaning.build_all(
        train_raw,
        is_train=True
    )

    # Clean validation data using training parameters
    base_val, val_variants, _, val_log = cleaning.build_all(
        val_raw,
        is_train=False,
        params=params
    )

    # Store all experiment results
    results = []

    # Run every variant separately
    for variant_name in train_variants:

        print("\n" + "=" * 60)
        print(f"Processing Variant: {variant_name}")
        print("=" * 60)

        train_df = train_variants[variant_name]

        val_df = val_variants[variant_name]

        # Prepare the three duplicate experiments
        experiments, original_validation = (
            cleaning.prepare_duplicate_experiments(
                train_df,
                val_df
            )
        )

        # Train and evaluate every duplicate experiment
        for experiment_name, experiment_train in experiments.items():

            print(f"\nTraining: {experiment_name}")

            result = train_and_evaluate(
                train_df=experiment_train,
                val_df=original_validation,
                variant_name=variant_name,
                experiment_name=experiment_name
            )

            results.append(result)

            print(f"MAE:  {result['MAE']:.4f}")
            print(f"RMSE: {result['RMSE']:.4f}")
            print(f"R2:   {result['R2']:.4f}")

    # Convert all results into one comparison table
    results_df = pd.DataFrame(results)

    # Sort experiments by lowest MAE
    results_df = results_df.sort_values(
        by="MAE",
        ascending=True
    ).reset_index(drop=True)

    return results_df


# ==========================================
# 7. Main
# ==========================================

if __name__ == "__main__":

    results_df = run_all_experiments()

    # Display final comparison
    print("\n\nFINAL MODEL COMPARISON")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save results
    results_df.to_csv(
        "random_forest_results.csv",
        index=False
    )

    print("\nResults saved to random_forest_results.csv")