import argparse
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    target_column = 'Transported'  # Updated to match actual column name

    if target_column not in train_df.columns:
        raise KeyError(f"Column '{target_column}' not found in train dataset. Available columns: {list(train_df.columns)}")
    
    return train_df, test_df

def preprocess_data(df):
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    df[cat_features] = df[cat_features].fillna("missing").astype(str)
    df = df.fillna(0)  # Fill numerical NaNs with 0
    return df, cat_features

def objective_catboost(trial, X_train, y_train, X_val, y_val, cat_features):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "verbose": 0
    }

    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_features)
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def main(train_csv, test_csv):
    train_df, test_df = load_data(train_csv, test_csv)
    print("Train columns:", train_df.columns)

    X, cat_features = preprocess_data(train_df.drop(columns=['Transported']))
    y = train_df['Transported']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val, cat_features), n_trials=20)
    
    print("Best trial:", study.best_trial.params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.train_csv, args.test_csv)

