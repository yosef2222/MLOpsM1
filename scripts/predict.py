import os
import pandas as pd
from catboost import CatBoostClassifier
import logging

# Ensure the data directory exists
os.makedirs("../data", exist_ok=True)

# Configure logging
logging.basicConfig(filename="../data/predict.log", level=logging.INFO)

def predict(test_csv_path, model_path, output_csv_path):
    """
    Generate predictions using a trained CatBoost model.

    Args:
        test_csv_path (str): Path to the test dataset CSV file.
        model_path (str): Path to the trained model.
        output_csv_path (str): Path to save the predictions CSV file.
    """
    logging.info("Starting prediction")

    # Load the test dataset
    test_df = pd.read_csv(test_csv_path)

    # Save PassengerId for the results
    passenger_ids = test_df["PassengerId"]

    # Preprocess the test data
    test_df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")

    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        test_df[col] = test_df[col].fillna(test_df[col].mean())

    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
        test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

    test_df[["Deck", "CabinNumber", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
    test_df["CabinNumber"] = pd.to_numeric(test_df["CabinNumber"], errors="coerce").fillna(0)
    test_df.drop("Cabin", axis=1, inplace=True)

    test_df = pd.get_dummies(test_df, columns=["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"], drop_first=True)

    # Load the trained model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Generate predictions
    predictions = model.predict(test_df)

    # Save predictions to a CSV file
    results = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": predictions
    })
    results.to_csv(output_csv_path, index=False)
    logging.info(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate predictions using a trained CatBoost model.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the predictions CSV file.")

    args = parser.parse_args()
    predict(args.test_csv, args.model_path, args.output_csv)
