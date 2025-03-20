import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import logging
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import fire
import os

logging.basicConfig(filename="../data/log_file.log", level=logging.INFO)

class MyClassifierModel:
    def __init__(self):
        self.model = None

    def train(self, dataset):
        logging.info("Starting training")
        
        # Load the dataset
        df = pd.read_csv(dataset)
        
        # Preprocessing (Feature Engineering, Encoding)
        df = self.preprocess(df, training=True)
        
        # Train-Test Split
        X = df.drop("Transported", axis=1)
        y = df["Transported"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load best hyperparameters
        best_params_path = "best_params.json"
        if not os.path.exists(best_params_path):
            raise FileNotFoundError(f"Hyperparameter file {best_params_path} not found. Run optimization first.")

        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        
        # Train CatBoost Model
        self.model = CatBoostClassifier(**best_params, verbose=0)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info(f"Validation Accuracy: {accuracy:.2f}")

        # Save Model
        os.makedirs("../model", exist_ok=True)
        self.model.save_model("../model/model.cbm")
        logging.info("Training completed and model saved.")

    def predict(self, dataset):
        logging.info("Starting prediction")
        
        # Load dataset
        df = pd.read_csv(dataset)
        passenger_ids = df["PassengerId"]
        df = self.preprocess(df, training=False)

        # Load Model
        self.model = CatBoostClassifier()
        self.model.load_model("../model/model.cbm")
        
        # Make Predictions
        predictions = self.model.predict(df)
        results = pd.DataFrame({"PassengerId": passenger_ids, "Transported": predictions})
        
        # Save Results
        os.makedirs("../data", exist_ok=True)
        results.to_csv("../data/results.csv", index=False)
        logging.info("Prediction completed and results saved.")

    def preprocess(self, df, training=True):
        """ Feature Engineering & Preprocessing """
        df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")

        # Define columns
        numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
        
        # Handle Missing Values
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Feature Engineering: Split "Cabin"
        df[["Deck", "CabinNumber", "Side"]] = df["Cabin"].str.split("/", expand=True)
        df.drop("Cabin", axis=1, inplace=True)
        df["CabinNumber"] = pd.to_numeric(df["CabinNumber"], errors="coerce").fillna(0)

        # Encode categorical variables
        df = pd.get_dummies(df, columns=["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"], drop_first=True)

        # Feature Engineering
        df["TotalSpending"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        df["Age_Spending"] = df["Age"] * df["TotalSpending"]
        df["High_Cabin"] = (df["CabinNumber"] > df["CabinNumber"].median()).astype(int)
        df.drop("CabinNumber", axis=1, inplace=True)

        return df

if __name__ == "__main__":
    fire.Fire(MyClassifierModel())
