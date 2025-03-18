import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import logging
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import fire

# Configure logging
logging.basicConfig(filename='data/log_file.log', level=logging.INFO)

class My_Classifier_Model:
    def __init__(self):
        self.model = None

    def train(self, dataset):
        logging.info("Starting training")
        
        # Load the dataset
        df = pd.read_csv(dataset)
        
        # Drop irrelevant columns (e.g., PassengerId, Name)
        df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")
        
        # Handle missing values
        numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Split Cabin into Deck, Number, and Side
        df[["Deck", "CabinNumber", "Side"]] = df["Cabin"].str.split("/", expand=True)
        df.drop("Cabin", axis=1, inplace=True)
        
        # Convert CabinNumber to numeric (if possible)
        df["CabinNumber"] = pd.to_numeric(df["CabinNumber"], errors="coerce")
        df["CabinNumber"] = df["CabinNumber"].fillna(0)
        
        # Encode categorical variables
        df = pd.get_dummies(df, columns=["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"], drop_first=True)
        
        # Separate features and target
        X = df.drop("Transported", axis=1)
        y = df["Transported"]
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the CatBoost model
        self.model = CatBoostClassifier(verbose=0)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info(f"Validation Accuracy: {accuracy:.2f}")
        
        # Save the model
        self.model.save_model("model/model.cbm")
        logging.info("Training completed and model saved")

    def predict(self, dataset):
	    logging.info("Starting prediction")
	    
	    # Load the dataset
	    df = pd.read_csv(dataset)
	    
	    # Save PassengerId for the results
	    passenger_ids = df["PassengerId"]
	    
	    # Drop irrelevant columns (e.g., Name)
	    df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")
	    
	    # Handle missing values
	    numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
	    for col in numerical_cols:
	        df[col] = df[col].fillna(df[col].mean())
	    
	    categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
	    for col in categorical_cols:
	        df[col] = df[col].fillna(df[col].mode()[0])
	    
	    # Split Cabin into Deck, Number, and Side
	    df[["Deck", "CabinNumber", "Side"]] = df["Cabin"].str.split("/", expand=True)
	    df.drop("Cabin", axis=1, inplace=True)
	    
	    # Convert CabinNumber to numeric (if possible)
	    df["CabinNumber"] = pd.to_numeric(df["CabinNumber"], errors="coerce")
	    df["CabinNumber"] = df["CabinNumber"].fillna(0)
	    
	    # Encode categorical variables
	    df = pd.get_dummies(df, columns=["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"], drop_first=True)
	    
	    # Load the model
	    self.model = CatBoostClassifier()
	    self.model.load_model("model/model.cbm")
	    
	    # Generate predictions
	    predictions = self.model.predict(df)
	    
	    # Combine PassengerId with predictions
	    results = pd.DataFrame({
	        "PassengerId": passenger_ids,
	        "Transported": predictions
	    })
	    
	    # Save results to CSV
	    results.to_csv("data/results.csv", index=False)
	    logging.info("Prediction completed and results saved")

if __name__ == "__main__":
    fire.Fire(My_Classifier_Model())
