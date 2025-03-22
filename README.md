# MLOpsM1
work by:
Абуелата Юсеф Осама Мохамед Элдсоуки 972301

# 🛠 Installation & Setup
1. **Clone the Repository:**
```
git clone <repository_url>
cd MLOpsM1
```
2. **Set up the environment:**
```
poetry install
poetry env activate  # Activate the virtual environment
```
```
cd dist/
pip install mlops_hw1-0.1.0-py3-none-any.whl 
```
# 🚀 Running the Model Script
```
cd mlops_hw1
python model.py train --dataset=../data/train.csv
python3 model.py predict --dataset=../data/test.csv
```

# 🛠 Resources Utilized
This project utilizes the following resources:

Optuna: A hyperparameter optimization framework to automate the search for the best model parameters.

CatBoost: A gradient boosting library for training machine learning models, particularly effective for categorical data. 

ClearML: A machine learning experiment tracking and model management tool.

Poetry: A dependency management and packaging tool for Python projects.
