# MLOpsM1
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
poetry shell  # Activate the virtual environment
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
