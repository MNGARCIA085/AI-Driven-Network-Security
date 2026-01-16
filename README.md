## AI-Driven-Network-Security

This project builds a machine-learning pipeline to classify network traffic from the CICIDS2017 dataset as **benign** or as a specific **attack type**. It includes preprocessing, multiple model families (tree-based and neural networks), hyperparameter exploration, evaluation on validation and test sets, and feature-importance/interpretability tools. The goal is to benchmark different approaches and understand which models generalize best for intrusion detection.



### Setup

#### 1. Create a virtual environment and install dependencies

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


#### 2. Run the main pipeline

```bash
python -m scripts.pipeline
```

#### 3. Run tests

```bash
pytest
```

### Docker usage

#### 1. Build the container

```bash
docker build -f Dockerfile -t ml_env:latest .
```

#### 2. Run the container

```bash
docker run -it -v $(pwd)/mlruns:/app/mlruns ml_env:latest 
```


### Running individual scripts

You can execute any module in the scripts/ directory directly and override parameters from the command line:


```bash
python -m scripts.tuning model_type=nn tuning.batch_size=64 tuning.epochs=5
```


mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000


# Delete the SQLite file
rm mlflow.db

# Delete all artifacts (by default in ./mlruns/)
rm -rf mlruns/