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

# Install the project package (src layout)
pip install -e .
```


### 2. Running individual scripts

You can execute any module in the scripts/ directory directly and override parameters from the command line:


```bash
python -m scripts.training -m model_type=tree,nn
python -m scripts.tuning model_type=nn model_type.tuning.num_samples=3
python -m scripts.inference
python -m scripts.evaluation
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


### MLFLow


#### Run server
```
...$ mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host localhost \
    --port 5000
```

#### Delete the SQLite file
```
rm mlflow.db
```

#### Delete all artifacts (by default in ./mlruns/)
```
rm -rf mlruns/
```

### References

The CICIDS2017 dataset consists of labeled network flows, including full packet payloads in pcap format, the corresponding profiles and the labeled flows (GeneratedLabelledFlows.zip) and CSV files for machine and deep learning purpose (MachineLearningCSV.zip) are publicly available for researchers. [Link](https://www.unb.ca/cic/datasets/ids-2017.html)

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.


**Note**. Given the restricted resources and the fact that this is a portfolio project, I will use a subset of the ICS Dataset.




