# Titanic Survival Classifier 🚢

An end-to-end machine learning project focused on predicting Titanic passenger survival. This project covers the full ML lifecycle — from data preprocessing to deployment on **Lightning AI**.

---

## 🔧 Technologies Used

- **[uv](https://github.com/astral-sh/uv)** – Dependency management  
- **[DVC](https://dvc.org/)** – Data version control  
- **[Hydra](https://hydra.cc/)** – Dynamic configuration management  
- **[MLflow](https://mlflow.org/)** via **[Dagshub](https://dagshub.com/)** – Experiment tracking and model registry  
- **[FastAPI](https://fastapi.tiangolo.com/)** + **[LitServe](https://lightning.ai/lightning-ai-components/lit-serve)** – Model serving API  
- **[Docker](https://www.docker.com/)** – Containerization  
- **[Docker Hub](https://hub.docker.com/)** – Image registry  
- **[Lightning AI](https://lightning.ai/)** – Deployment platform  

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
uv sync
```

### 2. Preprocess and Train the Model
```bash
uv run main.py
```
### 3. Start the Inference Server

```bash
python server.py
```