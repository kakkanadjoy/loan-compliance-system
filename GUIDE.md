# MLOps Pipeline — Complete Guide
## House Price Prediction · End-to-End Productionalization

---

## Project Structure

```
house-price-mlops/
│
├── src/
│   ├── data/
│   │   └── preprocess.py        # Load data, scale features, prep for inference
│   ├── model/
│   │   ├── train.py             # Train + log to MLflow
│   │   └── evaluate.py          # Quality gate (blocks bad models from deploying)
│   └── api/
│       └── app.py               # FastAPI prediction server
│
├── streamlit_app/
│   └── app.py                   # Streamlit frontend UI
│
├── tests/
│   └── test_api.py              # API unit tests
│
├── .github/workflows/
│   └── cicd.yml                 # GitHub Actions CI/CD pipeline
│
├── Dockerfile                   # Container for the API
├── Dockerfile.streamlit         # Container for the UI
├── docker-compose.yml           # Run everything locally
├── aws_setup.sh                 # One-time AWS infrastructure setup
├── Makefile                     # Shortcut commands
└── requirements.txt
```

---

## The Full Lifecycle (how data flows through the system)

```
Raw Data → Preprocess → Train → Evaluate → Docker Build → ECR Push → ECS Deploy
                ↓                                                          ↓
           MLflow tracks                                            FastAPI serves
           every run                                               predictions
                                                                          ↓
                                                                   Streamlit UI
                                                                   calls the API
```

---

## Section 1: Data Pipeline (`src/data/preprocess.py`)

**What it does:**
- Loads the California Housing dataset (built into sklearn, no CSV needed)
- Splits into train/test
- Fits a `StandardScaler` on training data only, saves it to disk

**Why save the scaler?**
When a user sends a prediction request, their raw numbers (income = 3.5, age = 25)
need to be scaled the same way training data was. You load the same scaler at
inference time. If you refit the scaler on new data, predictions break.

**Key concept — train/test split:**
```
fit_scaler=True   → during training:  scaler.fit_transform(X_train)
fit_scaler=False  → during inference: scaler.transform(X_new)
```
Never call `.fit()` on test data. That's data leakage.

---

## Section 2: MLflow Tracking (`src/model/train.py`)

MLflow is your experiment ledger. Every training run logs:
- **Parameters** — what settings you used (n_estimators=100, max_depth=10)
- **Metrics** — how well it performed (RMSE, MAE, R²)
- **Artifacts** — the model file itself

**Run the UI:**
```bash
mlflow ui   # opens http://localhost:5000
```
You'll see every run, compare them side by side, and click to download any model.

**Model Registry:**
`registered_model_name="HousePriceModel"` registers the model so you can version it:
- Version 1: baseline RandomForest
- Version 2: tuned RandomForest
- Version 3: XGBoost
...and promote the best one to "Production" in the UI.

---

## Section 3: Quality Gate (`src/model/evaluate.py`)

Before any model reaches production, it must pass:
```python
assert r2 > 0.7, "Model R² below threshold — pipeline failed!"
```
In the CI/CD pipeline, if this assertion fails, the whole deploy stops.
You never accidentally ship a broken model.

---

## Section 4: FastAPI (`src/api/app.py`)

FastAPI is the server that wraps your model. It:
1. Loads the model **once** at startup (not per request — that would be slow)
2. Exposes `/predict` — accepts JSON, returns a price
3. Uses Pydantic models for automatic input validation
4. Exposes `/health` — AWS pings this to check the container is alive

**Try it locally:**
```bash
make run-api
# then open http://localhost:8000/docs  ← auto-generated Swagger UI
```

**Why FastAPI over Flask?**
- Automatic API docs (Swagger UI at `/docs`)
- Pydantic validation catches bad inputs before they hit your model
- Async support — handles many requests simultaneously
- ~2-3x faster than Flask

---

## Section 5: Streamlit Frontend (`streamlit_app/app.py`)

Streamlit turns Python into a web UI with no HTML/CSS needed.
The sliders map to feature values → POST to the FastAPI `/predict` endpoint
→ display the returned price.

**Run locally:**
```bash
make run-ui   # opens http://localhost:8501
```

The `API_URL` environment variable lets you point the UI at localhost during
development and at the real AWS URL in production.

---

## Section 6: Docker & Containerization

**Why Docker?**
"It works on my machine" is not a deployment strategy.
Docker packages your code + Python version + all dependencies into one image
that runs identically everywhere — your laptop, CI, AWS.

**Multi-stage build (in the Dockerfile):**
```
Stage 1 (builder):  install all packages
Stage 2 (runtime):  copy only what's needed, discard build tools
```
Result: much smaller final image (~200MB vs ~800MB).

**docker-compose.yml** runs all three services together locally:
- API on :8000
- Streamlit on :8501  
- MLflow on :5000

```bash
make docker-up    # start everything
make docker-down  # stop everything
```

The services talk to each other by **service name** (not localhost).
In docker-compose, `http://api:8000` works because Docker creates
an internal network where service names resolve automatically.

---

## Section 7: CI/CD with GitHub Actions (`.github/workflows/cicd.yml`)

Every push to `main` triggers 4 jobs in sequence:

```
[test] → [train] → [build-and-push] → [deploy]
  ↑           ↑            ↑               ↑
pytest    python       docker build     aws ecs
tests     train.py     + ecr push       update
```

**If any job fails, the chain stops.** You can't deploy broken code.

**Artifacts between jobs:**
The trained model files (`artifacts/model.pkl`, `artifacts/scaler.pkl`)
are passed between jobs using `upload-artifact` / `download-artifact`.
GitHub Actions jobs run on separate machines, so files don't persist
automatically.

**Secrets:**
Never put AWS keys in code. Store them in:
`GitHub repo → Settings → Secrets and variables → Actions`

Then reference them as `${{ secrets.AWS_ACCESS_KEY_ID }}`.

---

## Section 8: AWS Deployment

### The services used:

| Service | What it does |
|---------|-------------|
| **ECR** | Private Docker registry (like DockerHub but AWS-managed) |
| **ECS Fargate** | Runs your containers — serverless, no servers to manage |
| **CloudWatch** | Stores container logs |
| **IAM** | Permissions — what the containers are allowed to do |

### Flow:
1. GitHub Actions builds the Docker image
2. Pushes it to ECR (tagged with the git commit hash)
3. Updates the ECS Task Definition with the new image tag
4. ECS does a **rolling deployment** — starts new container, waits for it
   to pass the health check, then stops the old one. Zero downtime.

### One-time setup:
```bash
aws configure    # enter your keys
make aws-setup   # creates ECR, ECS cluster, IAM roles, security group
```

### Getting your API's public URL after deploy:
```bash
# List running tasks
aws ecs list-tasks --cluster house-price-cluster

# Get task details (copy the task ARN from above)
aws ecs describe-tasks --cluster house-price-cluster --tasks <TASK_ARN>
# Look for the networkInterfaceId in the attachments

# Get the public IP
aws ec2 describe-network-interfaces --network-interface-ids <ENI_ID> \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```
Then set `API_URL=http://<PUBLIC_IP>:8000` in your Streamlit app.

---

## How to Run Everything

### Locally (no Docker):
```bash
make install       # install dependencies
make train         # train model, logs to MLflow
make evaluate      # check quality gate
make run-api       # start FastAPI on :8000
make run-ui        # start Streamlit on :8501 (new terminal)
mlflow ui          # view experiment tracking on :5000 (new terminal)
```

### Locally (with Docker):
```bash
make train         # train first to generate artifacts/
make docker-up     # start all 3 services in containers
```

### Production (AWS):
```bash
aws configure      # one-time
make aws-setup     # one-time
# Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to GitHub Secrets
git push origin main   # triggers full CI/CD pipeline
```

---

## What You've Learned

| Concept | Tool | Why |
|---------|------|-----|
| Experiment tracking | MLflow | Never lose a training run |
| Input validation | Pydantic | Catch bad data before it hits the model |
| Containerization | Docker | Reproducible everywhere |
| Orchestration | docker-compose | Run multi-service stack locally |
| Quality gates | pytest + evaluate.py | Block bad models from deploying |
| CI/CD | GitHub Actions | Automate the entire lifecycle |
| Container registry | AWS ECR | Store and version Docker images |
| Serverless containers | AWS ECS Fargate | Deploy without managing servers |
| Frontend | Streamlit | Turn Python into a UI in minutes |

---

## Next Steps (when you're ready to go deeper)

1. **Feature store** — use Feast to manage features across training and serving
2. **Model monitoring** — use Evidently to detect data drift in production
3. **Load balancer** — add an ALB in front of ECS for a stable URL + HTTPS
4. **S3 for artifacts** — store model files in S3 instead of baking them into the Docker image
5. **Hyperparameter tuning** — use MLflow's hyperopt integration to auto-tune
6. **A/B testing** — deploy two model versions and split traffic between them
