# 🚀 CV Inference System using FastAPI + NVIDIA Triton

A **production-ready machine learning inference system** built with **FastAPI** and **NVIDIA Triton Inference Server** to serve an open-source **ResNet50 computer vision model**.

The system supports **concurrent inference requests**, **dynamic batching**, **containerized deployment**, and **load testing** to demonstrate scalability.

---

- Demo Video: [Watch](https://github.com/aalekh12/Cv-Interface-System-/blob/master/Cv-Intreface-Demo.mp4)

# 📌 Features

* FastAPI REST API for inference
* NVIDIA Triton Inference Server for model serving
* ONNX ResNet50 model
* Async request handling
* Dynamic batching support
* Dockerized deployment
* Load testing with **k6**
* Health check endpoint
* Structured JSON response with latency

---

# 🏗️ System Architecture

```
Client
   │
   ▼
FastAPI API Server
   │
   ▼
Triton Client (HTTP)
   │
   ▼
NVIDIA Triton Inference Server
   │
   ▼
ResNet50 ONNX Model
```

### Flow

1. Client sends an image to `/predict`
2. FastAPI preprocesses the image
3. Image tensor is sent to Triton
4. Triton runs inference using ResNet50
5. Predictions are returned as structured JSON

---

# 📂 Project Structure

```
cv-inference-system
│
├── fastapi_app
│   ├── main.py
│   ├── triton_client.py
│   ├── utils.py
│   ├── config.py
│   └── schemas.py
│
├── model_repository
│   └── resnet50
│       ├── config.pbtxt
│       └── 1
│           └── model.onnx
│
├── docker
│   └── Dockerfile.fastapi
│
├── load_test
│   └── k6_test.js
│
├── docker-compose.yml
└── README.md
```

---

# 🧠 Model

The system uses **ResNet50**, a widely used convolutional neural network trained on **ImageNet**.

**Input**

```
Tensor shape: (1,3,224,224)
Data type: FP32
```

**Output**

```
Tensor shape: (1,1000)
Class probabilities for ImageNet classes
```

---

# ⚙️ Setup Instructions

## 1️⃣ Install Dependencies

Ensure the following are installed:

* Docker Desktop
* Python 3.10+
* pip
* k6 (for load testing)

---

## 2️⃣ Clone the Repository

```
git clone https://github.com/your-repo/cv-inference-system.git
cd cv-inference-system
```

---

## 3️⃣ Start the System

Run:

```
docker compose up --build
```

This launches:

* Triton Inference Server
* FastAPI service

---

## 4️⃣ Verify Services

### Triton Health

```
http://localhost:8000/v2/health/ready
```

Expected:

```
OK
```

---

### FastAPI Docs

```
http://localhost:9000/docs
```

---

# 🔮 API Usage

## Predict Endpoint

### POST `/predict`

Upload an image using multipart/form-data.

Example request:

```
curl -X POST http://localhost:9000/predict \
-F "image=@cat.jpg"
```

---

## Example Response

```json
{
  "predictions": [
    {
      "class_id": 624,
      "confidence": 0.91
    },
    {
      "class_id": 722,
      "confidence": 0.04
    }
  ],
  "latency_ms": 140.2
}
```

---

# ❤️ Health Check

Endpoint:

```
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

# 📊 Load Testing

Load testing is performed using **k6** to validate concurrency handling.

## Run Load Test

```
cd load_test
k6 run k6_test.js
```

### Test Configuration

```
Concurrent Users: 10
Duration: 30 seconds
```

---

## Example Results

```
Requests/sec: ~85
Average latency: 120 ms
Error rate: 0%
```

This demonstrates that the system can handle **10+ concurrent requests efficiently**.

---

# ⚡ Dynamic Batching

Dynamic batching is enabled in Triton using:

```
dynamic_batching {
  preferred_batch_size: [4,8]
  max_queue_delay_microseconds: 100
}
```

Benefits:

* Improved GPU/CPU utilization
* Reduced latency under load
* Higher throughput

---

# 📦 Containerization

The system is fully containerized using Docker.

Services:

| Service | Purpose                |
| ------- | ---------------------- |
| FastAPI | REST API for inference |
| Triton  | Model serving          |

Deployment is managed via:

```
docker-compose.yml
```

---

# 📈 Scalability

The system can scale horizontally by:

* Running multiple FastAPI replicas
* Placing a load balancer in front
* Running Triton with multiple instances

Example Kubernetes scaling:

```
replicas: 3
```

This allows the system to handle higher request volumes.

---

# 🛡️ Error Handling

The system includes:

* Input validation
* Graceful error responses
* Logging with Loguru
* Timeout handling

Example error response:

```json
{
  "error": "Inference service failed"
}
```

---

# 📊 Metrics (Optional Extension)

Prometheus metrics can be exposed via:

```
/metrics
```

Useful for monitoring:

* request rate
* latency
* error rates

---

# 📚 Technologies Used

| Technology    | Purpose                |
| ------------- | ---------------------- |
| FastAPI       | API framework          |
| NVIDIA Triton | Model inference server |
| ONNX          | Model format           |
| Docker        | Containerization       |
| k6            | Load testing           |
| NumPy         | Tensor processing      |
| Pillow        | Image preprocessing    |

---

# 📌 Conclusion

This project demonstrates a **production-ready inference pipeline** combining:

* scalable API infrastructure
* high-performance model serving
* containerized deployment
* load-tested concurrency handling

