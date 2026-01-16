# NeMo RL Web Configuration Tool

A web-based application for configuring and generating SLURM scripts for NeMo RL training jobs.

## Features

- **Visual Configuration Builder**: Form-based inputs for algorithm, model, dataset, and hyperparameters
- **Real-time Validation**: Configuration validation with helpful error messages and warnings
- **SLURM Script Generation**: Generate ready-to-run SLURM submission scripts
- **Model Search**: Search HuggingFace Hub for compatible models
- **Dark/Light Theme**: Toggle between dark and light modes

## Architecture

```
web/
├── frontend/          # React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── store/         # Zustand state management
│   │   ├── api/           # API client
│   │   └── types/         # TypeScript types
│   └── ...
├── backend/           # FastAPI + Pydantic
│   ├── main.py           # API endpoints
│   └── tests/            # Backend tests
└── .github/
    └── workflows/        # CI/CD pipeline
```

## Quick Start

### Prerequisites

- Node.js 18+ (with npm)
- Python 3.10+
- uv (recommended) or pip

### Development Setup

#### Frontend

```bash
cd web/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

#### Backend

```bash
cd web/backend

# Install dependencies (using uv - recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

API documentation is available at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### Running Tests

#### Frontend Tests

```bash
cd web/frontend

# Run tests in watch mode
npm test

# Run tests once
npm run test:run

# Run tests with coverage
npm run test:coverage
```

#### Backend Tests

```bash
cd web/backend

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Configuration Validation
```
POST /api/v1/config/validate
Content-Type: application/json

{
  "algorithm": "grpo",
  "model": "Qwen/Qwen2.5-1.5B",
  "dataset": "nvidia/OpenMathInstruct-2",
  "backend": "dtensor",
  "hyperparameters": {
    "learning_rate": 1e-6,
    "batch_size": 32,
    "max_steps": 1000,
    "num_generations_per_prompt": 16
  },
  "cluster": {
    "nodes": 1,
    "gpus_per_node": 8,
    "time_limit": "4:00:00"
  }
}
```

### Script Generation
```
POST /api/v1/script/generate
Content-Type: application/json

{
  "config": { ... },
  "cluster_preset": "dgx-cloud",
  "output_format": "slurm"
}
```

### Model Search
```
GET /api/v1/models/search?q=qwen&size=small
```

## Configuration Options

### Algorithms
- **GRPO**: Reinforcement learning from rewards
- **SFT**: Supervised fine-tuning on examples
- **DPO**: Preference optimization from pairs

### Backends
- **dtensor**: FSDP2 with PyTorch DTensor
- **megatron**: Megatron-LM with tensor/pipeline parallelism

### Cluster Presets
- **dgx-cloud**: NVIDIA DGX Cloud
- **nvidia-bcm**: NVIDIA BCM Cluster
- **generic**: Generic SLURM cluster

## Tech Stack

### Frontend
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Zustand (state management)
- Lucide React (icons)
- Vitest + Testing Library (testing)

### Backend
- FastAPI (web framework)
- Pydantic (data validation)
- Uvicorn (ASGI server)
- pytest (testing)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

See the main NeMo RL repository for license information.
