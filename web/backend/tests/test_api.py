"""
Tests for NeMo RL Web Configurator API
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# Health Check Tests
# =============================================================================

def test_health_check(client):
    """Test health check endpoint returns healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


# =============================================================================
# Config Validation Tests
# =============================================================================

def test_validate_valid_config(client):
    """Test validation of a valid configuration."""
    config = {
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
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["errors"]) == 0
    assert "resource_estimate" in data


def test_validate_config_with_warnings(client):
    """Test validation returns warnings for suboptimal config."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-1.5B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "dtensor",
        "hyperparameters": {
            "learning_rate": 1e-3,  # High learning rate should trigger warning
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 2  # Low generations should trigger warning
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True  # Warnings don't make config invalid
    assert len(data["warnings"]) > 0


def test_validate_invalid_model_name(client):
    """Test validation fails for invalid model name."""
    config = {
        "algorithm": "sft",
        "model": "invalid model name with spaces!",
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
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert any("model" in e["field"] for e in data["errors"])


def test_validate_invalid_gpu_count(client):
    """Test validation fails for invalid GPU count."""
    config = {
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
            "gpus_per_node": 3,  # Invalid - must be 1, 2, 4, or 8
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 422  # Pydantic validation error


# =============================================================================
# Script Generation Tests
# =============================================================================

def test_generate_script(client):
    """Test SLURM script generation."""
    request = {
        "config": {
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
        },
        "output_format": "slurm"
    }
    
    response = client.post("/api/v1/script/generate", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "script" in data
    assert "#SBATCH" in data["script"]
    assert "grpo" in data["script"]
    assert len(data["instructions"]) > 0
    # Should always include config.yaml
    assert "config.yaml" in data["files"]
    assert "Qwen/Qwen2.5-1.5B" in data["files"]["config.yaml"]


def test_generate_script_multinode(client):
    """Test SLURM script generation for multi-node includes ray.sub."""
    request = {
        "config": {
            "algorithm": "grpo",
            "model": "Qwen/Qwen2.5-7B",
            "dataset": "nvidia/OpenMathInstruct-2",
            "backend": "megatron",
            "hyperparameters": {
                "learning_rate": 1e-6,
                "batch_size": 32,
                "max_steps": 1000,
                "num_generations_per_prompt": 16
            },
            "cluster": {
                "nodes": 2,  # Multi-node
                "gpus_per_node": 8,
                "time_limit": "4:00:00"
            }
        },
        "output_format": "slurm"
    }
    
    response = client.post("/api/v1/script/generate", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "ray.sub" in data["files"]
    assert "config.yaml" in data["files"]
    # Multi-node script should reference ray.sub
    assert "ray.sub" in data["script"]


def test_generate_script_with_cluster_preset(client):
    """Test script generation with cluster preset."""
    request = {
        "config": {
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
        },
        "cluster_preset": "dgx-cloud",
        "output_format": "slurm"
    }
    
    response = client.post("/api/v1/script/generate", json=request)
    assert response.status_code == 200
    data = response.json()
    # DGX Cloud preset should set partition to batch
    assert "dgx-cloud" in data["script"].lower() or "partition" in data["script"]


def test_generate_script_returns_config_yaml(client):
    """
    VERIFY criterion: POST config - returns valid SLURM script with all required components.
    This test verifies config.yaml is included and contains all required fields.
    """
    request = {
        "config": {
            "algorithm": "sft",
            "model": "meta-llama/Llama-3.1-8B",
            "dataset": "nvidia/OpenMathInstruct-2",
            "backend": "megatron",
            "hyperparameters": {
                "learning_rate": 5e-6,
                "batch_size": 16,
                "max_steps": 500,
                "num_generations_per_prompt": 1,
                "tensor_parallel_size": 4
            },
            "cluster": {
                "nodes": 1,
                "gpus_per_node": 8,
                "time_limit": "8:00:00",
                "partition": "luna"
            }
        },
        "output_format": "slurm"
    }
    
    response = client.post("/api/v1/script/generate", json=request)
    assert response.status_code == 200
    data = response.json()
    
    # Verify SLURM script has all required components
    script = data["script"]
    assert "#SBATCH --job-name=" in script
    assert "#SBATCH --nodes=" in script
    assert "#SBATCH --gpus-per-node=" in script
    assert "#SBATCH --time=" in script
    
    # Verify config.yaml is included
    assert "config.yaml" in data["files"]
    config_yaml = data["files"]["config.yaml"]
    
    # Verify config.yaml contains all required sections
    assert "algorithm: sft" in config_yaml
    assert "meta-llama/Llama-3.1-8B" in config_yaml
    assert "backend: megatron" in config_yaml
    assert "tensor_parallel_size: 4" in config_yaml
    assert "learning_rate: 5e-06" in config_yaml or "learning_rate: 5.0e-06" in config_yaml
    assert "batch_size: 16" in config_yaml
    assert "max_steps: 500" in config_yaml
    assert "nodes: 1" in config_yaml
    assert "gpus_per_node: 8" in config_yaml
    
    # Verify instructions are provided
    assert len(data["instructions"]) >= 4
    assert any("sbatch" in instr.lower() for instr in data["instructions"])


def test_generate_script_instructions(client):
    """Test that script generation returns proper instructions."""
    request = {
        "config": {
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
        },
        "output_format": "slurm"
    }
    
    response = client.post("/api/v1/script/generate", json=request)
    assert response.status_code == 200
    data = response.json()
    
    # Check instructions cover key steps
    instructions_text = " ".join(data["instructions"]).lower()
    assert "save" in instructions_text
    assert "chmod" in instructions_text or "executable" in instructions_text
    assert "sbatch" in instructions_text
    assert "monitor" in instructions_text or "tail" in instructions_text


# =============================================================================
# Model Search Tests
# =============================================================================

def test_search_models(client):
    """Test model search returns results."""
    response = client.get("/api/v1/models/search", params={"q": "qwen"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert all("qwen" in m["id"].lower() for m in data["results"])


def test_search_models_no_results(client):
    """Test model search returns empty for non-existent query."""
    response = client.get("/api/v1/models/search", params={"q": "nonexistentmodel123"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []


def test_search_models_by_size(client):
    """Test model search can filter by size."""
    response = client.get("/api/v1/models/search", params={"q": "llama", "size": "small"})
    assert response.status_code == 200
    data = response.json()
    # All results should be small models (< 10GB)
    for model in data["results"]:
        assert model["size_gb"] < 10


def test_search_models_returns_metadata(client):
    """Test model search returns complete metadata."""
    response = client.get("/api/v1/models/search", params={"q": "qwen"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    
    for model in data["results"]:
        assert "id" in model
        assert "name" in model
        assert "size_gb" in model
        assert "parameters" in model
        assert "recommended_config" in model
        # Check recommended_config has expected fields
        assert "min_gpus" in model["recommended_config"]
        assert "tensor_parallel" in model["recommended_config"]


def test_search_models_sorted_by_size(client):
    """Test model search results are sorted by size (smallest first)."""
    response = client.get("/api/v1/models/search", params={"q": "qwen"})
    assert response.status_code == 200
    data = response.json()
    
    sizes = [m["size_gb"] for m in data["results"]]
    assert sizes == sorted(sizes), "Results should be sorted by size"


def test_search_models_with_limit(client):
    """Test model search respects limit parameter."""
    response = client.get("/api/v1/models/search", params={"q": "qwen", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 3


def test_search_models_large_size(client):
    """Test model search for large models."""
    response = client.get("/api/v1/models/search", params={"q": "llama", "size": "large"})
    assert response.status_code == 200
    data = response.json()
    # All results should be large models (>= 50GB)
    for model in data["results"]:
        assert model["size_gb"] >= 50


def test_search_models_medium_size(client):
    """Test model search for medium models."""
    response = client.get("/api/v1/models/search", params={"q": "qwen", "size": "medium"})
    assert response.status_code == 200
    data = response.json()
    # All results should be medium models (10-50GB)
    for model in data["results"]:
        assert 10 <= model["size_gb"] < 50


# =============================================================================
# Dataset Search Tests
# =============================================================================

def test_search_datasets(client):
    """Test dataset search returns results."""
    response = client.get("/api/v1/datasets/search", params={"q": "math"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_search_datasets_no_results(client):
    """Test dataset search returns empty for non-existent query."""
    response = client.get("/api/v1/datasets/search", params={"q": "nonexistent123xyz"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []


def test_search_datasets_returns_metadata(client):
    """Test dataset search returns complete metadata."""
    response = client.get("/api/v1/datasets/search", params={"q": "OpenMathInstruct"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    
    dataset = data["results"][0]
    assert "id" in dataset
    assert "name" in dataset
    assert "description" in dataset
    assert "size" in dataset
    assert "features" in dataset
    assert isinstance(dataset["features"], list)


def test_search_datasets_with_limit(client):
    """Test dataset search respects limit parameter."""
    response = client.get("/api/v1/datasets/search", params={"q": "a", "limit": 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 2


def test_search_datasets_by_description(client):
    """Test dataset search matches on description."""
    response = client.get("/api/v1/datasets/search", params={"q": "instruction"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0


# =============================================================================
# Algorithm-specific Tests
# =============================================================================

def test_validate_sft_config(client):
    """Test validation of SFT configuration."""
    config = {
        "algorithm": "sft",
        "model": "meta-llama/Llama-3.1-8B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "dtensor",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 16,
            "max_steps": 500,
            "num_generations_per_prompt": 1
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "2:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True


def test_validate_dpo_config(client):
    """Test validation of DPO configuration."""
    config = {
        "algorithm": "dpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/HelpSteer2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 5e-7,
            "batch_size": 8,
            "max_steps": 1000,
            "num_generations_per_prompt": 1
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True


# =============================================================================
# Enhanced Validation Tests
# =============================================================================

def test_validate_returns_resource_estimate(client):
    """Test validation returns resource estimates."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
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
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "resource_estimate" in data
    estimate = data["resource_estimate"]
    assert "memory_per_gpu" in estimate
    assert "recommended_gpus" in estimate
    assert "estimated_time" in estimate
    assert "GB" in estimate["memory_per_gpu"]


def test_validate_invalid_learning_rate_too_low(client):
    """Test validation fails for learning rate below minimum."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-1.5B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "dtensor",
        "hyperparameters": {
            "learning_rate": 1e-10,  # Below minimum
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
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 422  # Pydantic validation


def test_validate_invalid_time_limit_format(client):
    """Test validation fails for invalid time limit format."""
    config = {
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
            "time_limit": "4hours"  # Invalid format
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 422  # Pydantic validation error


def test_validate_megatron_backend_warning(client):
    """Test validation warns about Megatron with small models."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-1.5B",  # Small model
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",  # Megatron not optimal for small models
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
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    # Should have warning about small model with Megatron
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("small" in msg.lower() or "1.5b" in msg.lower() for msg in warning_messages)


def test_validate_large_model_gpu_warning(client):
    """Test validation warns when GPUs may be insufficient for model size."""
    config = {
        "algorithm": "grpo",
        "model": "meta-llama/Llama-3.1-70B",  # Large model
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16
        },
        "cluster": {
            "nodes": 1,  # Single node likely insufficient
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    # Should warn about GPU count for large model
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("gpu" in msg.lower() for msg in warning_messages)


def test_validate_dpo_odd_batch_size_info(client):
    """Test validation provides info for DPO with odd batch size."""
    config = {
        "algorithm": "dpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/HelpSteer2",
        "backend": "dtensor",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 31,  # Odd number - not ideal for DPO pairs
            "max_steps": 1000,
            "num_generations_per_prompt": 1
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    # Should have info about batch size for DPO
    all_messages = [w["message"] for w in data["warnings"]]
    assert any("pair" in msg.lower() or "even" in msg.lower() for msg in all_messages)


def test_validate_grpo_high_generations(client):
    """Test validation warns for very high generations per prompt."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-1.5B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "dtensor",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 48  # Very high
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("generation" in msg.lower() or "slow" in msg.lower() for msg in warning_messages)


def test_validate_multinode_partition_info(client):
    """Test validation suggests partition for multi-node jobs."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16
        },
        "cluster": {
            "nodes": 4,  # Multi-node
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
            # No partition specified
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("partition" in msg.lower() for msg in warning_messages)


# =============================================================================
# Tensor Parallelism Validation Tests
# =============================================================================

def test_validate_tensor_parallel_size_divides_evenly(client):
    """Test validation passes when tensor_parallel_size divides evenly into GPUs."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16,
            "tensor_parallel_size": 4  # 8 / 4 = 2 (divides evenly)
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    # Should not have tensor parallel errors
    error_fields = [e["field"] for e in data["errors"]]
    assert not any("tensor_parallel" in f for f in error_fields)


def test_validate_tensor_parallel_size_3_with_gpus_8_error(client):
    """
    VERIFY criterion: Enter invalid tensor_parallel_size=3 with gpus=8.
    Error 'must divide evenly' should appear.
    """
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16,
            "tensor_parallel_size": 3  # Invalid: 8 / 3 does not divide evenly
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    
    # Should be invalid
    assert data["valid"] is False
    
    # Should have error about tensor parallel size
    error_fields = [e["field"] for e in data["errors"]]
    error_messages = [e["message"] for e in data["errors"]]
    
    assert any("tensor_parallel" in f for f in error_fields)
    assert any("must divide evenly" in msg for msg in error_messages)


def test_validate_tensor_parallel_size_5_with_gpus_8_error(client):
    """Test tensor_parallel_size=5 fails with 8 GPUs (5 doesn't divide 8)."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16,
            "tensor_parallel_size": 5  # Invalid
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    error_messages = [e["message"] for e in data["errors"]]
    assert any("must divide evenly" in msg for msg in error_messages)


def test_validate_tensor_parallel_with_dtensor_warning(client):
    """Test tensor parallelism with DTensor backend shows warning."""
    config = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen2.5-7B",
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "dtensor",  # DTensor doesn't typically use tensor parallelism
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16,
            "tensor_parallel_size": 2  # Tensor parallelism with DTensor
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True  # Valid but with warning
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("megatron" in msg.lower() or "dtensor" in msg.lower() for msg in warning_messages)


def test_validate_large_model_suggests_tensor_parallelism(client):
    """Test large model with tensor_parallel_size=1 suggests enabling it."""
    config = {
        "algorithm": "grpo",
        "model": "meta-llama/Llama-3.1-70B",  # Large model
        "dataset": "nvidia/OpenMathInstruct-2",
        "backend": "megatron",
        "hyperparameters": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "max_steps": 1000,
            "num_generations_per_prompt": 16,
            "tensor_parallel_size": 1  # Not using tensor parallelism
        },
        "cluster": {
            "nodes": 2,
            "gpus_per_node": 8,
            "time_limit": "4:00:00"
        }
    }
    
    response = client.post("/api/v1/config/validate", json=config)
    assert response.status_code == 200
    data = response.json()
    # Should suggest tensor parallelism for large models
    warning_messages = [w["message"] for w in data["warnings"]]
    assert any("tensor" in msg.lower() and "parallel" in msg.lower() for msg in warning_messages)


def test_validate_tensor_parallel_valid_sizes(client):
    """Test all valid tensor_parallel_size values (1, 2, 4, 8) work with 8 GPUs."""
    valid_sizes = [1, 2, 4, 8]
    
    for tp_size in valid_sizes:
        config = {
            "algorithm": "grpo",
            "model": "Qwen/Qwen2.5-7B",
            "dataset": "nvidia/OpenMathInstruct-2",
            "backend": "megatron",
            "hyperparameters": {
                "learning_rate": 1e-6,
                "batch_size": 32,
                "max_steps": 1000,
                "num_generations_per_prompt": 16,
                "tensor_parallel_size": tp_size
            },
            "cluster": {
                "nodes": 1,
                "gpus_per_node": 8,
                "time_limit": "4:00:00"
            }
        }
        
        response = client.post("/api/v1/config/validate", json=config)
        assert response.status_code == 200
        data = response.json()
        # Should not have tensor parallel errors (might have other warnings)
        error_fields = [e["field"] for e in data["errors"]]
        assert not any("tensor_parallel" in f for f in error_fields), \
            f"tensor_parallel_size={tp_size} should be valid with 8 GPUs"
