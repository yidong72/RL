import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ValidationDisplay } from "./ValidationDisplay";
import { useConfigStore } from "../../store/configStore";
import { api } from "../../api/client";
import type { ValidationResult } from "../../types/config";

// Mock the API client
vi.mock("../../api/client", () => ({
  api: {
    validateConfig: vi.fn(),
  },
}));

// Helper to reset store state
function resetStore() {
  const store = useConfigStore.getState();
  store.resetConfig();
  store.setValidation(null);
  store.setIsValidating(false);
}

describe("ValidationDisplay", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("should display valid status after validation completes", async () => {
    const mockResult: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
      resource_estimate: {
        memory_per_gpu: "24GB",
        recommended_gpus: 2,
        estimated_time: "2-3 hours",
      },
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    // Wait for validation to complete
    await waitFor(() => {
      expect(screen.getByText("Valid Configuration")).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should display valid configuration status", async () => {
    const mockResult: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
      resource_estimate: {
        memory_per_gpu: "24GB",
        recommended_gpus: 2,
        estimated_time: "2-3 hours",
      },
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText("Configuration is valid")).toBeInTheDocument();
      expect(screen.getByText("Ready to generate training script")).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should display errors when configuration is invalid", async () => {
    const mockResult: ValidationResult = {
      valid: false,
      errors: [
        {
          field: "hyperparameters.tensor_parallel_size",
          message: "Tensor parallel size (3) must divide evenly into GPUs per node (8). Valid values: 1, 2, 4, or 8.",
          severity: "error",
        },
      ],
      warnings: [],
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText("Configuration has errors")).toBeInTheDocument();
      expect(screen.getByText(/Tensor parallel size.*must divide evenly/)).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should display warnings when present", async () => {
    const mockResult: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [
        {
          field: "hyperparameters.learning_rate",
          message: "Learning rate 1e-04 is high for fine-tuning. Typical range: 1e-6 to 5e-6",
          severity: "warning",
        },
      ],
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText("1 warning to review")).toBeInTheDocument();
      expect(screen.getByText(/Learning rate.*is high for fine-tuning/)).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should display resource estimates", async () => {
    const mockResult: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
      resource_estimate: {
        memory_per_gpu: "40GB",
        recommended_gpus: 4,
        estimated_time: "4-8 hours",
      },
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText("Resource Estimates")).toBeInTheDocument();
      expect(screen.getByText("40GB")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
      expect(screen.getByText("4-8 hours")).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should call validation API", async () => {
    const mockResult: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(api.validateConfig).toHaveBeenCalled();
    }, { timeout: 2000 });
  });

  it("should handle API errors gracefully", async () => {
    vi.mocked(api.validateConfig).mockRejectedValue(new Error("Network error"));

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText(/Unable to connect to validation service/)).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it("should show error count in header", async () => {
    const mockResult: ValidationResult = {
      valid: false,
      errors: [
        { field: "model", message: "Model not found", severity: "error" },
        { field: "dataset", message: "Dataset not found", severity: "error" },
      ],
      warnings: [],
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText("2 Errors")).toBeInTheDocument();
    }, { timeout: 2000 });
  });
});

describe("Tensor Parallel Validation Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("should validate tensor_parallel_size=3 with gpus=8 shows 'must divide evenly' error", async () => {
    // This is the exact VERIFY criterion from the acceptance criteria
    const mockResult: ValidationResult = {
      valid: false,
      errors: [
        {
          field: "hyperparameters.tensor_parallel_size",
          message: "Tensor parallel size (3) must divide evenly into GPUs per node (8). Valid values: 1, 2, 4, or 8.",
          severity: "error",
        },
      ],
      warnings: [],
    };
    vi.mocked(api.validateConfig).mockResolvedValue(mockResult);

    // Set tensor_parallel_size to 3 (invalid with 8 GPUs)
    const store = useConfigStore.getState();
    store.setHyperparameter("tensor_parallel_size", 3);
    store.setClusterConfig("gpus_per_node", 8);

    render(<ValidationDisplay />);

    await waitFor(() => {
      expect(screen.getByText(/must divide evenly/)).toBeInTheDocument();
    }, { timeout: 2000 });
  });
});
