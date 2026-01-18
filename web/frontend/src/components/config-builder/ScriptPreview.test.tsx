import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ScriptPreview } from "./ScriptPreview";
import { useConfigStore } from "../../store/configStore";
import { api } from "../../api/client";

// Mock the API client
vi.mock("../../api/client", () => ({
  api: {
    generateScript: vi.fn(),
  },
}));

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = vi.fn(() => "blob:test-url");
global.URL.revokeObjectURL = vi.fn();

describe("ScriptPreview", () => {
  const mockScriptResult = {
    script: `#!/bin/bash
#SBATCH --job-name=nemo-rl-grpo-training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00

export CONTAINER="nvcr.io/nvidia/nemo-rl:24.12"

python -m nemo_rl.train --config config.yaml`,
    files: {
      "config.yaml": `# NeMo RL Training Configuration
algorithm: grpo
model:
  name: Qwen/Qwen2.5-1.5B
  backend: dtensor`,
      "ray.sub": `#!/bin/bash
# Ray cluster startup script`,
    },
    instructions: [
      "1. Save the files to your working directory",
      "2. Make scripts executable",
      "3. Submit job to SLURM",
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store to default state
    useConfigStore.setState({
      config: {
        algorithm: "grpo",
        model: "Qwen/Qwen2.5-1.5B",
        dataset: "nvidia/OpenMathInstruct-2",
        backend: "dtensor",
        hyperparameters: {
          learning_rate: 1e-6,
          batch_size: 32,
          max_steps: 1000,
          num_generations_per_prompt: 16,
          tensor_parallel_size: 1,
        },
        cluster: {
          nodes: 1,
          gpus_per_node: 8,
          time_limit: "4:00:00",
        },
      },
      validation: {
        valid: true,
        errors: [],
        warnings: [],
      },
    });
    
    // Mock the API response
    vi.mocked(api.generateScript).mockResolvedValue(mockScriptResult);
  });


  it("renders the component with header", async () => {
    render(<ScriptPreview />);
    
    expect(screen.getByText("SLURM Script Preview")).toBeInTheDocument();
  });

  it("displays tab navigation", async () => {
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText("train.sh")).toBeInTheDocument();
    });
    expect(screen.getByText("config.yaml")).toBeInTheDocument();
  });

  it("shows Copy and Download buttons", async () => {
    render(<ScriptPreview />);
    
    expect(screen.getByText("Copy")).toBeInTheDocument();
    expect(screen.getByText("Download")).toBeInTheDocument();
  });

  it("calls generateScript API with debounce", async () => {
    render(<ScriptPreview />);
    
    // Wait for the debounced API call to happen
    await waitFor(() => {
      expect(api.generateScript).toHaveBeenCalled();
    }, { timeout: 1000 });
  });

  it("displays generated script content", async () => {
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText(/nemo-rl-grpo-training/)).toBeInTheDocument();
    });
  });

  it("switches tabs correctly", async () => {
    const user = userEvent.setup();
    render(<ScriptPreview />);
    
    // Wait for script to load
    await waitFor(() => {
      expect(screen.getByText(/nemo-rl-grpo-training/)).toBeInTheDocument();
    });
    
    // Click config.yaml tab
    await user.click(screen.getByText("config.yaml"));
    
    await waitFor(() => {
      expect(screen.getByText(/NeMo RL Training Configuration/)).toBeInTheDocument();
    });
  });

  it("copies script to clipboard and shows feedback", async () => {
    const user = userEvent.setup();
    render(<ScriptPreview />);
    
    // Wait for script to load
    await waitFor(() => {
      expect(screen.getByText(/nemo-rl-grpo-training/)).toBeInTheDocument();
    });
    
    // The copy button should be visible
    expect(screen.getByText("Copy")).toBeInTheDocument();
    
    // Click copy button
    await user.click(screen.getByText("Copy"));
    
    // Should show "Copied!" feedback
    await waitFor(() => {
      expect(screen.getByText("Copied!")).toBeInTheDocument();
    });
  });


  it("shows custom SLURM parameters section", async () => {
    const user = userEvent.setup();
    render(<ScriptPreview />);
    
    // Click to expand custom params
    await user.click(screen.getByText(/Custom SLURM Parameters/));
    
    expect(screen.getByText("Add Custom Parameter")).toBeInTheDocument();
  });

  it("adds custom SLURM parameters", async () => {
    const user = userEvent.setup();
    render(<ScriptPreview />);
    
    // Expand custom params
    await user.click(screen.getByText(/Custom SLURM Parameters/));
    
    // Add a custom param
    await user.click(screen.getByText("Add Custom Parameter"));
    
    // Should see input fields for name and value
    const inputs = screen.getAllByRole("textbox");
    expect(inputs.length).toBeGreaterThan(0);
  });

  it("shows instructions", async () => {
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText("Quick Start Instructions")).toBeInTheDocument();
    });
  });

  it("shows permissions hint", async () => {
    render(<ScriptPreview />);
    
    expect(screen.getByText(/make scripts executable/)).toBeInTheDocument();
  });

  it("disables download when validation fails", async () => {
    useConfigStore.setState({
      validation: {
        valid: false,
        errors: [{ field: "model", message: "Invalid", severity: "error" }],
        warnings: [],
      },
    });
    
    render(<ScriptPreview />);
    
    const downloadButton = screen.getByText("Download").closest("button");
    expect(downloadButton).toBeDisabled();
  });

  it("regenerates script when config changes", async () => {
    render(<ScriptPreview />);
    
    // Wait for initial generation
    await waitFor(() => {
      expect(api.generateScript).toHaveBeenCalledTimes(1);
    }, { timeout: 1000 });
    
    // Update config
    act(() => {
      useConfigStore.getState().setAlgorithm("sft");
    });
    
    // Wait for regeneration
    await waitFor(() => {
      expect(api.generateScript).toHaveBeenCalledTimes(2);
    }, { timeout: 1000 });
  });

  it("shows ray.sub tab for multi-node config", async () => {
    useConfigStore.setState({
      config: {
        ...useConfigStore.getState().config,
        cluster: {
          ...useConfigStore.getState().config.cluster,
          nodes: 2,
        },
      },
    });
    
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText("ray.sub")).toBeInTheDocument();
    });
  });

  it("shows Download All Files button", async () => {
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText("Download All Files")).toBeInTheDocument();
    });
  });

  it("handles API errors gracefully", async () => {
    vi.mocked(api.generateScript).mockRejectedValue(new Error("Network error"));
    
    render(<ScriptPreview />);
    
    await waitFor(() => {
      expect(screen.getByText(/Unable to generate script/)).toBeInTheDocument();
    });
  });
});
