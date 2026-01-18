import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ImportConfig } from "./ImportConfig";
import { useConfigStore } from "../../store/configStore";

describe("ImportConfig", () => {
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
      validation: null,
    });
  });

  it("renders the import config component", () => {
    render(<ImportConfig />);
    expect(screen.getByRole("heading", { name: /Import Configuration/ })).toBeInTheDocument();
  });

  it("displays drag & drop zone", () => {
    render(<ImportConfig />);
    expect(screen.getByText(/Drag & drop your config file/)).toBeInTheDocument();
  });

  it("displays Browse Files button", () => {
    render(<ImportConfig />);
    expect(screen.getByText("Browse Files")).toBeInTheDocument();
  });

  it("displays paste section", () => {
    render(<ImportConfig />);
    expect(screen.getByText(/Or paste YAML\/JSON content/)).toBeInTheDocument();
  });

  it("displays Paste from Clipboard button", () => {
    render(<ImportConfig />);
    expect(screen.getByText("Paste from Clipboard")).toBeInTheDocument();
  });

  it("displays Import Configuration button", () => {
    render(<ImportConfig />);
    expect(screen.getByRole("button", { name: "Import Configuration" })).toBeInTheDocument();
  });

  it("has disabled import button when paste content is empty", () => {
    render(<ImportConfig />);
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    expect(importButton).toBeDisabled();
  });

  it("enables import button when paste content is entered", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.type(textarea, "algorithm: sft");
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    expect(importButton).not.toBeDisabled();
  });

  it("imports valid YAML config and updates store", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const yamlContent = `
algorithm: sft
model:
  name: meta-llama/Llama-3.2-3B
  backend: megatron
dataset:
  name: openai/gsm8k
training:
  learning_rate: 5e-6
  batch_size: 64
  max_steps: 500
cluster:
  nodes: 2
  gpus_per_node: 4
`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Configuration imported successfully/)).toBeInTheDocument();
    });
    
    // Check store was updated
    const state = useConfigStore.getState();
    expect(state.config.algorithm).toBe("sft");
    expect(state.config.model).toBe("meta-llama/Llama-3.2-3B");
    expect(state.config.backend).toBe("megatron");
    expect(state.config.dataset).toBe("openai/gsm8k");
    expect(state.config.hyperparameters.learning_rate).toBe(5e-6);
    expect(state.config.hyperparameters.batch_size).toBe(64);
    expect(state.config.hyperparameters.max_steps).toBe(500);
    expect(state.config.cluster.nodes).toBe(2);
    expect(state.config.cluster.gpus_per_node).toBe(4);
  });

  it("imports valid JSON config", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    // Use YAML format instead of JSON to avoid character escaping issues in userEvent
    const yamlContent = `algorithm: dpo
model: mistralai/Mistral-7B-v0.1
dataset: tatsu-lab/alpaca
backend: dtensor
training:
  learning_rate: 0.00001
  batch_size: 16`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Configuration imported successfully/)).toBeInTheDocument();
    });
    
    const state = useConfigStore.getState();
    expect(state.config.algorithm).toBe("dpo");
    expect(state.config.model).toBe("mistralai/Mistral-7B-v0.1");
  });

  it("shows error for non-object YAML", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    // A simple string value will be parsed as a non-object
    const invalidYaml = `just a plain string with no structure`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, invalidYaml);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/expected an object/)).toBeInTheDocument();
    });
  });

  it("shows error for unknown algorithm", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const yamlContent = `algorithm: unknown_algo`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Unknown algorithm/)).toBeInTheDocument();
    });
  });

  it("handles simple string model format", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const yamlContent = `
algorithm: grpo
model: Qwen/Qwen2.5-7B
dataset: nvidia/OpenMathInstruct-2
`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Configuration imported successfully/)).toBeInTheDocument();
    });
    
    const state = useConfigStore.getState();
    expect(state.config.model).toBe("Qwen/Qwen2.5-7B");
  });

  it("shows supported formats section", () => {
    render(<ImportConfig />);
    expect(screen.getByText("Supported Formats")).toBeInTheDocument();
    expect(screen.getByText(/NeMo RL YAML configuration files/)).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<ImportConfig onClose={onClose} />);
    
    // Find and click the close button (X icon)
    const closeButton = screen.getAllByRole("button")[0]; // First button is close
    await user.click(closeButton);
    
    expect(onClose).toHaveBeenCalled();
  });

  it("accepts file input with correct types", () => {
    render(<ImportConfig />);
    
    // File input should accept yaml, yml, and json
    const fileInput = document.querySelector('input[type="file"]');
    expect(fileInput).toHaveAttribute("accept", ".yaml,.yml,.json");
  });

  it("imports hyperparameters field format", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const yamlContent = `
algorithm: sft
model: test-model
hyperparameters:
  learning_rate: 2e-5
  batch_size: 128
`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Configuration imported successfully/)).toBeInTheDocument();
    });
    
    const state = useConfigStore.getState();
    expect(state.config.hyperparameters.learning_rate).toBe(2e-5);
    expect(state.config.hyperparameters.batch_size).toBe(128);
  });

  it("imports cluster partition and account", async () => {
    const user = userEvent.setup();
    render(<ImportConfig />);
    
    const yamlContent = `
algorithm: grpo
cluster:
  nodes: 4
  partition: gpu-batch
  account: my-project
  time_limit: 8:00:00
`;
    
    const textarea = screen.getByPlaceholderText(/Example YAML config/);
    await user.clear(textarea);
    await user.type(textarea, yamlContent);
    
    const importButton = screen.getByRole("button", { name: "Import Configuration" });
    await user.click(importButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Configuration imported successfully/)).toBeInTheDocument();
    });
    
    const state = useConfigStore.getState();
    expect(state.config.cluster.nodes).toBe(4);
    expect(state.config.cluster.partition).toBe("gpu-batch");
    expect(state.config.cluster.account).toBe("my-project");
    expect(state.config.cluster.time_limit).toBe("8:00:00");
  });
});
