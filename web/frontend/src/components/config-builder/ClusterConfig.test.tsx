import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ClusterConfig } from "./ClusterConfig";
import { useConfigStore } from "../../store/configStore";

describe("ClusterConfig", () => {
  beforeEach(() => {
    // Reset store state before each test
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
        },
        cluster: {
          nodes: 1,
          gpus_per_node: 8,
          time_limit: "4:00:00",
        },
      },
    });
  });

  it("renders the cluster config with header", () => {
    render(<ClusterConfig />);
    expect(screen.getByText("Cluster Configuration")).toBeInTheDocument();
    expect(
      screen.getByText("Configure compute resources and SLURM settings")
    ).toBeInTheDocument();
  });

  it("displays cluster preset selector", () => {
    render(<ClusterConfig />);
    expect(screen.getByText("Cluster Preset")).toBeInTheDocument();
    expect(screen.getByText("Generic SLURM")).toBeInTheDocument();
  });

  it("displays backend selector with both options", () => {
    render(<ClusterConfig />);
    expect(screen.getByText("Training Backend")).toBeInTheDocument();
    expect(screen.getByText("DTensor (FSDP2)")).toBeInTheDocument();
    expect(screen.getByText("Megatron-LM")).toBeInTheDocument();
  });

  it("displays resource configuration fields", () => {
    render(<ClusterConfig />);
    expect(screen.getByText("Number of Nodes")).toBeInTheDocument();
    expect(screen.getByText("GPUs per Node")).toBeInTheDocument();
    expect(screen.getByText("Time Limit")).toBeInTheDocument();
    expect(screen.getByText("Partition")).toBeInTheDocument();
    expect(screen.getByText("Account")).toBeInTheDocument();
  });

  it("shows DTensor as selected by default", () => {
    render(<ClusterConfig />);
    const dtensorButton = screen.getByText("DTensor (FSDP2)").closest("button");
    expect(dtensorButton).toHaveClass("border-primary");
  });

  it("updates backend when clicking Megatron", () => {
    render(<ClusterConfig />);
    const megatronButton = screen.getByText("Megatron-LM").closest("button");
    
    fireEvent.click(megatronButton!);

    expect(useConfigStore.getState().config.backend).toBe("megatron");
  });

  it("updates nodes when changing input", () => {
    render(<ClusterConfig />);
    const nodesInput = screen.getByLabelText(/Number of Nodes/i);

    fireEvent.change(nodesInput, { target: { value: "2" } });

    expect(useConfigStore.getState().config.cluster.nodes).toBe(2);
  });

  it("updates time limit when changing input", () => {
    render(<ClusterConfig />);
    const timeLimitInput = screen.getByLabelText(/Time Limit/i);

    fireEvent.change(timeLimitInput, { target: { value: "8:00:00" } });

    expect(useConfigStore.getState().config.cluster.time_limit).toBe("8:00:00");
  });

  it("updates partition when changing input", () => {
    render(<ClusterConfig />);
    const partitionInput = screen.getByLabelText(/Partition/i);

    fireEvent.change(partitionInput, { target: { value: "luna" } });

    expect(useConfigStore.getState().config.cluster.partition).toBe("luna");
  });

  it("updates account when changing input", () => {
    render(<ClusterConfig />);
    const accountInput = screen.getByLabelText(/Account/i);

    fireEvent.change(accountInput, { target: { value: "myproject" } });

    expect(useConfigStore.getState().config.cluster.account).toBe("myproject");
  });

  it("populates fields when selecting DGX Cloud preset", () => {
    render(<ClusterConfig />);

    // Open preset dropdown
    const presetButton = screen.getByText("Generic SLURM").closest("button");
    fireEvent.click(presetButton!);

    // Select DGX Cloud
    const dgxOption = screen.getByText("DGX Cloud");
    fireEvent.click(dgxOption);

    // Verify fields were populated
    const state = useConfigStore.getState().config;
    expect(state.cluster.partition).toBe("batch");
    expect(state.backend).toBe("dtensor");
  });

  it("populates fields when selecting BCM preset", () => {
    render(<ClusterConfig />);

    // Open preset dropdown
    const presetButton = screen.getByText("Generic SLURM").closest("button");
    fireEvent.click(presetButton!);

    // Select BCM
    const bcmOption = screen.getByText("BCM (Base Command Manager)");
    fireEvent.click(bcmOption);

    // Verify fields were populated
    const state = useConfigStore.getState().config;
    expect(state.cluster.partition).toBe("luna");
    expect(state.cluster.time_limit).toBe("8:00:00");
    expect(state.backend).toBe("megatron");
  });

  it("displays resource summary with total GPUs", () => {
    render(<ClusterConfig />);
    
    expect(screen.getByText("Resource Summary")).toBeInTheDocument();
    expect(screen.getByText("Total GPUs")).toBeInTheDocument();
    // Multiple "8" values exist (Total GPUs and GPUs/Node)
    const eightValues = screen.getAllByText("8");
    expect(eightValues.length).toBeGreaterThanOrEqual(2);
  });

  it("updates total GPUs when nodes change", () => {
    render(<ClusterConfig />);
    const nodesInput = screen.getByLabelText(/Number of Nodes/i);

    fireEvent.change(nodesInput, { target: { value: "2" } });

    // Should show 16 total GPUs (2 nodes * 8 gpus)
    expect(screen.getByText("16")).toBeInTheDocument();
  });

  it("displays backend hint for DTensor", () => {
    render(<ClusterConfig />);
    expect(
      screen.getByText(/Best for: Models up to 7B/i)
    ).toBeInTheDocument();
  });

  it("displays backend hint for Megatron", () => {
    render(<ClusterConfig />);
    expect(
      screen.getByText(/Best for: Models >7B/i)
    ).toBeInTheDocument();
  });

  it("displays help tooltips", () => {
    render(<ClusterConfig />);
    const helpIcons = screen.getAllByLabelText("More information");
    expect(helpIcons.length).toBeGreaterThan(0);
  });

  it("allows manual config changes after preset selection", () => {
    render(<ClusterConfig />);

    // Select DGX Cloud preset
    const presetButton = screen.getByText("Generic SLURM").closest("button");
    fireEvent.click(presetButton!);
    const dgxOption = screen.getByText("DGX Cloud");
    fireEvent.click(dgxOption);

    // Manually change values
    const nodesInput = screen.getByLabelText(/Number of Nodes/i);
    fireEvent.change(nodesInput, { target: { value: "4" } });

    const timeLimitInput = screen.getByLabelText(/Time Limit/i);
    fireEvent.change(timeLimitInput, { target: { value: "12:00:00" } });

    // Verify manual changes were applied
    const state = useConfigStore.getState().config;
    expect(state.cluster.nodes).toBe(4);
    expect(state.cluster.time_limit).toBe("12:00:00");
  });

  it("shows GPUs per node dropdown with options", () => {
    render(<ClusterConfig />);

    // Open GPUs per node dropdown
    const gpusButton = screen.getByText("8 GPUs").closest("button");
    fireEvent.click(gpusButton!);

    // Verify options are visible
    expect(screen.getByText("1 GPU")).toBeInTheDocument();
    expect(screen.getByText("2 GPUs")).toBeInTheDocument();
    expect(screen.getByText("4 GPUs")).toBeInTheDocument();
  });

  it("updates GPUs per node when selecting from dropdown", () => {
    render(<ClusterConfig />);

    // Open GPUs per node dropdown
    const gpusButton = screen.getByText("8 GPUs").closest("button");
    fireEvent.click(gpusButton!);

    // Select 4 GPUs
    const fourGpusOption = screen.getByText("4 GPUs");
    fireEvent.click(fourGpusOption);

    expect(useConfigStore.getState().config.cluster.gpus_per_node).toBe(4);
  });

  it("displays appropriate backend summary message for DTensor", () => {
    render(<ClusterConfig />);
    expect(
      screen.getByText(/Using DTensor \(FSDP2\) with 8 GPUs/i)
    ).toBeInTheDocument();
  });

  it("displays appropriate backend summary message for Megatron", () => {
    render(<ClusterConfig />);

    // Select Megatron backend
    const megatronButton = screen.getByText("Megatron-LM").closest("button");
    fireEvent.click(megatronButton!);

    expect(
      screen.getByText(/Using Megatron-LM with 8 GPUs/i)
    ).toBeInTheDocument();
  });
});
