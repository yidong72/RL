import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HyperparameterForm } from "./HyperparameterForm";
import { useConfigStore } from "../../store/configStore";

describe("HyperparameterForm", () => {
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
          tensor_parallel_size: 1,
        },
        cluster: {
          nodes: 1,
          gpus_per_node: 8,
          time_limit: "4:00:00",
        },
      },
    });
  });

  it("renders the hyperparameter form with header", () => {
    render(<HyperparameterForm />);
    expect(screen.getByText("Hyperparameters")).toBeInTheDocument();
    expect(
      screen.getByText(/Configure training hyperparameters/i)
    ).toBeInTheDocument();
  });

  it("displays all common hyperparameters", () => {
    render(<HyperparameterForm />);
    expect(screen.getByText("Learning Rate")).toBeInTheDocument();
    expect(screen.getByText("Batch Size")).toBeInTheDocument();
    expect(screen.getByText("Max Steps")).toBeInTheDocument();
  });

  it("displays GRPO-specific parameters when algorithm is GRPO", () => {
    render(<HyperparameterForm />);
    // Check for the label element specifically
    expect(screen.getByLabelText("Generations per Prompt")).toBeInTheDocument();
  });

  it("hides GRPO-specific parameters when algorithm is SFT", () => {
    useConfigStore.setState({
      config: {
        ...useConfigStore.getState().config,
        algorithm: "sft",
      },
    });
    render(<HyperparameterForm />);
    expect(screen.queryByText("Generations per Prompt")).not.toBeInTheDocument();
  });

  it("hides GRPO-specific parameters when algorithm is DPO", () => {
    useConfigStore.setState({
      config: {
        ...useConfigStore.getState().config,
        algorithm: "dpo",
      },
    });
    render(<HyperparameterForm />);
    expect(screen.queryByText("Generations per Prompt")).not.toBeInTheDocument();
  });

  it("updates store when changing learning rate", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Learning Rate");

    fireEvent.change(input, { target: { value: "5e-6" } });

    expect(useConfigStore.getState().config.hyperparameters.learning_rate).toBe(5e-6);
  });

  it("updates store when changing batch size", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Batch Size");

    fireEvent.change(input, { target: { value: "64" } });

    expect(useConfigStore.getState().config.hyperparameters.batch_size).toBe(64);
  });

  it("updates store when changing max steps", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Max Steps");

    fireEvent.change(input, { target: { value: "5000" } });

    expect(useConfigStore.getState().config.hyperparameters.max_steps).toBe(5000);
  });

  it("updates store when changing generations per prompt", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Generations per Prompt");

    fireEvent.change(input, { target: { value: "8" } });

    expect(
      useConfigStore.getState().config.hyperparameters.num_generations_per_prompt
    ).toBe(8);
  });

  it("shows validation error for out-of-range learning rate", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Learning Rate");

    // Set value above max (1e-2)
    fireEvent.change(input, { target: { value: "0.1" } });

    expect(screen.getByText(/Maximum value is/i)).toBeInTheDocument();
  });

  it("shows validation error for negative batch size", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Batch Size");

    fireEvent.change(input, { target: { value: "0" } });

    expect(screen.getByText(/Minimum value is/i)).toBeInTheDocument();
  });

  it("shows validation error for batch size above max", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Batch Size");

    fireEvent.change(input, { target: { value: "2000" } });

    expect(screen.getByText(/Maximum value is/i)).toBeInTheDocument();
  });

  it("displays help tooltips for each parameter", () => {
    render(<HyperparameterForm />);

    // There should be help icons for each parameter
    const helpIcons = screen.getAllByLabelText("More information");
    expect(helpIcons.length).toBeGreaterThan(0);
  });

  it("shows validation summary as valid when all params are correct", () => {
    render(<HyperparameterForm />);
    expect(
      screen.getByText("All hyperparameters are valid")
    ).toBeInTheDocument();
  });

  it("shows validation summary as invalid when params have errors", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Batch Size");

    // Set invalid value
    fireEvent.change(input, { target: { value: "0" } });

    expect(
      screen.getByText("Some hyperparameters have validation errors")
    ).toBeInTheDocument();
  });

  it("displays range information for each parameter", () => {
    render(<HyperparameterForm />);

    // Check for range display
    expect(screen.getByText(/Range: 1 - 1,024/)).toBeInTheDocument(); // batch size
    expect(screen.getByText(/Range: 1 - 1,000,000/)).toBeInTheDocument(); // max steps
  });

  it("displays default values for each parameter", () => {
    render(<HyperparameterForm />);

    // Check for default display
    expect(screen.getByText(/Default: 32/)).toBeInTheDocument(); // batch size
    expect(screen.getByText(/Default: 1000/)).toBeInTheDocument(); // max steps
  });

  it("shows algorithm-specific tips for GRPO", () => {
    render(<HyperparameterForm />);
    expect(
      screen.getByText(/GRPO-specific/i)
    ).toBeInTheDocument();
  });

  it("shows algorithm-specific tips for SFT", () => {
    useConfigStore.setState({
      config: {
        ...useConfigStore.getState().config,
        algorithm: "sft",
      },
    });
    render(<HyperparameterForm />);
    expect(screen.getByText(/SFT tip/i)).toBeInTheDocument();
  });

  it("shows algorithm-specific tips for DPO", () => {
    useConfigStore.setState({
      config: {
        ...useConfigStore.getState().config,
        algorithm: "dpo",
      },
    });
    render(<HyperparameterForm />);
    expect(screen.getByText(/DPO tip/i)).toBeInTheDocument();
  });

  it("displays tensor parallel size field", () => {
    render(<HyperparameterForm />);
    expect(screen.getByText("Tensor Parallel Size")).toBeInTheDocument();
  });

  it("updates store when changing tensor parallel size", () => {
    render(<HyperparameterForm />);
    const input = screen.getByLabelText("Tensor Parallel Size");

    fireEvent.change(input, { target: { value: "4" } });

    expect(
      useConfigStore.getState().config.hyperparameters.tensor_parallel_size
    ).toBe(4);
  });
});
