import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DatasetSelector } from "./DatasetSelector";
import { useConfigStore } from "../../store/configStore";

describe("DatasetSelector", () => {
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

  it("renders the dataset selector with header", () => {
    render(<DatasetSelector />);
    expect(screen.getByText("Select Dataset")).toBeInTheDocument();
    expect(
      screen.getByText("Search HuggingFace Hub datasets or enter a local dataset path")
    ).toBeInTheDocument();
  });

  it("displays popular datasets", () => {
    render(<DatasetSelector />);
    expect(screen.getByText("OpenMathInstruct-2")).toBeInTheDocument();
    expect(screen.getByText("HelpSteer2")).toBeInTheDocument();
    expect(screen.getByText("Alpaca")).toBeInTheDocument();
    expect(screen.getByText("GSM8K")).toBeInTheDocument();
  });

  it("filters datasets based on search query", () => {
    render(<DatasetSelector />);
    const searchInput = screen.getByPlaceholderText(
      "Search datasets (e.g., 'OpenMathInstruct', 'gsm8k')..."
    );

    fireEvent.change(searchInput, { target: { value: "math" } });

    // Should show math-related datasets
    expect(screen.getByText("OpenMathInstruct-2")).toBeInTheDocument();
    expect(screen.getByText("MATH")).toBeInTheDocument();
    expect(screen.getByText("GSM8K")).toBeInTheDocument();

    // Should not show non-math datasets
    expect(screen.queryByText("Alpaca")).not.toBeInTheDocument();
    expect(screen.queryByText("GitHub Code")).not.toBeInTheDocument();
  });

  it("shows no results message when search has no matches", () => {
    render(<DatasetSelector />);
    const searchInput = screen.getByPlaceholderText(
      "Search datasets (e.g., 'OpenMathInstruct', 'gsm8k')..."
    );

    fireEvent.change(searchInput, { target: { value: "nonexistent_dataset_xyz" } });

    expect(
      screen.getByText("No datasets found. Try a different search term or use a local path.")
    ).toBeInTheDocument();
  });

  it("updates store when selecting a dataset", () => {
    render(<DatasetSelector />);

    // Click on GSM8K
    const gsm8kCard = screen.getByText("GSM8K").closest("button");
    expect(gsm8kCard).toBeInTheDocument();
    fireEvent.click(gsm8kCard!);

    // Verify store updated
    expect(useConfigStore.getState().config.dataset).toBe("openai/gsm8k");
  });

  it("shows selected state for the current dataset", () => {
    render(<DatasetSelector />);

    // OpenMathInstruct-2 should be selected by default
    const selectedBadges = screen.getAllByText("Selected");
    expect(selectedBadges.length).toBeGreaterThan(0);
  });

  it("allows entering a local path", () => {
    render(<DatasetSelector />);

    const pathInput = screen.getByPlaceholderText(
      "/path/to/dataset or organization/dataset-name"
    );
    fireEvent.change(pathInput, { target: { value: "/data/my-dataset" } });

    const usePathButton = screen.getByText("Use Path");
    fireEvent.click(usePathButton);

    expect(useConfigStore.getState().config.dataset).toBe("/data/my-dataset");
  });

  it("disables Use Path button when input is empty", () => {
    render(<DatasetSelector />);

    const usePathButton = screen.getByText("Use Path");
    expect(usePathButton).toBeDisabled();
  });

  it("displays dataset features", () => {
    render(<DatasetSelector />);

    // OpenMathInstruct-2 features (may appear multiple times in card and info box)
    expect(screen.getAllByText("problem").length).toBeGreaterThan(0);
    expect(screen.getAllByText("solution").length).toBeGreaterThan(0);
    expect(screen.getAllByText("answer").length).toBeGreaterThan(0);
  });

  it("displays dataset size and download count", () => {
    render(<DatasetSelector />);

    expect(screen.getByText("14.5M examples")).toBeInTheDocument();
    expect(screen.getByText("50,000 downloads")).toBeInTheDocument();
  });

  it("displays current selection info box", () => {
    render(<DatasetSelector />);

    expect(screen.getByText("Selected dataset:")).toBeInTheDocument();
    expect(screen.getByText("nvidia/OpenMathInstruct-2")).toBeInTheDocument();
    expect(screen.getByText("View on HF")).toBeInTheDocument();
  });

  it("searches by description as well", () => {
    render(<DatasetSelector />);
    const searchInput = screen.getByPlaceholderText(
      "Search datasets (e.g., 'OpenMathInstruct', 'gsm8k')..."
    );

    // Search for "reward" which is in HelpSteer2's description
    fireEvent.change(searchInput, { target: { value: "reward" } });

    expect(screen.getByText("HelpSteer2")).toBeInTheDocument();
  });

  it("shows 'OpenMathInstruct' when searching for that term", () => {
    render(<DatasetSelector />);
    const searchInput = screen.getByPlaceholderText(
      "Search datasets (e.g., 'OpenMathInstruct', 'gsm8k')..."
    );

    fireEvent.change(searchInput, { target: { value: "OpenMathInstruct" } });

    expect(screen.getByText("OpenMathInstruct-2")).toBeInTheDocument();
  });
});
