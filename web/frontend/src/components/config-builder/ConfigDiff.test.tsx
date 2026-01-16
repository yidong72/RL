import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConfigDiff, computeDiff, getDiffStats, configToYaml } from "./ConfigDiff";
import { useConfigStore } from "../../store/configStore";
import type { TrainingConfig } from "../../types/config";

// Mock the config store
vi.mock("../../store/configStore", () => ({
  useConfigStore: vi.fn(),
}));

describe("ConfigDiff", () => {
  const mockOnClose = vi.fn();
  const mockConfig: TrainingConfig = {
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
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
      config: mockConfig,
    });
  });

  describe("Modal behavior", () => {
    it("should not render when closed", () => {
      render(<ConfigDiff isOpen={false} onClose={mockOnClose} />);
      expect(screen.queryByText("Config Diff Viewer")).not.toBeInTheDocument();
    });

    it("should render when open", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Config Diff Viewer")).toBeInTheDocument();
    });

    it("should call onClose when close button is clicked", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      const closeButton = screen.getByLabelText("Close diff viewer");
      fireEvent.click(closeButton);
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should call onClose when backdrop is clicked", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      const backdrop = document.querySelector(".bg-black\\/50");
      fireEvent.click(backdrop!);
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe("Compare source selection", () => {
    it("should default to template comparison", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // Find the Template button in the compare source selector group
      const buttons = screen.getAllByRole("button");
      const templateButton = buttons.find(btn => btn.textContent === "Template");
      expect(templateButton?.getAttribute("aria-pressed")).toBe("true");
    });

    it("should switch to custom YAML comparison", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      const customButton = screen.getByRole("button", { name: /custom yaml/i });
      fireEvent.click(customButton);
      expect(customButton.getAttribute("aria-pressed")).toBe("true");
      expect(screen.getByPlaceholderText("Paste YAML configuration to compare...")).toBeInTheDocument();
    });

    it("should hide custom YAML textarea when template is selected", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      expect(screen.queryByPlaceholderText("Paste YAML configuration to compare...")).not.toBeInTheDocument();
    });
  });

  describe("Template selection", () => {
    it("should display template selector when template source is selected", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // Should show the first template name by default
      expect(screen.getByText("Math Problem Solving")).toBeInTheDocument();
    });

    it("should open template dropdown when clicked", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // The template selector button has id="template-select"
      const templateSelector = screen.getByRole("button", { expanded: false });
      fireEvent.click(templateSelector);
      // Should show template options
      expect(screen.getByRole("listbox")).toBeInTheDocument();
    });

    it("should change selected template when option is clicked", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // Find the button with the template name inside it
      const templateSelector = document.getElementById("template-select");
      fireEvent.click(templateSelector!);
      
      // Click on a different template
      const option = screen.getByRole("option", { name: /deepscaler math/i });
      fireEvent.click(option);
      
      // Should show the new template name in the button
      expect(screen.getByText("DeepScaleR Math")).toBeInTheDocument();
    });
  });

  describe("Diff display", () => {
    it("should show side-by-side comparison headers", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/\(Base\)/)).toBeInTheDocument();
      expect(screen.getByText(/\(Changed\)/)).toBeInTheDocument();
    });

    it("should show diff stats", () => {
      // Render with different config to trigger changes
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: {
          ...mockConfig,
          model: "different-model",
          hyperparameters: {
            ...mockConfig.hyperparameters,
            batch_size: 64,
          },
        },
      });
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // Should show modification stats (use getAllByText since there are multiple)
      const modifiedElements = screen.getAllByText(/modified/i);
      expect(modifiedElements.length).toBeGreaterThan(0);
    });

    it("should show empty state when custom YAML is not provided", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      const customButton = screen.getByRole("button", { name: /custom yaml/i });
      fireEvent.click(customButton);
      expect(screen.getByText("No comparison data")).toBeInTheDocument();
    });
  });

  describe("Copy functionality", () => {
    it("should have a copy diff button", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByRole("button", { name: /copy diff/i })).toBeInTheDocument();
    });
  });

  describe("Color coding", () => {
    it("should display color legend in footer", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/green = added/i)).toBeInTheDocument();
      expect(screen.getByText(/red = removed/i)).toBeInTheDocument();
      expect(screen.getByText(/yellow = modified/i)).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("should have proper aria attributes for modal", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      const modal = screen.getByRole("dialog");
      expect(modal).toHaveAttribute("aria-modal", "true");
      expect(modal).toHaveAttribute("aria-labelledby", "config-diff-title");
    });

    it("should have proper aria attributes for compare source buttons", () => {
      render(<ConfigDiff isOpen={true} onClose={mockOnClose} />);
      // Find the Template button specifically
      const buttons = screen.getAllByRole("button");
      const templateButton = buttons.find(btn => btn.textContent === "Template");
      expect(templateButton).toHaveAttribute("aria-pressed");
    });
  });
});

describe("computeDiff", () => {
  it("should identify unchanged lines", () => {
    const left = "algorithm: grpo\nmodel: test";
    const right = "algorithm: grpo\nmodel: test";
    const diff = computeDiff(left, right);
    expect(diff.every(line => line.type === "unchanged")).toBe(true);
  });

  it("should identify added lines", () => {
    const left = "algorithm: grpo";
    const right = "algorithm: grpo\nmodel: test";
    const diff = computeDiff(left, right);
    const addedLines = diff.filter(line => line.type === "added");
    expect(addedLines.length).toBeGreaterThan(0);
  });

  it("should identify removed lines", () => {
    const left = "algorithm: grpo\nmodel: test";
    const right = "algorithm: grpo";
    const diff = computeDiff(left, right);
    const removedLines = diff.filter(line => line.type === "removed");
    expect(removedLines.length).toBeGreaterThan(0);
  });

  it("should identify modified lines with same key", () => {
    const left = "  batch_size: 32";
    const right = "  batch_size: 64";
    const diff = computeDiff(left, right);
    const modifiedLines = diff.filter(line => line.type === "modified");
    expect(modifiedLines.length).toBeGreaterThan(0);
  });

  it("should preserve line numbers", () => {
    const left = "line1\nline2\nline3";
    const right = "line1\nline2\nline3";
    const diff = computeDiff(left, right);
    expect(diff[0].lineNumber.left).toBe(1);
    expect(diff[0].lineNumber.right).toBe(1);
    expect(diff[2].lineNumber.left).toBe(3);
    expect(diff[2].lineNumber.right).toBe(3);
  });
});

describe("getDiffStats", () => {
  it("should count added lines", () => {
    const diff = [
      { type: "added" as const, lineNumber: { right: 1 }, content: { right: "test" } },
      { type: "added" as const, lineNumber: { right: 2 }, content: { right: "test2" } },
    ];
    const stats = getDiffStats(diff);
    expect(stats.added).toBe(2);
    expect(stats.removed).toBe(0);
    expect(stats.modified).toBe(0);
  });

  it("should count removed lines", () => {
    const diff = [
      { type: "removed" as const, lineNumber: { left: 1 }, content: { left: "test" } },
    ];
    const stats = getDiffStats(diff);
    expect(stats.removed).toBe(1);
  });

  it("should count modified lines", () => {
    const diff = [
      { type: "modified" as const, lineNumber: { left: 1, right: 1 }, content: { left: "old", right: "new" } },
      { type: "modified" as const, lineNumber: { left: 2, right: 2 }, content: { left: "old2", right: "new2" } },
    ];
    const stats = getDiffStats(diff);
    expect(stats.modified).toBe(2);
  });

  it("should return zero counts for empty diff", () => {
    const stats = getDiffStats([]);
    expect(stats.added).toBe(0);
    expect(stats.removed).toBe(0);
    expect(stats.modified).toBe(0);
  });
});

describe("configToYaml", () => {
  it("should convert config to valid YAML", () => {
    const config: TrainingConfig = {
      algorithm: "grpo",
      model: "test-model",
      dataset: "test-dataset",
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
    };
    const yaml = configToYaml(config);
    expect(yaml).toContain("algorithm: grpo");
    expect(yaml).toContain("name: test-model");
    expect(yaml).toContain("name: test-dataset");
    expect(yaml).toContain("batch_size: 32");
    expect(yaml).toContain("max_steps: 1000");
  });

  it("should include num_generations_per_prompt for GRPO", () => {
    const config: TrainingConfig = {
      algorithm: "grpo",
      model: "test-model",
      dataset: "test-dataset",
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
    };
    const yaml = configToYaml(config);
    expect(yaml).toContain("num_generations_per_prompt: 16");
  });

  it("should not include num_generations_per_prompt for SFT", () => {
    const config: TrainingConfig = {
      algorithm: "sft",
      model: "test-model",
      dataset: "test-dataset",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 1e-6,
        batch_size: 32,
        max_steps: 1000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 8,
        time_limit: "4:00:00",
      },
    };
    const yaml = configToYaml(config);
    expect(yaml).not.toContain("num_generations_per_prompt");
  });

  it("should include tensor_parallel_size when greater than 1", () => {
    const config: TrainingConfig = {
      algorithm: "grpo",
      model: "test-model",
      dataset: "test-dataset",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 1e-6,
        batch_size: 32,
        max_steps: 1000,
        num_generations_per_prompt: 16,
        tensor_parallel_size: 2,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 8,
        time_limit: "4:00:00",
      },
    };
    const yaml = configToYaml(config);
    expect(yaml).toContain("tensor_parallel_size: 2");
  });
});
