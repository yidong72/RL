import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TestHarness, executeRewardFunction, simulateRewardFunction, DEFAULT_TEST_CASES } from "./TestHarness";

describe("TestHarness", () => {
  const mockOnClose = vi.fn();
  const sampleCode = `def reward_function(prompt: str, response: str) -> float:
    """Simple reward function"""
    return 1.0 if len(response) > 10 else 0.0
`;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Modal behavior", () => {
    it("should not render when closed", () => {
      render(
        <TestHarness isOpen={false} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.queryByText("Reward Function Test Harness")).not.toBeInTheDocument();
    });

    it("should render when open", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByText("Reward Function Test Harness")).toBeInTheDocument();
    });

    it("should call onClose when close button is clicked", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const closeButton = screen.getByLabelText("Close test harness");
      fireEvent.click(closeButton);
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should call onClose when backdrop is clicked", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const backdrop = document.querySelector('[aria-hidden="true"]');
      fireEvent.click(backdrop!);
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe("Input fields", () => {
    it("should render prompt input", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByLabelText(/Prompt/)).toBeInTheDocument();
    });

    it("should render response input", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByLabelText(/Model Response/)).toBeInTheDocument();
    });

    it("should allow typing in prompt field", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const promptInput = screen.getByLabelText(/Prompt/);
      fireEvent.change(promptInput, { target: { value: "Test prompt" } });
      expect(promptInput).toHaveValue("Test prompt");
    });

    it("should allow typing in response field", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const responseInput = screen.getByLabelText(/Model Response/);
      fireEvent.change(responseInput, { target: { value: "Test response" } });
      expect(responseInput).toHaveValue("Test response");
    });
  });

  describe("Preset test cases", () => {
    it("should show preset test cases", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByText("Quick Test Presets")).toBeInTheDocument();
    });

    it("should load preset when clicked", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const presetButtons = screen.getAllByRole("button");
      // Find the first preset button (contains truncated prompt text)
      const presetButton = presetButtons.find(btn => 
        btn.textContent?.includes("What is 2 + 2")
      );
      fireEvent.click(presetButton!);
      
      const promptInput = screen.getByLabelText(/Prompt/);
      expect(promptInput).toHaveValue(DEFAULT_TEST_CASES[0].prompt);
    });
  });

  describe("Run test button", () => {
    it("should show Run Test button", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByRole("button", { name: /Run Test/i })).toBeInTheDocument();
    });

    it("should disable Run Test when prompt is empty", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const runButton = screen.getByRole("button", { name: /Run Test/i });
      expect(runButton).toBeDisabled();
    });

    it("should show error when running test without inputs", async () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      
      // Fill only prompt
      const promptInput = screen.getByLabelText(/Prompt/);
      fireEvent.change(promptInput, { target: { value: "Test" } });
      
      // Button should still be disabled because response is empty
      const runButton = screen.getByRole("button", { name: /Run Test/i });
      expect(runButton).toBeDisabled();
    });

    it("should run test when both inputs are provided", async () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      
      // Fill both inputs
      const promptInput = screen.getByLabelText(/Prompt/);
      fireEvent.change(promptInput, { target: { value: "Test prompt" } });
      
      const responseInput = screen.getByLabelText(/Model Response/);
      fireEvent.change(responseInput, { target: { value: "Test response with enough length" } });
      
      // Run test
      const runButton = screen.getByRole("button", { name: /Run Test/i });
      expect(runButton).not.toBeDisabled();
      
      fireEvent.click(runButton);
      
      // Wait for result
      await waitFor(() => {
        expect(screen.getByRole("alert")).toBeInTheDocument();
      });
    });
  });

  describe("Clear button", () => {
    it("should show Clear button", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByRole("button", { name: /Clear/i })).toBeInTheDocument();
    });

    it("should clear inputs when clicked", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      
      // Fill inputs
      const promptInput = screen.getByLabelText(/Prompt/);
      fireEvent.change(promptInput, { target: { value: "Test prompt" } });
      
      const responseInput = screen.getByLabelText(/Model Response/);
      fireEvent.change(responseInput, { target: { value: "Test response" } });
      
      // Clear
      const clearButton = screen.getByRole("button", { name: /Clear/i });
      fireEvent.click(clearButton);
      
      expect(promptInput).toHaveValue("");
      expect(responseInput).toHaveValue("");
    });
  });

  describe("Accessibility", () => {
    it("should have proper aria attributes for modal", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      const modal = screen.getByRole("dialog");
      expect(modal).toHaveAttribute("aria-modal", "true");
      expect(modal).toHaveAttribute("aria-labelledby", "test-harness-title");
    });

    it("should have proper labels for inputs", () => {
      render(
        <TestHarness isOpen={true} onClose={mockOnClose} rewardCode={sampleCode} />
      );
      expect(screen.getByLabelText(/Prompt/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Model Response/)).toBeInTheDocument();
    });
  });
});

describe("executeRewardFunction", () => {
  it("should return error when no reward_function is defined", async () => {
    const result = await executeRewardFunction("# no function here", "prompt", "response");
    expect(result.success).toBe(false);
    expect(result.error).toContain("No 'reward_function' found");
  });

  it("should return success for valid reward function", async () => {
    const code = `def reward_function(prompt, response):
    return 1.0
`;
    const result = await executeRewardFunction(code, "test prompt", "test response");
    expect(result.success).toBe(true);
    expect(result.score).toBeDefined();
    expect(result.executionTime).toBeDefined();
  });
});

describe("simulateRewardFunction", () => {
  it("should return score for exact match pattern", () => {
    const code = "exact match comparison";
    const score = simulateRewardFunction(code, "prompt", "This is a longer response");
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should return score for length-based pattern", () => {
    const code = "word_count length based";
    const score = simulateRewardFunction(code, "prompt", "Short");
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should return score for keyword pattern", () => {
    const code = "keyword contains check";
    const score = simulateRewardFunction(code, "prompt", "therefore the solution is the answer");
    expect(score).toBeGreaterThan(0);
  });

  it("should return score for regex pattern", () => {
    const code = "re.search regex pattern";
    const score = simulateRewardFunction(code, "prompt", "The answer is 42");
    expect(score).toBe(1.0);
  });

  it("should return 0 for regex pattern without match", () => {
    const code = "re.search regex pattern";
    const score = simulateRewardFunction(code, "prompt", "No numbers here");
    expect(score).toBe(0.0);
  });
});

describe("DEFAULT_TEST_CASES", () => {
  it("should have at least 3 test cases", () => {
    expect(DEFAULT_TEST_CASES.length).toBeGreaterThanOrEqual(3);
  });

  it("should have required fields for all test cases", () => {
    DEFAULT_TEST_CASES.forEach(tc => {
      expect(tc.id).toBeDefined();
      expect(tc.prompt).toBeDefined();
      expect(tc.response).toBeDefined();
    });
  });

  it("should have expectedScore defined for test cases", () => {
    DEFAULT_TEST_CASES.forEach(tc => {
      expect(tc.expectedScore).toBeDefined();
      expect(tc.expectedScore).toBeGreaterThanOrEqual(0);
      expect(tc.expectedScore).toBeLessThanOrEqual(1);
    });
  });
});
