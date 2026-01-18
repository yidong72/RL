import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RewardEditor, REWARD_TEMPLATES, validatePythonCode } from "./RewardEditor";
import { useConfigStore } from "../../store/configStore";

// Mock the config store
vi.mock("../../store/configStore", () => ({
  useConfigStore: vi.fn(),
}));

// Mock Monaco Editor
vi.mock("@monaco-editor/react", () => ({
  default: ({ value, onChange }: { value: string; onChange: (val: string) => void }) => (
    <textarea
      data-testid="monaco-editor"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

describe("RewardEditor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Rendering based on algorithm", () => {
    it("should render when algorithm is GRPO", () => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "grpo" },
        isDarkMode: false,
      });
      render(<RewardEditor />);
      expect(screen.getByText("Reward Function Editor")).toBeInTheDocument();
    });

    it("should not render when algorithm is SFT", () => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "sft" },
        isDarkMode: false,
      });
      const { container } = render(<RewardEditor />);
      expect(container).toBeEmptyDOMElement();
    });

    it("should not render when algorithm is DPO", () => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "dpo" },
        isDarkMode: false,
      });
      const { container } = render(<RewardEditor />);
      expect(container).toBeEmptyDOMElement();
    });
  });

  describe("Editor rendering", () => {
    beforeEach(() => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "grpo" },
        isDarkMode: false,
      });
    });

    it("should render the Monaco editor", () => {
      render(<RewardEditor />);
      expect(screen.getByTestId("monaco-editor")).toBeInTheDocument();
    });

    it("should display validation status", () => {
      render(<RewardEditor />);
      expect(screen.getByText("Valid Python")).toBeInTheDocument();
    });

    it("should display template rewards section", () => {
      render(<RewardEditor />);
      expect(screen.getByText("Template Rewards")).toBeInTheDocument();
    });

    it("should display help text", () => {
      render(<RewardEditor />);
      expect(screen.getByText("How to use:")).toBeInTheDocument();
    });

    it("should display the file name header", () => {
      render(<RewardEditor />);
      expect(screen.getByText("reward_function.py")).toBeInTheDocument();
    });

    it("should have copy button", () => {
      render(<RewardEditor />);
      expect(screen.getByText("Copy")).toBeInTheDocument();
    });
  });

  describe("Template selection", () => {
    beforeEach(() => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "grpo" },
        isDarkMode: false,
      });
    });

    it("should expand templates when clicking header", () => {
      render(<RewardEditor />);
      const header = screen.getByText("Template Rewards");
      fireEvent.click(header);
      // Should show all template names
      expect(screen.getByText("Exact Match")).toBeInTheDocument();
      expect(screen.getByText("Regex Match")).toBeInTheDocument();
      expect(screen.getByText("Contains Keywords")).toBeInTheDocument();
    });

    it("should display template descriptions when expanded", () => {
      render(<RewardEditor />);
      fireEvent.click(screen.getByText("Template Rewards"));
      expect(screen.getByText(/Returns 1.0 if response exactly matches/)).toBeInTheDocument();
    });

    it("should insert template code when selecting a template", () => {
      render(<RewardEditor />);
      fireEvent.click(screen.getByText("Template Rewards"));
      fireEvent.click(screen.getByText("Regex Match"));
      
      const editor = screen.getByTestId("monaco-editor") as HTMLTextAreaElement;
      expect(editor.value).toContain("import re");
      expect(editor.value).toContain("pattern");
    });

    it("should close templates panel after selection", () => {
      render(<RewardEditor />);
      fireEvent.click(screen.getByText("Template Rewards"));
      expect(screen.getByText("Exact Match")).toBeInTheDocument();
      
      fireEvent.click(screen.getByText("Regex Match"));
      // The expanded templates should be hidden after selection
      expect(screen.queryByText(/Returns 1.0 if response matches a regex/)).not.toBeInTheDocument();
    });

    it("should show selected template name in header", () => {
      render(<RewardEditor />);
      expect(screen.getByText(/\(Exact Match\)/)).toBeInTheDocument();
      
      fireEvent.click(screen.getByText("Template Rewards"));
      fireEvent.click(screen.getByText("Length-Based"));
      
      expect(screen.getByText(/\(Length-Based\)/)).toBeInTheDocument();
    });
  });

  describe("Code editing", () => {
    beforeEach(() => {
      (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
        config: { algorithm: "grpo" },
        isDarkMode: false,
      });
    });

    it("should update code when editing", () => {
      render(<RewardEditor />);
      const editor = screen.getByTestId("monaco-editor") as HTMLTextAreaElement;
      
      fireEvent.change(editor, { target: { value: "def reward_function(): return 1.0" } });
      expect(editor.value).toBe("def reward_function(): return 1.0");
    });

    it("should validate code on change", () => {
      render(<RewardEditor />);
      const editor = screen.getByTestId("monaco-editor") as HTMLTextAreaElement;
      
      // Enter invalid code
      fireEvent.change(editor, { target: { value: "def bad_function(" } });
      
      // Should show validation error
      expect(screen.queryByText("Valid Python")).not.toBeInTheDocument();
    });
  });
});

describe("validatePythonCode", () => {
  describe("Valid code", () => {
    it("should validate correct reward function", () => {
      const code = `def reward_function(prompt: str, response: str) -> float:
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it("should validate code with imports", () => {
      const code = `import re

def reward_function(prompt: str, response: str) -> float:
    pattern = r'\\d+'
    return 1.0 if re.search(pattern, response) else 0.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(true);
    });

    it("should allow comments and docstrings", () => {
      const code = `# This is a comment
def reward_function(prompt: str, response: str) -> float:
    """Docstring here."""
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(true);
    });
  });

  describe("Syntax errors", () => {
    it("should detect unclosed parentheses", () => {
      const code = `def reward_function(prompt, response:
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.message.includes("Unclosed bracket"))).toBe(true);
    });

    it("should detect unclosed brackets", () => {
      const code = `def reward_function(prompt, response):
    data = [1, 2, 3
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(false);
    });

    it("should detect mismatched brackets", () => {
      const code = `def reward_function(prompt, response):
    data = [1, 2, 3)
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.message.includes("Mismatched"))).toBe(true);
    });

    it("should detect missing colon in function definition", () => {
      const code = `def reward_function(prompt, response)
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.message.includes(":"))).toBe(true);
    });

    it("should detect Python 2 print syntax", () => {
      const code = `def reward_function(prompt, response):
    print "debug"
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.message.includes("print()"))).toBe(true);
    });
  });

  describe("Warnings", () => {
    it("should warn when reward_function is not defined", () => {
      const code = `def my_reward(prompt, response):
    return 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.warnings.some(w => w.message.includes("reward_function"))).toBe(true);
    });

    it("should warn about missing return statement", () => {
      const code = `def reward_function(prompt, response):
    score = 1.0`;
      
      const result = validatePythonCode(code);
      expect(result.warnings.some(w => w.message.includes("return"))).toBe(true);
    });
  });
});

describe("REWARD_TEMPLATES", () => {
  it("should have at least 4 templates", () => {
    expect(REWARD_TEMPLATES.length).toBeGreaterThanOrEqual(4);
  });

  it("should include exact_match template", () => {
    const template = REWARD_TEMPLATES.find(t => t.id === "exact_match");
    expect(template).toBeDefined();
    expect(template?.name).toBe("Exact Match");
  });

  it("should include regex_match template", () => {
    const template = REWARD_TEMPLATES.find(t => t.id === "regex_match");
    expect(template).toBeDefined();
    expect(template?.name).toBe("Regex Match");
  });

  it("should include contains template", () => {
    const template = REWARD_TEMPLATES.find(t => t.id === "contains");
    expect(template).toBeDefined();
    expect(template?.name).toBe("Contains Keywords");
  });

  it("should include length_based template", () => {
    const template = REWARD_TEMPLATES.find(t => t.id === "length_based");
    expect(template).toBeDefined();
    expect(template?.name).toBe("Length-Based");
  });

  it("all templates should have valid code structure", () => {
    REWARD_TEMPLATES.forEach(template => {
      expect(template.code).toContain("def");
      expect(template.code).toContain("return");
    });
  });

  it("all templates should pass basic validation", () => {
    REWARD_TEMPLATES.forEach(template => {
      const result = validatePythonCode(template.code);
      if (!result.valid) {
        console.log(`Template "${template.id}" validation failed:`, result.errors);
      }
      expect(result.valid, `Template "${template.id}" should be valid`).toBe(true);
      expect(result.errors, `Template "${template.id}" should have no errors`).toHaveLength(0);
    });
  });

  it("all templates should have required fields", () => {
    REWARD_TEMPLATES.forEach(template => {
      expect(template.id).toBeDefined();
      expect(template.name).toBeDefined();
      expect(template.description).toBeDefined();
      expect(template.icon).toBeDefined();
      expect(template.code).toBeDefined();
    });
  });
});
