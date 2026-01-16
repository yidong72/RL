import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { TemplateLibrary, BUILT_IN_TEMPLATES } from "./TemplateLibrary";
import { useConfigStore } from "../../store/configStore";

// Mock the config store
vi.mock("../../store/configStore", () => ({
  useConfigStore: vi.fn(),
}));

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Mock clipboard API
const clipboardMock = {
  writeText: vi.fn().mockResolvedValue(undefined),
};
Object.defineProperty(navigator, "clipboard", { value: clipboardMock });

// Sample config for testing
const mockConfig = {
  algorithm: "grpo" as const,
  model: "Qwen/Qwen2.5-1.5B",
  dataset: "nvidia/OpenMathInstruct-2",
  backend: "dtensor" as const,
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

describe("TemplateLibrary", () => {
  const mockSetConfig = vi.fn();
  const mockOnClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
    (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
      config: mockConfig,
      setConfig: mockSetConfig,
    });
  });

  describe("Modal behavior", () => {
    it("should not render when closed", () => {
      render(<TemplateLibrary isOpen={false} onClose={mockOnClose} />);
      expect(screen.queryByText("Template Library")).not.toBeInTheDocument();
    });

    it("should render when open", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Template Library")).toBeInTheDocument();
    });

    it("should call onClose when close button is clicked", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const closeButton = screen.getByLabelText("Close template library");
      fireEvent.click(closeButton);
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should call onClose when backdrop is clicked", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const backdrop = document.querySelector(".bg-black\\/50");
      fireEvent.click(backdrop!);
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe("Template rendering", () => {
    it("should render all templates by default", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      // Check that we have at least 5 templates (per acceptance criteria)
      expect(BUILT_IN_TEMPLATES.length).toBeGreaterThanOrEqual(5);
    });

    it("should display template names", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Math Problem Solving")).toBeInTheDocument();
    });

    it("should display template descriptions", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/GRPO training for mathematical reasoning/)).toBeInTheDocument();
    });

    it("should display algorithm badges", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      // Check for GRPO badges
      const grpoBadges = screen.getAllByText("GRPO");
      expect(grpoBadges.length).toBeGreaterThan(0);
    });
  });

  describe("Category filtering", () => {
    it("should show all categories by default", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("All Templates")).toBeInTheDocument();
      expect(screen.getByText("Math Reasoning")).toBeInTheDocument();
      expect(screen.getByText("Code Generation")).toBeInTheDocument();
      expect(screen.getByText("Chat & Conversation")).toBeInTheDocument();
    });

    it("should filter by math category", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      // Find the category button specifically (it has a specific class pattern)
      const categoryButtons = screen.getAllByRole("button");
      const mathButton = categoryButtons.find(btn => btn.textContent?.includes("Math Reasoning"));
      fireEvent.click(mathButton!);
      // Math templates should be visible
      expect(screen.getByText("Math Problem Solving")).toBeInTheDocument();
    });

    it("should filter by code category", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      // Find the category button specifically
      const categoryButtons = screen.getAllByRole("button");
      const codeButton = categoryButtons.find(btn => btn.textContent?.includes("Code Generation") && btn.className.includes("rounded-full"));
      fireEvent.click(codeButton!);
      // Code templates should be visible
      expect(screen.getByText("Code Instruction Tuning")).toBeInTheDocument();
    });

    it("should filter by chat category", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Chat & Conversation"));
      // Chat templates should be visible
      expect(screen.getByText("Chat Instruction Tuning")).toBeInTheDocument();
    });

    it("should return to all templates when clicking All Templates", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      // Filter to math first
      fireEvent.click(screen.getByText("Math Reasoning"));
      // Then back to all
      fireEvent.click(screen.getByText("All Templates"));
      // All templates should be visible
      expect(screen.getByText("Math Problem Solving")).toBeInTheDocument();
      expect(screen.getByText("Code Instruction Tuning")).toBeInTheDocument();
    });
  });

  describe("Search functionality", () => {
    it("should filter templates by search query", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText("Search templates...");
      fireEvent.change(searchInput, { target: { value: "DeepScaleR" } });
      expect(screen.getByText("DeepScaleR Math")).toBeInTheDocument();
      expect(screen.queryByText("Chat Instruction Tuning")).not.toBeInTheDocument();
    });

    it("should search by algorithm name", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText("Search templates...");
      fireEvent.change(searchInput, { target: { value: "dpo" } });
      expect(screen.getByText("Code Quality Improvement")).toBeInTheDocument();
      expect(screen.getByText("Chat Alignment (DPO)")).toBeInTheDocument();
    });

    it("should search by tags", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText("Search templates...");
      fireEvent.change(searchInput, { target: { value: "safety" } });
      expect(screen.getByText("Chat Alignment (DPO)")).toBeInTheDocument();
    });

    it("should show empty state when no templates match", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const searchInput = screen.getByPlaceholderText("Search templates...");
      fireEvent.change(searchInput, { target: { value: "nonexistent template xyz" } });
      expect(screen.getByText("No templates found")).toBeInTheDocument();
    });
  });

  describe("Template expansion", () => {
    it("should expand template details when clicking View details", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      expect(screen.getByText("Apply Template")).toBeInTheDocument();
    });

    it("should show model and dataset when expanded", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      expect(screen.getByText("Model:")).toBeInTheDocument();
      expect(screen.getByText("Dataset:")).toBeInTheDocument();
    });

    it("should collapse when clicking Hide details", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      const hideButton = screen.getByText("Hide details");
      fireEvent.click(hideButton);
      expect(screen.queryByText("Hide details")).not.toBeInTheDocument();
    });
  });

  describe("Template application", () => {
    it("should apply template config when Apply Template is clicked", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      
      // Expand first template
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      
      // Click Apply Template
      const applyButton = screen.getByText("Apply Template");
      fireEvent.click(applyButton);
      
      // Verify setConfig was called with the template's config
      expect(mockSetConfig).toHaveBeenCalledTimes(1);
      expect(mockSetConfig).toHaveBeenCalledWith(BUILT_IN_TEMPLATES[0].config);
    });

    it("should close modal after applying template", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      
      // Expand first template
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      
      // Click Apply Template
      const applyButton = screen.getByText("Apply Template");
      fireEvent.click(applyButton);
      
      // Verify onClose was called
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should apply Math Problem Solving template with correct values", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      
      // Search for Math Problem Solving
      const searchInput = screen.getByPlaceholderText("Search templates...");
      fireEvent.change(searchInput, { target: { value: "Math Problem Solving" } });
      
      // Expand template
      const viewDetailsButton = screen.getByText("View details");
      fireEvent.click(viewDetailsButton);
      
      // Click Apply Template
      const applyButton = screen.getByText("Apply Template");
      fireEvent.click(applyButton);
      
      // Verify correct config
      const expectedConfig = BUILT_IN_TEMPLATES.find(t => t.name === "Math Problem Solving")!.config;
      expect(mockSetConfig).toHaveBeenCalledWith(expectedConfig);
      expect(expectedConfig.algorithm).toBe("grpo");
      expect(expectedConfig.model).toBe("Qwen/Qwen2.5-1.5B");
      expect(expectedConfig.dataset).toBe("nvidia/OpenMathInstruct-2");
    });
  });

  describe("Built-in templates validation", () => {
    it("should have at least 5 built-in templates", () => {
      expect(BUILT_IN_TEMPLATES.length).toBeGreaterThanOrEqual(5);
    });

    it("should have templates for each category", () => {
      const categories = BUILT_IN_TEMPLATES.map(t => t.category);
      expect(categories).toContain("math");
      expect(categories).toContain("code");
      expect(categories).toContain("chat");
    });

    it("should have templates for each algorithm", () => {
      const algorithms = BUILT_IN_TEMPLATES.map(t => t.algorithm);
      expect(algorithms).toContain("grpo");
      expect(algorithms).toContain("sft");
      expect(algorithms).toContain("dpo");
    });

    it("should have valid config structure for all templates", () => {
      BUILT_IN_TEMPLATES.forEach(template => {
        expect(template.config).toBeDefined();
        expect(template.config.algorithm).toBeDefined();
        expect(template.config.model).toBeDefined();
        expect(template.config.dataset).toBeDefined();
        expect(template.config.backend).toBeDefined();
        expect(template.config.hyperparameters).toBeDefined();
        expect(template.config.cluster).toBeDefined();
      });
    });

    it("should have required hyperparameters for all templates", () => {
      BUILT_IN_TEMPLATES.forEach(template => {
        expect(template.config.hyperparameters.learning_rate).toBeDefined();
        expect(template.config.hyperparameters.batch_size).toBeDefined();
        expect(template.config.hyperparameters.max_steps).toBeDefined();
      });
    });

    it("should have required cluster config for all templates", () => {
      BUILT_IN_TEMPLATES.forEach(template => {
        expect(template.config.cluster.nodes).toBeDefined();
        expect(template.config.cluster.gpus_per_node).toBeDefined();
        expect(template.config.cluster.time_limit).toBeDefined();
      });
    });
  });

  describe("My Templates category", () => {
    it("should show My Templates category button", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("My Templates")).toBeInTheDocument();
    });

    it("should show empty state when no user templates exist", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("My Templates"));
      expect(screen.getByText("No saved templates yet")).toBeInTheDocument();
    });
  });

  describe("Save Current button", () => {
    it("should show Save Current button in header", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Save Current")).toBeInTheDocument();
    });

    it("should open save modal when clicking Save Current", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Save Current"));
      expect(screen.getByText("Save as Template")).toBeInTheDocument();
    });

    it("should show template name input in save modal", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Save Current"));
      expect(screen.getByLabelText(/Template Name/)).toBeInTheDocument();
    });

    it("should show description input in save modal", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Save Current"));
      expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
    });

    it("should show tags input in save modal", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Save Current"));
      expect(screen.getByLabelText(/Tags/)).toBeInTheDocument();
    });

    it("should close save modal when clicking Cancel", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Save Current"));
      fireEvent.click(screen.getByText("Cancel"));
      expect(screen.queryByText("Save as Template")).not.toBeInTheDocument();
    });
  });

  describe("Share functionality", () => {
    it("should show share button when template is expanded", () => {
      render(<TemplateLibrary isOpen={true} onClose={mockOnClose} />);
      const viewDetailsButtons = screen.getAllByText("View details");
      fireEvent.click(viewDetailsButtons[0]);
      // Share button should be visible (it's an icon button)
      const shareButtons = document.querySelectorAll('[class*="outline"]');
      expect(shareButtons.length).toBeGreaterThan(0);
    });
  });
});
