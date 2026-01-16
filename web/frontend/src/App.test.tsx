import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import App from "./App";
import { useConfigStore } from "./store/configStore";

// Mock the config store
vi.mock("./store/configStore", () => ({
  useConfigStore: vi.fn(),
}));

// Mock Monaco Editor
vi.mock("@monaco-editor/react", () => ({
  default: ({ value, onChange }: { value: string; onChange?: (val: string) => void }) => (
    <textarea
      data-testid="monaco-editor"
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
    />
  ),
}));

// Mock resize observer
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = ResizeObserverMock;

describe("App - Responsive Design", () => {
  const mockSetAlgorithm = vi.fn();
  const mockSetModel = vi.fn();
  const mockSetDataset = vi.fn();
  const mockSetBackend = vi.fn();
  const mockSetHyperparameter = vi.fn();
  const mockSetClusterConfig = vi.fn();
  const mockSetConfig = vi.fn();
  const mockToggleDarkMode = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
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
      isDarkMode: false,
      setAlgorithm: mockSetAlgorithm,
      setModel: mockSetModel,
      setDataset: mockSetDataset,
      setBackend: mockSetBackend,
      setHyperparameter: mockSetHyperparameter,
      setClusterConfig: mockSetClusterConfig,
      setConfig: mockSetConfig,
      toggleDarkMode: mockToggleDarkMode,
      validation: null,
      isValidating: false,
      setValidation: vi.fn(),
      setIsValidating: vi.fn(),
    });
  });

  describe("Header", () => {
    it("should render the header with logo and title", () => {
      render(<App />);
      expect(screen.getByText("NR")).toBeInTheDocument();
      // Both full and short title should be present (hidden/shown based on screen size)
      expect(screen.getByText("NeMo RL Configurator")).toBeInTheDocument();
      expect(screen.getByText("NeMo RL")).toBeInTheDocument();
    });

    it("should have editor mode toggle (Form/YAML)", () => {
      render(<App />);
      expect(screen.getAllByText("Form").length).toBeGreaterThan(0);
      expect(screen.getAllByText("YAML").length).toBeGreaterThan(0);
    });

    it("should have Import and Templates buttons", () => {
      render(<App />);
      // These appear in both desktop nav and mobile menu
      expect(screen.getAllByText("Import").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Templates").length).toBeGreaterThanOrEqual(1);
    });

    it("should have mobile menu button", () => {
      render(<App />);
      expect(screen.getByLabelText("Open menu")).toBeInTheDocument();
    });
  });

  describe("Mobile Menu", () => {
    it("should open mobile menu when clicking menu button", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      expect(screen.getByText("Menu")).toBeInTheDocument();
      expect(screen.getByLabelText("Close menu")).toBeInTheDocument();
    });

    it("should close mobile menu when clicking close button", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      const closeButton = screen.getByLabelText("Close menu");
      fireEvent.click(closeButton);
      
      expect(screen.queryByText("Menu")).not.toBeInTheDocument();
    });

    it("should have editor mode toggle in mobile menu", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      expect(screen.getByText("Editor Mode")).toBeInTheDocument();
    });

    it("should have Import Config option in mobile menu", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      expect(screen.getByText("Import Config")).toBeInTheDocument();
    });

    it("should have theme toggle in mobile menu", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      expect(screen.getByText("Dark Mode")).toBeInTheDocument();
    });

    it("should toggle theme when clicking theme option", () => {
      render(<App />);
      const menuButton = screen.getByLabelText("Open menu");
      fireEvent.click(menuButton);
      
      const themeOption = screen.getByText("Dark Mode");
      fireEvent.click(themeOption);
      
      expect(mockToggleDarkMode).toHaveBeenCalled();
    });
  });

  describe("Mobile View Tabs", () => {
    it("should render mobile view tabs", () => {
      render(<App />);
      expect(screen.getByLabelText("Configuration panel")).toBeInTheDocument();
      expect(screen.getByLabelText("Preview panel")).toBeInTheDocument();
    });

    it("should show Configure and Preview tab labels", () => {
      render(<App />);
      expect(screen.getByText("Configure")).toBeInTheDocument();
      expect(screen.getByText("Preview")).toBeInTheDocument();
    });

    it("should switch to preview view when clicking Preview tab", () => {
      render(<App />);
      const previewTab = screen.getByLabelText("Preview panel");
      fireEvent.click(previewTab);
      
      // Preview should show validation and script preview
      // We can verify the tab is now active
      expect(previewTab).toHaveClass("text-primary");
    });

    it("should switch back to config view when clicking Configure tab", () => {
      render(<App />);
      
      // First switch to preview
      const previewTab = screen.getByLabelText("Preview panel");
      fireEvent.click(previewTab);
      
      // Then switch back to config
      const configTab = screen.getByLabelText("Configuration panel");
      fireEvent.click(configTab);
      
      expect(configTab).toHaveClass("text-primary");
    });
  });

  describe("Editor Mode Toggle", () => {
    it("should switch to YAML mode when clicking YAML button", () => {
      render(<App />);
      // Click the YAML button (there are multiple - one in desktop nav, one in mobile menu)
      const yamlButtons = screen.getAllByText("YAML");
      fireEvent.click(yamlButtons[0]);
      
      // Should show Monaco editor (may appear in both desktop and mobile layouts)
      const editors = screen.getAllByTestId("monaco-editor");
      expect(editors.length).toBeGreaterThan(0);
    });
  });

  describe("Import Modal", () => {
    it("should open import modal when clicking Import", () => {
      render(<App />);
      // Click Import in the navigation
      const importButtons = screen.getAllByText("Import");
      fireEvent.click(importButtons[0]);
      
      // Should show import modal content (title appears in the modal)
      const importTitles = screen.getAllByText(/Import/i);
      expect(importTitles.length).toBeGreaterThan(0);
    });
  });

  describe("Template Library Modal", () => {
    it("should open template library when clicking Templates", () => {
      render(<App />);
      const templatesButtons = screen.getAllByText("Templates");
      fireEvent.click(templatesButtons[0]);
      
      // Should show template library
      expect(screen.getByText("Template Library")).toBeInTheDocument();
    });
  });

  describe("Main Layout", () => {
    it("should render configuration components", () => {
      render(<App />);
      // Algorithm selector appears in both desktop and mobile layouts
      const algorithmHeaders = screen.getAllByText("Select Training Algorithm");
      expect(algorithmHeaders.length).toBeGreaterThan(0);
    });

    it("should have preview section with script preview", () => {
      render(<App />);
      // ScriptPreview should be rendered (appears in both desktop and mobile layouts)
      // It has a title "Generated Script" or similar
      const scriptPreviews = screen.getAllByText(/Script/i);
      expect(scriptPreviews.length).toBeGreaterThan(0);
    });
  });

  describe("Responsive breakpoints", () => {
    // Note: Testing actual CSS breakpoints would require a different approach
    // (e.g., using matchMedia mocks or visual regression testing)
    // Here we test that the responsive classes are applied correctly

    it("should have responsive grid classes for desktop layout", () => {
      const { container } = render(<App />);
      // Check for the presence of responsive grid classes
      const desktopGrid = container.querySelector('.lg\\:grid-cols-2');
      expect(desktopGrid).toBeInTheDocument();
    });

    it("should have hidden class for mobile bottom tabs on desktop", () => {
      const { container } = render(<App />);
      // Mobile tabs should have md:hidden class equivalent
      const bottomTabs = container.querySelector('.md\\:hidden.fixed.bottom-0');
      expect(bottomTabs).toBeInTheDocument();
    });

    it("should have hidden class for footer on mobile", () => {
      const { container } = render(<App />);
      // Footer should be hidden on mobile
      const footer = container.querySelector('footer.hidden.md\\:block');
      expect(footer).toBeInTheDocument();
    });
  });
});

describe("App - Dark Mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useConfigStore as ReturnType<typeof vi.fn>).mockReturnValue({
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
      isDarkMode: true,
      setAlgorithm: vi.fn(),
      setModel: vi.fn(),
      setDataset: vi.fn(),
      setBackend: vi.fn(),
      setHyperparameter: vi.fn(),
      setClusterConfig: vi.fn(),
      setConfig: vi.fn(),
      toggleDarkMode: vi.fn(),
      validation: null,
      isValidating: false,
      setValidation: vi.fn(),
      setIsValidating: vi.fn(),
    });
  });

  it("should show Light Mode option in mobile menu when in dark mode", () => {
    render(<App />);
    const menuButton = screen.getByLabelText("Open menu");
    fireEvent.click(menuButton);
    
    expect(screen.getByText("Light Mode")).toBeInTheDocument();
  });
});
