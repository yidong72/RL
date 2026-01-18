import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { 
  HelpModal, 
  TutorialModal, 
  HelpButton, 
  HELP_TOPICS, 
  TUTORIAL_STEPS 
} from "./HelpSystem";

describe("HelpModal", () => {
  const mockOnClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Modal behavior", () => {
    it("should not render when closed", () => {
      render(<HelpModal isOpen={false} onClose={mockOnClose} />);
      expect(screen.queryByText("Help & Documentation")).not.toBeInTheDocument();
    });

    it("should render when open", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Help & Documentation")).toBeInTheDocument();
    });

    it("should call onClose when close button is clicked", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      const closeButton = screen.getByLabelText("Close help");
      fireEvent.click(closeButton);
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should call onClose when backdrop is clicked", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      const backdrop = document.querySelector(".bg-black\\/50");
      fireEvent.click(backdrop!);
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe("Topic navigation", () => {
    it("should display all help topics in sidebar", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      // Topics appear both in sidebar and content header, so use getAllByText
      HELP_TOPICS.forEach(topic => {
        const elements = screen.getAllByText(topic.title);
        expect(elements.length).toBeGreaterThan(0);
      });
    });

    it("should display first topic by default", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      // First topic is "Getting Started"
      expect(screen.getByText("Quick introduction to NeMo RL Configurator")).toBeInTheDocument();
    });

    it("should switch topics when clicking sidebar item", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      // Click on "Training Algorithms" topic
      fireEvent.click(screen.getByText("Training Algorithms"));
      // Should show algorithms content
      expect(screen.getByText("Learn about GRPO, SFT, and DPO")).toBeInTheDocument();
    });

    it("should display topic content", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      // First topic should have Quick Start section
      expect(screen.getByText(/Select your training algorithm/)).toBeInTheDocument();
    });

    it("should open with specific topic when initialTopic is provided", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} initialTopic="algorithms" />);
      // Should show algorithms content
      expect(screen.getByText("GRPO (Group Relative Policy Optimization)")).toBeInTheDocument();
    });
  });

  describe("External links", () => {
    it("should have link to full documentation", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Full Documentation")).toBeInTheDocument();
    });

    it("should have link to GitHub", () => {
      render(<HelpModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("GitHub Repository")).toBeInTheDocument();
    });
  });
});

describe("TutorialModal", () => {
  const mockOnClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Modal behavior", () => {
    it("should not render when closed", () => {
      render(<TutorialModal isOpen={false} onClose={mockOnClose} />);
      expect(screen.queryByText("Welcome to NeMo RL!")).not.toBeInTheDocument();
    });

    it("should render when open", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Welcome to NeMo RL!")).toBeInTheDocument();
    });

    it("should show step progress indicator", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(`Step 1 of ${TUTORIAL_STEPS.length}`)).toBeInTheDocument();
    });
  });

  describe("Navigation", () => {
    it("should start at first step", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Welcome to NeMo RL!")).toBeInTheDocument();
    });

    it("should go to next step when clicking Next", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Next"));
      // Should show step 2
      expect(screen.getByText("Step 1: Choose Your Algorithm")).toBeInTheDocument();
    });

    it("should not show Back button on first step", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.queryByText("Back")).not.toBeInTheDocument();
    });

    it("should show Back button on second step", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Next"));
      expect(screen.getByText("Back")).toBeInTheDocument();
    });

    it("should go to previous step when clicking Back", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      // Go to step 2
      fireEvent.click(screen.getByText("Next"));
      // Go back to step 1
      fireEvent.click(screen.getByText("Back"));
      expect(screen.getByText("Welcome to NeMo RL!")).toBeInTheDocument();
    });

    it("should show Get Started on last step", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      // Navigate to last step
      for (let i = 0; i < TUTORIAL_STEPS.length - 1; i++) {
        fireEvent.click(screen.getByText("Next"));
      }
      expect(screen.getByText("Get Started")).toBeInTheDocument();
    });

    it("should close modal when clicking Get Started on last step", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      // Navigate to last step
      for (let i = 0; i < TUTORIAL_STEPS.length - 1; i++) {
        fireEvent.click(screen.getByText("Next"));
      }
      fireEvent.click(screen.getByText("Get Started"));
      expect(mockOnClose).toHaveBeenCalled();
    });

    it("should have Skip tutorial option", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Skip tutorial")).toBeInTheDocument();
    });

    it("should close modal when clicking Skip tutorial", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      fireEvent.click(screen.getByText("Skip tutorial"));
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe("Step content", () => {
    it("should display step description", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText("Let's create your first training configuration")).toBeInTheDocument();
    });

    it("should display step content", () => {
      render(<TutorialModal isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByText(/This tutorial will guide you/)).toBeInTheDocument();
    });
  });
});

describe("HelpButton", () => {
  it("should render help icon button", () => {
    render(<HelpButton topic="algorithms" />);
    expect(screen.getByLabelText("Show help")).toBeInTheDocument();
  });

  it("should open help modal when clicked", () => {
    render(<HelpButton topic="algorithms" />);
    fireEvent.click(screen.getByLabelText("Show help"));
    expect(screen.getByText("Help & Documentation")).toBeInTheDocument();
  });

  it("should open to specific topic", () => {
    render(<HelpButton topic="algorithms" />);
    fireEvent.click(screen.getByLabelText("Show help"));
    // Should show algorithms content
    expect(screen.getByText("Learn about GRPO, SFT, and DPO")).toBeInTheDocument();
  });
});

describe("HELP_TOPICS", () => {
  it("should have at least 5 topics", () => {
    expect(HELP_TOPICS.length).toBeGreaterThanOrEqual(5);
  });

  it("should have required fields for all topics", () => {
    HELP_TOPICS.forEach(topic => {
      expect(topic.id).toBeDefined();
      expect(topic.title).toBeDefined();
      expect(topic.icon).toBeDefined();
      expect(topic.description).toBeDefined();
      expect(topic.content).toBeDefined();
    });
  });

  it("should include getting-started topic", () => {
    const topic = HELP_TOPICS.find(t => t.id === "getting-started");
    expect(topic).toBeDefined();
  });

  it("should include algorithms topic", () => {
    const topic = HELP_TOPICS.find(t => t.id === "algorithms");
    expect(topic).toBeDefined();
  });

  it("should include cluster topic", () => {
    const topic = HELP_TOPICS.find(t => t.id === "cluster");
    expect(topic).toBeDefined();
  });
});

describe("TUTORIAL_STEPS", () => {
  it("should have at least 5 steps", () => {
    expect(TUTORIAL_STEPS.length).toBeGreaterThanOrEqual(5);
  });

  it("should have required fields for all steps", () => {
    TUTORIAL_STEPS.forEach(step => {
      expect(step.id).toBeDefined();
      expect(step.title).toBeDefined();
      expect(step.icon).toBeDefined();
      expect(step.description).toBeDefined();
      expect(step.content).toBeDefined();
    });
  });

  it("should start with welcome step", () => {
    expect(TUTORIAL_STEPS[0].id).toBe("welcome");
  });

  it("should end with complete step", () => {
    expect(TUTORIAL_STEPS[TUTORIAL_STEPS.length - 1].id).toBe("complete");
  });

  it("should cover all main configuration areas", () => {
    const stepIds = TUTORIAL_STEPS.map(s => s.id);
    expect(stepIds).toContain("algorithm");
    expect(stepIds).toContain("model");
    expect(stepIds).toContain("dataset");
    expect(stepIds).toContain("hyperparams");
    expect(stepIds).toContain("cluster");
  });
});
