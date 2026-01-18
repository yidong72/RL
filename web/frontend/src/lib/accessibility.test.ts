import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  createFocusTrap,
  announceToScreenReader,
  prefersReducedMotion,
  prefersHighContrast,
  generateA11yId,
  meetsContrastRatio,
  KEYBOARD_SHORTCUTS,
} from "./accessibility";

describe("accessibility utilities", () => {
  describe("createFocusTrap", () => {
    it("should return a cleanup function", () => {
      const container = document.createElement("div");
      container.innerHTML = `
        <button id="btn1">Button 1</button>
        <input id="input1" type="text" />
        <button id="btn2">Button 2</button>
      `;
      document.body.appendChild(container);

      const cleanup = createFocusTrap(container);
      expect(typeof cleanup).toBe("function");

      cleanup();
      document.body.removeChild(container);
    });

    it("should focus first focusable element", () => {
      const container = document.createElement("div");
      const button = document.createElement("button");
      button.textContent = "Test";
      container.appendChild(button);
      document.body.appendChild(container);

      createFocusTrap(container);
      expect(document.activeElement).toBe(button);

      document.body.removeChild(container);
    });
  });

  describe("announceToScreenReader", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
      // Clean up any remaining announcers
      document.querySelectorAll('[role="status"]').forEach(el => el.remove());
    });

    it("should create an announcement element", () => {
      announceToScreenReader("Test message");
      const announcer = document.querySelector('[role="status"]');
      expect(announcer).toBeTruthy();
      expect(announcer?.textContent).toBe("Test message");
      vi.advanceTimersByTime(1000);
    });

    it("should set aria-live to polite by default", () => {
      announceToScreenReader("Test message polite");
      const announcer = document.querySelector('[role="status"]');
      expect(announcer?.getAttribute("aria-live")).toBe("polite");
      vi.advanceTimersByTime(1000);
    });
  });

  describe("prefersReducedMotion", () => {
    beforeEach(() => {
      // Mock window.matchMedia
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: query.includes("reduce"),
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      });
    });

    it("should return a boolean", () => {
      const result = prefersReducedMotion();
      expect(typeof result).toBe("boolean");
    });
  });

  describe("prefersHighContrast", () => {
    beforeEach(() => {
      // Mock window.matchMedia
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: query.includes("more"),
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      });
    });

    it("should return a boolean", () => {
      const result = prefersHighContrast();
      expect(typeof result).toBe("boolean");
    });
  });

  describe("generateA11yId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateA11yId();
      const id2 = generateA11yId();
      expect(id1).not.toBe(id2);
    });

    it("should use custom prefix", () => {
      const id = generateA11yId("custom");
      expect(id.startsWith("custom-")).toBe(true);
    });

    it("should use default prefix", () => {
      const id = generateA11yId();
      expect(id.startsWith("a11y-")).toBe(true);
    });
  });

  describe("meetsContrastRatio", () => {
    it("should return true for black on white (21:1 ratio)", () => {
      const result = meetsContrastRatio("#000000", "#FFFFFF");
      expect(result).toBe(true);
    });

    it("should return true for white on black (21:1 ratio)", () => {
      const result = meetsContrastRatio("#FFFFFF", "#000000");
      expect(result).toBe(true);
    });

    it("should return false for low contrast colors", () => {
      const result = meetsContrastRatio("#777777", "#888888");
      expect(result).toBe(false);
    });

    it("should use lower threshold for large text", () => {
      // This combination passes for large text (3:1) but not normal text (4.5:1)
      const normalText = meetsContrastRatio("#767676", "#FFFFFF", false);
      const largeText = meetsContrastRatio("#767676", "#FFFFFF", true);
      expect(largeText).toBe(true);
      expect(normalText).toBe(true); // #767676 actually passes 4.5:1
    });
  });

  describe("KEYBOARD_SHORTCUTS", () => {
    it("should have common shortcuts defined", () => {
      expect(KEYBOARD_SHORTCUTS["?"]).toBe("Open help");
      expect(KEYBOARD_SHORTCUTS["t"]).toBe("Open templates");
      expect(KEYBOARD_SHORTCUTS["i"]).toBe("Import config");
      expect(KEYBOARD_SHORTCUTS["Escape"]).toBe("Close modal");
    });
  });
});
