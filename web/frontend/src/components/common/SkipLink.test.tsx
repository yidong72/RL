import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SkipLink, SkipLinks } from "./SkipLink";

describe("SkipLink", () => {
  it("should render with correct href", () => {
    render(<SkipLink href="#main-content">Skip to main</SkipLink>);
    const link = screen.getByRole("link", { name: /skip to main/i });
    expect(link).toHaveAttribute("href", "#main-content");
  });

  it("should render children text", () => {
    render(<SkipLink href="#test">Test Link</SkipLink>);
    expect(screen.getByText("Test Link")).toBeInTheDocument();
  });

  it("should have accessibility-focused styles", () => {
    render(<SkipLink href="#main">Skip</SkipLink>);
    const link = screen.getByRole("link");
    expect(link.className).toContain("focus:");
  });

  it("should accept custom className", () => {
    render(<SkipLink href="#main" className="custom-class">Skip</SkipLink>);
    const link = screen.getByRole("link");
    expect(link.className).toContain("custom-class");
  });
});

describe("SkipLinks", () => {
  it("should render navigation with skip links", () => {
    render(<SkipLinks />);
    const nav = screen.getByRole("navigation", { name: /skip links/i });
    expect(nav).toBeInTheDocument();
  });

  it("should have skip to main content link", () => {
    render(<SkipLinks />);
    const link = screen.getByRole("link", { name: /skip to main content/i });
    expect(link).toHaveAttribute("href", "#main-content");
  });

  it("should have skip to configuration link", () => {
    render(<SkipLinks />);
    const link = screen.getByRole("link", { name: /skip to configuration/i });
    expect(link).toHaveAttribute("href", "#config-panel");
  });
});
