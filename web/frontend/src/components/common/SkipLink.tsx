import { cn } from "../../lib/utils";

/**
 * Skip Link Props
 */
interface SkipLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

/**
 * Skip Link Component
 * Provides a way for keyboard users to skip to main content
 * Following WCAG 2.1 Success Criterion 2.4.1
 */
export function SkipLink({ href, children, className }: SkipLinkProps) {
  return (
    <a
      href={href}
      className={cn(
        // Visually hidden by default
        "absolute -top-10 left-4 z-[100] px-4 py-2 rounded-md",
        // Styling
        "bg-primary text-primary-foreground font-medium text-sm",
        "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        // Visible on focus
        "focus:top-4 transition-all",
        className
      )}
    >
      {children}
    </a>
  );
}

/**
 * Skip Links Group Component
 * Multiple skip links for navigation
 */
export function SkipLinks() {
  return (
    <div role="navigation" aria-label="Skip links">
      <SkipLink href="#main-content">
        Skip to main content
      </SkipLink>
      <SkipLink href="#config-panel" className="focus:left-40">
        Skip to configuration
      </SkipLink>
    </div>
  );
}
