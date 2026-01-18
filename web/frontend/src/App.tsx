import { useState, useEffect, useCallback } from "react";
import { 
  Moon, 
  Sun, 
  Github, 
  FileText, 
  Upload, 
  Code2, 
  Menu, 
  X,
  Settings,
  Eye,
  LayoutGrid,
  HelpCircle,
  GraduationCap
} from "lucide-react";
import { 
  AlgorithmSelector, 
  ModelSelector, 
  DatasetSelector, 
  HyperparameterForm, 
  ClusterConfig,
  ValidationDisplay,
  ScriptPreview,
  ImportConfig,
  YamlEditor,
  TemplateLibrary,
  RewardEditor,
  HelpModal,
  TutorialModal
} from "./components/config-builder";
import { Button, SkipLinks } from "./components/common";
import { useConfigStore } from "./store/configStore";
import { cn } from "./lib/utils";
import { announceToScreenReader } from "./lib/accessibility";

type EditorMode = "form" | "yaml";
type MobileView = "config" | "preview";

interface HeaderProps {
  editorMode: EditorMode;
  onModeChange: (mode: EditorMode) => void;
  onImportClick: () => void;
  onTemplatesClick: () => void;
  onHelpClick: () => void;
  onTutorialClick: () => void;
}

/**
 * Mobile Menu Component
 */
function MobileMenu({ 
  isOpen, 
  onClose,
  editorMode,
  onModeChange,
  onImportClick,
  onTemplatesClick,
  onHelpClick,
  onTutorialClick
}: { 
  isOpen: boolean; 
  onClose: () => void;
  editorMode: EditorMode;
  onModeChange: (mode: EditorMode) => void;
  onImportClick: () => void;
  onTemplatesClick: () => void;
  onHelpClick: () => void;
  onTutorialClick: () => void;
}) {
  const { isDarkMode, toggleDarkMode } = useConfigStore();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 md:hidden">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Menu */}
      <div className="absolute top-0 right-0 w-72 h-full bg-background border-l shadow-xl">
        <div className="flex items-center justify-between p-4 border-b">
          <span className="font-semibold">Menu</span>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="p-4 space-y-2">
          {/* Editor Mode Toggle */}
          <div className="pb-4 border-b">
            <p className="text-xs text-muted-foreground mb-2">Editor Mode</p>
            <div className="flex items-center rounded-lg border p-1 bg-muted/30">
              <button
                className={cn(
                  "flex-1 px-3 py-2 text-sm rounded-md transition-colors",
                  editorMode === "form"
                    ? "bg-background shadow text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => {
                  onModeChange("form");
                  onClose();
                }}
              >
                Form
              </button>
              <button
                className={cn(
                  "flex-1 px-3 py-2 text-sm rounded-md transition-colors flex items-center justify-center gap-1",
                  editorMode === "yaml"
                    ? "bg-background shadow text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => {
                  onModeChange("yaml");
                  onClose();
                }}
              >
                <Code2 className="h-3 w-3" />
                YAML
              </button>
            </div>
          </div>

          {/* Actions */}
          <button 
            onClick={() => { onImportClick(); onClose(); }}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors text-left"
          >
            <Upload className="h-4 w-4" />
            <span>Import Config</span>
          </button>
          
          <button 
            onClick={() => { onTemplatesClick(); onClose(); }}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors text-left"
          >
            <FileText className="h-4 w-4" />
            <span>Templates</span>
          </button>

          <button 
            onClick={() => { onHelpClick(); onClose(); }}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors text-left"
          >
            <HelpCircle className="h-4 w-4" />
            <span>Help</span>
          </button>

          <button 
            onClick={() => { onTutorialClick(); onClose(); }}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors text-left"
          >
            <GraduationCap className="h-4 w-4" />
            <span>Tutorial</span>
          </button>

          <a
            href="https://docs.nvidia.com/nemo-rl"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors"
          >
            <LayoutGrid className="h-4 w-4" />
            <span>Documentation</span>
          </a>

          <a
            href="https://github.com/NVIDIA/NeMo-RL"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors"
          >
            <Github className="h-4 w-4" />
            <span>GitHub</span>
          </a>

          {/* Theme Toggle */}
          <div className="pt-4 border-t">
            <button
              onClick={() => { toggleDarkMode(); onClose(); }}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted transition-colors text-left"
            >
              {isDarkMode ? (
                <>
                  <Sun className="h-4 w-4" />
                  <span>Light Mode</span>
                </>
              ) : (
                <>
                  <Moon className="h-4 w-4" />
                  <span>Dark Mode</span>
                </>
              )}
            </button>
          </div>
        </nav>
      </div>
    </div>
  );
}

/**
 * Mobile View Tabs - for switching between config and preview on mobile
 */
function MobileViewTabs({ 
  activeView, 
  onViewChange 
}: { 
  activeView: MobileView; 
  onViewChange: (view: MobileView) => void;
}) {
  return (
    <div className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-background border-t shadow-lg">
      <div className="flex">
        <button
          onClick={() => onViewChange("config")}
          className={cn(
            "flex-1 flex flex-col items-center gap-1 py-3 px-4 transition-colors",
            activeView === "config"
              ? "text-primary bg-primary/5"
              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
          )}
          aria-label="Configuration panel"
        >
          <Settings className="h-5 w-5" />
          <span className="text-xs font-medium">Configure</span>
        </button>
        <button
          onClick={() => onViewChange("preview")}
          className={cn(
            "flex-1 flex flex-col items-center gap-1 py-3 px-4 transition-colors",
            activeView === "preview"
              ? "text-primary bg-primary/5"
              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
          )}
          aria-label="Preview panel"
        >
          <Eye className="h-5 w-5" />
          <span className="text-xs font-medium">Preview</span>
        </button>
      </div>
    </div>
  );
}

function Header({ editorMode, onModeChange, onImportClick, onTemplatesClick, onHelpClick, onTutorialClick }: HeaderProps) {
  const { isDarkMode, toggleDarkMode } = useConfigStore();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-30">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">NR</span>
            </div>
            <span className="font-semibold text-lg hidden sm:inline">NeMo RL Configurator</span>
            <span className="font-semibold text-lg sm:hidden">NeMo RL</span>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-4">
            {/* Editor Mode Toggle */}
            <div className="flex items-center rounded-lg border p-1 bg-muted/30">
              <button
                className={cn(
                  "px-3 py-1 text-sm rounded-md transition-colors",
                  editorMode === "form"
                    ? "bg-background shadow text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => onModeChange("form")}
              >
                Form
              </button>
              <button
                className={cn(
                  "px-3 py-1 text-sm rounded-md transition-colors flex items-center gap-1",
                  editorMode === "yaml"
                    ? "bg-background shadow text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => onModeChange("yaml")}
              >
                <Code2 className="h-3 w-3" />
                YAML
              </button>
            </div>
            
            <Button variant="ghost" size="sm" onClick={onImportClick}>
              <Upload className="h-4 w-4 mr-2" />
              Import
            </Button>
            <Button variant="ghost" size="sm" onClick={onTemplatesClick}>
              <FileText className="h-4 w-4 mr-2" />
              Templates
            </Button>
            <Button variant="ghost" size="sm" onClick={onHelpClick}>
              <HelpCircle className="h-4 w-4 mr-2" />
              Help
            </Button>
            <a
              href="https://docs.nvidia.com/nemo-rl"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground h-8 px-3"
            >
              Docs
            </a>
            <a
              href="https://github.com/NVIDIA/NeMo-RL"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground h-9 w-9"
            >
              <Github className="h-4 w-4" />
            </a>
            <Button variant="ghost" size="icon" onClick={toggleDarkMode}>
              {isDarkMode ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </Button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="md:hidden p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
        </div>
      </header>

      {/* Mobile Menu */}
      <MobileMenu
        isOpen={isMobileMenuOpen}
        onClose={() => setIsMobileMenuOpen(false)}
        editorMode={editorMode}
        onModeChange={onModeChange}
        onImportClick={onImportClick}
        onTemplatesClick={onTemplatesClick}
        onHelpClick={onHelpClick}
        onTutorialClick={onTutorialClick}
      />
    </>
  );
}

/**
 * Import Config Modal
 */
function ImportModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative z-10 w-full max-w-2xl mx-4 bg-background rounded-lg shadow-xl border max-h-[90vh] overflow-y-auto">
        <div className="p-4 sm:p-6">
          <ImportConfig onClose={onClose} />
        </div>
      </div>
    </div>
  );
}


function App() {
  const [editorMode, setEditorMode] = useState<EditorMode>("form");
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [isTemplatesOpen, setIsTemplatesOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isTutorialOpen, setIsTutorialOpen] = useState(false);
  const [mobileView, setMobileView] = useState<MobileView>("config");

  // Keyboard shortcuts handler
  const handleKeyboardShortcuts = useCallback((e: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    const target = e.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
      return;
    }

    // Escape key closes any open modal
    if (e.key === "Escape") {
      if (isImportOpen) setIsImportOpen(false);
      if (isTemplatesOpen) setIsTemplatesOpen(false);
      if (isHelpOpen) setIsHelpOpen(false);
      if (isTutorialOpen) setIsTutorialOpen(false);
      return;
    }

    // Keyboard shortcuts (without modifiers)
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      switch (e.key) {
        case "?":
          e.preventDefault();
          setIsHelpOpen(true);
          announceToScreenReader("Opening help dialog");
          break;
        case "t":
          if (!isTemplatesOpen) {
            e.preventDefault();
            setIsTemplatesOpen(true);
            announceToScreenReader("Opening template library");
          }
          break;
        case "i":
          if (!isImportOpen) {
            e.preventDefault();
            setIsImportOpen(true);
            announceToScreenReader("Opening import dialog");
          }
          break;
      }
    }
  }, [isImportOpen, isTemplatesOpen, isHelpOpen, isTutorialOpen]);

  // Register keyboard shortcuts
  useEffect(() => {
    document.addEventListener("keydown", handleKeyboardShortcuts);
    return () => document.removeEventListener("keydown", handleKeyboardShortcuts);
  }, [handleKeyboardShortcuts]);

  return (
    <div className="min-h-screen bg-background">
      {/* Skip Links for keyboard navigation - WCAG 2.4.1 */}
      <SkipLinks />

      <Header 
        editorMode={editorMode} 
        onModeChange={setEditorMode}
        onImportClick={() => setIsImportOpen(true)}
        onTemplatesClick={() => setIsTemplatesOpen(true)}
        onHelpClick={() => setIsHelpOpen(true)}
        onTutorialClick={() => setIsTutorialOpen(true)}
      />

      <main 
        id="main-content" 
        className="container mx-auto px-4 py-4 sm:py-6 lg:py-8 pb-20 md:pb-8"
        role="main"
        aria-label="Configuration builder"
      >
        {/* Desktop & Tablet Layout: Grid with side-by-side (lg) or stacked (md) panels */}
        <div className="hidden md:grid md:grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Configuration Panel */}
          <section 
            id="config-panel" 
            className="space-y-6 lg:space-y-8"
            aria-label="Training configuration"
          >
            {editorMode === "form" ? (
              <>
                <AlgorithmSelector />
                <ModelSelector />
                <DatasetSelector />
                <HyperparameterForm />
                <RewardEditor />
                <ClusterConfig />
              </>
            ) : (
              <YamlEditor />
            )}
          </section>

          {/* Preview Panel */}
          <aside 
            className="lg:sticky lg:top-20 space-y-6 lg:max-h-[calc(100vh-8rem)] lg:overflow-y-auto"
            aria-label="Validation and preview"
          >
            <ValidationDisplay />
            <ScriptPreview />
          </aside>
        </div>

        {/* Mobile Layout: Single column with tab-based navigation */}
        <div className="md:hidden">
          {mobileView === "config" ? (
            <section 
              id="config-panel-mobile" 
              className="space-y-6"
              aria-label="Training configuration"
            >
              {editorMode === "form" ? (
                <>
                  <AlgorithmSelector />
                  <ModelSelector />
                  <DatasetSelector />
                  <HyperparameterForm />
                  <RewardEditor />
                  <ClusterConfig />
                </>
              ) : (
                <YamlEditor />
              )}
            </section>
          ) : (
            <section 
              className="space-y-6"
              aria-label="Validation and preview"
            >
              <ValidationDisplay />
              <ScriptPreview />
            </section>
          )}
        </div>
      </main>

      {/* Mobile View Tabs */}
      <MobileViewTabs activeView={mobileView} onViewChange={setMobileView} />

      {/* Import Modal */}
      <ImportModal isOpen={isImportOpen} onClose={() => setIsImportOpen(false)} />

      {/* Template Library Modal */}
      <TemplateLibrary isOpen={isTemplatesOpen} onClose={() => setIsTemplatesOpen(false)} />

      {/* Help Modal */}
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />

      {/* Tutorial Modal */}
      <TutorialModal isOpen={isTutorialOpen} onClose={() => setIsTutorialOpen(false)} />

      {/* Footer - hidden on mobile to make room for bottom tabs */}
      <footer className="hidden md:block border-t py-6 mt-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>NeMo RL Web Configurator - NVIDIA</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
