import { useEffect, useCallback, useRef, useState } from "react";
import { 
  Copy, 
  Download, 
  Loader2, 
  FileCode, 
  Settings, 
  Terminal,
  ChevronDown,
  Plus,
  X,
  Check
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { api } from "../../api/client";
import { Button } from "../common/Button";
import type { ScriptGenerationResult } from "../../types/config";

// Debounce delay for script generation
const SCRIPT_GENERATION_DEBOUNCE_MS = 300;

// Custom SLURM parameter type
interface CustomSlurmParam {
  id: string;
  name: string;
  value: string;
}

/**
 * Hook for debounced script generation
 */
function useScriptGeneration(customParams: CustomSlurmParam[]) {
  const { config, validation } = useConfigStore();
  const [scriptResult, setScriptResult] = useState<ScriptGenerationResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const generate = useCallback(async () => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsGenerating(true);
    setError(null);

    try {
      const result = await api.generateScript(config);
      
      // Apply custom SLURM parameters to the script
      let modifiedScript = result.script;
      if (customParams.length > 0) {
        const customDirectives = customParams
          .filter(p => p.name && p.value)
          .map(p => `#SBATCH --${p.name}=${p.value}`)
          .join("\n");
        
        if (customDirectives) {
          // Insert custom parameters after the last #SBATCH line
          const lines = modifiedScript.split("\n");
          const lastSbatchIndex = lines.reduce((lastIdx, line, idx) => 
            line.startsWith("#SBATCH") ? idx : lastIdx, -1
          );
          
          if (lastSbatchIndex >= 0) {
            lines.splice(lastSbatchIndex + 1, 0, customDirectives);
            modifiedScript = lines.join("\n");
          }
        }
      }

      setScriptResult({
        ...result,
        script: modifiedScript
      });
    } catch (err) {
      if (err instanceof Error && err.name !== "AbortError") {
        console.error("Script generation error:", err);
        setError("Unable to generate script. Please check if the backend is running.");
      }
    } finally {
      setIsGenerating(false);
    }
  }, [config, customParams]);

  // Debounced generation effect
  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      generate();
    }, SCRIPT_GENERATION_DEBOUNCE_MS);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [generate]);

  return { scriptResult, isGenerating, error, validation };
}

/**
 * Tab type for file viewer
 */
type ScriptTab = "slurm" | "config" | "ray";

/**
 * Syntax highlighting for bash/YAML content
 */
function SyntaxHighlightedCode({ content, language }: { content: string; language: "bash" | "yaml" }) {
  const highlightLine = (line: string): React.ReactNode => {
    if (language === "bash") {
      // Comment lines
      if (line.trim().startsWith("#")) {
        return <span className="text-gray-500 dark:text-gray-400">{line}</span>;
      }
      // SBATCH directives
      if (line.startsWith("#SBATCH")) {
        const [directive, ...rest] = line.split("=");
        return (
          <>
            <span className="text-purple-600 dark:text-purple-400">{directive}</span>
            {rest.length > 0 && <span className="text-gray-600 dark:text-gray-300">=</span>}
            <span className="text-green-600 dark:text-green-400">{rest.join("=")}</span>
          </>
        );
      }
      // Export statements
      if (line.trim().startsWith("export ")) {
        const match = line.match(/^(\s*)(export\s+)(\w+)(=)(.*)$/);
        if (match) {
          return (
            <>
              {match[1]}
              <span className="text-blue-600 dark:text-blue-400">{match[2]}</span>
              <span className="text-yellow-600 dark:text-yellow-400">{match[3]}</span>
              <span className="text-gray-600 dark:text-gray-300">{match[4]}</span>
              <span className="text-green-600 dark:text-green-400">{match[5]}</span>
            </>
          );
        }
      }
      // Commands
      const keywords = ["python", "bash", "mkdir", "echo", "ray", "srun", "sbatch", "if", "then", "else", "fi", "for", "do", "done"];
      for (const keyword of keywords) {
        if (line.trim().startsWith(keyword + " ") || line.trim() === keyword) {
          const idx = line.indexOf(keyword);
          return (
            <>
              {line.slice(0, idx)}
              <span className="text-blue-600 dark:text-blue-400">{keyword}</span>
              {line.slice(idx + keyword.length)}
            </>
          );
        }
      }
    } else if (language === "yaml") {
      // Comment lines
      if (line.trim().startsWith("#")) {
        return <span className="text-gray-500 dark:text-gray-400">{line}</span>;
      }
      // Key-value pairs
      const match = line.match(/^(\s*)(\w[\w_-]*)(:)(.*)$/);
      if (match) {
        return (
          <>
            {match[1]}
            <span className="text-blue-600 dark:text-blue-400">{match[2]}</span>
            <span className="text-gray-600 dark:text-gray-300">{match[3]}</span>
            <span className="text-green-600 dark:text-green-400">{match[4]}</span>
          </>
        );
      }
    }
    return line;
  };

  return (
    <pre className="text-sm font-mono whitespace-pre-wrap">
      <code>
        {content.split("\n").map((line, idx) => (
          <div key={idx} className="leading-relaxed">
            {highlightLine(line)}
          </div>
        ))}
      </code>
    </pre>
  );
}

/**
 * Custom SLURM Parameter Input
 */
interface CustomParamInputProps {
  param: CustomSlurmParam;
  onChange: (id: string, name: string, value: string) => void;
  onRemove: (id: string) => void;
}

function CustomParamInput({ param, onChange, onRemove }: CustomParamInputProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">#SBATCH --</span>
      <input
        type="text"
        value={param.name}
        onChange={(e) => onChange(param.id, e.target.value, param.value)}
        placeholder="param-name"
        className="flex-1 px-2 py-1 text-sm border rounded bg-background focus:outline-none focus:ring-1 focus:ring-primary"
      />
      <span className="text-sm text-muted-foreground">=</span>
      <input
        type="text"
        value={param.value}
        onChange={(e) => onChange(param.id, param.name, e.target.value)}
        placeholder="value"
        className="flex-1 px-2 py-1 text-sm border rounded bg-background focus:outline-none focus:ring-1 focus:ring-primary"
      />
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onRemove(param.id)}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}

/**
 * ScriptPreview component with live preview, custom params, and file tabs
 */
export function ScriptPreview() {
  const { config, validation } = useConfigStore();
  const [activeTab, setActiveTab] = useState<ScriptTab>("slurm");
  const [copied, setCopied] = useState(false);
  const [showCustomParams, setShowCustomParams] = useState(false);
  const [customParams, setCustomParams] = useState<CustomSlurmParam[]>([]);
  
  const { scriptResult, isGenerating, error } = useScriptGeneration(customParams);

  // Add a new custom parameter
  const addCustomParam = () => {
    setCustomParams([
      ...customParams,
      { id: `param-${Date.now()}`, name: "", value: "" }
    ]);
    setShowCustomParams(true);
  };

  // Update a custom parameter
  const updateCustomParam = (id: string, name: string, value: string) => {
    setCustomParams(customParams.map(p => 
      p.id === id ? { ...p, name, value } : p
    ));
  };

  // Remove a custom parameter
  const removeCustomParam = (id: string) => {
    setCustomParams(customParams.filter(p => p.id !== id));
  };

  // Get content for current tab
  const getCurrentContent = (): { content: string; language: "bash" | "yaml" } => {
    if (!scriptResult) {
      return { content: "Generating script...", language: "bash" };
    }

    switch (activeTab) {
      case "slurm":
        return { content: scriptResult.script, language: "bash" };
      case "config":
        return { content: scriptResult.files["config.yaml"] || "", language: "yaml" };
      case "ray":
        return { content: scriptResult.files["ray.sub"] || "# Ray cluster script not needed for single-node training", language: "bash" };
      default:
        return { content: "", language: "bash" };
    }
  };

  // Copy to clipboard
  const handleCopy = async () => {
    const { content } = getCurrentContent();
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Download file
  const handleDownload = () => {
    const { content } = getCurrentContent();
    const filenames: Record<ScriptTab, string> = {
      slurm: `nemo_rl_${config.algorithm}_train.sh`,
      config: "config.yaml",
      ray: "ray.sub"
    };
    const mimeTypes: Record<ScriptTab, string> = {
      slurm: "text/x-shellscript",
      config: "text/yaml",
      ray: "text/x-shellscript"
    };

    const blob = new Blob([content], { type: mimeTypes[activeTab] });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filenames[activeTab];
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Download all files as a bundle
  const handleDownloadAll = () => {
    if (!scriptResult) return;

    // Download main script
    const mainBlob = new Blob([scriptResult.script], { type: "text/x-shellscript" });
    const mainUrl = URL.createObjectURL(mainBlob);
    const mainLink = document.createElement("a");
    mainLink.href = mainUrl;
    mainLink.download = `nemo_rl_${config.algorithm}_train.sh`;
    mainLink.click();
    URL.revokeObjectURL(mainUrl);

    // Download config.yaml
    if (scriptResult.files["config.yaml"]) {
      setTimeout(() => {
        const configBlob = new Blob([scriptResult.files["config.yaml"]], { type: "text/yaml" });
        const configUrl = URL.createObjectURL(configBlob);
        const configLink = document.createElement("a");
        configLink.href = configUrl;
        configLink.download = "config.yaml";
        configLink.click();
        URL.revokeObjectURL(configUrl);
      }, 100);
    }

    // Download ray.sub if present
    if (scriptResult.files["ray.sub"]) {
      setTimeout(() => {
        const rayBlob = new Blob([scriptResult.files["ray.sub"]], { type: "text/x-shellscript" });
        const rayUrl = URL.createObjectURL(rayBlob);
        const rayLink = document.createElement("a");
        rayLink.href = rayUrl;
        rayLink.download = "ray.sub";
        rayLink.click();
        URL.revokeObjectURL(rayUrl);
      }, 200);
    }
  };

  const isValid = validation?.valid ?? true;
  const { content, language } = getCurrentContent();
  const hasRaySub = scriptResult?.files["ray.sub"] !== undefined;

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Terminal className="h-5 w-5" />
          SLURM Script Preview
          {isGenerating && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? (
              <Check className="h-4 w-4 mr-2" />
            ) : (
              <Copy className="h-4 w-4 mr-2" />
            )}
            {copied ? "Copied!" : "Copy"}
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={handleDownload}
            disabled={!isValid || !scriptResult}
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex items-center gap-2 mb-3 border-b">
        <button
          className={cn(
            "flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
            activeTab === "slurm"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          )}
          onClick={() => setActiveTab("slurm")}
        >
          <Terminal className="h-4 w-4" />
          train.sh
        </button>
        <button
          className={cn(
            "flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
            activeTab === "config"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          )}
          onClick={() => setActiveTab("config")}
        >
          <FileCode className="h-4 w-4" />
          config.yaml
        </button>
        {hasRaySub && (
          <button
            className={cn(
              "flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
              activeTab === "ray"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
            onClick={() => setActiveTab("ray")}
          >
            <Settings className="h-4 w-4" />
            ray.sub
          </button>
        )}
      </div>

      {/* Custom SLURM Parameters */}
      {activeTab === "slurm" && (
        <div className="mb-3">
          <button
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowCustomParams(!showCustomParams)}
          >
            <ChevronDown className={cn(
              "h-4 w-4 transition-transform",
              showCustomParams && "rotate-180"
            )} />
            Custom SLURM Parameters ({customParams.length})
          </button>
          
          {showCustomParams && (
            <div className="mt-3 p-3 bg-muted/30 rounded-lg space-y-2">
              {customParams.map(param => (
                <CustomParamInput
                  key={param.id}
                  param={param}
                  onChange={updateCustomParam}
                  onRemove={removeCustomParam}
                />
              ))}
              <Button
                variant="outline"
                size="sm"
                onClick={addCustomParam}
                className="w-full"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Custom Parameter
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="mb-3 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Code preview */}
      <div className="overflow-auto rounded-lg border bg-muted/50 max-h-96">
        <div className="p-4">
          <SyntaxHighlightedCode content={content} language={language} />
        </div>
      </div>

      {/* Instructions */}
      {scriptResult?.instructions && scriptResult.instructions.length > 0 && (
        <div className="mt-4 p-4 bg-muted/30 rounded-lg">
          <h4 className="text-sm font-medium mb-2">Quick Start Instructions</h4>
          <ol className="text-sm text-muted-foreground space-y-1">
            {scriptResult.instructions.map((instruction, idx) => (
              <li key={idx} className="font-mono text-xs">{instruction}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Download All Button */}
      {scriptResult && (
        <div className="mt-4 flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownloadAll}
            disabled={!isValid}
          >
            <Download className="h-4 w-4 mr-2" />
            Download All Files
          </Button>
        </div>
      )}

      {/* Permissions hint */}
      <p className="mt-2 text-xs text-muted-foreground">
        After downloading, make scripts executable: <code className="bg-muted px-1 rounded">chmod +x train.sh{hasRaySub ? " ray.sub" : ""}</code>
      </p>
    </div>
  );
}
