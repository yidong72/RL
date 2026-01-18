import { useState, useMemo, useCallback } from "react";
import yaml from "js-yaml";
import { 
  GitCompare, 
  ChevronDown,
  ArrowLeftRight,
  Copy,
  Check,
  X,
  Plus,
  Minus,
  Edit3
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { Button } from "../common/Button";
import type { TrainingConfig } from "../../types/config";
import { BUILT_IN_TEMPLATES } from "./TemplateLibrary";

/**
 * Diff line type
 */
type DiffType = "added" | "removed" | "modified" | "unchanged";

/**
 * Diff line interface
 */
interface DiffLine {
  type: DiffType;
  lineNumber: { left?: number; right?: number };
  content: { left?: string; right?: string };
  path?: string; // The YAML path for context
}

/**
 * Compare source options
 */
type CompareSource = "template" | "custom";

/**
 * Convert TrainingConfig to YAML string for display
 */
function configToYaml(config: TrainingConfig): string {
  const yamlObj = {
    algorithm: config.algorithm,
    model: {
      name: config.model,
      backend: config.backend,
      ...(config.hyperparameters.tensor_parallel_size && 
          config.hyperparameters.tensor_parallel_size > 1 
        ? { tensor_parallel_size: config.hyperparameters.tensor_parallel_size }
        : {}),
    },
    dataset: {
      name: config.dataset,
    },
    training: {
      learning_rate: config.hyperparameters.learning_rate,
      batch_size: config.hyperparameters.batch_size,
      max_steps: config.hyperparameters.max_steps,
      ...(config.algorithm === "grpo" 
        ? { num_generations_per_prompt: config.hyperparameters.num_generations_per_prompt }
        : {}),
    },
    cluster: {
      nodes: config.cluster.nodes,
      gpus_per_node: config.cluster.gpus_per_node,
      time_limit: config.cluster.time_limit,
      ...(config.cluster.partition ? { partition: config.cluster.partition } : {}),
      ...(config.cluster.account ? { account: config.cluster.account } : {}),
    },
  };

  return yaml.dump(yamlObj, { 
    indent: 2,
    lineWidth: 120,
    noRefs: true,
  });
}

/**
 * Compute diff between two YAML strings
 */
function computeDiff(leftYaml: string, rightYaml: string): DiffLine[] {
  const leftLines = leftYaml.split("\n");
  const rightLines = rightYaml.split("\n");
  const result: DiffLine[] = [];

  // Simple LCS-based diff algorithm
  const lcs = computeLCS(leftLines, rightLines);
  
  let leftIdx = 0;
  let rightIdx = 0;
  let lcsIdx = 0;

  while (leftIdx < leftLines.length || rightIdx < rightLines.length) {
    if (lcsIdx < lcs.length && 
        leftIdx < leftLines.length && 
        rightIdx < rightLines.length &&
        leftLines[leftIdx] === lcs[lcsIdx] && 
        rightLines[rightIdx] === lcs[lcsIdx]) {
      // Lines match - unchanged
      result.push({
        type: "unchanged",
        lineNumber: { left: leftIdx + 1, right: rightIdx + 1 },
        content: { left: leftLines[leftIdx], right: rightLines[rightIdx] },
      });
      leftIdx++;
      rightIdx++;
      lcsIdx++;
    } else if (leftIdx < leftLines.length && 
               (lcsIdx >= lcs.length || leftLines[leftIdx] !== lcs[lcsIdx])) {
      // Check if this is a modification (similar key, different value)
      if (rightIdx < rightLines.length && 
          isSameKey(leftLines[leftIdx], rightLines[rightIdx])) {
        result.push({
          type: "modified",
          lineNumber: { left: leftIdx + 1, right: rightIdx + 1 },
          content: { left: leftLines[leftIdx], right: rightLines[rightIdx] },
        });
        leftIdx++;
        rightIdx++;
      } else {
        // Line removed from left
        result.push({
          type: "removed",
          lineNumber: { left: leftIdx + 1 },
          content: { left: leftLines[leftIdx] },
        });
        leftIdx++;
      }
    } else if (rightIdx < rightLines.length && 
               (lcsIdx >= lcs.length || rightLines[rightIdx] !== lcs[lcsIdx])) {
      // Line added in right
      result.push({
        type: "added",
        lineNumber: { right: rightIdx + 1 },
        content: { right: rightLines[rightIdx] },
      });
      rightIdx++;
    }
  }

  return result;
}

/**
 * Compute Longest Common Subsequence
 */
function computeLCS(left: string[], right: string[]): string[] {
  const m = left.length;
  const n = right.length;
  
  // DP table
  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
  
  // Fill the DP table
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (left[i - 1] === right[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  
  // Backtrack to find the LCS
  const lcs: string[] = [];
  let i = m, j = n;
  while (i > 0 && j > 0) {
    if (left[i - 1] === right[j - 1]) {
      lcs.unshift(left[i - 1]);
      i--;
      j--;
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  
  return lcs;
}

/**
 * Check if two lines have the same YAML key
 */
function isSameKey(left: string, right: string): boolean {
  const leftMatch = left.match(/^(\s*)([a-zA-Z_][a-zA-Z0-9_]*):/);
  const rightMatch = right.match(/^(\s*)([a-zA-Z_][a-zA-Z0-9_]*):/);
  
  if (leftMatch && rightMatch) {
    // Same key and same indentation level
    return leftMatch[1] === rightMatch[1] && leftMatch[2] === rightMatch[2];
  }
  
  return false;
}

/**
 * Get statistics from diff
 */
function getDiffStats(diff: DiffLine[]): { added: number; removed: number; modified: number } {
  return diff.reduce(
    (acc, line) => {
      if (line.type === "added") acc.added++;
      if (line.type === "removed") acc.removed++;
      if (line.type === "modified") acc.modified++;
      return acc;
    },
    { added: 0, removed: 0, modified: 0 }
  );
}

/**
 * ConfigDiff Props
 */
interface ConfigDiffProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Config Diff Viewer Component
 * Compares current configuration with templates or custom YAML
 */
export function ConfigDiff({ isOpen, onClose }: ConfigDiffProps) {
  const { config } = useConfigStore();
  const [compareSource, setCompareSource] = useState<CompareSource>("template");
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>(BUILT_IN_TEMPLATES[0]?.id || "");
  const [customYaml, setCustomYaml] = useState("");
  const [copied, setCopied] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Get current config as YAML
  const currentYaml = useMemo(() => configToYaml(config), [config]);

  // Get comparison config as YAML
  const comparisonYaml = useMemo(() => {
    if (compareSource === "template") {
      const template = BUILT_IN_TEMPLATES.find(t => t.id === selectedTemplateId);
      return template ? configToYaml(template.config) : "";
    }
    return customYaml;
  }, [compareSource, selectedTemplateId, customYaml]);

  // Compute diff
  const diff = useMemo(() => {
    if (!comparisonYaml) return [];
    return computeDiff(comparisonYaml, currentYaml);
  }, [comparisonYaml, currentYaml]);

  // Diff statistics
  const stats = useMemo(() => getDiffStats(diff), [diff]);

  // Copy diff to clipboard
  const handleCopyDiff = useCallback(async () => {
    const diffText = diff.map(line => {
      const prefix = line.type === "added" ? "+" : 
                    line.type === "removed" ? "-" : 
                    line.type === "modified" ? "~" : " ";
      const content = line.content.right || line.content.left || "";
      return `${prefix} ${content}`;
    }).join("\n");
    
    await navigator.clipboard.writeText(diffText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [diff]);

  // Get selected template name
  const selectedTemplateName = useMemo(() => {
    const template = BUILT_IN_TEMPLATES.find(t => t.id === selectedTemplateId);
    return template?.name || "Select template";
  }, [selectedTemplateId]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      
      {/* Modal */}
      <div 
        className="relative z-10 w-full max-w-5xl mx-4 bg-background rounded-lg shadow-xl border max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="config-diff-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <GitCompare className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h2 id="config-diff-title" className="text-lg font-semibold">Config Diff Viewer</h2>
              <p className="text-xs text-muted-foreground">
                Compare current configuration with templates or previous versions
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close diff viewer"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b shrink-0 space-y-3">
          <div className="flex flex-wrap items-center gap-4">
            {/* Compare source selector */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium" id="compare-source-label">Compare with:</label>
              <div className="flex rounded-lg border overflow-hidden" role="group" aria-labelledby="compare-source-label">
                <button
                  type="button"
                  onClick={() => setCompareSource("template")}
                  className={cn(
                    "px-3 py-1.5 text-sm transition-colors",
                    compareSource === "template" 
                      ? "bg-primary text-primary-foreground" 
                      : "hover:bg-muted"
                  )}
                  aria-pressed={compareSource === "template"}
                >
                  Template
                </button>
                <button
                  type="button"
                  onClick={() => setCompareSource("custom")}
                  className={cn(
                    "px-3 py-1.5 text-sm transition-colors border-l",
                    compareSource === "custom" 
                      ? "bg-primary text-primary-foreground" 
                      : "hover:bg-muted"
                  )}
                  aria-pressed={compareSource === "custom"}
                >
                  Custom YAML
                </button>
              </div>
            </div>

            {/* Template selector */}
            {compareSource === "template" && (
              <div className="relative">
                <label htmlFor="template-select" className="sr-only">Select template to compare</label>
                <button
                  id="template-select"
                  type="button"
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="flex items-center gap-2 px-3 py-1.5 border rounded-lg text-sm hover:bg-muted transition-colors"
                  aria-haspopup="listbox"
                  aria-expanded={showDropdown}
                >
                  <span>{selectedTemplateName}</span>
                  <ChevronDown className={cn(
                    "h-4 w-4 transition-transform",
                    showDropdown && "rotate-180"
                  )} aria-hidden="true" />
                </button>
                
                {showDropdown && (
                  <div 
                    className="absolute top-full left-0 mt-1 w-64 bg-background border rounded-lg shadow-lg z-20 max-h-60 overflow-auto"
                    role="listbox"
                    aria-label="Template options"
                  >
                    {BUILT_IN_TEMPLATES.map((template) => (
                      <button
                        key={template.id}
                        type="button"
                        role="option"
                        aria-selected={selectedTemplateId === template.id}
                        onClick={() => {
                          setSelectedTemplateId(template.id);
                          setShowDropdown(false);
                        }}
                        className={cn(
                          "w-full text-left px-3 py-2 text-sm hover:bg-muted transition-colors",
                          selectedTemplateId === template.id && "bg-primary/10"
                        )}
                      >
                        <div className="font-medium">{template.name}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {template.algorithm.toUpperCase()} - {template.description}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Stats */}
            <div className="flex items-center gap-3 ml-auto text-sm">
              {stats.added > 0 && (
                <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                  <Plus className="h-3 w-3" aria-hidden="true" />
                  <span>{stats.added} added</span>
                </span>
              )}
              {stats.removed > 0 && (
                <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
                  <Minus className="h-3 w-3" aria-hidden="true" />
                  <span>{stats.removed} removed</span>
                </span>
              )}
              {stats.modified > 0 && (
                <span className="flex items-center gap-1 text-yellow-600 dark:text-yellow-400">
                  <Edit3 className="h-3 w-3" aria-hidden="true" />
                  <span>{stats.modified} modified</span>
                </span>
              )}
              {stats.added === 0 && stats.removed === 0 && stats.modified === 0 && comparisonYaml && (
                <span className="text-muted-foreground">No differences</span>
              )}
            </div>
          </div>

          {/* Custom YAML input */}
          {compareSource === "custom" && (
            <div>
              <label htmlFor="custom-yaml-input" className="sr-only">Paste custom YAML to compare</label>
              <textarea
                id="custom-yaml-input"
                value={customYaml}
                onChange={(e) => setCustomYaml(e.target.value)}
                placeholder="Paste YAML configuration to compare..."
                className="w-full h-32 p-3 text-sm font-mono border rounded-lg bg-muted/30 resize-none focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          )}
        </div>

        {/* Diff View */}
        <div className="flex-1 overflow-auto p-4">
          {!comparisonYaml ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <ArrowLeftRight className="h-12 w-12 text-muted-foreground/50 mb-4" aria-hidden="true" />
              <h3 className="font-medium text-muted-foreground">No comparison data</h3>
              <p className="text-sm text-muted-foreground/70 mt-1">
                {compareSource === "template" 
                  ? "Select a template to compare"
                  : "Paste YAML configuration to compare"}
              </p>
            </div>
          ) : (
            <div className="border rounded-lg overflow-hidden">
              {/* Column headers */}
              <div className="flex bg-muted/50 border-b text-sm font-medium">
                <div className="flex-1 px-4 py-2 border-r">
                  {compareSource === "template" ? selectedTemplateName : "Comparison"} (Base)
                </div>
                <div className="flex-1 px-4 py-2">
                  Current Config (Changed)
                </div>
              </div>

              {/* Diff lines */}
              <div className="font-mono text-sm">
                {diff.map((line, idx) => (
                  <div 
                    key={idx} 
                    className={cn(
                      "flex border-b last:border-b-0",
                      line.type === "added" && "bg-green-50 dark:bg-green-950/30",
                      line.type === "removed" && "bg-red-50 dark:bg-red-950/30",
                      line.type === "modified" && "bg-yellow-50 dark:bg-yellow-950/30"
                    )}
                  >
                    {/* Left side (base) */}
                    <div className={cn(
                      "flex-1 flex items-start border-r",
                      line.type === "removed" && "bg-red-100/50 dark:bg-red-900/20",
                      line.type === "modified" && "bg-yellow-100/50 dark:bg-yellow-900/20"
                    )}>
                      <span className="w-10 text-right pr-2 py-1 text-muted-foreground select-none border-r bg-muted/30">
                        {line.lineNumber.left || ""}
                      </span>
                      <span className="w-6 text-center py-1 select-none">
                        {line.type === "removed" && (
                          <Minus className="h-3 w-3 mx-auto text-red-600 dark:text-red-400" aria-label="Removed" />
                        )}
                        {line.type === "modified" && (
                          <Edit3 className="h-3 w-3 mx-auto text-yellow-600 dark:text-yellow-400" aria-label="Modified" />
                        )}
                      </span>
                      <pre className={cn(
                        "flex-1 py-1 pr-2 whitespace-pre-wrap break-all",
                        line.type === "removed" && "text-red-700 dark:text-red-300",
                        line.type === "modified" && "text-yellow-700 dark:text-yellow-300"
                      )}>
                        {line.content.left || ""}
                      </pre>
                    </div>

                    {/* Right side (current) */}
                    <div className={cn(
                      "flex-1 flex items-start",
                      line.type === "added" && "bg-green-100/50 dark:bg-green-900/20",
                      line.type === "modified" && "bg-yellow-100/50 dark:bg-yellow-900/20"
                    )}>
                      <span className="w-10 text-right pr-2 py-1 text-muted-foreground select-none border-r bg-muted/30">
                        {line.lineNumber.right || ""}
                      </span>
                      <span className="w-6 text-center py-1 select-none">
                        {line.type === "added" && (
                          <Plus className="h-3 w-3 mx-auto text-green-600 dark:text-green-400" aria-label="Added" />
                        )}
                        {line.type === "modified" && (
                          <Edit3 className="h-3 w-3 mx-auto text-yellow-600 dark:text-yellow-400" aria-label="Modified" />
                        )}
                      </span>
                      <pre className={cn(
                        "flex-1 py-1 pr-2 whitespace-pre-wrap break-all",
                        line.type === "added" && "text-green-700 dark:text-green-300",
                        line.type === "modified" && "text-yellow-700 dark:text-yellow-300"
                      )}>
                        {line.content.right || ""}
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t shrink-0 flex items-center justify-between bg-muted/30">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <GitCompare className="h-4 w-4" aria-hidden="true" />
            <span>
              Showing differences between configurations. 
              <span className="text-green-600 dark:text-green-400 ml-1">Green = added</span>, 
              <span className="text-red-600 dark:text-red-400 ml-1">Red = removed</span>, 
              <span className="text-yellow-600 dark:text-yellow-400 ml-1">Yellow = modified</span>
            </span>
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleCopyDiff}
            disabled={diff.length === 0}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 mr-2" aria-hidden="true" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 mr-2" aria-hidden="true" />
                Copy Diff
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// Export for testing
export { computeDiff, getDiffStats, configToYaml };
export type { DiffLine, DiffType };
