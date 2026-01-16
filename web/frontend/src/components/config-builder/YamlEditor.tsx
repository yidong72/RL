import { useCallback, useEffect, useRef, useState } from "react";
import Editor, { Monaco, OnMount } from "@monaco-editor/react";
import yaml from "js-yaml";
import { 
  Code2, 
  AlertTriangle, 
  CheckCircle2, 
  RefreshCw,
  Copy,
  Check
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { Button } from "../common/Button";
import type { TrainingConfig, Algorithm, Backend } from "../../types/config";
import type * as MonacoEditor from "monaco-editor";

interface YamlEditorProps {
  className?: string;
}

interface YamlValidationError {
  line: number;
  message: string;
}

/**
 * Convert TrainingConfig to YAML string
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
 * Parse YAML string and extract TrainingConfig
 */
function yamlToConfig(yamlStr: string): { 
  config: Partial<TrainingConfig>; 
  errors: YamlValidationError[];
} {
  const errors: YamlValidationError[] = [];
  const config: Partial<TrainingConfig> = {};

  try {
    const parsed = yaml.load(yamlStr) as Record<string, unknown>;
    
    if (!parsed || typeof parsed !== "object") {
      errors.push({ line: 1, message: "Invalid configuration: expected an object" });
      return { config, errors };
    }

    // Extract algorithm
    if (parsed.algorithm) {
      const alg = String(parsed.algorithm).toLowerCase();
      if (["grpo", "sft", "dpo"].includes(alg)) {
        config.algorithm = alg as Algorithm;
      } else {
        errors.push({ line: 1, message: `Unknown algorithm: ${parsed.algorithm}` });
      }
    }

    // Extract model
    if (parsed.model) {
      if (typeof parsed.model === "string") {
        config.model = parsed.model;
      } else if (typeof parsed.model === "object" && parsed.model !== null) {
        const modelObj = parsed.model as Record<string, unknown>;
        if (modelObj.name) {
          config.model = String(modelObj.name);
        }
        if (modelObj.backend) {
          const backend = String(modelObj.backend).toLowerCase();
          if (["dtensor", "megatron"].includes(backend)) {
            config.backend = backend as Backend;
          }
        }
      }
    }

    // Extract dataset
    if (parsed.dataset) {
      if (typeof parsed.dataset === "string") {
        config.dataset = parsed.dataset;
      } else if (typeof parsed.dataset === "object" && parsed.dataset !== null) {
        const datasetObj = parsed.dataset as Record<string, unknown>;
        if (datasetObj.name) {
          config.dataset = String(datasetObj.name);
        }
      }
    }

    // Extract hyperparameters
    const hyperparameters: Partial<TrainingConfig["hyperparameters"]> = {};
    const trainingObj = (parsed.training || parsed.hyperparameters || {}) as Record<string, unknown>;

    if (trainingObj.learning_rate !== undefined) {
      const lr = Number(trainingObj.learning_rate);
      if (!isNaN(lr) && lr > 0) {
        hyperparameters.learning_rate = lr;
      }
    }
    if (trainingObj.batch_size !== undefined) {
      const bs = Number(trainingObj.batch_size);
      if (!isNaN(bs) && bs > 0) {
        hyperparameters.batch_size = Math.floor(bs);
      }
    }
    if (trainingObj.max_steps !== undefined) {
      const ms = Number(trainingObj.max_steps);
      if (!isNaN(ms) && ms > 0) {
        hyperparameters.max_steps = Math.floor(ms);
      }
    }
    if (trainingObj.num_generations_per_prompt !== undefined) {
      const ng = Number(trainingObj.num_generations_per_prompt);
      if (!isNaN(ng) && ng > 0) {
        hyperparameters.num_generations_per_prompt = Math.floor(ng);
      }
    }
    
    // Extract tensor_parallel_size from model or training
    const tps = trainingObj.tensor_parallel_size ?? 
      (typeof parsed.model === "object" && parsed.model !== null 
        ? (parsed.model as Record<string, unknown>).tensor_parallel_size 
        : undefined);
    if (tps !== undefined) {
      const tpsNum = Number(tps);
      if (!isNaN(tpsNum) && tpsNum > 0) {
        hyperparameters.tensor_parallel_size = Math.floor(tpsNum);
      }
    }

    if (Object.keys(hyperparameters).length > 0) {
      config.hyperparameters = hyperparameters as TrainingConfig["hyperparameters"];
    }

    // Extract cluster config
    const clusterObj = (parsed.cluster || {}) as Record<string, unknown>;
    const cluster: Partial<TrainingConfig["cluster"]> = {};

    if (clusterObj.nodes !== undefined) {
      const nodes = Number(clusterObj.nodes);
      if (!isNaN(nodes) && nodes > 0) {
        cluster.nodes = Math.floor(nodes);
      }
    }
    if (clusterObj.gpus_per_node !== undefined) {
      const gpus = Number(clusterObj.gpus_per_node);
      if (!isNaN(gpus) && gpus > 0) {
        cluster.gpus_per_node = Math.floor(gpus);
      }
    }
    if (clusterObj.time_limit !== undefined) {
      cluster.time_limit = String(clusterObj.time_limit);
    }
    if (clusterObj.partition !== undefined) {
      cluster.partition = String(clusterObj.partition);
    }
    if (clusterObj.account !== undefined) {
      cluster.account = String(clusterObj.account);
    }

    if (Object.keys(cluster).length > 0) {
      config.cluster = cluster as TrainingConfig["cluster"];
    }

  } catch (err) {
    if (err instanceof yaml.YAMLException) {
      errors.push({ 
        line: err.mark?.line ? err.mark.line + 1 : 1, 
        message: err.message 
      });
    } else {
      errors.push({ line: 1, message: (err as Error).message });
    }
  }

  return { config, errors };
}

/**
 * YamlEditor component with Monaco Editor and bi-directional sync
 */
export function YamlEditor({ className }: YamlEditorProps) {
  const { config, isDarkMode, setAlgorithm, setModel, setDataset, setBackend, setHyperparameter, setClusterConfig } = useConfigStore();
  const [yamlContent, setYamlContent] = useState(() => configToYaml(config));
  const [validationErrors, setValidationErrors] = useState<YamlValidationError[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [copied, setCopied] = useState(false);
  const editorRef = useRef<MonacoEditor.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const lastConfigRef = useRef<string>(JSON.stringify(config));
  const isInternalUpdateRef = useRef(false);

  // Handle editor mount
  const handleEditorMount: OnMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    
    // Configure YAML language settings
    monaco.languages.setLanguageConfiguration("yaml", {
      comments: {
        lineComment: "#",
      },
      brackets: [
        ["{", "}"],
        ["[", "]"],
      ],
      autoClosingPairs: [
        { open: "{", close: "}" },
        { open: "[", close: "]" },
        { open: '"', close: '"' },
        { open: "'", close: "'" },
      ],
    });
  }, []);

  // Update editor markers for validation errors
  const updateEditorMarkers = useCallback((errors: YamlValidationError[]) => {
    if (!editorRef.current || !monacoRef.current) return;

    const model = editorRef.current.getModel();
    if (!model) return;

    const markers: MonacoEditor.editor.IMarkerData[] = errors.map(error => ({
      severity: monacoRef.current!.MarkerSeverity.Error,
      message: error.message,
      startLineNumber: error.line,
      startColumn: 1,
      endLineNumber: error.line,
      endColumn: model.getLineMaxColumn(error.line) || 1,
    }));

    monacoRef.current.editor.setModelMarkers(model, "yaml-validation", markers);
  }, []);

  // Handle YAML content change from editor
  const handleEditorChange = useCallback((value: string | undefined) => {
    if (!value || isInternalUpdateRef.current) return;
    
    setYamlContent(value);
    setIsSyncing(true);

    const { config: parsedConfig, errors } = yamlToConfig(value);
    setValidationErrors(errors);
    updateEditorMarkers(errors);

    if (errors.length === 0) {
      // Apply parsed config to store
      isInternalUpdateRef.current = true;
      
      if (parsedConfig.algorithm) {
        setAlgorithm(parsedConfig.algorithm);
      }
      if (parsedConfig.model) {
        setModel(parsedConfig.model);
      }
      if (parsedConfig.dataset) {
        setDataset(parsedConfig.dataset);
      }
      if (parsedConfig.backend) {
        setBackend(parsedConfig.backend);
      }
      if (parsedConfig.hyperparameters) {
        Object.entries(parsedConfig.hyperparameters).forEach(([key, value]) => {
          setHyperparameter(key as keyof TrainingConfig["hyperparameters"], value);
        });
      }
      if (parsedConfig.cluster) {
        Object.entries(parsedConfig.cluster).forEach(([key, value]) => {
          setClusterConfig(key as keyof TrainingConfig["cluster"], value);
        });
      }
      
      // Update last config to prevent re-sync
      lastConfigRef.current = JSON.stringify(useConfigStore.getState().config);
      
      setTimeout(() => {
        isInternalUpdateRef.current = false;
      }, 100);
    }

    setIsSyncing(false);
  }, [setAlgorithm, setModel, setDataset, setBackend, setHyperparameter, setClusterConfig, updateEditorMarkers]);

  // Sync from form to editor when config changes
  // This is intentional bi-directional sync - we need to update editor when form changes
  useEffect(() => {
    const currentConfig = JSON.stringify(config);
    if (currentConfig !== lastConfigRef.current && !isInternalUpdateRef.current) {
      lastConfigRef.current = currentConfig;
      const newYaml = configToYaml(config);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional bi-directional sync
      setYamlContent(newYaml);
      setValidationErrors([]);
      if (monacoRef.current && editorRef.current) {
        const model = editorRef.current.getModel();
        if (model) {
          monacoRef.current.editor.setModelMarkers(model, "yaml-validation", []);
        }
      }
    }
  }, [config]);

  // Refresh editor content from store
  const handleRefresh = useCallback(() => {
    const newYaml = configToYaml(config);
    setYamlContent(newYaml);
    setValidationErrors([]);
    lastConfigRef.current = JSON.stringify(config);
    if (monacoRef.current && editorRef.current) {
      const model = editorRef.current.getModel();
      if (model) {
        monacoRef.current.editor.setModelMarkers(model, "yaml-validation", []);
      }
    }
  }, [config]);

  // Copy YAML to clipboard
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(yamlContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [yamlContent]);

  const isValid = validationErrors.length === 0;

  return (
    <div className={cn("flex flex-col", className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Code2 className="h-5 w-5" />
          YAML Editor
          {isSyncing && (
            <span className="text-xs text-muted-foreground">(syncing...)</span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          {/* Validation status */}
          <div className={cn(
            "flex items-center gap-1 px-2 py-1 rounded text-xs",
            isValid 
              ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
              : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
          )}>
            {isValid ? (
              <>
                <CheckCircle2 className="h-3 w-3" />
                Valid YAML
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" />
                {validationErrors.length} error{validationErrors.length !== 1 ? "s" : ""}
              </>
            )}
          </div>
          
          <Button variant="ghost" size="sm" onClick={handleRefresh} title="Refresh from form">
            <RefreshCw className="h-4 w-4" />
          </Button>
          
          <Button variant="ghost" size="sm" onClick={handleCopy}>
            {copied ? (
              <Check className="h-4 w-4" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Validation errors */}
      {validationErrors.length > 0 && (
        <div className="mb-3 space-y-1">
          {validationErrors.map((error, idx) => (
            <div 
              key={idx} 
              className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded"
            >
              <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
              <span>Line {error.line}: {error.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Monaco Editor */}
      <div className="flex-1 border rounded-lg overflow-hidden min-h-[400px]">
        <Editor
          height="400px"
          language="yaml"
          value={yamlContent}
          theme={isDarkMode ? "vs-dark" : "light"}
          onChange={handleEditorChange}
          onMount={handleEditorMount}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            wordWrap: "on",
            folding: true,
            renderLineHighlight: "line",
            selectOnLineNumbers: true,
            roundedSelection: true,
            cursorBlinking: "smooth",
            cursorSmoothCaretAnimation: "on",
            smoothScrolling: true,
            padding: { top: 8, bottom: 8 },
          }}
        />
      </div>

      {/* Help text */}
      <p className="mt-2 text-xs text-muted-foreground">
        Edit the YAML directly. Changes sync automatically to the form when valid.
      </p>
    </div>
  );
}
