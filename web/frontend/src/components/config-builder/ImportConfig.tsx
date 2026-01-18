import { useState, useCallback, useRef } from "react";
import { 
  Upload, 
  FileText, 
  Clipboard, 
  AlertCircle, 
  CheckCircle2,
  X 
} from "lucide-react";
import yaml from "js-yaml";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { Button } from "../common/Button";
import type { TrainingConfig, Algorithm, Backend } from "../../types/config";

interface ImportError {
  field?: string;
  message: string;
}

interface ImportConfigProps {
  onClose?: () => void;
}

/**
 * Validate and normalize imported config data
 */
function validateAndNormalizeConfig(data: unknown): { 
  config: Partial<TrainingConfig>; 
  errors: ImportError[]; 
  warnings: string[];
} {
  const errors: ImportError[] = [];
  const warnings: string[] = [];
  const config: Partial<TrainingConfig> = {};

  if (!data || typeof data !== "object") {
    errors.push({ message: "Invalid configuration: expected an object" });
    return { config, errors, warnings };
  }

  const obj = data as Record<string, unknown>;

  // Extract algorithm
  if (obj.algorithm) {
    const alg = String(obj.algorithm).toLowerCase();
    if (["grpo", "sft", "dpo"].includes(alg)) {
      config.algorithm = alg as Algorithm;
    } else {
      errors.push({ field: "algorithm", message: `Unknown algorithm: ${obj.algorithm}` });
    }
  }

  // Extract model
  if (obj.model) {
    if (typeof obj.model === "string") {
      config.model = obj.model;
    } else if (typeof obj.model === "object" && obj.model !== null) {
      const modelObj = obj.model as Record<string, unknown>;
      if (modelObj.name) {
        config.model = String(modelObj.name);
      }
      if (modelObj.backend) {
        const backend = String(modelObj.backend).toLowerCase();
        if (["dtensor", "megatron"].includes(backend)) {
          config.backend = backend as Backend;
        } else {
          warnings.push(`Unknown backend: ${modelObj.backend}, using default`);
        }
      }
    }
  }

  // Extract dataset
  if (obj.dataset) {
    if (typeof obj.dataset === "string") {
      config.dataset = obj.dataset;
    } else if (typeof obj.dataset === "object" && obj.dataset !== null) {
      const datasetObj = obj.dataset as Record<string, unknown>;
      if (datasetObj.name) {
        config.dataset = String(datasetObj.name);
      }
    }
  }

  // Extract backend (top-level)
  if (obj.backend && !config.backend) {
    const backend = String(obj.backend).toLowerCase();
    if (["dtensor", "megatron"].includes(backend)) {
      config.backend = backend as Backend;
    }
  }

  // Extract hyperparameters
  const hyperparameters: Partial<TrainingConfig["hyperparameters"]> = {};
  const trainingObj = (obj.training || obj.hyperparameters || {}) as Record<string, unknown>;

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
  if (trainingObj.tensor_parallel_size !== undefined || 
      (typeof obj.model === "object" && (obj.model as Record<string, unknown>).tensor_parallel_size !== undefined)) {
    const tps = Number(
      trainingObj.tensor_parallel_size ?? 
      (obj.model as Record<string, unknown>).tensor_parallel_size
    );
    if (!isNaN(tps) && tps > 0) {
      hyperparameters.tensor_parallel_size = Math.floor(tps);
    }
  }

  if (Object.keys(hyperparameters).length > 0) {
    config.hyperparameters = hyperparameters as TrainingConfig["hyperparameters"];
  }

  // Extract cluster config
  const clusterObj = (obj.cluster || {}) as Record<string, unknown>;
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

  // Add warnings for unrecognized fields
  const knownFields = ["algorithm", "model", "dataset", "backend", "training", "hyperparameters", "cluster", "checkpointing", "logging"];
  Object.keys(obj).forEach(key => {
    if (!knownFields.includes(key)) {
      warnings.push(`Unrecognized field: ${key}`);
    }
  });

  return { config, errors, warnings };
}

/**
 * ImportConfig component for importing YAML/JSON configurations
 */
export function ImportConfig({ onClose }: ImportConfigProps) {
  const { setAlgorithm, setModel, setDataset, setBackend, setHyperparameter, setClusterConfig } = useConfigStore();
  const [isDragging, setIsDragging] = useState(false);
  const [pasteContent, setPasteContent] = useState("");
  const [importStatus, setImportStatus] = useState<"idle" | "success" | "error">("idle");
  const [errors, setErrors] = useState<ImportError[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Parse content and apply to store
   */
  const handleImport = useCallback((content: string) => {
    setErrors([]);
    setWarnings([]);

    try {
      // Try parsing as YAML (which also handles JSON)
      let parsed: unknown;
      try {
        parsed = yaml.load(content);
      } catch (yamlError) {
        // Try JSON as fallback
        try {
          parsed = JSON.parse(content);
        } catch {
          throw new Error("Invalid YAML/JSON format: " + (yamlError as Error).message);
        }
      }

      const { config, errors: validationErrors, warnings: validationWarnings } = validateAndNormalizeConfig(parsed);

      if (validationErrors.length > 0) {
        setErrors(validationErrors);
        setImportStatus("error");
        return;
      }

      setWarnings(validationWarnings);

      // Apply config to store
      if (config.algorithm) {
        setAlgorithm(config.algorithm);
      }
      if (config.model) {
        setModel(config.model);
      }
      if (config.dataset) {
        setDataset(config.dataset);
      }
      if (config.backend) {
        setBackend(config.backend);
      }
      if (config.hyperparameters) {
        Object.entries(config.hyperparameters).forEach(([key, value]) => {
          setHyperparameter(key as keyof TrainingConfig["hyperparameters"], value);
        });
      }
      if (config.cluster) {
        Object.entries(config.cluster).forEach(([key, value]) => {
          setClusterConfig(key as keyof TrainingConfig["cluster"], value);
        });
      }

      setImportStatus("success");
      
      // Clear paste content after successful import
      setPasteContent("");
      
      // Close modal after a short delay
      if (onClose) {
        setTimeout(onClose, 1500);
      }
    } catch (err) {
      setErrors([{ message: (err as Error).message }]);
      setImportStatus("error");
    }
  }, [setAlgorithm, setModel, setDataset, setBackend, setHyperparameter, setClusterConfig, onClose]);

  /**
   * Handle file upload
   */
  const handleFileUpload = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result;
      if (typeof content === "string") {
        handleImport(content);
      }
    };
    reader.onerror = () => {
      setErrors([{ message: "Failed to read file" }]);
      setImportStatus("error");
    };
    reader.readAsText(file);
  }, [handleImport]);

  /**
   * Handle drag events
   */
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  /**
   * Handle paste from clipboard
   */
  const handlePasteFromClipboard = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      setPasteContent(text);
    } catch {
      setErrors([{ message: "Unable to read from clipboard. Please paste manually." }]);
    }
  }, []);

  /**
   * Handle file input change
   */
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Import Configuration
        </h3>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Status messages */}
      {importStatus === "success" && (
        <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800 rounded-lg">
          <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
          <div>
            <p className="font-medium text-green-700 dark:text-green-300">Configuration imported successfully</p>
            {warnings.length > 0 && (
              <p className="text-sm text-green-600 dark:text-green-400">
                {warnings.length} warning{warnings.length !== 1 ? "s" : ""}: {warnings[0]}
                {warnings.length > 1 && ` (+${warnings.length - 1} more)`}
              </p>
            )}
          </div>
        </div>
      )}

      {errors.length > 0 && (
        <div className="space-y-2">
          {errors.map((error, idx) => (
            <div key={idx} className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                {error.field && (
                  <p className="text-sm font-medium text-red-700 dark:text-red-300">{error.field}</p>
                )}
                <p className="text-sm text-red-600 dark:text-red-400">{error.message}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Drag & Drop Zone */}
      <div
        className={cn(
          "relative border-2 border-dashed rounded-lg p-8 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50"
        )}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml,.json"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleFileInputChange}
        />
        <div className="space-y-4">
          <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center">
            <FileText className="h-6 w-6 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium">
              {isDragging ? "Drop your file here" : "Drag & drop your config file"}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Supports .yaml, .yml, and .json files
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
          >
            Browse Files
          </Button>
        </div>
      </div>

      {/* Paste Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium">Or paste YAML/JSON content</h4>
          <Button
            variant="ghost"
            size="sm"
            onClick={handlePasteFromClipboard}
          >
            <Clipboard className="h-4 w-4 mr-2" />
            Paste from Clipboard
          </Button>
        </div>
        <textarea
          value={pasteContent}
          onChange={(e) => setPasteContent(e.target.value)}
          placeholder={`# Example YAML config
algorithm: grpo
model:
  name: Qwen/Qwen2.5-1.5B
  backend: dtensor
dataset:
  name: nvidia/OpenMathInstruct-2
training:
  learning_rate: 1e-6
  batch_size: 32
  max_steps: 1000`}
          className="w-full h-48 px-3 py-2 text-sm font-mono bg-muted/30 border rounded-lg resize-none focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <Button
          variant="default"
          className="w-full"
          onClick={() => handleImport(pasteContent)}
          disabled={!pasteContent.trim()}
        >
          Import Configuration
        </Button>
      </div>

      {/* Supported Formats */}
      <div className="p-4 bg-muted/30 rounded-lg">
        <h4 className="text-sm font-medium mb-2">Supported Formats</h4>
        <ul className="text-xs text-muted-foreground space-y-1">
          <li>• NeMo RL YAML configuration files</li>
          <li>• Simplified JSON format with training parameters</li>
          <li>• Nested model/dataset objects or flat string values</li>
        </ul>
      </div>
    </div>
  );
}
