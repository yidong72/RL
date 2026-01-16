import { useState, useCallback, useMemo } from "react";
import { HelpCircle, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import type { Algorithm, HyperparametersConfig } from "../../types/config";

// Hyperparameter metadata with validation rules and tooltips
interface HyperparameterMeta {
  key: keyof HyperparametersConfig;
  label: string;
  tooltip: string;
  min: number;
  max: number;
  step: number;
  default: number;
  type: "number" | "float";
  algorithms?: Algorithm[]; // If specified, only show for these algorithms
}

const HYPERPARAMETER_META: HyperparameterMeta[] = [
  {
    key: "learning_rate",
    label: "Learning Rate",
    tooltip: "Controls how much to adjust model weights during training. Lower values (1e-6 to 5e-6) are typical for fine-tuning.",
    min: 1e-8,
    max: 1e-2,
    step: 1e-7,
    default: 1e-6,
    type: "float",
  },
  {
    key: "batch_size",
    label: "Batch Size",
    tooltip: "Number of samples processed before updating model weights. Larger batches require more GPU memory but can improve training stability.",
    min: 1,
    max: 1024,
    step: 1,
    default: 32,
    type: "number",
  },
  {
    key: "max_steps",
    label: "Max Steps",
    tooltip: "Total number of training steps to run. More steps allow more learning but increase training time.",
    min: 1,
    max: 1000000,
    step: 1,
    default: 1000,
    type: "number",
  },
  {
    key: "num_generations_per_prompt",
    label: "Generations per Prompt",
    tooltip: "Number of response generations per prompt for GRPO. More generations improve reward estimation but increase compute cost.",
    min: 1,
    max: 64,
    step: 1,
    default: 16,
    type: "number",
    algorithms: ["grpo"], // Only show for GRPO
  },
  {
    key: "tensor_parallel_size",
    label: "Tensor Parallel Size",
    tooltip: "Number of GPUs for tensor parallelism. Must divide evenly into GPUs per node. Use 1 for no tensor parallelism, or 2/4/8 for larger models with Megatron backend.",
    min: 1,
    max: 8,
    step: 1,
    default: 1,
    type: "number",
  },
];

interface TooltipProps {
  content: string;
}

function Tooltip({ content }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-flex">
      <button
        type="button"
        className="p-1 hover:bg-muted rounded-full transition-colors"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label="More information"
      >
        <HelpCircle className="h-4 w-4 text-muted-foreground" />
      </button>
      {isVisible && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 w-64 p-2 bg-popover border border-border rounded-md shadow-md text-xs text-popover-foreground">
          {content}
          <div className="absolute left-1/2 -translate-x-1/2 top-full border-4 border-transparent border-t-popover" />
        </div>
      )}
    </div>
  );
}

interface ValidationState {
  isValid: boolean;
  message?: string;
}

function validateValue(
  value: number,
  meta: HyperparameterMeta
): ValidationState {
  if (isNaN(value)) {
    return { isValid: false, message: "Please enter a valid number" };
  }
  if (value < meta.min) {
    return {
      isValid: false,
      message: `Minimum value is ${meta.type === "float" ? meta.min.toExponential() : meta.min}`,
    };
  }
  if (value > meta.max) {
    return {
      isValid: false,
      message: `Maximum value is ${meta.type === "float" ? meta.max.toExponential() : meta.max}`,
    };
  }
  return { isValid: true };
}

interface HyperparameterInputProps {
  meta: HyperparameterMeta;
  value: number;
  onChange: (value: number) => void;
}

function HyperparameterInput({ meta, value, onChange }: HyperparameterInputProps) {
  const [localValue, setLocalValue] = useState(
    meta.type === "float" ? value.toExponential() : String(value)
  );
  const [isFocused, setIsFocused] = useState(false);

  const validation = useMemo(() => validateValue(value, meta), [value, meta]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const inputValue = e.target.value;
      setLocalValue(inputValue);

      const numValue = parseFloat(inputValue);
      if (!isNaN(numValue)) {
        onChange(numValue);
      }
    },
    [onChange]
  );

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    // Reformat to proper display format on blur
    if (meta.type === "float") {
      setLocalValue(value.toExponential());
    } else {
      setLocalValue(String(value));
    }
  }, [meta.type, value]);

  const handleFocus = useCallback(() => {
    setIsFocused(true);
  }, []);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <label
            htmlFor={`hp-${meta.key}`}
            className="text-sm font-medium leading-none"
          >
            {meta.label}
          </label>
          <Tooltip content={meta.tooltip} />
        </div>
        {!validation.isValid && (
          <span className="flex items-center gap-1 text-xs text-destructive">
            <AlertCircle className="h-3 w-3" />
            {validation.message}
          </span>
        )}
        {validation.isValid && !isFocused && (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        )}
      </div>
      <div className="relative">
        <input
          id={`hp-${meta.key}`}
          type="text"
          value={localValue}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={handleFocus}
          className={cn(
            "flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm shadow-sm transition-colors",
            "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1",
            validation.isValid
              ? "border-input focus-visible:ring-ring"
              : "border-destructive focus-visible:ring-destructive"
          )}
          aria-invalid={!validation.isValid}
          aria-describedby={!validation.isValid ? `${meta.key}-error` : undefined}
        />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>
          Range: {meta.type === "float" ? meta.min.toExponential() : meta.min} -{" "}
          {meta.type === "float" ? meta.max.toExponential() : meta.max.toLocaleString()}
        </span>
        <span>Default: {meta.type === "float" ? meta.default.toExponential() : meta.default}</span>
      </div>
    </div>
  );
}

export function HyperparameterForm() {
  const { config, setHyperparameter } = useConfigStore();
  const { hyperparameters, algorithm } = config;

  // Filter hyperparameters based on current algorithm
  const visibleParams = HYPERPARAMETER_META.filter((meta) => {
    if (!meta.algorithms) return true;
    return meta.algorithms.includes(algorithm);
  });

  const handleChange = useCallback(
    (key: keyof HyperparametersConfig, value: number) => {
      setHyperparameter(key, value);
    },
    [setHyperparameter]
  );

  // Calculate if all validations pass
  const allValid = visibleParams.every((meta) =>
    validateValue(hyperparameters[meta.key] as number, meta).isValid
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Hyperparameters</h2>
        <p className="text-sm text-muted-foreground">
          Configure training hyperparameters. Hover over <HelpCircle className="h-3 w-3 inline" /> for explanations.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visibleParams.map((meta) => (
          <HyperparameterInput
            key={meta.key}
            meta={meta}
            value={hyperparameters[meta.key] as number}
            onChange={(value) => handleChange(meta.key, value)}
          />
        ))}
      </div>

      {/* Validation summary */}
      <div
        className={cn(
          "p-4 rounded-lg border",
          allValid
            ? "bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800"
            : "bg-amber-50 border-amber-200 dark:bg-amber-950/20 dark:border-amber-800"
        )}
      >
        <div className="flex items-center gap-2">
          {allValid ? (
            <>
              <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-green-700 dark:text-green-300">
                All hyperparameters are valid
              </span>
            </>
          ) : (
            <>
              <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
              <span className="text-sm font-medium text-amber-700 dark:text-amber-300">
                Some hyperparameters have validation errors
              </span>
            </>
          )}
        </div>
        <div className="mt-2 text-xs text-muted-foreground">
          {algorithm === "grpo" && (
            <p>
              GRPO-specific: <span className="font-medium">Generations per Prompt</span> controls
              how many responses are sampled for each prompt.
            </p>
          )}
          {algorithm === "sft" && (
            <p>
              SFT tip: Use a lower learning rate (1e-6 to 1e-5) for stable fine-tuning.
            </p>
          )}
          {algorithm === "dpo" && (
            <p>
              DPO tip: Batch size should be divisible by 2 since DPO uses preference pairs.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
