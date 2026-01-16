import { useEffect, useCallback, useRef } from "react";
import { AlertCircle, AlertTriangle, Info, CheckCircle2, Loader2, Cpu, Clock, Zap } from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { api } from "../../api/client";
import type { ValidationError, ResourceEstimate } from "../../types/config";

// Debounce delay in milliseconds
const VALIDATION_DEBOUNCE_MS = 500;

/**
 * Hook to perform debounced validation of the config
 */
function useValidation() {
  const { config, validation, isValidating, setValidation, setIsValidating } = useConfigStore();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const validate = useCallback(async () => {
    // Cancel any pending validation
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsValidating(true);
    try {
      const result = await api.validateConfig(config);
      setValidation(result);
    } catch (error) {
      // Only set error if not aborted
      if (error instanceof Error && error.name !== "AbortError") {
        console.error("Validation error:", error);
        // Set a fallback validation result on error
        setValidation({
          valid: false,
          errors: [{
            field: "connection",
            message: "Unable to connect to validation service. Please check if the backend is running.",
            severity: "error"
          }],
          warnings: []
        });
      }
    } finally {
      setIsValidating(false);
    }
  }, [config, setValidation, setIsValidating]);

  // Debounced validation effect
  useEffect(() => {
    // Clear existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Set new timeout for debounced validation
    timeoutRef.current = setTimeout(() => {
      validate();
    }, VALIDATION_DEBOUNCE_MS);

    // Cleanup
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [validate]);

  return { validation, isValidating };
}

/**
 * Component to display a single validation error/warning
 */
interface ValidationItemProps {
  error: ValidationError;
}

function ValidationItem({ error }: ValidationItemProps) {
  const severityStyles = {
    error: {
      icon: AlertCircle,
      containerClass: "bg-red-50 border-red-200 dark:bg-red-950/20 dark:border-red-800",
      iconClass: "text-red-600 dark:text-red-400",
      textClass: "text-red-700 dark:text-red-300",
    },
    warning: {
      icon: AlertTriangle,
      containerClass: "bg-amber-50 border-amber-200 dark:bg-amber-950/20 dark:border-amber-800",
      iconClass: "text-amber-600 dark:text-amber-400",
      textClass: "text-amber-700 dark:text-amber-300",
    },
    info: {
      icon: Info,
      containerClass: "bg-blue-50 border-blue-200 dark:bg-blue-950/20 dark:border-blue-800",
      iconClass: "text-blue-600 dark:text-blue-400",
      textClass: "text-blue-700 dark:text-blue-300",
    },
  };

  const style = severityStyles[error.severity];
  const Icon = style.icon;

  return (
    <div className={cn("flex items-start gap-3 p-3 rounded-lg border", style.containerClass)}>
      <Icon className={cn("h-5 w-5 flex-shrink-0 mt-0.5", style.iconClass)} />
      <div className="flex-1 min-w-0">
        <p className={cn("text-sm font-medium", style.textClass)}>
          {error.field.replace(/\./g, " > ")}
        </p>
        <p className={cn("text-sm mt-0.5", style.textClass)}>{error.message}</p>
      </div>
    </div>
  );
}

/**
 * Component to display resource estimates
 */
interface ResourceEstimateDisplayProps {
  estimate: ResourceEstimate;
}

function ResourceEstimateDisplay({ estimate }: ResourceEstimateDisplayProps) {
  return (
    <div className="p-4 rounded-lg border bg-muted/30">
      <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
        <Zap className="h-4 w-4" />
        Resource Estimates
      </h4>
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="flex items-center justify-center mb-1">
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </div>
          <p className="text-xs text-muted-foreground">Memory/GPU</p>
          <p className="text-sm font-medium">{estimate.memory_per_gpu}</p>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center mb-1">
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </div>
          <p className="text-xs text-muted-foreground">Recommended GPUs</p>
          <p className="text-sm font-medium">{estimate.recommended_gpus}</p>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center mb-1">
            <Clock className="h-4 w-4 text-muted-foreground" />
          </div>
          <p className="text-xs text-muted-foreground">Est. Time</p>
          <p className="text-sm font-medium">{estimate.estimated_time}</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Main ValidationDisplay component
 * Shows real-time validation results including errors, warnings, and resource estimates
 */
export function ValidationDisplay() {
  const { validation, isValidating } = useValidation();

  // Loading state
  if (isValidating && !validation) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Configuration Validation</h3>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Validating...
          </div>
        </div>
        <div className="p-8 flex items-center justify-center border rounded-lg">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  // No validation result yet
  if (!validation) {
    return null;
  }

  const hasErrors = validation.errors.length > 0;
  const hasWarnings = validation.warnings.length > 0;

  return (
    <div className="space-y-4">
      {/* Header with status */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Configuration Validation</h3>
        <div className="flex items-center gap-2">
          {isValidating && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Validating...
            </div>
          )}
          {!isValidating && (
            validation.valid ? (
              <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
                <CheckCircle2 className="h-4 w-4" />
                Valid Configuration
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
                <AlertCircle className="h-4 w-4" />
                {validation.errors.length} Error{validation.errors.length !== 1 ? "s" : ""}
              </div>
            )
          )}
        </div>
      </div>

      {/* Validation summary */}
      <div
        className={cn(
          "p-4 rounded-lg border",
          validation.valid
            ? "bg-green-50 border-green-200 dark:bg-green-950/20 dark:border-green-800"
            : "bg-red-50 border-red-200 dark:bg-red-950/20 dark:border-red-800"
        )}
      >
        <div className="flex items-center gap-3">
          {validation.valid ? (
            <>
              <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
              <div>
                <p className="font-medium text-green-700 dark:text-green-300">
                  Configuration is valid
                </p>
                <p className="text-sm text-green-600 dark:text-green-400">
                  {hasWarnings
                    ? `${validation.warnings.length} warning${validation.warnings.length !== 1 ? "s" : ""} to review`
                    : "Ready to generate training script"}
                </p>
              </div>
            </>
          ) : (
            <>
              <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
              <div>
                <p className="font-medium text-red-700 dark:text-red-300">
                  Configuration has errors
                </p>
                <p className="text-sm text-red-600 dark:text-red-400">
                  Please fix the {validation.errors.length} error{validation.errors.length !== 1 ? "s" : ""} below before generating scripts
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Errors section */}
      {hasErrors && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-red-700 dark:text-red-300 flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Errors ({validation.errors.length})
          </h4>
          <div className="space-y-2">
            {validation.errors.map((error, index) => (
              <ValidationItem key={`error-${index}`} error={error} />
            ))}
          </div>
        </div>
      )}

      {/* Warnings section */}
      {hasWarnings && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-amber-700 dark:text-amber-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Warnings & Suggestions ({validation.warnings.length})
          </h4>
          <div className="space-y-2">
            {validation.warnings.map((warning, index) => (
              <ValidationItem key={`warning-${index}`} error={warning} />
            ))}
          </div>
        </div>
      )}

      {/* Resource estimates */}
      {validation.resource_estimate && (
        <ResourceEstimateDisplay estimate={validation.resource_estimate} />
      )}
    </div>
  );
}
