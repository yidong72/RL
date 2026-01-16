import { useState, useCallback } from "react";
import { Search, ExternalLink, HardDrive } from "lucide-react";
import { Input } from "../common/Input";
import { Button } from "../common/Button";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import type { ModelInfo } from "../../types/config";

// Mock data for model search (in production, this would come from the API)
const POPULAR_MODELS: ModelInfo[] = [
  {
    id: "Qwen/Qwen2.5-1.5B",
    name: "Qwen2.5-1.5B",
    size_gb: 3.0,
    parameters: "1.5B",
    recommended_config: { min_gpus: 1, tensor_parallel: false },
  },
  {
    id: "Qwen/Qwen2.5-7B",
    name: "Qwen2.5-7B",
    size_gb: 14.0,
    parameters: "7B",
    recommended_config: { min_gpus: 2, tensor_parallel: true },
  },
  {
    id: "meta-llama/Llama-3.1-8B",
    name: "Llama-3.1-8B",
    size_gb: 16.0,
    parameters: "8B",
    recommended_config: { min_gpus: 2, tensor_parallel: true },
  },
  {
    id: "Qwen/Qwen2.5-32B",
    name: "Qwen2.5-32B",
    size_gb: 64.0,
    parameters: "32B",
    recommended_config: { min_gpus: 8, tensor_parallel: true },
  },
];

interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  onSelect: () => void;
}

function ModelCard({ model, isSelected, onSelect }: ModelCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "flex items-center justify-between p-4 rounded-lg border transition-all duration-200",
        "hover:border-primary/50 text-left w-full",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border bg-card"
      )}
    >
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium">{model.name}</span>
          {isSelected && (
            <span className="text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">
              Selected
            </span>
          )}
        </div>
        <div className="text-sm text-muted-foreground mt-1">
          <span>{model.parameters} parameters</span>
          <span className="mx-2">•</span>
          <span>{model.size_gb} GB</span>
        </div>
      </div>
      <div className="text-right text-xs text-muted-foreground">
        {model.recommended_config && (
          <div>
            Min GPUs: {model.recommended_config.min_gpus}
            {model.recommended_config.tensor_parallel && (
              <span className="ml-2 text-amber-600 dark:text-amber-400">TP required</span>
            )}
          </div>
        )}
      </div>
    </button>
  );
}

export function ModelSelector() {
  const { config, setModel } = useConfigStore();
  const [searchQuery, setSearchQuery] = useState("");
  const [isLocalPath, setIsLocalPath] = useState(false);
  const [localPath, setLocalPath] = useState("");

  const filteredModels = POPULAR_MODELS.filter(
    (model) =>
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectModel = useCallback(
    (modelId: string) => {
      setModel(modelId);
      setIsLocalPath(false);
    },
    [setModel]
  );

  const handleLocalPathSubmit = useCallback(() => {
    if (localPath.trim()) {
      setModel(localPath.trim());
      setIsLocalPath(true);
    }
  }, [localPath, setModel]);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Select Model</h2>
        <p className="text-sm text-muted-foreground">
          Search HuggingFace Hub or enter a local model path
        </p>
      </div>

      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search models (e.g., 'qwen', 'llama')..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex h-10 w-full rounded-md border border-input bg-background pl-10 pr-3 py-2 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>

      {/* Model list */}
      <div className="space-y-2 max-h-60 overflow-y-auto">
        {filteredModels.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            isSelected={config.model === model.id && !isLocalPath}
            onSelect={() => handleSelectModel(model.id)}
          />
        ))}
        {filteredModels.length === 0 && searchQuery && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No models found. Try a different search term or use a local path.
          </p>
        )}
      </div>

      {/* Local path input */}
      <div className="border-t pt-4">
        <div className="flex items-center gap-2 mb-2">
          <HardDrive className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Or enter local model path</span>
        </div>
        <div className="flex gap-2">
          <Input
            placeholder="/path/to/model or organization/model-name"
            value={localPath}
            onChange={(e) => setLocalPath(e.target.value)}
            className="flex-1"
          />
          <Button
            type="button"
            variant="outline"
            onClick={handleLocalPathSubmit}
            disabled={!localPath.trim()}
          >
            Use Path
          </Button>
        </div>
        {isLocalPath && config.model === localPath && (
          <p className="text-xs text-primary mt-2">
            Using local path: {config.model}
          </p>
        )}
      </div>

      {/* Current selection */}
      <div className="p-3 rounded-lg bg-muted/50 border">
        <div className="text-sm">
          <span className="text-muted-foreground">Selected model: </span>
          <span className="font-medium">{config.model}</span>
          <a
            href={`https://huggingface.co/${config.model}`}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 inline-flex items-center text-primary hover:underline text-xs"
          >
            View on HF <ExternalLink className="h-3 w-3 ml-1" />
          </a>
        </div>
      </div>
    </div>
  );
}
