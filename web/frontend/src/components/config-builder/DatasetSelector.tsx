import { useState, useCallback } from "react";
import { Search, Database, HardDrive, ExternalLink, FileText, Users } from "lucide-react";
import { Input } from "../common/Input";
import { Button } from "../common/Button";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import type { DatasetInfo } from "../../types/config";

// Mock data for dataset search (in production, this would come from the API)
const POPULAR_DATASETS: DatasetInfo[] = [
  {
    id: "nvidia/OpenMathInstruct-2",
    name: "OpenMathInstruct-2",
    description: "Large-scale math instruction dataset for training math reasoning models",
    size: "14.5M examples",
    features: ["problem", "solution", "answer"],
    downloads: 50000,
  },
  {
    id: "nvidia/HelpSteer2",
    name: "HelpSteer2",
    description: "Human preference dataset for reward modeling and alignment",
    size: "21K examples",
    features: ["prompt", "response", "helpfulness", "correctness"],
    downloads: 25000,
  },
  {
    id: "tatsu-lab/alpaca",
    name: "Alpaca",
    description: "Instruction-following dataset based on self-instruct",
    size: "52K examples",
    features: ["instruction", "input", "output"],
    downloads: 100000,
  },
  {
    id: "openai/gsm8k",
    name: "GSM8K",
    description: "Grade school math word problems with step-by-step solutions",
    size: "8.5K train examples",
    features: ["question", "answer"],
    downloads: 75000,
  },
  {
    id: "hendrycks/competition_math",
    name: "MATH",
    description: "Competition mathematics problems from AMC, AIME, etc.",
    size: "12.5K examples",
    features: ["problem", "level", "type", "solution"],
    downloads: 40000,
  },
  {
    id: "codeparrot/github-code",
    name: "GitHub Code",
    description: "Large-scale code dataset from GitHub repositories",
    size: "1.2B files",
    features: ["code", "language", "license"],
    downloads: 60000,
  },
];

interface DatasetCardProps {
  dataset: DatasetInfo;
  isSelected: boolean;
  onSelect: () => void;
}

function DatasetCard({ dataset, isSelected, onSelect }: DatasetCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "flex flex-col p-4 rounded-lg border transition-all duration-200 text-left w-full",
        "hover:border-primary/50",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border bg-card"
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground shrink-0" />
          <span className="font-medium">{dataset.name}</span>
        </div>
        {isSelected && (
          <span className="text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">
            Selected
          </span>
        )}
      </div>
      <p className="text-sm text-muted-foreground mt-2 line-clamp-2">
        {dataset.description}
      </p>
      <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <FileText className="h-3 w-3" />
          {dataset.size}
        </span>
        <span className="flex items-center gap-1">
          <Users className="h-3 w-3" />
          {dataset.downloads.toLocaleString()} downloads
        </span>
      </div>
      <div className="flex flex-wrap gap-1 mt-2">
        {dataset.features.slice(0, 4).map((feature) => (
          <span
            key={feature}
            className={cn(
              "text-xs px-2 py-0.5 rounded-full",
              isSelected
                ? "bg-primary/10 text-primary"
                : "bg-muted text-muted-foreground"
            )}
          >
            {feature}
          </span>
        ))}
        {dataset.features.length > 4 && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
            +{dataset.features.length - 4} more
          </span>
        )}
      </div>
    </button>
  );
}

export function DatasetSelector() {
  const { config, setDataset } = useConfigStore();
  const [searchQuery, setSearchQuery] = useState("");
  const [isLocalPath, setIsLocalPath] = useState(false);
  const [localPath, setLocalPath] = useState("");

  const filteredDatasets = POPULAR_DATASETS.filter(
    (dataset) =>
      dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dataset.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dataset.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectDataset = useCallback(
    (datasetId: string) => {
      setDataset(datasetId);
      setIsLocalPath(false);
    },
    [setDataset]
  );

  const handleLocalPathSubmit = useCallback(() => {
    if (localPath.trim()) {
      setDataset(localPath.trim());
      setIsLocalPath(true);
    }
  }, [localPath, setDataset]);

  const selectedDataset = POPULAR_DATASETS.find((d) => d.id === config.dataset);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Select Dataset</h2>
        <p className="text-sm text-muted-foreground">
          Search HuggingFace Hub datasets or enter a local dataset path
        </p>
      </div>

      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search datasets (e.g., 'OpenMathInstruct', 'gsm8k')..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex h-10 w-full rounded-md border border-input bg-background pl-10 pr-3 py-2 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>

      {/* Dataset list */}
      <div className="space-y-2 max-h-72 overflow-y-auto">
        {filteredDatasets.map((dataset) => (
          <DatasetCard
            key={dataset.id}
            dataset={dataset}
            isSelected={config.dataset === dataset.id && !isLocalPath}
            onSelect={() => handleSelectDataset(dataset.id)}
          />
        ))}
        {filteredDatasets.length === 0 && searchQuery && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No datasets found. Try a different search term or use a local path.
          </p>
        )}
      </div>

      {/* Local path input */}
      <div className="border-t pt-4">
        <div className="flex items-center gap-2 mb-2">
          <HardDrive className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Or enter local dataset path</span>
        </div>
        <div className="flex gap-2">
          <Input
            placeholder="/path/to/dataset or organization/dataset-name"
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
        {isLocalPath && config.dataset === localPath && (
          <p className="text-xs text-primary mt-2">
            Using local path: {config.dataset}
          </p>
        )}
      </div>

      {/* Current selection info */}
      <div className="p-3 rounded-lg bg-muted/50 border">
        <div className="text-sm">
          <span className="text-muted-foreground">Selected dataset: </span>
          <span className="font-medium">{config.dataset}</span>
          <a
            href={`https://huggingface.co/datasets/${config.dataset}`}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 inline-flex items-center text-primary hover:underline text-xs"
          >
            View on HF <ExternalLink className="h-3 w-3 ml-1" />
          </a>
        </div>
        {selectedDataset && (
          <div className="mt-2 text-xs text-muted-foreground">
            <p>{selectedDataset.description}</p>
            <p className="mt-1">
              <span className="font-medium">Features:</span>{" "}
              {selectedDataset.features.join(", ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
