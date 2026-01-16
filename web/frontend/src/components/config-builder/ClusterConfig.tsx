import { useState, useCallback } from "react";
import { Server, Cpu, Clock, HelpCircle, ChevronDown, Check } from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import type { Backend, ClusterConfig as ClusterConfigType } from "../../types/config";

// Cluster presets
interface ClusterPreset {
  id: string;
  name: string;
  description: string;
  defaults: Partial<ClusterConfigType> & { backend?: Backend };
}

const CLUSTER_PRESETS: ClusterPreset[] = [
  {
    id: "dgx-cloud",
    name: "DGX Cloud",
    description: "NVIDIA DGX Cloud environment with H100 GPUs",
    defaults: {
      nodes: 1,
      gpus_per_node: 8,
      time_limit: "4:00:00",
      partition: "batch",
      backend: "dtensor",
    },
  },
  {
    id: "bcm",
    name: "BCM (Base Command Manager)",
    description: "NVIDIA BCM-managed cluster with Luna partition",
    defaults: {
      nodes: 1,
      gpus_per_node: 8,
      time_limit: "8:00:00",
      partition: "luna",
      backend: "megatron",
    },
  },
  {
    id: "generic",
    name: "Generic SLURM",
    description: "Standard SLURM cluster - configure manually",
    defaults: {
      nodes: 1,
      gpus_per_node: 8,
      time_limit: "4:00:00",
      partition: undefined,
      account: undefined,
    },
  },
];

// Backend info
const BACKEND_INFO: Record<Backend, { name: string; description: string; hint: string }> = {
  dtensor: {
    name: "DTensor (FSDP2)",
    description: "PyTorch Distributed Tensor for efficient sharding",
    hint: "Best for: Models up to 7B on single node, or multi-node with FSDP",
  },
  megatron: {
    name: "Megatron-LM",
    description: "NVIDIA Megatron for large-scale tensor parallelism",
    hint: "Best for: Models >7B, tensor/pipeline parallelism required",
  },
};

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

interface SelectDropdownProps {
  label: string;
  value: string;
  options: { value: string; label: string; description?: string }[];
  onChange: (value: string) => void;
  tooltip?: string;
}

function SelectDropdown({ label, value, options, onChange, tooltip }: SelectDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const selectedOption = options.find((o) => o.value === value);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <label className="text-sm font-medium">{label}</label>
        {tooltip && <Tooltip content={tooltip} />}
      </div>
      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className={cn(
            "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm",
            "hover:bg-accent hover:text-accent-foreground",
            "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          )}
        >
          <span>{selectedOption?.label || "Select..."}</span>
          <ChevronDown className={cn("h-4 w-4 transition-transform", isOpen && "rotate-180")} />
        </button>
        {isOpen && (
          <div className="absolute top-full mt-1 w-full z-50 bg-popover border border-border rounded-md shadow-lg">
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={cn(
                  "flex w-full items-center justify-between px-3 py-2 text-sm hover:bg-accent",
                  option.value === value && "bg-accent"
                )}
              >
                <div className="text-left">
                  <div className="font-medium">{option.label}</div>
                  {option.description && (
                    <div className="text-xs text-muted-foreground">{option.description}</div>
                  )}
                </div>
                {option.value === value && <Check className="h-4 w-4" />}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function ClusterConfig() {
  const { config, setBackend, setClusterConfig } = useConfigStore();
  const { backend, cluster } = config;
  const [selectedPreset, setSelectedPreset] = useState<string>("generic");

  const handlePresetChange = useCallback(
    (presetId: string) => {
      setSelectedPreset(presetId);
      const preset = CLUSTER_PRESETS.find((p) => p.id === presetId);
      if (preset) {
        // Apply preset defaults
        if (preset.defaults.nodes !== undefined) {
          setClusterConfig("nodes", preset.defaults.nodes);
        }
        if (preset.defaults.gpus_per_node !== undefined) {
          setClusterConfig("gpus_per_node", preset.defaults.gpus_per_node);
        }
        if (preset.defaults.time_limit !== undefined) {
          setClusterConfig("time_limit", preset.defaults.time_limit);
        }
        if (preset.defaults.partition !== undefined) {
          setClusterConfig("partition", preset.defaults.partition);
        }
        if (preset.defaults.account !== undefined) {
          setClusterConfig("account", preset.defaults.account);
        }
        if (preset.defaults.backend !== undefined) {
          setBackend(preset.defaults.backend);
        }
      }
    },
    [setBackend, setClusterConfig]
  );

  const handleNodesChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseInt(e.target.value, 10);
      if (!isNaN(value) && value >= 1 && value <= 256) {
        setClusterConfig("nodes", value);
      }
    },
    [setClusterConfig]
  );

  const handleGpusChange = useCallback(
    (value: string) => {
      const numValue = parseInt(value, 10);
      if ([1, 2, 4, 8].includes(numValue)) {
        setClusterConfig("gpus_per_node", numValue);
      }
    },
    [setClusterConfig]
  );

  const handleTimeLimitChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setClusterConfig("time_limit", e.target.value);
    },
    [setClusterConfig]
  );

  const handlePartitionChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setClusterConfig("partition", e.target.value || undefined);
    },
    [setClusterConfig]
  );

  const handleAccountChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setClusterConfig("account", e.target.value || undefined);
    },
    [setClusterConfig]
  );

  const totalGpus = cluster.nodes * cluster.gpus_per_node;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Cluster Configuration</h2>
        <p className="text-sm text-muted-foreground">
          Configure compute resources and SLURM settings
        </p>
      </div>

      {/* Cluster Preset */}
      <SelectDropdown
        label="Cluster Preset"
        value={selectedPreset}
        options={CLUSTER_PRESETS.map((p) => ({
          value: p.id,
          label: p.name,
          description: p.description,
        }))}
        onChange={handlePresetChange}
        tooltip="Select a preset to auto-populate cluster settings for your environment"
      />

      {/* Backend Selector */}
      <div className="space-y-2">
        <div className="flex items-center gap-1">
          <label className="text-sm font-medium">Training Backend</label>
          <Tooltip content="The backend determines how model weights are distributed across GPUs" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(Object.keys(BACKEND_INFO) as Backend[]).map((backendKey) => {
            const info = BACKEND_INFO[backendKey];
            const isSelected = backend === backendKey;
            return (
              <button
                key={backendKey}
                type="button"
                onClick={() => setBackend(backendKey)}
                className={cn(
                  "flex flex-col p-4 rounded-lg border transition-all duration-200 text-left",
                  "hover:border-primary/50",
                  "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                  isSelected ? "border-primary bg-primary/5" : "border-border bg-card"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{info.name}</span>
                  {isSelected && (
                    <span className="text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">
                      Selected
                    </span>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
                <p className="text-xs text-primary mt-2">{info.hint}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Resource Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Nodes */}
        <div className="space-y-2">
          <div className="flex items-center gap-1">
            <label htmlFor="nodes" className="text-sm font-medium">
              <Server className="h-4 w-4 inline mr-1" />
              Number of Nodes
            </label>
            <Tooltip content="Total compute nodes for training. More nodes enable larger models and faster training." />
          </div>
          <input
            id="nodes"
            type="number"
            min={1}
            max={256}
            value={cluster.nodes}
            onChange={handleNodesChange}
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <p className="text-xs text-muted-foreground">Range: 1-256 nodes</p>
        </div>

        {/* GPUs per Node */}
        <SelectDropdown
          label="GPUs per Node"
          value={String(cluster.gpus_per_node)}
          options={[
            { value: "1", label: "1 GPU", description: "Single GPU training" },
            { value: "2", label: "2 GPUs", description: "Basic parallelism" },
            { value: "4", label: "4 GPUs", description: "Medium parallelism" },
            { value: "8", label: "8 GPUs", description: "Full node (standard)" },
          ]}
          onChange={handleGpusChange}
          tooltip="Number of GPUs per node. 8 GPUs is standard for DGX/HGX systems."
        />

        {/* Time Limit */}
        <div className="space-y-2">
          <div className="flex items-center gap-1">
            <label htmlFor="time_limit" className="text-sm font-medium">
              <Clock className="h-4 w-4 inline mr-1" />
              Time Limit
            </label>
            <Tooltip content="Maximum job duration in HH:MM:SS format. Job will be terminated when time limit is reached." />
          </div>
          <input
            id="time_limit"
            type="text"
            value={cluster.time_limit}
            onChange={handleTimeLimitChange}
            placeholder="HH:MM:SS"
            pattern="^\d+:\d{2}:\d{2}$"
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <p className="text-xs text-muted-foreground">Format: HH:MM:SS (e.g., 4:00:00)</p>
        </div>

        {/* Partition */}
        <div className="space-y-2">
          <div className="flex items-center gap-1">
            <label htmlFor="partition" className="text-sm font-medium">
              Partition
            </label>
            <Tooltip content="SLURM partition/queue name. Check your cluster documentation for available partitions." />
          </div>
          <input
            id="partition"
            type="text"
            value={cluster.partition || ""}
            onChange={handlePartitionChange}
            placeholder="e.g., batch, luna, gpu"
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <p className="text-xs text-muted-foreground">Optional: SLURM partition name</p>
        </div>

        {/* Account */}
        <div className="space-y-2">
          <div className="flex items-center gap-1">
            <label htmlFor="account" className="text-sm font-medium">
              Account
            </label>
            <Tooltip content="SLURM account for job billing/allocation tracking." />
          </div>
          <input
            id="account"
            type="text"
            value={cluster.account || ""}
            onChange={handleAccountChange}
            placeholder="e.g., myproject"
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <p className="text-xs text-muted-foreground">Optional: SLURM account name</p>
        </div>
      </div>

      {/* Summary */}
      <div className="p-4 rounded-lg bg-muted/50 border">
        <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          Resource Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Total GPUs</p>
            <p className="font-medium text-lg">{totalGpus}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Nodes</p>
            <p className="font-medium text-lg">{cluster.nodes}</p>
          </div>
          <div>
            <p className="text-muted-foreground">GPUs/Node</p>
            <p className="font-medium text-lg">{cluster.gpus_per_node}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Backend</p>
            <p className="font-medium text-lg">{BACKEND_INFO[backend].name.split(" ")[0]}</p>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          {backend === "megatron"
            ? `Using Megatron-LM with ${totalGpus} GPUs enables tensor/pipeline parallelism for large models.`
            : `Using DTensor (FSDP2) with ${totalGpus} GPUs provides efficient sharding for distributed training.`}
        </p>
      </div>
    </div>
  );
}
