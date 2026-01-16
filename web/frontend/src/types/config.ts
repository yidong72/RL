/**
 * NeMo RL Web Configurator - Configuration Types
 */

export type Algorithm = "grpo" | "sft" | "dpo";

export type Backend = "dtensor" | "megatron";

export interface HyperparametersConfig {
  learning_rate: number;
  batch_size: number;
  max_steps: number;
  num_generations_per_prompt?: number; // GRPO only
  tensor_parallel_size?: number; // For Megatron backend
}

export interface ClusterConfig {
  nodes: number;
  gpus_per_node: number;
  time_limit: string;
  partition?: string;
  account?: string;
}

export interface TrainingConfig {
  algorithm: Algorithm;
  model: string;
  dataset: string;
  backend: Backend;
  hyperparameters: HyperparametersConfig;
  cluster: ClusterConfig;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: "error" | "warning" | "info";
}

export interface ResourceEstimate {
  memory_per_gpu: string;
  recommended_gpus: number;
  estimated_time: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  resource_estimate?: ResourceEstimate;
}

export interface ModelInfo {
  id: string;
  name: string;
  size_gb: number;
  parameters: string;
  recommended_config?: {
    min_gpus: number;
    tensor_parallel: boolean;
  };
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  size: string;
  features: string[];
  downloads: number;
}

export interface ScriptGenerationResult {
  script: string;
  files: {
    "ray.sub"?: string;
    "config.yaml": string;  // Always included
  };
  instructions: string[];
}

export const ALGORITHM_INFO: Record<Algorithm, { 
  name: string; 
  description: string; 
  useCases: string[];
}> = {
  grpo: {
    name: "GRPO",
    description: "Reinforcement learning from rewards",
    useCases: ["Math problem solving", "Code generation", "Reasoning tasks"],
  },
  sft: {
    name: "SFT",
    description: "Supervised fine-tuning on examples",
    useCases: ["Instruction following", "Domain adaptation", "Style transfer"],
  },
  dpo: {
    name: "DPO",
    description: "Preference optimization from pairs",
    useCases: ["Alignment", "Safety tuning", "Quality improvements"],
  },
};

export const DEFAULT_CONFIG: TrainingConfig = {
  algorithm: "grpo",
  model: "Qwen/Qwen2.5-1.5B",
  dataset: "nvidia/OpenMathInstruct-2",
  backend: "dtensor",
  hyperparameters: {
    learning_rate: 1e-6,
    batch_size: 32,
    max_steps: 1000,
    num_generations_per_prompt: 16,
    tensor_parallel_size: 1,
  },
  cluster: {
    nodes: 1,
    gpus_per_node: 8,
    time_limit: "4:00:00",
  },
};
