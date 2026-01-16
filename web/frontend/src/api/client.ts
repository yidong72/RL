/**
 * API Client for NeMo RL Web Configurator Backend
 */

import type { 
  TrainingConfig, 
  ValidationResult, 
  ModelInfo, 
  ScriptGenerationResult 
} from "../types/config";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  details?: unknown;
  
  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `HTTP error ${response.status}`,
      response.status,
      errorData
    );
  }

  return response.json();
}

export const api = {
  /**
   * Validate a training configuration
   */
  async validateConfig(config: TrainingConfig): Promise<ValidationResult> {
    return fetchApi<ValidationResult>("/api/v1/config/validate", {
      method: "POST",
      body: JSON.stringify(config),
    });
  },

  /**
   * Generate SLURM script from configuration
   */
  async generateScript(
    config: TrainingConfig,
    clusterPreset?: string
  ): Promise<ScriptGenerationResult> {
    return fetchApi<ScriptGenerationResult>("/api/v1/script/generate", {
      method: "POST",
      body: JSON.stringify({
        config,
        cluster_preset: clusterPreset,
        output_format: "slurm",
      }),
    });
  },

  /**
   * Search for models on HuggingFace Hub
   */
  async searchModels(
    query: string,
    size?: "small" | "medium" | "large"
  ): Promise<{ results: ModelInfo[] }> {
    const params = new URLSearchParams({ q: query });
    if (size) params.append("size", size);
    
    return fetchApi<{ results: ModelInfo[] }>(
      `/api/v1/models/search?${params.toString()}`
    );
  },

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string }> {
    return fetchApi<{ status: string }>("/api/v1/health");
  },
};

export { ApiError };
