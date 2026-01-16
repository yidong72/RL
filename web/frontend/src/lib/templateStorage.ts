import type { TrainingConfig, Algorithm } from "../types/config";

/**
 * User template interface
 */
export interface UserTemplate {
  id: string;
  name: string;
  description: string;
  algorithm: Algorithm;
  tags: string[];
  config: TrainingConfig;
  createdAt: string;
  updatedAt: string;
}

// Storage key for user templates
const USER_TEMPLATES_KEY = "nemo-rl-user-templates";

/**
 * Get all user templates from localStorage
 */
export function getUserTemplates(): UserTemplate[] {
  if (typeof window === "undefined") return [];
  
  try {
    const stored = localStorage.getItem(USER_TEMPLATES_KEY);
    if (!stored) return [];
    
    const templates = JSON.parse(stored) as UserTemplate[];
    // Validate structure
    return templates.filter(t => 
      t.id && t.name && t.config && t.algorithm
    );
  } catch {
    console.error("Failed to parse user templates from localStorage");
    return [];
  }
}

/**
 * Save a user template to localStorage
 */
export function saveUserTemplate(template: Omit<UserTemplate, "id" | "createdAt" | "updatedAt">): UserTemplate {
  const templates = getUserTemplates();
  
  const newTemplate: UserTemplate = {
    ...template,
    id: `user-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  
  templates.push(newTemplate);
  localStorage.setItem(USER_TEMPLATES_KEY, JSON.stringify(templates));
  
  return newTemplate;
}

/**
 * Update an existing user template
 */
export function updateUserTemplate(id: string, updates: Partial<Omit<UserTemplate, "id" | "createdAt">>): UserTemplate | null {
  const templates = getUserTemplates();
  const index = templates.findIndex(t => t.id === id);
  
  if (index === -1) return null;
  
  templates[index] = {
    ...templates[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };
  
  localStorage.setItem(USER_TEMPLATES_KEY, JSON.stringify(templates));
  return templates[index];
}

/**
 * Delete a user template
 */
export function deleteUserTemplate(id: string): boolean {
  const templates = getUserTemplates();
  const filtered = templates.filter(t => t.id !== id);
  
  if (filtered.length === templates.length) return false;
  
  localStorage.setItem(USER_TEMPLATES_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Encode a config to a shareable URL string
 */
export function encodeConfigToShareableLink(config: TrainingConfig, name?: string): string {
  const payload = {
    n: name || "Shared Config",
    c: config,
    v: 1, // version for future compatibility
  };
  
  const json = JSON.stringify(payload);
  const base64 = btoa(encodeURIComponent(json));
  
  // Create URL with hash parameter
  const url = new URL(window.location.href);
  url.hash = "";
  url.search = "";
  url.searchParams.set("template", base64);
  
  return url.toString();
}

/**
 * Decode a shared config from URL
 */
export function decodeConfigFromUrl(url: string): { name: string; config: TrainingConfig } | null {
  try {
    const urlObj = new URL(url);
    const encoded = urlObj.searchParams.get("template");
    
    if (!encoded) return null;
    
    const json = decodeURIComponent(atob(encoded));
    const payload = JSON.parse(json) as { n: string; c: TrainingConfig; v: number };
    
    // Validate the config structure
    if (!payload.c || !payload.c.algorithm) return null;
    
    return {
      name: payload.n || "Shared Config",
      config: payload.c,
    };
  } catch {
    console.error("Failed to decode shared config from URL");
    return null;
  }
}

/**
 * Check if current URL has a shared template
 */
export function hasSharedTemplate(): boolean {
  if (typeof window === "undefined") return false;
  
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.has("template");
}

/**
 * Get shared template from current URL
 */
export function getSharedTemplateFromUrl(): { name: string; config: TrainingConfig } | null {
  if (typeof window === "undefined") return null;
  return decodeConfigFromUrl(window.location.href);
}

/**
 * Clear template parameter from URL
 */
export function clearTemplateFromUrl(): void {
  if (typeof window === "undefined") return;
  
  const url = new URL(window.location.href);
  url.searchParams.delete("template");
  window.history.replaceState({}, "", url.toString());
}

/**
 * Generate a shortened shareable link using base64 compression
 * This compresses the JSON before base64 encoding
 */
export function encodeConfigToCompressedLink(config: TrainingConfig, name?: string): string {
  // Create minimal payload to reduce URL length
  const payload = {
    n: name || "Shared",
    a: config.algorithm,
    m: config.model,
    d: config.dataset,
    b: config.backend,
    h: config.hyperparameters,
    c: config.cluster,
  };
  
  const json = JSON.stringify(payload);
  const base64 = btoa(encodeURIComponent(json));
  
  const url = new URL(window.location.origin + window.location.pathname);
  url.searchParams.set("t", base64);
  
  return url.toString();
}

/**
 * Decode compressed config from URL
 */
export function decodeCompressedConfigFromUrl(url: string): { name: string; config: TrainingConfig } | null {
  try {
    const urlObj = new URL(url);
    const encoded = urlObj.searchParams.get("t") || urlObj.searchParams.get("template");
    
    if (!encoded) return null;
    
    const json = decodeURIComponent(atob(encoded));
    const payload = JSON.parse(json);
    
    // Handle both compressed and full format
    if (payload.c && typeof payload.c === "object" && payload.c.algorithm) {
      // Full format
      return {
        name: payload.n || "Shared Config",
        config: payload.c,
      };
    }
    
    // Compressed format
    if (!payload.a) return null;
    
    const config: TrainingConfig = {
      algorithm: payload.a,
      model: payload.m,
      dataset: payload.d,
      backend: payload.b,
      hyperparameters: payload.h,
      cluster: payload.c,
    };
    
    return {
      name: payload.n || "Shared Config",
      config,
    };
  } catch {
    console.error("Failed to decode shared config from URL");
    return null;
  }
}
