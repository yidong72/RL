import { create } from "zustand";
import type { 
  TrainingConfig, 
  ValidationResult, 
  Algorithm, 
  Backend
} from "../types/config";

// Theme storage key
const THEME_STORAGE_KEY = "nemo-rl-theme";

/**
 * Get the initial theme preference
 * Priority: 1. localStorage 2. system preference 3. light mode (default)
 */
function getInitialTheme(): boolean {
  // Check if we're in browser environment
  if (typeof window === "undefined") {
    return false;
  }

  // Check localStorage first
  const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme !== null) {
    return storedTheme === "dark";
  }

  // Fall back to system preference
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    return true;
  }

  // Default to light mode
  return false;
}

/**
 * Apply theme to document
 */
function applyTheme(isDark: boolean): void {
  if (typeof document !== "undefined") {
    document.documentElement.classList.toggle("dark", isDark);
  }
}

/**
 * Persist theme to localStorage
 */
function persistTheme(isDark: boolean): void {
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(THEME_STORAGE_KEY, isDark ? "dark" : "light");
  }
}

interface ConfigState {
  // Config state
  config: TrainingConfig;
  
  // Validation state
  validation: ValidationResult | null;
  isValidating: boolean;
  
  // UI state
  isDarkMode: boolean;
  
  // Actions
  setAlgorithm: (algorithm: Algorithm) => void;
  setModel: (model: string) => void;
  setDataset: (dataset: string) => void;
  setBackend: (backend: Backend) => void;
  setHyperparameter: <K extends keyof TrainingConfig["hyperparameters"]>(
    key: K,
    value: TrainingConfig["hyperparameters"][K]
  ) => void;
  setClusterConfig: <K extends keyof TrainingConfig["cluster"]>(
    key: K,
    value: TrainingConfig["cluster"][K]
  ) => void;
  setConfig: (config: Partial<TrainingConfig>) => void;
  resetConfig: () => void;
  
  // Validation actions
  setValidation: (validation: ValidationResult | null) => void;
  setIsValidating: (isValidating: boolean) => void;
  
  // Theme actions
  toggleDarkMode: () => void;
  setDarkMode: (isDark: boolean) => void;
  initializeTheme: () => void;
}

const DEFAULT_CONFIG_STATE: TrainingConfig = {
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

// Get initial theme synchronously for SSR compatibility
const initialTheme = typeof window !== "undefined" ? getInitialTheme() : false;

export const useConfigStore = create<ConfigState>((set) => ({
  // Initial state
  config: { ...DEFAULT_CONFIG_STATE },
  validation: null,
  isValidating: false,
  isDarkMode: initialTheme,

  // Actions
  setAlgorithm: (algorithm) =>
    set((state) => ({
      config: { ...state.config, algorithm },
    })),

  setModel: (model) =>
    set((state) => ({
      config: { ...state.config, model },
    })),

  setDataset: (dataset) =>
    set((state) => ({
      config: { ...state.config, dataset },
    })),

  setBackend: (backend) =>
    set((state) => ({
      config: { ...state.config, backend },
    })),

  setHyperparameter: (key, value) =>
    set((state) => ({
      config: {
        ...state.config,
        hyperparameters: {
          ...state.config.hyperparameters,
          [key]: value,
        },
      },
    })),

  setClusterConfig: (key, value) =>
    set((state) => ({
      config: {
        ...state.config,
        cluster: {
          ...state.config.cluster,
          [key]: value,
        },
      },
    })),

  setConfig: (config) =>
    set((state) => ({
      config: { ...state.config, ...config },
    })),

  resetConfig: () =>
    set({
      config: { ...DEFAULT_CONFIG_STATE },
      validation: null,
    }),

  setValidation: (validation) => set({ validation }),
  setIsValidating: (isValidating) => set({ isValidating }),

  toggleDarkMode: () =>
    set((state) => {
      const newMode = !state.isDarkMode;
      // Apply theme to document
      applyTheme(newMode);
      // Persist to localStorage
      persistTheme(newMode);
      return { isDarkMode: newMode };
    }),

  setDarkMode: (isDark: boolean) =>
    set(() => {
      // Apply theme to document
      applyTheme(isDark);
      // Persist to localStorage
      persistTheme(isDark);
      return { isDarkMode: isDark };
    }),

  initializeTheme: () =>
    set(() => {
      const isDark = getInitialTheme();
      // Apply theme to document
      applyTheme(isDark);
      return { isDarkMode: isDark };
    }),
}));

// Initialize theme on store creation (browser only)
if (typeof window !== "undefined") {
  // Apply initial theme to document
  applyTheme(initialTheme);
  
  // Listen for system preference changes
  if (window.matchMedia) {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", (e) => {
      // Only update if user hasn't set a preference
      const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
      if (storedTheme === null) {
        const store = useConfigStore.getState();
        store.setDarkMode(e.matches);
      }
    });
  }
}
