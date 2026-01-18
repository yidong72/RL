import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useConfigStore } from './configStore';

describe('configStore', () => {
  beforeEach(() => {
    // Clear localStorage mock
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(() => null),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
    
    // Reset store to initial state before each test
    useConfigStore.setState({
      config: {
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
      },
      validation: null,
      isValidating: false,
      isDarkMode: false,
    });
  });

  it('should have default config values', () => {
    const state = useConfigStore.getState();
    expect(state.config.algorithm).toBe('grpo');
    expect(state.config.model).toBe('Qwen/Qwen2.5-1.5B');
    expect(state.config.backend).toBe('dtensor');
  });

  it('should update algorithm', () => {
    const { setAlgorithm } = useConfigStore.getState();
    setAlgorithm('sft');
    expect(useConfigStore.getState().config.algorithm).toBe('sft');
  });

  it('should update model', () => {
    const { setModel } = useConfigStore.getState();
    setModel('meta-llama/Llama-3.1-8B');
    expect(useConfigStore.getState().config.model).toBe('meta-llama/Llama-3.1-8B');
  });

  it('should update dataset', () => {
    const { setDataset } = useConfigStore.getState();
    setDataset('custom/dataset');
    expect(useConfigStore.getState().config.dataset).toBe('custom/dataset');
  });

  it('should update backend', () => {
    const { setBackend } = useConfigStore.getState();
    setBackend('megatron');
    expect(useConfigStore.getState().config.backend).toBe('megatron');
  });

  it('should update hyperparameters', () => {
    const { setHyperparameter } = useConfigStore.getState();
    setHyperparameter('learning_rate', 5e-6);
    expect(useConfigStore.getState().config.hyperparameters.learning_rate).toBe(5e-6);
    
    setHyperparameter('batch_size', 64);
    expect(useConfigStore.getState().config.hyperparameters.batch_size).toBe(64);
  });

  it('should update cluster config', () => {
    const { setClusterConfig } = useConfigStore.getState();
    setClusterConfig('nodes', 2);
    expect(useConfigStore.getState().config.cluster.nodes).toBe(2);
    
    setClusterConfig('time_limit', '8:00:00');
    expect(useConfigStore.getState().config.cluster.time_limit).toBe('8:00:00');
  });

  it('should reset config to defaults', () => {
    const { setAlgorithm, setModel, resetConfig } = useConfigStore.getState();
    
    // Modify config
    setAlgorithm('dpo');
    setModel('custom-model');
    
    // Reset
    resetConfig();
    
    const state = useConfigStore.getState();
    expect(state.config.algorithm).toBe('grpo');
    expect(state.config.model).toBe('Qwen/Qwen2.5-1.5B');
  });

  it('should toggle dark mode', () => {
    const { toggleDarkMode } = useConfigStore.getState();
    expect(useConfigStore.getState().isDarkMode).toBe(false);
    
    toggleDarkMode();
    expect(useConfigStore.getState().isDarkMode).toBe(true);
    
    toggleDarkMode();
    expect(useConfigStore.getState().isDarkMode).toBe(false);
  });

  it('should set validation result', () => {
    const { setValidation } = useConfigStore.getState();
    const validationResult = {
      valid: true,
      errors: [],
      warnings: [{ field: 'test', message: 'warning', severity: 'warning' as const }],
    };
    
    setValidation(validationResult);
    expect(useConfigStore.getState().validation).toEqual(validationResult);
  });

  it('should set validating state', () => {
    const { setIsValidating } = useConfigStore.getState();
    
    setIsValidating(true);
    expect(useConfigStore.getState().isValidating).toBe(true);
    
    setIsValidating(false);
    expect(useConfigStore.getState().isValidating).toBe(false);
  });

  it('should set dark mode directly', () => {
    const { setDarkMode } = useConfigStore.getState();
    
    setDarkMode(true);
    expect(useConfigStore.getState().isDarkMode).toBe(true);
    
    setDarkMode(false);
    expect(useConfigStore.getState().isDarkMode).toBe(false);
  });

  it('should persist theme to localStorage when toggling', () => {
    const { toggleDarkMode } = useConfigStore.getState();
    
    toggleDarkMode();
    expect(localStorage.setItem).toHaveBeenCalledWith('nemo-rl-theme', 'dark');
    
    toggleDarkMode();
    expect(localStorage.setItem).toHaveBeenCalledWith('nemo-rl-theme', 'light');
  });

  it('should persist theme to localStorage when setting directly', () => {
    const { setDarkMode } = useConfigStore.getState();
    
    setDarkMode(true);
    expect(localStorage.setItem).toHaveBeenCalledWith('nemo-rl-theme', 'dark');
    
    setDarkMode(false);
    expect(localStorage.setItem).toHaveBeenCalledWith('nemo-rl-theme', 'light');
  });

  it('should update tensor_parallel_size hyperparameter', () => {
    const { setHyperparameter } = useConfigStore.getState();
    
    setHyperparameter('tensor_parallel_size', 4);
    expect(useConfigStore.getState().config.hyperparameters.tensor_parallel_size).toBe(4);
  });
});
