import { describe, it, expect } from 'vitest';
import { ALGORITHM_INFO, DEFAULT_CONFIG } from './config';
import type { Algorithm } from './config';

describe('config types', () => {
  describe('ALGORITHM_INFO', () => {
    it('should have info for all algorithms', () => {
      const algorithms: Algorithm[] = ['grpo', 'sft', 'dpo'];
      
      algorithms.forEach(algo => {
        expect(ALGORITHM_INFO[algo]).toBeDefined();
        expect(ALGORITHM_INFO[algo].name).toBeTruthy();
        expect(ALGORITHM_INFO[algo].description).toBeTruthy();
        expect(ALGORITHM_INFO[algo].useCases.length).toBeGreaterThan(0);
      });
    });

    it('should have correct GRPO info', () => {
      expect(ALGORITHM_INFO.grpo.name).toBe('GRPO');
      expect(ALGORITHM_INFO.grpo.description).toContain('Reinforcement');
    });

    it('should have correct SFT info', () => {
      expect(ALGORITHM_INFO.sft.name).toBe('SFT');
      expect(ALGORITHM_INFO.sft.description).toContain('Supervised');
    });

    it('should have correct DPO info', () => {
      expect(ALGORITHM_INFO.dpo.name).toBe('DPO');
      expect(ALGORITHM_INFO.dpo.description).toContain('Preference');
    });
  });

  describe('DEFAULT_CONFIG', () => {
    it('should have valid default algorithm', () => {
      expect(['grpo', 'sft', 'dpo']).toContain(DEFAULT_CONFIG.algorithm);
    });

    it('should have valid default backend', () => {
      expect(['dtensor', 'megatron']).toContain(DEFAULT_CONFIG.backend);
    });

    it('should have valid hyperparameters', () => {
      expect(DEFAULT_CONFIG.hyperparameters.learning_rate).toBeGreaterThan(0);
      expect(DEFAULT_CONFIG.hyperparameters.batch_size).toBeGreaterThan(0);
      expect(DEFAULT_CONFIG.hyperparameters.max_steps).toBeGreaterThan(0);
    });

    it('should have valid cluster config', () => {
      expect(DEFAULT_CONFIG.cluster.nodes).toBeGreaterThan(0);
      expect(DEFAULT_CONFIG.cluster.gpus_per_node).toBeGreaterThan(0);
      expect(DEFAULT_CONFIG.cluster.time_limit).toMatch(/^\d+:\d{2}:\d{2}$/);
    });
  });
});
