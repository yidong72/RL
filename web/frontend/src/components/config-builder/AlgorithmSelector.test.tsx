import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AlgorithmSelector } from './AlgorithmSelector';
import { useConfigStore } from '../../store/configStore';

describe('AlgorithmSelector', () => {
  beforeEach(() => {
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

  it('renders all three algorithm options', () => {
    render(<AlgorithmSelector />);
    
    // Check that all three algorithm buttons are displayed
    expect(screen.getByRole('button', { name: /Select GRPO algorithm/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Select SFT algorithm/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Select DPO algorithm/i })).toBeInTheDocument();
  });

  it('displays algorithm descriptions', () => {
    render(<AlgorithmSelector />);
    
    // Check descriptions are shown
    expect(screen.getByText(/Reinforcement learning from rewards/i)).toBeInTheDocument();
    expect(screen.getByText(/Supervised fine-tuning on examples/i)).toBeInTheDocument();
    expect(screen.getByText(/Preference optimization from pairs/i)).toBeInTheDocument();
  });

  it('shows GRPO selected by default', () => {
    render(<AlgorithmSelector />);
    
    // Check that GRPO is initially selected
    const grpoButton = screen.getByRole('button', { name: /Select GRPO algorithm/i });
    expect(grpoButton).toHaveAttribute('aria-pressed', 'true');
  });

  it('updates state when clicking on SFT', () => {
    render(<AlgorithmSelector />);
    
    // Click on SFT
    const sftButton = screen.getByRole('button', { name: /Select SFT algorithm/i });
    fireEvent.click(sftButton);
    
    // Verify state was updated
    expect(useConfigStore.getState().config.algorithm).toBe('sft');
  });

  it('updates state when clicking on DPO', () => {
    render(<AlgorithmSelector />);
    
    // Click on DPO
    const dpoButton = screen.getByRole('button', { name: /Select DPO algorithm/i });
    fireEvent.click(dpoButton);
    
    // Verify state was updated
    expect(useConfigStore.getState().config.algorithm).toBe('dpo');
  });

  it('shows contextual help for selected algorithm', () => {
    render(<AlgorithmSelector />);
    
    // Default is GRPO, should show GRPO help - look for the text within paragraph
    expect(screen.getByText(/is best for:/i)).toBeInTheDocument();
    
    // Click on SFT
    const sftButton = screen.getByRole('button', { name: /Select SFT algorithm/i });
    fireEvent.click(sftButton);
    
    // Help text should still be visible (it updates based on selection)
    expect(screen.getByText(/is best for:/i)).toBeInTheDocument();
  });

  it('displays use cases for each algorithm', () => {
    render(<AlgorithmSelector />);
    
    // Check for some use case tags
    expect(screen.getByText('Math problem solving')).toBeInTheDocument();
    expect(screen.getByText('Instruction following')).toBeInTheDocument();
    expect(screen.getByText('Alignment')).toBeInTheDocument();
  });

  it('visually indicates the selected algorithm', () => {
    render(<AlgorithmSelector />);
    
    // GRPO should have aria-pressed=true initially
    const grpoButton = screen.getByRole('button', { name: /Select GRPO algorithm/i });
    expect(grpoButton).toHaveAttribute('aria-pressed', 'true');
    
    // SFT and DPO should have aria-pressed=false
    const sftButton = screen.getByRole('button', { name: /Select SFT algorithm/i });
    const dpoButton = screen.getByRole('button', { name: /Select DPO algorithm/i });
    expect(sftButton).toHaveAttribute('aria-pressed', 'false');
    expect(dpoButton).toHaveAttribute('aria-pressed', 'false');
    
    // After clicking SFT
    fireEvent.click(sftButton);
    expect(sftButton).toHaveAttribute('aria-pressed', 'true');
    expect(grpoButton).toHaveAttribute('aria-pressed', 'false');
  });

  it('has accessible button labels', () => {
    render(<AlgorithmSelector />);
    
    // All buttons should have accessible labels
    expect(screen.getByRole('button', { name: /Select GRPO algorithm/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Select SFT algorithm/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Select DPO algorithm/i })).toBeInTheDocument();
  });
});
