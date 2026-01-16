import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ModelSelector } from './ModelSelector';
import { useConfigStore } from '../../store/configStore';

describe('ModelSelector', () => {
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

  it('renders the model selector component', () => {
    render(<ModelSelector />);
    
    expect(screen.getByText('Select Model')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search models/i)).toBeInTheDocument();
  });

  it('displays the search input field', () => {
    render(<ModelSelector />);
    
    const searchInput = screen.getByPlaceholderText(/Search models/i);
    expect(searchInput).toBeInTheDocument();
  });

  it('shows popular models by default', () => {
    render(<ModelSelector />);
    
    // Check that popular models are displayed
    expect(screen.getByText('Qwen2.5-1.5B')).toBeInTheDocument();
    expect(screen.getByText('Qwen2.5-7B')).toBeInTheDocument();
    expect(screen.getByText('Llama-3.1-8B')).toBeInTheDocument();
  });

  it('filters models based on search query', async () => {
    render(<ModelSelector />);
    
    const searchInput = screen.getByPlaceholderText(/Search models/i);
    fireEvent.change(searchInput, { target: { value: 'llama' } });
    
    // Should show only llama models
    await waitFor(() => {
      expect(screen.getByText('Llama-3.1-8B')).toBeInTheDocument();
    });
    
    // Qwen models should be filtered out
    expect(screen.queryByText('Qwen2.5-1.5B')).not.toBeInTheDocument();
  });

  it('shows no results message for non-matching query', async () => {
    render(<ModelSelector />);
    
    const searchInput = screen.getByPlaceholderText(/Search models/i);
    fireEvent.change(searchInput, { target: { value: 'nonexistent123' } });
    
    await waitFor(() => {
      expect(screen.getByText(/No models found/i)).toBeInTheDocument();
    });
  });

  it('updates state when selecting a model', () => {
    render(<ModelSelector />);
    
    // Click on a different model (Llama)
    const llamaModel = screen.getByText('Llama-3.1-8B');
    const modelCard = llamaModel.closest('button');
    if (modelCard) {
      fireEvent.click(modelCard);
    }
    
    // Verify state was updated
    expect(useConfigStore.getState().config.model).toBe('meta-llama/Llama-3.1-8B');
  });

  it('displays model info (size, parameters)', () => {
    render(<ModelSelector />);
    
    // Check that model info is displayed - there may be multiple matches
    expect(screen.getAllByText(/parameters/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/GB/).length).toBeGreaterThan(0);
  });

  it('shows recommended config for models', () => {
    render(<ModelSelector />);
    
    // Check for recommended GPU info
    expect(screen.getAllByText(/Min GPUs:/i).length).toBeGreaterThan(0);
  });

  it('shows local path input section', () => {
    render(<ModelSelector />);
    
    expect(screen.getByText(/Or enter local model path/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/\/path\/to\/model/i)).toBeInTheDocument();
  });

  it('allows entering a custom local path', () => {
    render(<ModelSelector />);
    
    const localPathInput = screen.getByPlaceholderText(/\/path\/to\/model/i);
    fireEvent.change(localPathInput, { target: { value: '/custom/model/path' } });
    
    const usePathButton = screen.getByText('Use Path');
    fireEvent.click(usePathButton);
    
    expect(useConfigStore.getState().config.model).toBe('/custom/model/path');
  });

  it('disables Use Path button when input is empty', () => {
    render(<ModelSelector />);
    
    const usePathButton = screen.getByText('Use Path');
    expect(usePathButton).toBeDisabled();
  });

  it('shows current selection', () => {
    render(<ModelSelector />);
    
    // Should show the default model as selected
    expect(screen.getByText(/Selected model:/i)).toBeInTheDocument();
    expect(screen.getByText('Qwen/Qwen2.5-1.5B')).toBeInTheDocument();
  });

  it('shows HuggingFace link for selected model', () => {
    render(<ModelSelector />);
    
    const hfLink = screen.getByText(/View on HF/i);
    expect(hfLink).toBeInTheDocument();
    expect(hfLink).toHaveAttribute('href', 'https://huggingface.co/Qwen/Qwen2.5-1.5B');
  });

  it('indicates which model is currently selected', () => {
    render(<ModelSelector />);
    
    // The default model should show as selected
    expect(screen.getByText('Selected')).toBeInTheDocument();
  });

  it('clears local path flag when selecting from list', () => {
    render(<ModelSelector />);
    
    // First, enter a local path
    const localPathInput = screen.getByPlaceholderText(/\/path\/to\/model/i);
    fireEvent.change(localPathInput, { target: { value: '/custom/path' } });
    const usePathButton = screen.getByText('Use Path');
    fireEvent.click(usePathButton);
    
    // Now select a model from the list
    const qwenModel = screen.getByText('Qwen2.5-7B');
    const modelCard = qwenModel.closest('button');
    if (modelCard) {
      fireEvent.click(modelCard);
    }
    
    // State should be updated to the list model
    expect(useConfigStore.getState().config.model).toBe('Qwen/Qwen2.5-7B');
  });
});
