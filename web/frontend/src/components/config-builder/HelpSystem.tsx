import { useState, useCallback } from "react";
import {
  HelpCircle,
  X,
  ChevronRight,
  ChevronLeft,
  Book,
  Sparkles,
  Settings,
  Cpu,
  Database,
  Server,
  Play,
  CheckCircle2,
  Code2,
  Layers,
  ExternalLink
} from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "../common/Button";

/**
 * Help topic definition
 */
interface HelpTopic {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  content: React.ReactNode;
}

/**
 * Tutorial step definition
 */
interface TutorialStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

/**
 * Help topics with detailed content
 */
const HELP_TOPICS: HelpTopic[] = [
  {
    id: "getting-started",
    title: "Getting Started",
    icon: <Play className="h-5 w-5" />,
    description: "Quick introduction to NeMo RL Configurator",
    content: (
      <div className="space-y-4">
        <p>
          Welcome to the NeMo RL Web Configurator! This tool helps you create training 
          configurations for NVIDIA's NeMo RL framework.
        </p>
        <h4 className="font-semibold">Quick Start:</h4>
        <ol className="list-decimal list-inside space-y-2 text-sm">
          <li>Select your training algorithm (GRPO, SFT, or DPO)</li>
          <li>Choose a pre-trained model from HuggingFace</li>
          <li>Select your training dataset</li>
          <li>Configure hyperparameters for your use case</li>
          <li>Set up cluster resources</li>
          <li>Download the generated SLURM script</li>
        </ol>
        <div className="p-3 bg-muted/50 rounded-lg">
          <p className="text-sm">
            <strong>Tip:</strong> Use the Templates feature to start with pre-configured 
            setups for common use cases like math reasoning or code generation.
          </p>
        </div>
      </div>
    ),
  },
  {
    id: "algorithms",
    title: "Training Algorithms",
    icon: <Sparkles className="h-5 w-5" />,
    description: "Learn about GRPO, SFT, and DPO",
    content: (
      <div className="space-y-4">
        <h4 className="font-semibold">GRPO (Group Relative Policy Optimization)</h4>
        <p className="text-sm">
          GRPO is a reinforcement learning algorithm that trains models using reward signals. 
          It generates multiple responses per prompt and uses the rewards to update the policy.
        </p>
        <ul className="list-disc list-inside text-sm space-y-1 ml-2">
          <li><strong>Best for:</strong> Math reasoning, code generation, logical tasks</li>
          <li><strong>Requires:</strong> A reward function to evaluate responses</li>
          <li><strong>Key parameter:</strong> <code className="bg-muted px-1 rounded">num_generations_per_prompt</code></li>
        </ul>

        <h4 className="font-semibold mt-4">SFT (Supervised Fine-Tuning)</h4>
        <p className="text-sm">
          SFT trains the model on input-output pairs using standard supervised learning.
          It's the simplest approach and works well for instruction following.
        </p>
        <ul className="list-disc list-inside text-sm space-y-1 ml-2">
          <li><strong>Best for:</strong> Instruction following, domain adaptation, style transfer</li>
          <li><strong>Requires:</strong> Labeled training data</li>
          <li><strong>Key parameter:</strong> <code className="bg-muted px-1 rounded">learning_rate</code></li>
        </ul>

        <h4 className="font-semibold mt-4">DPO (Direct Preference Optimization)</h4>
        <p className="text-sm">
          DPO learns from preference pairs without an explicit reward model.
          It directly optimizes for preferred vs rejected response pairs.
        </p>
        <ul className="list-disc list-inside text-sm space-y-1 ml-2">
          <li><strong>Best for:</strong> Alignment, safety tuning, quality improvements</li>
          <li><strong>Requires:</strong> Preference data (chosen vs rejected pairs)</li>
          <li><strong>Key parameter:</strong> <code className="bg-muted px-1 rounded">beta</code> (implicit reward strength)</li>
        </ul>
      </div>
    ),
  },
  {
    id: "models",
    title: "Model Selection",
    icon: <Cpu className="h-5 w-5" />,
    description: "Choosing the right model for your task",
    content: (
      <div className="space-y-4">
        <p>
          NeMo RL supports models from HuggingFace Hub. The choice of model depends on 
          your task requirements, available compute, and performance needs.
        </p>
        
        <h4 className="font-semibold">Popular Models:</h4>
        <div className="space-y-2 text-sm">
          <div className="p-2 bg-muted/50 rounded">
            <strong>Qwen2.5 (0.5B-72B)</strong> - Excellent for math and reasoning
          </div>
          <div className="p-2 bg-muted/50 rounded">
            <strong>Llama 3 (8B-70B)</strong> - Strong general-purpose models
          </div>
          <div className="p-2 bg-muted/50 rounded">
            <strong>CodeQwen/StarCoder</strong> - Specialized for code
          </div>
        </div>

        <h4 className="font-semibold mt-4">Memory Requirements:</h4>
        <ul className="list-disc list-inside text-sm space-y-1">
          <li>1-3B params: 1 GPU (24GB+ VRAM)</li>
          <li>7-8B params: 2-4 GPUs (16-24GB each)</li>
          <li>13-14B params: 4-8 GPUs</li>
          <li>70B+ params: 8+ GPUs with tensor parallelism</li>
        </ul>

        <div className="p-3 bg-muted/50 rounded-lg mt-4">
          <p className="text-sm">
            <strong>Tip:</strong> Start with smaller models (1.5B-3B) for experimentation, 
            then scale up once you've validated your approach.
          </p>
        </div>
      </div>
    ),
  },
  {
    id: "datasets",
    title: "Dataset Configuration",
    icon: <Database className="h-5 w-5" />,
    description: "Understanding training data requirements",
    content: (
      <div className="space-y-4">
        <p>
          The dataset you choose should match your training objective and contain 
          the types of examples you want your model to learn from.
        </p>

        <h4 className="font-semibold">Dataset Format by Algorithm:</h4>
        <div className="space-y-3 text-sm">
          <div className="p-2 border rounded">
            <strong>GRPO:</strong> Prompts with optional reference answers. The reward 
            function evaluates generated responses.
          </div>
          <div className="p-2 border rounded">
            <strong>SFT:</strong> Input-output pairs. The model learns to generate 
            the expected output given the input.
          </div>
          <div className="p-2 border rounded">
            <strong>DPO:</strong> Preference pairs with chosen and rejected responses 
            for the same prompt.
          </div>
        </div>

        <h4 className="font-semibold mt-4">Recommended Datasets:</h4>
        <ul className="list-disc list-inside text-sm space-y-1">
          <li><strong>Math:</strong> nvidia/OpenMathInstruct-2, gsm8k</li>
          <li><strong>Code:</strong> bigcode/the-stack-v2, CodeAlpaca</li>
          <li><strong>Chat:</strong> HuggingFaceH4/ultrachat_200k</li>
          <li><strong>Alignment:</strong> Anthropic/hh-rlhf</li>
        </ul>
      </div>
    ),
  },
  {
    id: "hyperparameters",
    title: "Hyperparameters",
    icon: <Settings className="h-5 w-5" />,
    description: "Key training parameters explained",
    content: (
      <div className="space-y-4">
        <h4 className="font-semibold">Common Parameters:</h4>
        
        <div className="space-y-3 text-sm">
          <div className="p-2 border rounded">
            <div className="flex justify-between items-center mb-1">
              <strong>Learning Rate</strong>
              <code className="bg-muted px-1 rounded text-xs">1e-6 to 1e-4</code>
            </div>
            <p className="text-muted-foreground text-xs">
              How fast the model updates. Lower = more stable, higher = faster but riskier.
              Start with 1e-6 for RL, 2e-5 for SFT.
            </p>
          </div>

          <div className="p-2 border rounded">
            <div className="flex justify-between items-center mb-1">
              <strong>Batch Size</strong>
              <code className="bg-muted px-1 rounded text-xs">8 to 128</code>
            </div>
            <p className="text-muted-foreground text-xs">
              Samples processed per step. Larger = more stable gradients but more memory.
              Typically 16-32 for most cases.
            </p>
          </div>

          <div className="p-2 border rounded">
            <div className="flex justify-between items-center mb-1">
              <strong>Max Steps</strong>
              <code className="bg-muted px-1 rounded text-xs">100 to 10000</code>
            </div>
            <p className="text-muted-foreground text-xs">
              Total training iterations. Monitor validation metrics to determine optimal value.
            </p>
          </div>
        </div>

        <h4 className="font-semibold mt-4">GRPO-Specific:</h4>
        <div className="p-2 border rounded text-sm">
          <div className="flex justify-between items-center mb-1">
            <strong>Generations per Prompt</strong>
            <code className="bg-muted px-1 rounded text-xs">4 to 32</code>
          </div>
          <p className="text-muted-foreground text-xs">
            Number of responses to generate for each prompt. More = better reward signal 
            but slower training. 16 is a good default.
          </p>
        </div>
      </div>
    ),
  },
  {
    id: "cluster",
    title: "Cluster Configuration",
    icon: <Server className="h-5 w-5" />,
    description: "Setting up distributed training",
    content: (
      <div className="space-y-4">
        <h4 className="font-semibold">Compute Resources:</h4>
        
        <div className="space-y-2 text-sm">
          <div className="p-2 border rounded">
            <strong>Nodes:</strong> Number of machines. More nodes = more parallelism.
            Start with 1 for experimentation.
          </div>
          <div className="p-2 border rounded">
            <strong>GPUs per Node:</strong> Typically 8 for DGX systems.
            Use 4 or fewer for smaller experiments.
          </div>
          <div className="p-2 border rounded">
            <strong>Time Limit:</strong> Maximum job duration (HH:MM:SS).
            Account for initialization and checkpoint saving time.
          </div>
        </div>

        <h4 className="font-semibold mt-4">Backend Selection:</h4>
        <div className="space-y-2 text-sm">
          <div className="p-2 bg-muted/50 rounded">
            <strong>DTensor:</strong> PyTorch's native distributed training.
            Good default choice, easier to debug.
          </div>
          <div className="p-2 bg-muted/50 rounded">
            <strong>Megatron:</strong> Optimized for large models with tensor/pipeline 
            parallelism. Better throughput but more complex.
          </div>
        </div>

        <h4 className="font-semibold mt-4">Tensor Parallelism:</h4>
        <p className="text-sm">
          Split model layers across GPUs. Must evenly divide total GPUs.
          Use 1 for small models, 2-8 for larger models.
        </p>
      </div>
    ),
  },
  {
    id: "reward-functions",
    title: "Reward Functions",
    icon: <Code2 className="h-5 w-5" />,
    description: "Creating custom rewards for GRPO",
    content: (
      <div className="space-y-4">
        <p>
          For GRPO training, you need a reward function that evaluates generated 
          responses. The reward signal guides the model toward better outputs.
        </p>

        <h4 className="font-semibold">Function Signature:</h4>
        <div className="p-2 bg-muted rounded font-mono text-xs">
          def reward_function(prompt: str, response: str) -{'>'} float:
        </div>

        <h4 className="font-semibold mt-4">Built-in Templates:</h4>
        <ul className="list-disc list-inside text-sm space-y-1">
          <li><strong>Exact Match:</strong> 1.0 if response matches expected</li>
          <li><strong>Regex Match:</strong> 1.0 if response matches pattern</li>
          <li><strong>Contains Keywords:</strong> Score based on keyword presence</li>
          <li><strong>Length-Based:</strong> Score based on response length</li>
          <li><strong>Math Correctness:</strong> Validates numerical answers</li>
        </ul>

        <h4 className="font-semibold mt-4">Best Practices:</h4>
        <ul className="list-disc list-inside text-sm space-y-1">
          <li>Return values between 0.0 and 1.0</li>
          <li>Higher scores for better responses</li>
          <li>Consider partial credit for partial solutions</li>
          <li>Test your reward function before training</li>
        </ul>
      </div>
    ),
  },
  {
    id: "templates",
    title: "Using Templates",
    icon: <Layers className="h-5 w-5" />,
    description: "Quick start with pre-configured setups",
    content: (
      <div className="space-y-4">
        <p>
          Templates provide pre-configured training setups for common use cases.
          They include optimized hyperparameters and recommended models/datasets.
        </p>

        <h4 className="font-semibold">Available Categories:</h4>
        <div className="space-y-2 text-sm">
          <div className="p-2 border rounded">
            <strong>Math Reasoning:</strong> GRPO training for mathematical problem solving.
            Uses OpenMathInstruct-2 and Qwen models.
          </div>
          <div className="p-2 border rounded">
            <strong>Code Generation:</strong> Training for code writing and understanding.
            Uses code-specific models and datasets.
          </div>
          <div className="p-2 border rounded">
            <strong>Chat & Conversation:</strong> Building helpful conversational AI.
            Uses chat datasets and instruction-tuned models.
          </div>
        </div>

        <h4 className="font-semibold mt-4">How to Use:</h4>
        <ol className="list-decimal list-inside text-sm space-y-1">
          <li>Click "Templates" in the header</li>
          <li>Browse or search for your use case</li>
          <li>Click "Apply Template" to load settings</li>
          <li>Customize any values as needed</li>
          <li>Generate your SLURM script</li>
        </ol>
      </div>
    ),
  },
];

/**
 * Tutorial steps for onboarding
 */
const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "welcome",
    title: "Welcome to NeMo RL!",
    description: "Let's create your first training configuration",
    icon: <Book className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          This tutorial will guide you through creating a training configuration
          for NeMo RL. We'll cover the essential steps to get you training quickly.
        </p>
        <p className="text-sm text-muted-foreground">
          You can exit this tutorial at any time and come back later.
        </p>
      </div>
    ),
  },
  {
    id: "algorithm",
    title: "Step 1: Choose Your Algorithm",
    description: "Select how you want to train your model",
    icon: <Sparkles className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          Start by selecting your training algorithm:
        </p>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-amber-500 mt-1 shrink-0" />
            <span><strong>GRPO</strong> - Best for math, code, and reasoning tasks</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-sky-500 mt-1 shrink-0" />
            <span><strong>SFT</strong> - Best for instruction following and domain adaptation</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-rose-500 mt-1 shrink-0" />
            <span><strong>DPO</strong> - Best for alignment and preference learning</span>
          </li>
        </ul>
      </div>
    ),
  },
  {
    id: "model",
    title: "Step 2: Select a Model",
    description: "Choose a pre-trained model to fine-tune",
    icon: <Cpu className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          Search for a model from HuggingFace or enter a local path.
          Start with smaller models (1.5B-3B) for experimentation.
        </p>
        <div className="p-3 bg-muted/50 rounded-lg text-sm">
          <strong>Recommended for beginners:</strong>
          <ul className="mt-1 space-y-1">
            <li>• Qwen/Qwen2.5-1.5B (general purpose)</li>
            <li>• Qwen/Qwen2.5-Coder-1.5B (code)</li>
            <li>• meta-llama/Llama-3.2-3B (chat)</li>
          </ul>
        </div>
      </div>
    ),
  },
  {
    id: "dataset",
    title: "Step 3: Choose Your Dataset",
    description: "Select training data that matches your task",
    icon: <Database className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          The dataset should contain examples similar to what you want 
          your model to learn.
        </p>
        <div className="p-3 bg-muted/50 rounded-lg text-sm">
          <strong>Popular choices:</strong>
          <ul className="mt-1 space-y-1">
            <li>• nvidia/OpenMathInstruct-2 (math)</li>
            <li>• bigcode/the-stack-v2-train-smol-ids (code)</li>
            <li>• HuggingFaceH4/ultrachat_200k (chat)</li>
          </ul>
        </div>
      </div>
    ),
  },
  {
    id: "hyperparams",
    title: "Step 4: Configure Hyperparameters",
    description: "Fine-tune training settings",
    icon: <Settings className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          Adjust the training hyperparameters. Defaults work well for most cases.
        </p>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <strong className="w-28 shrink-0">Learning Rate:</strong>
            <span>1e-6 for GRPO, 2e-5 for SFT</span>
          </li>
          <li className="flex items-start gap-2">
            <strong className="w-28 shrink-0">Batch Size:</strong>
            <span>16-32 for most experiments</span>
          </li>
          <li className="flex items-start gap-2">
            <strong className="w-28 shrink-0">Max Steps:</strong>
            <span>Start with 100-500 for testing</span>
          </li>
        </ul>
      </div>
    ),
  },
  {
    id: "cluster",
    title: "Step 5: Set Up Cluster",
    description: "Configure compute resources",
    icon: <Server className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          Configure your compute cluster. Use presets for common setups.
        </p>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <strong className="w-20 shrink-0">Nodes:</strong>
            <span>Start with 1 for experimentation</span>
          </li>
          <li className="flex items-start gap-2">
            <strong className="w-20 shrink-0">GPUs:</strong>
            <span>4-8 GPUs is typical for most models</span>
          </li>
          <li className="flex items-start gap-2">
            <strong className="w-20 shrink-0">Backend:</strong>
            <span>DTensor is recommended for beginners</span>
          </li>
        </ul>
      </div>
    ),
  },
  {
    id: "complete",
    title: "You're Ready!",
    description: "Download your script and start training",
    icon: <CheckCircle2 className="h-6 w-6" />,
    content: (
      <div className="space-y-3">
        <p>
          Your configuration is complete! You can now:
        </p>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
            <span>Review the generated SLURM script in the preview panel</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
            <span>Download the script and config files</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
            <span>Submit your job to your SLURM cluster</span>
          </li>
        </ul>
        <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg text-sm text-green-700 dark:text-green-300">
          <strong>Next steps:</strong> Run <code className="bg-green-200 dark:bg-green-800/50 px-1 rounded">sbatch train.sh</code> on your cluster
        </div>
      </div>
    ),
  },
];

/**
 * Help Modal Component
 */
interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTopic?: string;
}

export function HelpModal({ isOpen, onClose, initialTopic }: HelpModalProps) {
  const [selectedTopic, setSelectedTopic] = useState<string>(initialTopic || HELP_TOPICS[0].id);

  const currentTopic = HELP_TOPICS.find(t => t.id === selectedTopic) || HELP_TOPICS[0];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative z-10 w-full max-w-3xl mx-4 bg-background rounded-lg shadow-xl border max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Book className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Help & Documentation</h2>
              <p className="text-xs text-muted-foreground">
                Learn how to use NeMo RL Configurator
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close help"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-56 border-r overflow-y-auto shrink-0 p-2">
            <nav className="space-y-1">
              {HELP_TOPICS.map((topic) => (
                <button
                  key={topic.id}
                  onClick={() => setSelectedTopic(topic.id)}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors",
                    selectedTopic === topic.id
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted text-muted-foreground hover:text-foreground"
                  )}
                >
                  {topic.icon}
                  <span className="truncate">{topic.title}</span>
                </button>
              ))}
            </nav>

            {/* External links */}
            <div className="mt-4 pt-4 border-t space-y-1">
              <a
                href="https://docs.nvidia.com/nemo-rl"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                <span>Full Documentation</span>
              </a>
              <a
                href="https://github.com/NVIDIA/NeMo-RL"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                <span>GitHub Repository</span>
              </a>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-primary/10 text-primary">
                {currentTopic.icon}
              </div>
              <div>
                <h3 className="font-semibold">{currentTopic.title}</h3>
                <p className="text-sm text-muted-foreground">{currentTopic.description}</p>
              </div>
            </div>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              {currentTopic.content}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Tutorial Modal Component
 */
interface TutorialModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function TutorialModal({ isOpen, onClose }: TutorialModalProps) {
  const [currentStep, setCurrentStep] = useState(0);

  const handleNext = useCallback(() => {
    if (currentStep < TUTORIAL_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
    }
  }, [currentStep, onClose]);

  const handlePrev = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const handleSkip = useCallback(() => {
    onClose();
  }, [onClose]);

  if (!isOpen) return null;

  const step = TUTORIAL_STEPS[currentStep];
  const isLastStep = currentStep === TUTORIAL_STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={handleSkip}
      />
      
      {/* Modal */}
      <div className="relative z-10 w-full max-w-lg mx-4 bg-background rounded-lg shadow-xl border">
        {/* Progress */}
        <div className="px-6 pt-6">
          <div className="flex items-center gap-1">
            {TUTORIAL_STEPS.map((_, index) => (
              <div
                key={index}
                className={cn(
                  "flex-1 h-1 rounded-full transition-colors",
                  index <= currentStep ? "bg-primary" : "bg-muted"
                )}
              />
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-right">
            Step {currentStep + 1} of {TUTORIAL_STEPS.length}
          </p>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 rounded-full bg-primary/10 text-primary">
              {step.icon}
            </div>
            <div>
              <h3 className="text-lg font-semibold">{step.title}</h3>
              <p className="text-sm text-muted-foreground">{step.description}</p>
            </div>
          </div>
          <div className="text-sm">
            {step.content}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t bg-muted/30">
          <button
            onClick={handleSkip}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip tutorial
          </button>
          <div className="flex items-center gap-2">
            {currentStep > 0 && (
              <Button variant="ghost" size="sm" onClick={handlePrev}>
                <ChevronLeft className="h-4 w-4 mr-1" />
                Back
              </Button>
            )}
            <Button size="sm" onClick={handleNext}>
              {isLastStep ? "Get Started" : "Next"}
              {!isLastStep && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Contextual Help Button Component
 * Use this next to form fields for inline help
 */
interface HelpButtonProps {
  topic: string;
  className?: string;
}

export function HelpButton({ topic, className }: HelpButtonProps) {
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setIsHelpOpen(true)}
        className={cn(
          "p-1 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
          className
        )}
        aria-label="Show help"
      >
        <HelpCircle className="h-4 w-4" />
      </button>
      <HelpModal 
        isOpen={isHelpOpen} 
        onClose={() => setIsHelpOpen(false)}
        initialTopic={topic}
      />
    </>
  );
}

// Export types and constants for testing
export { HELP_TOPICS, TUTORIAL_STEPS };
export type { HelpTopic, TutorialStep };
