import { useState, useMemo, useCallback, useEffect } from "react";
import { 
  Search, 
  Calculator, 
  Code2, 
  MessageSquare, 
  Zap, 
  Book, 
  Check,
  X,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Scale,
  BookOpen,
  User,
  Save,
  Share2,
  Trash2,
  Edit3,
  Link,
  Copy,
  Plus
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { Button } from "../common/Button";
import type { TrainingConfig, Algorithm } from "../../types/config";
import {
  getUserTemplates,
  saveUserTemplate,
  deleteUserTemplate,
  updateUserTemplate,
  encodeConfigToShareableLink,
  type UserTemplate,
} from "../../lib/templateStorage";

/**
 * Template category type
 */
type TemplateCategory = "math" | "code" | "chat" | "all" | "user";

/**
 * Template definition
 */
interface Template {
  id: string;
  name: string;
  description: string;
  category: TemplateCategory;
  algorithm: Algorithm;
  tags: string[];
  config: TrainingConfig;
  isUserTemplate?: boolean;
  createdAt?: string;
  updatedAt?: string;
}

/**
 * Built-in templates from NeMo RL documentation recipes
 */
const BUILT_IN_TEMPLATES: Template[] = [
  // Math Reasoning Templates
  {
    id: "math-grpo-qwen2.5",
    name: "Math Problem Solving",
    description: "GRPO training for mathematical reasoning using Qwen2.5 model with OpenMathInstruct-2 dataset. Optimized for step-by-step problem solving.",
    category: "math",
    algorithm: "grpo",
    tags: ["GRPO", "Math", "Qwen", "Reasoning"],
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
  },
  {
    id: "math-grpo-deepscaler",
    name: "DeepScaleR Math",
    description: "Advanced GRPO configuration for scaling mathematical reasoning capabilities. Based on DeepScaleR training methodology.",
    category: "math",
    algorithm: "grpo",
    tags: ["GRPO", "Math", "DeepScaleR", "Advanced"],
    config: {
      algorithm: "grpo",
      model: "Qwen/Qwen2.5-7B",
      dataset: "nvidia/OpenMathInstruct-2",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 5e-7,
        batch_size: 64,
        max_steps: 2000,
        num_generations_per_prompt: 32,
        tensor_parallel_size: 2,
      },
      cluster: {
        nodes: 2,
        gpus_per_node: 8,
        time_limit: "8:00:00",
      },
    },
  },
  {
    id: "math-sft-instruct",
    name: "Math Instruction Tuning",
    description: "Supervised fine-tuning on mathematical instruction data. Good starting point before GRPO training.",
    category: "math",
    algorithm: "sft",
    tags: ["SFT", "Math", "Instruction", "Foundation"],
    config: {
      algorithm: "sft",
      model: "Qwen/Qwen2.5-1.5B",
      dataset: "nvidia/OpenMathInstruct-2",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 2e-5,
        batch_size: 16,
        max_steps: 5000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 4,
        time_limit: "2:00:00",
      },
    },
  },
  // Code Generation Templates
  {
    id: "code-grpo-generation",
    name: "Code Generation (GRPO)",
    description: "GRPO training for code generation tasks. Uses execution-based reward signal for improved code quality.",
    category: "code",
    algorithm: "grpo",
    tags: ["GRPO", "Code", "Generation", "Execution"],
    config: {
      algorithm: "grpo",
      model: "Qwen/Qwen2.5-Coder-1.5B",
      dataset: "bigcode/the-stack-v2-train-smol-ids",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 1e-6,
        batch_size: 24,
        max_steps: 1500,
        num_generations_per_prompt: 8,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 8,
        time_limit: "6:00:00",
      },
    },
  },
  {
    id: "code-sft-instruct",
    name: "Code Instruction Tuning",
    description: "Supervised fine-tuning for code understanding and instruction following. Foundation for code assistants.",
    category: "code",
    algorithm: "sft",
    tags: ["SFT", "Code", "Instruction", "Assistant"],
    config: {
      algorithm: "sft",
      model: "Qwen/Qwen2.5-Coder-1.5B",
      dataset: "HuggingFaceH4/CodeAlpaca-20k",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 2e-5,
        batch_size: 8,
        max_steps: 3000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 4,
        time_limit: "3:00:00",
      },
    },
  },
  {
    id: "code-dpo-quality",
    name: "Code Quality Improvement",
    description: "DPO training to improve code quality through preference learning. Uses pairs of good vs bad code examples.",
    category: "code",
    algorithm: "dpo",
    tags: ["DPO", "Code", "Quality", "Preference"],
    config: {
      algorithm: "dpo",
      model: "Qwen/Qwen2.5-Coder-1.5B",
      dataset: "nvidia/code-preferences",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 5e-7,
        batch_size: 8,
        max_steps: 1000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 4,
        time_limit: "2:00:00",
      },
    },
  },
  // Chat/Conversation Templates
  {
    id: "chat-sft-instruction",
    name: "Chat Instruction Tuning",
    description: "Supervised fine-tuning for conversational AI. Creates helpful, harmless, and honest assistants.",
    category: "chat",
    algorithm: "sft",
    tags: ["SFT", "Chat", "Instruction", "Assistant"],
    config: {
      algorithm: "sft",
      model: "meta-llama/Llama-3.2-3B",
      dataset: "HuggingFaceH4/ultrachat_200k",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 2e-5,
        batch_size: 16,
        max_steps: 10000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 8,
        time_limit: "8:00:00",
      },
    },
  },
  {
    id: "chat-dpo-alignment",
    name: "Chat Alignment (DPO)",
    description: "DPO training for aligning chat models with human preferences. Improves safety and helpfulness.",
    category: "chat",
    algorithm: "dpo",
    tags: ["DPO", "Chat", "Alignment", "Safety"],
    config: {
      algorithm: "dpo",
      model: "meta-llama/Llama-3.2-3B",
      dataset: "Anthropic/hh-rlhf",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 5e-7,
        batch_size: 8,
        max_steps: 2000,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 4,
        time_limit: "4:00:00",
      },
    },
  },
  {
    id: "chat-grpo-helpfulness",
    name: "Chat Helpfulness (GRPO)",
    description: "GRPO training to maximize helpfulness in conversations using reward model feedback.",
    category: "chat",
    algorithm: "grpo",
    tags: ["GRPO", "Chat", "Helpfulness", "Reward"],
    config: {
      algorithm: "grpo",
      model: "meta-llama/Llama-3.2-3B",
      dataset: "HuggingFaceH4/ultrachat_200k",
      backend: "dtensor",
      hyperparameters: {
        learning_rate: 1e-6,
        batch_size: 32,
        max_steps: 1000,
        num_generations_per_prompt: 8,
        tensor_parallel_size: 1,
      },
      cluster: {
        nodes: 1,
        gpus_per_node: 8,
        time_limit: "6:00:00",
      },
    },
  },
];

/**
 * Category icons and metadata
 */
const CATEGORY_INFO: Record<TemplateCategory, { 
  name: string; 
  icon: React.ReactNode;
  description: string;
}> = {
  all: {
    name: "All Templates",
    icon: <Sparkles className="h-4 w-4" />,
    description: "Browse all available templates",
  },
  user: {
    name: "My Templates",
    icon: <User className="h-4 w-4" />,
    description: "Your saved custom templates",
  },
  math: {
    name: "Math Reasoning",
    icon: <Calculator className="h-4 w-4" />,
    description: "Mathematical problem solving and reasoning",
  },
  code: {
    name: "Code Generation",
    icon: <Code2 className="h-4 w-4" />,
    description: "Code writing and understanding",
  },
  chat: {
    name: "Chat & Conversation",
    icon: <MessageSquare className="h-4 w-4" />,
    description: "Conversational AI and assistants",
  },
};

/**
 * Algorithm icons for template cards
 */
const ALGORITHM_ICONS: Record<Algorithm, React.ReactNode> = {
  grpo: <Sparkles className="h-4 w-4" />,
  sft: <BookOpen className="h-4 w-4" />,
  dpo: <Scale className="h-4 w-4" />,
};

/**
 * Save Template Modal Props
 */
interface SaveTemplateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description: string, tags: string[]) => void;
  initialName?: string;
  initialDescription?: string;
  initialTags?: string[];
  isEditing?: boolean;
}

/**
 * Save Template Modal Component
 */
function SaveTemplateModal({ 
  isOpen, 
  onClose, 
  onSave,
  initialName = "",
  initialDescription = "",
  initialTags = [],
  isEditing = false,
}: SaveTemplateModalProps) {
  const [name, setName] = useState(initialName);
  const [description, setDescription] = useState(initialDescription);
  const [tagsInput, setTagsInput] = useState(initialTags.join(", "));

  useEffect(() => {
    setName(initialName);
    setDescription(initialDescription);
    setTagsInput(initialTags.join(", "));
  }, [initialName, initialDescription, initialTags, isOpen]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const tags = tagsInput.split(",").map(t => t.trim()).filter(Boolean);
    onSave(name, description, tags);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-md mx-4 bg-background rounded-lg shadow-xl border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">
            {isEditing ? "Edit Template" : "Save as Template"}
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="template-name" className="block text-sm font-medium mb-1">
              Template Name *
            </label>
            <input
              id="template-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Custom Training Config"
              required
              className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          
          <div>
            <label htmlFor="template-description" className="block text-sm font-medium mb-1">
              Description
            </label>
            <textarea
              id="template-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this template is for..."
              rows={3}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            />
          </div>
          
          <div>
            <label htmlFor="template-tags" className="block text-sm font-medium mb-1">
              Tags (comma-separated)
            </label>
            <input
              id="template-tags"
              type="text"
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              placeholder="GRPO, Math, Custom"
              className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim()}>
              <Save className="h-4 w-4 mr-2" />
              {isEditing ? "Update" : "Save"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

/**
 * Share Link Modal Props
 */
interface ShareLinkModalProps {
  isOpen: boolean;
  onClose: () => void;
  shareUrl: string;
}

/**
 * Share Link Modal Component
 */
function ShareLinkModal({ isOpen, onClose, shareUrl }: ShareLinkModalProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-lg mx-4 bg-background rounded-lg shadow-xl border p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Link className="h-5 w-5" />
            </div>
            <h3 className="text-lg font-semibold">Share Template</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        
        <p className="text-sm text-muted-foreground mb-4">
          Share this link with others to let them load your configuration:
        </p>
        
        <div className="flex gap-2">
          <input
            type="text"
            value={shareUrl}
            readOnly
            className="flex-1 px-3 py-2 border rounded-lg text-sm bg-muted/30 font-mono text-xs"
          />
          <Button onClick={handleCopy}>
            {copied ? (
              <>
                <Check className="h-4 w-4 mr-2" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </>
            )}
          </Button>
        </div>
        
        <p className="text-xs text-muted-foreground mt-4">
          This link contains your entire configuration encoded in the URL.
          Anyone with this link can load your config settings.
        </p>
      </div>
    </div>
  );
}

/**
 * Template Card Component
 */
interface TemplateCardProps {
  template: Template;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onApply: () => void;
  onShare?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
}

function TemplateCard({ 
  template, 
  isExpanded, 
  onToggleExpand, 
  onApply,
  onShare,
  onEdit,
  onDelete,
}: TemplateCardProps) {
  const categoryInfo = CATEGORY_INFO[template.category === "user" ? "user" : template.category];

  return (
    <div
      className={cn(
        "border rounded-lg p-4 transition-all duration-200",
        "hover:border-primary/50 hover:shadow-sm",
        isExpanded && "border-primary/50 bg-primary/5",
        template.isUserTemplate && "border-dashed"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1">
          {/* Category Icon */}
          <div className={cn(
            "p-2 rounded-lg shrink-0",
            template.isUserTemplate && "bg-violet-500/10 text-violet-500",
            !template.isUserTemplate && template.category === "math" && "bg-blue-500/10 text-blue-500",
            !template.isUserTemplate && template.category === "code" && "bg-green-500/10 text-green-500",
            !template.isUserTemplate && template.category === "chat" && "bg-purple-500/10 text-purple-500"
          )}>
            {template.isUserTemplate ? <User className="h-4 w-4" /> : categoryInfo.icon}
          </div>

          {/* Title and Description */}
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm leading-tight">{template.name}</h3>
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {template.description}
            </p>
          </div>
        </div>

        {/* Algorithm Badge */}
        <div className={cn(
          "flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium shrink-0",
          template.algorithm === "grpo" && "bg-amber-500/10 text-amber-600 dark:text-amber-400",
          template.algorithm === "sft" && "bg-sky-500/10 text-sky-600 dark:text-sky-400",
          template.algorithm === "dpo" && "bg-rose-500/10 text-rose-600 dark:text-rose-400"
        )}>
          {ALGORITHM_ICONS[template.algorithm]}
          <span>{template.algorithm.toUpperCase()}</span>
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mt-3">
        {template.tags.slice(0, 4).map((tag) => (
          <span
            key={tag}
            className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground"
          >
            {tag}
          </span>
        ))}
        {template.isUserTemplate && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-violet-500/10 text-violet-600 dark:text-violet-400">
            Custom
          </span>
        )}
      </div>

      {/* Expand/Collapse Button */}
      <button
        type="button"
        onClick={onToggleExpand}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground mt-3 transition-colors"
      >
        {isExpanded ? (
          <>
            <ChevronUp className="h-3 w-3" />
            <span>Hide details</span>
          </>
        ) : (
          <>
            <ChevronDown className="h-3 w-3" />
            <span>View details</span>
          </>
        )}
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t space-y-4">
          {/* Configuration Summary */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Model:</span>
              <p className="font-mono text-[11px] truncate">{template.config.model}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Dataset:</span>
              <p className="font-mono text-[11px] truncate">{template.config.dataset}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Learning Rate:</span>
              <p className="font-mono text-[11px]">{template.config.hyperparameters.learning_rate}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Batch Size:</span>
              <p className="font-mono text-[11px]">{template.config.hyperparameters.batch_size}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Max Steps:</span>
              <p className="font-mono text-[11px]">{template.config.hyperparameters.max_steps}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Cluster:</span>
              <p className="font-mono text-[11px]">
                {template.config.cluster.nodes}N × {template.config.cluster.gpus_per_node}G
              </p>
            </div>
          </div>

          {/* User template metadata */}
          {template.isUserTemplate && template.updatedAt && (
            <div className="text-xs text-muted-foreground">
              Last updated: {new Date(template.updatedAt).toLocaleDateString()}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2">
            <Button 
              onClick={onApply}
              className="flex-1"
              size="sm"
            >
              <Check className="h-4 w-4 mr-2" />
              Apply Template
            </Button>
            
            {onShare && (
              <Button 
                onClick={onShare}
                variant="outline"
                size="sm"
              >
                <Share2 className="h-4 w-4" />
              </Button>
            )}
            
            {template.isUserTemplate && onEdit && (
              <Button 
                onClick={onEdit}
                variant="outline"
                size="sm"
              >
                <Edit3 className="h-4 w-4" />
              </Button>
            )}
            
            {template.isUserTemplate && onDelete && (
              <Button 
                onClick={onDelete}
                variant="outline"
                size="sm"
                className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950/30"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Template Library Modal Props
 */
interface TemplateLibraryProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Template Library Component
 * Provides searchable, filterable template selection with user templates support
 */
export function TemplateLibrary({ isOpen, onClose }: TemplateLibraryProps) {
  const { config, setConfig } = useConfigStore();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<TemplateCategory>("all");
  const [expandedTemplate, setExpandedTemplate] = useState<string | null>(null);
  const [userTemplates, setUserTemplates] = useState<UserTemplate[]>([]);
  
  // Modal states
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [shareUrl, setShareUrl] = useState("");
  const [editingTemplate, setEditingTemplate] = useState<UserTemplate | null>(null);

  // Load user templates from localStorage
  useEffect(() => {
    if (isOpen) {
      setUserTemplates(getUserTemplates());
    }
  }, [isOpen]);

  // Convert user templates to Template format
  const userTemplatesAsTemplates: Template[] = useMemo(() => {
    return userTemplates.map(ut => ({
      id: ut.id,
      name: ut.name,
      description: ut.description,
      category: "user" as TemplateCategory,
      algorithm: ut.algorithm,
      tags: ut.tags,
      config: ut.config,
      isUserTemplate: true,
      createdAt: ut.createdAt,
      updatedAt: ut.updatedAt,
    }));
  }, [userTemplates]);

  // Combine all templates
  const allTemplates = useMemo(() => {
    return [...userTemplatesAsTemplates, ...BUILT_IN_TEMPLATES];
  }, [userTemplatesAsTemplates]);

  // Filter templates based on search and category
  const filteredTemplates = useMemo(() => {
    return allTemplates.filter((template) => {
      // Category filter
      if (selectedCategory === "user") {
        if (!template.isUserTemplate) return false;
      } else if (selectedCategory !== "all") {
        if (template.category !== selectedCategory && !template.isUserTemplate) {
          return false;
        }
        if (template.isUserTemplate && selectedCategory !== "all") {
          return false;
        }
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          template.name.toLowerCase().includes(query) ||
          template.description.toLowerCase().includes(query) ||
          template.tags.some((tag) => tag.toLowerCase().includes(query)) ||
          template.algorithm.toLowerCase().includes(query)
        );
      }

      return true;
    });
  }, [allTemplates, searchQuery, selectedCategory]);

  // Handle template application
  const handleApplyTemplate = useCallback((template: Template) => {
    setConfig(template.config);
    onClose();
  }, [setConfig, onClose]);

  // Toggle template expansion
  const handleToggleExpand = useCallback((templateId: string) => {
    setExpandedTemplate(expandedTemplate === templateId ? null : templateId);
  }, [expandedTemplate]);

  // Handle save current config as template
  const handleSaveCurrentConfig = useCallback((name: string, description: string, tags: string[]) => {
    if (editingTemplate) {
      // Update existing template
      updateUserTemplate(editingTemplate.id, {
        name,
        description,
        tags,
        config,
        algorithm: config.algorithm,
      });
    } else {
      // Save new template
      saveUserTemplate({
        name,
        description,
        algorithm: config.algorithm,
        tags,
        config,
      });
    }
    
    // Refresh user templates
    setUserTemplates(getUserTemplates());
    setIsSaveModalOpen(false);
    setEditingTemplate(null);
  }, [config, editingTemplate]);

  // Handle share template
  const handleShareTemplate = useCallback((template: Template) => {
    const url = encodeConfigToShareableLink(template.config, template.name);
    setShareUrl(url);
    setIsShareModalOpen(true);
  }, []);

  // Handle edit user template
  const handleEditTemplate = useCallback((template: Template) => {
    const userTemplate = userTemplates.find(ut => ut.id === template.id);
    if (userTemplate) {
      setEditingTemplate(userTemplate);
      setIsSaveModalOpen(true);
    }
  }, [userTemplates]);

  // Handle delete user template
  const handleDeleteTemplate = useCallback((templateId: string) => {
    if (confirm("Are you sure you want to delete this template?")) {
      deleteUserTemplate(templateId);
      setUserTemplates(getUserTemplates());
    }
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative z-10 w-full max-w-3xl mx-4 bg-background rounded-lg shadow-xl border max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Book className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Template Library</h2>
              <p className="text-xs text-muted-foreground">
                Pre-configured training setups and your saved templates
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={() => {
                setEditingTemplate(null);
                setIsSaveModalOpen(true);
              }}
              variant="outline"
              size="sm"
            >
              <Plus className="h-4 w-4 mr-2" />
              Save Current
            </Button>
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              aria-label="Close template library"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="p-4 border-b shrink-0 space-y-3">
          {/* Search Input */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={cn(
                "w-full pl-10 pr-4 py-2 text-sm rounded-lg border bg-background",
                "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent",
                "placeholder:text-muted-foreground"
              )}
            />
          </div>

          {/* Category Filters */}
          <div className="flex flex-wrap gap-2">
            {(Object.keys(CATEGORY_INFO) as TemplateCategory[]).map((category) => {
              const info = CATEGORY_INFO[category];
              let count = 0;
              if (category === "all") {
                count = allTemplates.length;
              } else if (category === "user") {
                count = userTemplates.length;
              } else {
                count = BUILT_IN_TEMPLATES.filter((t) => t.category === category).length;
              }

              return (
                <button
                  key={category}
                  type="button"
                  onClick={() => setSelectedCategory(category)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-full transition-colors",
                    selectedCategory === category
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                  )}
                >
                  {info.icon}
                  <span>{info.name}</span>
                  <span className={cn(
                    "px-1.5 py-0.5 rounded-full text-[10px]",
                    selectedCategory === category
                      ? "bg-primary-foreground/20"
                      : "bg-background"
                  )}>
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Template List */}
        <div className="flex-1 overflow-y-auto p-4">
          {filteredTemplates.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              {selectedCategory === "user" && userTemplates.length === 0 ? (
                <>
                  <User className="h-12 w-12 text-muted-foreground/50 mb-4" />
                  <h3 className="font-medium text-muted-foreground">No saved templates yet</h3>
                  <p className="text-sm text-muted-foreground/70 mt-1">
                    Click "Save Current" to save your current configuration as a template
                  </p>
                </>
              ) : (
                <>
                  <Search className="h-12 w-12 text-muted-foreground/50 mb-4" />
                  <h3 className="font-medium text-muted-foreground">No templates found</h3>
                  <p className="text-sm text-muted-foreground/70 mt-1">
                    Try adjusting your search or filter criteria
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredTemplates.map((template) => (
                <TemplateCard
                  key={template.id}
                  template={template}
                  isExpanded={expandedTemplate === template.id}
                  onToggleExpand={() => handleToggleExpand(template.id)}
                  onApply={() => handleApplyTemplate(template)}
                  onShare={() => handleShareTemplate(template)}
                  onEdit={template.isUserTemplate ? () => handleEditTemplate(template) : undefined}
                  onDelete={template.isUserTemplate ? () => handleDeleteTemplate(template.id) : undefined}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t shrink-0 bg-muted/30">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Zap className="h-4 w-4" />
            <span>
              {filteredTemplates.length} template{filteredTemplates.length !== 1 ? "s" : ""} available
              {userTemplates.length > 0 && ` (${userTemplates.length} custom)`}.
              Click "Apply Template" to populate the configuration form.
            </span>
          </div>
        </div>
      </div>

      {/* Save Template Modal */}
      <SaveTemplateModal
        isOpen={isSaveModalOpen}
        onClose={() => {
          setIsSaveModalOpen(false);
          setEditingTemplate(null);
        }}
        onSave={handleSaveCurrentConfig}
        initialName={editingTemplate?.name || ""}
        initialDescription={editingTemplate?.description || ""}
        initialTags={editingTemplate?.tags || []}
        isEditing={!!editingTemplate}
      />

      {/* Share Link Modal */}
      <ShareLinkModal
        isOpen={isShareModalOpen}
        onClose={() => setIsShareModalOpen(false)}
        shareUrl={shareUrl}
      />
    </div>
  );
}

// Export templates and storage functions for testing
export { BUILT_IN_TEMPLATES };
export type { Template, TemplateCategory };
