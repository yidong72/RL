import { useCallback, useEffect, useRef, useState } from "react";
import Editor, { Monaco, OnMount } from "@monaco-editor/react";
import { 
  Code2, 
  AlertTriangle, 
  CheckCircle2, 
  Play,
  Layers,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Sparkles,
  Hash,
  Type,
  Ruler,
  Search,
  FlaskConical
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import { Button } from "../common/Button";
import { TestHarness } from "./TestHarness";
import type * as MonacoEditor from "monaco-editor";

/**
 * Reward function template type
 */
interface RewardTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  code: string;
}

/**
 * Validation result
 */
interface ValidationResult {
  valid: boolean;
  errors: { line: number; message: string }[];
  warnings: { line: number; message: string }[];
}

/**
 * Built-in reward function templates
 */
const REWARD_TEMPLATES: RewardTemplate[] = [
  {
    id: "exact_match",
    name: "Exact Match",
    description: "Returns 1.0 if response exactly matches expected answer, 0.0 otherwise",
    icon: <Check className="h-4 w-4" />,
    code: `def reward_function(prompt: str, response: str, expected: str = None) -> float:
    """
    Exact match reward function.
    Returns 1.0 if response matches expected answer exactly.
    
    Args:
        prompt: The input prompt
        response: The model's response
        expected: The expected answer (optional, can be extracted from prompt)
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if expected is None:
        # Extract expected from prompt if not provided
        # Customize this based on your data format
        return 0.0
    
    # Normalize strings for comparison
    normalized_response = response.strip().lower()
    normalized_expected = expected.strip().lower()
    
    return 1.0 if normalized_response == normalized_expected else 0.0
`,
  },
  {
    id: "regex_match",
    name: "Regex Match",
    description: "Returns 1.0 if response matches a regex pattern",
    icon: <Search className="h-4 w-4" />,
    code: `import re

def reward_function(prompt: str, response: str) -> float:
    """
    Regex-based reward function.
    Returns 1.0 if response matches the expected pattern.
    
    Args:
        prompt: The input prompt
        response: The model's response
    
    Returns:
        1.0 if pattern matches, 0.0 otherwise
    """
    # Define your pattern - customize as needed
    # Example: Match a number in a box like \\boxed{42}
    pattern = r'\\\\boxed\\{([^}]+)\\}'
    
    match = re.search(pattern, response)
    if match:
        # You can also extract and validate the matched content
        extracted = match.group(1)
        # Add additional validation logic here
        return 1.0
    
    return 0.0
`,
  },
  {
    id: "contains",
    name: "Contains Keywords",
    description: "Returns score based on presence of required keywords",
    icon: <Type className="h-4 w-4" />,
    code: `def reward_function(prompt: str, response: str) -> float:
    """
    Keyword-based reward function.
    Returns a score based on how many required keywords are present.
    
    Args:
        prompt: The input prompt
        response: The model's response
    
    Returns:
        Score from 0.0 to 1.0 based on keyword presence
    """
    # Define required keywords - customize as needed
    required_keywords = ["therefore", "solution", "answer"]
    
    # Count how many keywords are present
    response_lower = response.lower()
    present_count = sum(1 for kw in required_keywords if kw in response_lower)
    
    # Return proportional score
    return present_count / len(required_keywords) if required_keywords else 0.0
`,
  },
  {
    id: "length_based",
    name: "Length-Based",
    description: "Rewards responses within a target length range",
    icon: <Ruler className="h-4 w-4" />,
    code: `def reward_function(prompt: str, response: str) -> float:
    """
    Length-based reward function.
    Rewards responses that fall within a target length range.
    
    Args:
        prompt: The input prompt
        response: The model's response
    
    Returns:
        Score from 0.0 to 1.0 based on response length
    """
    # Define target length range (in tokens/words)
    min_length = 50
    max_length = 500
    optimal_length = 200
    
    # Count words (or use a tokenizer for token count)
    word_count = len(response.split())
    
    # Penalize if too short or too long
    if word_count < min_length:
        return word_count / min_length
    elif word_count > max_length:
        return max(0.0, 1.0 - (word_count - max_length) / max_length)
    else:
        # Optimal range - slight preference for optimal length
        distance_from_optimal = abs(word_count - optimal_length)
        return max(0.5, 1.0 - distance_from_optimal / (max_length - min_length))
`,
  },
  {
    id: "math_correctness",
    name: "Math Correctness",
    description: "Validates mathematical answers with numerical tolerance",
    icon: <Hash className="h-4 w-4" />,
    code: `import re

def reward_function(prompt: str, response: str, expected: float = None) -> float:
    """
    Math correctness reward function with numerical tolerance.
    Extracts numerical answer and compares with expected value.
    
    Args:
        prompt: The input prompt
        response: The model's response
        expected: The expected numerical answer
    
    Returns:
        1.0 if answer is correct (within tolerance), 0.0 otherwise
    """
    tolerance = 1e-6
    
    # Try to extract the final answer from response
    # Common patterns: "The answer is X", "= X", "\\boxed{X}"
    patterns = [
        r'\\\\boxed\\{([^}]+)\\}',
        r'[Tt]he answer is[:\\s]+([\\d.\\-]+)',
        r'=\\s*([\\d.\\-]+)\\s*$',
    ]
    
    extracted_answer = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                extracted_answer = float(match.group(1))
                break
            except ValueError:
                continue
    
    if extracted_answer is None or expected is None:
        return 0.0
    
    # Check if answer is within tolerance
    return 1.0 if abs(extracted_answer - expected) < tolerance else 0.0
`,
  },
  {
    id: "custom_lambda",
    name: "Custom Lambda",
    description: "Simple lambda function for quick customization",
    icon: <Sparkles className="h-4 w-4" />,
    code: `# Simple lambda-style reward function
# Customize this for quick experiments

def reward_function(prompt: str, response: str) -> float:
    """
    Custom reward function - modify as needed.
    
    Args:
        prompt: The input prompt
        response: The model's response
    
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Example: Reward longer, more detailed responses
    score = min(1.0, len(response) / 1000)
    
    # Add your custom logic here
    # Examples:
    # - Check for specific patterns
    # - Validate against expected output
    # - Use external APIs for evaluation
    
    return score
`,
  },
];

/**
 * Basic Python syntax validation
 * Checks for common syntax errors and best practices
 */
function validatePythonCode(code: string): ValidationResult {
  const errors: { line: number; message: string }[] = [];
  const warnings: { line: number; message: string }[] = [];
  const lines = code.split("\n");

  // Track indentation
  let inFunction = false;
  let hasReturnStatement = false;
  let hasRewardFunction = false;
  let braceStack: { char: string; line: number }[] = [];

  lines.forEach((line, index) => {
    const lineNum = index + 1;
    const trimmed = line.trim();

    // Skip empty lines and comments
    if (trimmed === "" || trimmed.startsWith("#")) {
      return;
    }

    // Check for function definition
    if (trimmed.startsWith("def ")) {
      if (trimmed.includes("reward_function")) {
        hasRewardFunction = true;
      }
      inFunction = true;
      hasReturnStatement = false;

      // Check function signature
      if (!trimmed.includes("(") || !trimmed.includes(")")) {
        errors.push({ line: lineNum, message: "Invalid function definition: missing parentheses" });
      }
      if (!trimmed.endsWith(":")) {
        errors.push({ line: lineNum, message: "Function definition must end with ':'" });
      }
    }

    // Check for return statement in function
    if (inFunction && trimmed.startsWith("return ")) {
      hasReturnStatement = true;
    }

    // Check bracket matching (skip content inside strings)
    // Remove string content to avoid false positives with brackets in strings
    let lineWithoutStrings = line;
    // Remove double-quoted strings
    lineWithoutStrings = lineWithoutStrings.replace(/"[^"]*"/g, '""');
    // Remove single-quoted strings
    lineWithoutStrings = lineWithoutStrings.replace(/'[^']*'/g, "''");
    // Remove triple-quoted strings (simplified - doesn't handle multiline)
    lineWithoutStrings = lineWithoutStrings.replace(/"""[^"]*"""/g, '""""""');
    lineWithoutStrings = lineWithoutStrings.replace(/'''[^']*'''/g, "''''''");
    
    for (const char of lineWithoutStrings) {
      if (char === "(" || char === "[" || char === "{") {
        braceStack.push({ char, line: lineNum });
      } else if (char === ")" || char === "]" || char === "}") {
        const expected = char === ")" ? "(" : char === "]" ? "[" : "{";
        if (braceStack.length === 0) {
          errors.push({ line: lineNum, message: `Unmatched closing bracket '${char}'` });
        } else if (braceStack[braceStack.length - 1].char !== expected) {
          errors.push({ line: lineNum, message: `Mismatched brackets: expected '${expected}', got '${braceStack[braceStack.length - 1].char}'` });
        } else {
          braceStack.pop();
        }
      }
    }

    // Check for common syntax issues
    if (trimmed.includes("print ") && !trimmed.includes("print(")) {
      errors.push({ line: lineNum, message: "Use print() function (Python 3 syntax)" });
    }

    // Check for tabs mixed with spaces (Python style issue)
    if (line.includes("\t") && line.includes("    ")) {
      warnings.push({ line: lineNum, message: "Mixed tabs and spaces - use consistent indentation" });
    }
  });

  // Check for unclosed brackets
  braceStack.forEach(({ char, line }) => {
    errors.push({ line, message: `Unclosed bracket '${char}'` });
  });

  // Warn if no reward_function defined
  if (!hasRewardFunction) {
    warnings.push({ line: 1, message: "No 'reward_function' found - ensure your function is named correctly" });
  }

  // Warn if function has no return statement
  if (inFunction && !hasReturnStatement && hasRewardFunction) {
    warnings.push({ line: 1, message: "reward_function should have a return statement" });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * RewardEditor Props
 */
interface RewardEditorProps {
  className?: string;
}

/**
 * Reward Function Code Editor Component
 */
export function RewardEditor({ className }: RewardEditorProps) {
  const { config, isDarkMode } = useConfigStore();
  const [code, setCode] = useState(REWARD_TEMPLATES[0].code);
  const [validation, setValidation] = useState<ValidationResult>({ valid: true, errors: [], warnings: [] });
  const [isTemplatesExpanded, setIsTemplatesExpanded] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string>(REWARD_TEMPLATES[0].id);
  const [copied, setCopied] = useState(false);
  const [isTestHarnessOpen, setIsTestHarnessOpen] = useState(false);
  const editorRef = useRef<MonacoEditor.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);

  // Only show for GRPO algorithm
  const isGrpo = config.algorithm === "grpo";

  // Handle editor mount
  const handleEditorMount: OnMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Configure Python language settings
    monaco.languages.setLanguageConfiguration("python", {
      comments: {
        lineComment: "#",
        blockComment: ['"""', '"""'],
      },
      brackets: [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"],
      ],
      autoClosingPairs: [
        { open: "{", close: "}" },
        { open: "[", close: "]" },
        { open: "(", close: ")" },
        { open: '"', close: '"', notIn: ["string"] },
        { open: "'", close: "'", notIn: ["string"] },
      ],
      indentationRules: {
        increaseIndentPattern: /^\s*(def|class|for|if|elif|else|while|try|except|finally|with|async).*:\s*$/,
        decreaseIndentPattern: /^\s*(elif|else|except|finally).*:\s*$/,
      },
    });
  }, []);

  // Update editor markers for validation
  const updateEditorMarkers = useCallback((result: ValidationResult) => {
    if (!editorRef.current || !monacoRef.current) return;

    const model = editorRef.current.getModel();
    if (!model) return;

    const markers: MonacoEditor.editor.IMarkerData[] = [
      ...result.errors.map(err => ({
        severity: monacoRef.current!.MarkerSeverity.Error,
        message: err.message,
        startLineNumber: err.line,
        startColumn: 1,
        endLineNumber: err.line,
        endColumn: model.getLineMaxColumn(err.line) || 1,
      })),
      ...result.warnings.map(warn => ({
        severity: monacoRef.current!.MarkerSeverity.Warning,
        message: warn.message,
        startLineNumber: warn.line,
        startColumn: 1,
        endLineNumber: warn.line,
        endColumn: model.getLineMaxColumn(warn.line) || 1,
      })),
    ];

    monacoRef.current.editor.setModelMarkers(model, "python-validation", markers);
  }, []);

  // Handle code change
  const handleCodeChange = useCallback((value: string | undefined) => {
    if (!value) return;
    setCode(value);
    
    // Validate the code
    const result = validatePythonCode(value);
    setValidation(result);
    updateEditorMarkers(result);
  }, [updateEditorMarkers]);

  // Handle template selection
  const handleTemplateSelect = useCallback((template: RewardTemplate) => {
    setCode(template.code);
    setSelectedTemplate(template.id);
    
    // Validate the new code
    const result = validatePythonCode(template.code);
    setValidation(result);
    updateEditorMarkers(result);
    
    // Close templates panel
    setIsTemplatesExpanded(false);
  }, [updateEditorMarkers]);

  // Copy code to clipboard
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  // Initial validation
  useEffect(() => {
    const result = validatePythonCode(code);
    setValidation(result);
  }, [code]);

  // Don't render for non-GRPO algorithms
  if (!isGrpo) {
    return null;
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Code2 className="h-5 w-5" />
            Reward Function Editor
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Write a custom reward function for GRPO training
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Validation status */}
          <div className={cn(
            "flex items-center gap-1 px-2 py-1 rounded text-xs",
            validation.valid 
              ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
              : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
          )}>
            {validation.valid ? (
              <>
                <CheckCircle2 className="h-3 w-3" />
                Valid Python
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" />
                {validation.errors.length} error{validation.errors.length !== 1 ? "s" : ""}
              </>
            )}
          </div>
          
          {validation.warnings.length > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">
              <AlertTriangle className="h-3 w-3" />
              {validation.warnings.length} warning{validation.warnings.length !== 1 ? "s" : ""}
            </div>
          )}
        </div>
      </div>

      {/* Template selector */}
      <div className="border rounded-lg">
        <button
          type="button"
          onClick={() => setIsTemplatesExpanded(!isTemplatesExpanded)}
          className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium text-sm">Template Rewards</span>
            <span className="text-xs text-muted-foreground">
              ({REWARD_TEMPLATES.find(t => t.id === selectedTemplate)?.name || "Custom"})
            </span>
          </div>
          {isTemplatesExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>

        {isTemplatesExpanded && (
          <div className="border-t p-3 grid grid-cols-2 md:grid-cols-3 gap-2">
            {REWARD_TEMPLATES.map((template) => (
              <button
                key={template.id}
                type="button"
                onClick={() => handleTemplateSelect(template)}
                className={cn(
                  "flex flex-col items-start p-3 rounded-lg border text-left transition-all",
                  "hover:border-primary/50 hover:bg-primary/5",
                  selectedTemplate === template.id && "border-primary bg-primary/5"
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className={cn(
                    "p-1 rounded",
                    selectedTemplate === template.id 
                      ? "bg-primary/20 text-primary"
                      : "bg-muted text-muted-foreground"
                  )}>
                    {template.icon}
                  </div>
                  <span className="font-medium text-sm">{template.name}</span>
                </div>
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {template.description}
                </p>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Validation errors/warnings */}
      {(validation.errors.length > 0 || validation.warnings.length > 0) && (
        <div className="space-y-2">
          {validation.errors.map((error, idx) => (
            <div 
              key={`error-${idx}`}
              className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded"
            >
              <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
              <span>Line {error.line}: {error.message}</span>
            </div>
          ))}
          {validation.warnings.map((warning, idx) => (
            <div 
              key={`warning-${idx}`}
              className="flex items-start gap-2 text-xs text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 px-3 py-2 rounded"
            >
              <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
              <span>Line {warning.line}: {warning.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Monaco Editor */}
      <div className="border rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b">
          <span className="text-xs font-medium text-muted-foreground">reward_function.py</span>
          <div className="flex items-center gap-1">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => setIsTestHarnessOpen(true)} 
              className="h-7"
              disabled={!validation.valid}
              title={validation.valid ? "Test reward function" : "Fix errors before testing"}
            >
              <FlaskConical className="h-3 w-3 mr-1" />
              <span className="text-xs">Test</span>
            </Button>
            <Button variant="ghost" size="sm" onClick={handleCopy} className="h-7">
              {copied ? (
                <>
                  <Check className="h-3 w-3 mr-1" />
                  <span className="text-xs">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3 mr-1" />
                  <span className="text-xs">Copy</span>
                </>
              )}
            </Button>
          </div>
        </div>
        <Editor
          height="350px"
          language="python"
          value={code}
          theme={isDarkMode ? "vs-dark" : "light"}
          onChange={handleCodeChange}
          onMount={handleEditorMount}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 4,
            wordWrap: "on",
            folding: true,
            renderLineHighlight: "line",
            selectOnLineNumbers: true,
            roundedSelection: true,
            cursorBlinking: "smooth",
            cursorSmoothCaretAnimation: "on",
            smoothScrolling: true,
            padding: { top: 8, bottom: 8 },
          }}
        />
      </div>

      {/* Help text */}
      <div className="p-3 rounded-lg bg-muted/50 border">
        <div className="flex items-start gap-2 text-xs text-muted-foreground">
          <Play className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-foreground mb-1">How to use:</p>
            <ul className="space-y-1 list-disc list-inside">
              <li>Your function must be named <code className="bg-muted px-1 rounded">reward_function</code></li>
              <li>It should accept <code className="bg-muted px-1 rounded">prompt</code> and <code className="bg-muted px-1 rounded">response</code> as arguments</li>
              <li>Return a float between 0.0 and 1.0</li>
              <li>The function will be called for each generated response during training</li>
              <li>Click <strong>Test</strong> to try your function with sample data</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Test Harness Modal */}
      <TestHarness
        isOpen={isTestHarnessOpen}
        onClose={() => setIsTestHarnessOpen(false)}
        rewardCode={code}
      />
    </div>
  );
}

// Export templates for testing
export { REWARD_TEMPLATES, validatePythonCode };
export type { RewardTemplate, ValidationResult };
