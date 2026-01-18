import { useState, useCallback } from "react";
import { 
  Play, 
  AlertTriangle, 
  CheckCircle2,
  RefreshCw,
  FlaskConical,
  MessageSquare,
  FileText,
  Sparkles,
  Clock,
  X
} from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "../common/Button";

/**
 * Test result interface
 */
interface TestResult {
  success: boolean;
  score?: number;
  executionTime?: number;
  error?: string;
  traceback?: string;
}

/**
 * Test case interface
 */
interface TestCase {
  id: string;
  prompt: string;
  response: string;
  expectedScore?: number;
}

/**
 * Default test cases for quick testing
 */
const DEFAULT_TEST_CASES: TestCase[] = [
  {
    id: "math-correct",
    prompt: "What is 2 + 2?",
    response: "The answer is 4. I can verify this because 2 + 2 equals 4.",
    expectedScore: 1.0,
  },
  {
    id: "math-wrong",
    prompt: "What is 5 * 3?",
    response: "The answer is 12.",
    expectedScore: 0.0,
  },
  {
    id: "reasoning",
    prompt: "Explain why the sky is blue.",
    response: "The sky appears blue due to Rayleigh scattering. Sunlight interacts with molecules in the atmosphere, scattering shorter blue wavelengths more than other colors, making the sky appear blue to our eyes.",
    expectedScore: 1.0,
  },
];

/**
 * Execute reward function in browser (sandboxed evaluation)
 * This simulates backend execution - in production this would call an API
 */
async function executeRewardFunction(
  code: string,
  prompt: string,
  response: string
): Promise<TestResult> {
  const startTime = performance.now();
  
  try {
    // Create a sandboxed function execution context
    // Note: In production, this should be executed on the backend for security
    
    // Check if the code defines a reward_function
    if (!code.includes("def reward_function")) {
      return {
        success: false,
        error: "No 'reward_function' found in code",
      };
    }

    // For the frontend test harness, we'll do a simple regex-based simulation
    // Real execution would happen on the Python backend
    
    // Simulate basic reward function patterns
    const score = simulateRewardFunction(code, prompt, response);
    
    const executionTime = performance.now() - startTime;
    
    return {
      success: true,
      score,
      executionTime,
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Execution failed",
      traceback: err instanceof Error ? err.stack : undefined,
    };
  }
}

/**
 * Simulate reward function execution based on common patterns
 * This is a client-side approximation - real execution would be server-side
 */
function simulateRewardFunction(code: string, prompt: string, response: string): number {
  // Exact match pattern
  if (code.includes("exact match") || code.includes("normalized_response == normalized_expected")) {
    // Simple length-based heuristic for demo
    return response.length > 20 ? 0.8 : 0.3;
  }
  
  // Length-based pattern
  if (code.includes("length") || code.includes("word_count")) {
    const wordCount = response.split(/\s+/).length;
    const minLength = 50;
    const maxLength = 500;
    
    if (wordCount < minLength) {
      return wordCount / minLength;
    } else if (wordCount > maxLength) {
      return Math.max(0, 1 - (wordCount - maxLength) / maxLength);
    }
    return 0.8 + Math.random() * 0.2;
  }
  
  // Keyword pattern
  if (code.includes("keyword") || code.includes("contains")) {
    const keywords = ["therefore", "solution", "answer", "because", "result"];
    const responseLower = response.toLowerCase();
    const matchCount = keywords.filter(kw => responseLower.includes(kw)).length;
    return Math.min(1.0, matchCount / 3);
  }
  
  // Regex pattern
  if (code.includes("regex") || code.includes("re.search")) {
    // Check for boxed answer pattern
    if (response.includes("\\boxed{") || response.match(/the answer is \d+/i)) {
      return 1.0;
    }
    return 0.0;
  }
  
  // Math correctness pattern
  if (code.includes("math") || code.includes("numerical")) {
    // Look for numbers in response
    const numbers = response.match(/\d+(\.\d+)?/g);
    if (numbers && numbers.length > 0) {
      return 0.7 + Math.random() * 0.3;
    }
    return 0.3;
  }
  
  // Default: Random score for custom functions
  return 0.5 + Math.random() * 0.5;
}

/**
 * TestHarness Props
 */
interface TestHarnessProps {
  isOpen: boolean;
  onClose: () => void;
  rewardCode: string;
}

/**
 * Test Harness Component
 * Allows testing reward functions with sample prompts and responses
 */
export function TestHarness({ isOpen, onClose, rewardCode }: TestHarnessProps) {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [result, setResult] = useState<TestResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  // Handle test execution
  const handleRunTest = useCallback(async () => {
    if (!prompt.trim() || !response.trim()) {
      setResult({
        success: false,
        error: "Please provide both a prompt and response to test",
      });
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const testResult = await executeRewardFunction(rewardCode, prompt, response);
      setResult(testResult);
    } finally {
      setIsRunning(false);
    }
  }, [rewardCode, prompt, response]);

  // Load preset test case
  const handleLoadPreset = useCallback((testCase: TestCase) => {
    setPrompt(testCase.prompt);
    setResponse(testCase.response);
    setSelectedPreset(testCase.id);
    setResult(null);
  }, []);

  // Clear form
  const handleClear = useCallback(() => {
    setPrompt("");
    setResponse("");
    setResult(null);
    setSelectedPreset(null);
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      
      {/* Modal */}
      <div 
        className="relative z-10 w-full max-w-3xl mx-4 bg-background rounded-lg shadow-xl border max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="test-harness-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <FlaskConical className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h2 id="test-harness-title" className="text-lg font-semibold">
                Reward Function Test Harness
              </h2>
              <p className="text-xs text-muted-foreground">
                Test your reward function with sample prompts and responses
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close test harness"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Preset Test Cases */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Quick Test Presets
            </label>
            <div className="grid grid-cols-3 gap-2">
              {DEFAULT_TEST_CASES.map((testCase) => (
                <button
                  key={testCase.id}
                  type="button"
                  onClick={() => handleLoadPreset(testCase)}
                  className={cn(
                    "p-3 text-left border rounded-lg text-xs transition-all",
                    "hover:border-primary/50 hover:bg-primary/5",
                    selectedPreset === testCase.id && "border-primary bg-primary/5"
                  )}
                >
                  <div className="font-medium truncate">{testCase.prompt.slice(0, 25)}...</div>
                  <div className="text-muted-foreground mt-1">
                    Expected: {testCase.expectedScore?.toFixed(1)}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Prompt Input */}
          <div>
            <label htmlFor="test-prompt" className="block text-sm font-medium mb-2">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" aria-hidden="true" />
                Prompt
              </div>
            </label>
            <textarea
              id="test-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter the prompt that would be given to the model..."
              rows={3}
              className={cn(
                "w-full px-3 py-2 border rounded-lg text-sm resize-none",
                "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
              )}
            />
          </div>

          {/* Response Input */}
          <div>
            <label htmlFor="test-response" className="block text-sm font-medium mb-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4" aria-hidden="true" />
                Model Response
              </div>
            </label>
            <textarea
              id="test-response"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder="Enter the model's response to evaluate..."
              rows={5}
              className={cn(
                "w-full px-3 py-2 border rounded-lg text-sm resize-none",
                "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
              )}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <Button
              onClick={handleRunTest}
              disabled={isRunning || !prompt.trim() || !response.trim()}
              className="flex-1"
            >
              {isRunning ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" aria-hidden="true" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" aria-hidden="true" />
                  Run Test
                </>
              )}
            </Button>
            <Button variant="outline" onClick={handleClear}>
              Clear
            </Button>
          </div>

          {/* Result Display */}
          {result && (
            <div
              className={cn(
                "p-4 rounded-lg border",
                result.success
                  ? "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-900"
                  : "bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900"
              )}
              role="alert"
              aria-live="polite"
            >
              <div className="flex items-start gap-3">
                {result.success ? (
                  <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 shrink-0 mt-0.5" aria-hidden="true" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0 mt-0.5" aria-hidden="true" />
                )}
                
                <div className="flex-1 min-w-0">
                  <h3 className={cn(
                    "font-semibold text-sm",
                    result.success ? "text-green-800 dark:text-green-200" : "text-red-800 dark:text-red-200"
                  )}>
                    {result.success ? "Test Passed" : "Test Failed"}
                  </h3>
                  
                  {result.success && result.score !== undefined && (
                    <div className="mt-3 space-y-2">
                      {/* Score Display */}
                      <div className="flex items-center gap-3">
                        <Sparkles className="h-4 w-4 text-amber-500" aria-hidden="true" />
                        <span className="text-sm font-medium">Reward Score:</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full transition-all duration-500",
                                result.score >= 0.7 && "bg-green-500",
                                result.score >= 0.4 && result.score < 0.7 && "bg-yellow-500",
                                result.score < 0.4 && "bg-red-500"
                              )}
                              style={{ width: `${result.score * 100}%` }}
                            />
                          </div>
                          <span className="text-lg font-bold">
                            {result.score.toFixed(3)}
                          </span>
                        </div>
                      </div>
                      
                      {/* Execution Time */}
                      {result.executionTime !== undefined && (
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" aria-hidden="true" />
                          <span>Execution time: {result.executionTime.toFixed(2)}ms</span>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Error Display */}
                  {result.error && (
                    <div className="mt-2">
                      <p className="text-sm text-red-700 dark:text-red-300">
                        {result.error}
                      </p>
                      {result.traceback && (
                        <pre className="mt-2 p-2 bg-red-100 dark:bg-red-900/50 rounded text-xs overflow-x-auto">
                          {result.traceback}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t shrink-0 bg-muted/30">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <FlaskConical className="h-4 w-4" aria-hidden="true" />
            <span>
              Note: This is a simulation. In production, reward functions are executed 
              on the server with proper Python sandboxing.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export for testing
export { executeRewardFunction, simulateRewardFunction, DEFAULT_TEST_CASES };
export type { TestResult, TestCase };
