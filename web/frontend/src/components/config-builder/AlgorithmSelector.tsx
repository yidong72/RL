import { Check, Sparkles, BookOpen, Scale } from "lucide-react";
import { cn } from "../../lib/utils";
import { useConfigStore } from "../../store/configStore";
import type { Algorithm } from "../../types/config";
import { ALGORITHM_INFO } from "../../types/config";

const ALGORITHM_ICONS: Record<Algorithm, React.ReactNode> = {
  grpo: <Sparkles className="h-8 w-8" />,
  sft: <BookOpen className="h-8 w-8" />,
  dpo: <Scale className="h-8 w-8" />,
};

interface AlgorithmCardProps {
  algorithm: Algorithm;
  isSelected: boolean;
  onSelect: () => void;
}

function AlgorithmCard({ algorithm, isSelected, onSelect }: AlgorithmCardProps) {
  const info = ALGORITHM_INFO[algorithm];

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "relative flex flex-col items-center p-6 rounded-xl border-2 transition-all duration-200",
        "hover:border-primary/50 hover:shadow-md",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
        isSelected
          ? "border-primary bg-primary/5 shadow-md"
          : "border-border bg-card"
      )}
      aria-pressed={isSelected}
      aria-label={`Select ${info.name} algorithm`}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute top-3 right-3 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Check className="h-4 w-4" />
        </div>
      )}

      {/* Icon */}
      <div
        className={cn(
          "mb-4 p-3 rounded-full",
          isSelected ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
        )}
      >
        {ALGORITHM_ICONS[algorithm]}
      </div>

      {/* Title */}
      <h3 className="text-lg font-semibold mb-2">{info.name}</h3>

      {/* Description */}
      <p className="text-sm text-muted-foreground text-center mb-4">
        {info.description}
      </p>

      {/* Use cases */}
      <div className="flex flex-wrap gap-1 justify-center">
        {info.useCases.slice(0, 2).map((useCase) => (
          <span
            key={useCase}
            className={cn(
              "text-xs px-2 py-1 rounded-full",
              isSelected
                ? "bg-primary/10 text-primary"
                : "bg-muted text-muted-foreground"
            )}
          >
            {useCase}
          </span>
        ))}
      </div>
    </button>
  );
}

export function AlgorithmSelector() {
  const { config, setAlgorithm } = useConfigStore();
  const selectedAlgorithm = config.algorithm;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Select Training Algorithm</h2>
        <p className="text-sm text-muted-foreground">
          Choose the algorithm that best fits your training objective
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(Object.keys(ALGORITHM_INFO) as Algorithm[]).map((algorithm) => (
          <AlgorithmCard
            key={algorithm}
            algorithm={algorithm}
            isSelected={selectedAlgorithm === algorithm}
            onSelect={() => setAlgorithm(algorithm)}
          />
        ))}
      </div>

      {/* Contextual help */}
      <div className="p-4 rounded-lg bg-muted/50 border border-border">
        <p className="text-sm">
          <span className="font-medium">{ALGORITHM_INFO[selectedAlgorithm].name}</span>
          {" "}is best for:{" "}
          {ALGORITHM_INFO[selectedAlgorithm].useCases.join(", ")}.
        </p>
      </div>
    </div>
  );
}
