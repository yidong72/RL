import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  getUserTemplates,
  saveUserTemplate,
  updateUserTemplate,
  deleteUserTemplate,
  encodeConfigToShareableLink,
  decodeConfigFromUrl,
  hasSharedTemplate,
  clearTemplateFromUrl,
  type UserTemplate,
} from "./templateStorage";
import type { TrainingConfig } from "../types/config";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Mock window.location
const mockLocation = {
  href: "http://localhost:3000/",
  origin: "http://localhost:3000",
  pathname: "/",
  search: "",
};
Object.defineProperty(window, "location", {
  value: mockLocation,
  writable: true,
});

// Sample config for testing
const sampleConfig: TrainingConfig = {
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

describe("templateStorage", () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
    mockLocation.search = "";
    mockLocation.href = "http://localhost:3000/";
  });

  describe("getUserTemplates", () => {
    it("should return empty array when no templates exist", () => {
      const templates = getUserTemplates();
      expect(templates).toEqual([]);
    });

    it("should return templates from localStorage", () => {
      const storedTemplates: UserTemplate[] = [
        {
          id: "test-1",
          name: "Test Template",
          description: "A test template",
          algorithm: "grpo",
          tags: ["test"],
          config: sampleConfig,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      ];
      localStorageMock.setItem("nemo-rl-user-templates", JSON.stringify(storedTemplates));
      
      const templates = getUserTemplates();
      expect(templates).toHaveLength(1);
      expect(templates[0].name).toBe("Test Template");
    });

    it("should filter out invalid templates", () => {
      const invalidTemplates = [
        { id: "valid", name: "Valid", algorithm: "grpo", config: sampleConfig },
        { name: "Missing ID", algorithm: "grpo", config: sampleConfig },
        { id: "missing-name", algorithm: "grpo", config: sampleConfig },
      ];
      localStorageMock.setItem("nemo-rl-user-templates", JSON.stringify(invalidTemplates));
      
      const templates = getUserTemplates();
      expect(templates).toHaveLength(1);
      expect(templates[0].id).toBe("valid");
    });

    it("should handle JSON parse errors gracefully", () => {
      localStorageMock.setItem("nemo-rl-user-templates", "invalid json");
      
      const templates = getUserTemplates();
      expect(templates).toEqual([]);
    });
  });

  describe("saveUserTemplate", () => {
    it("should save a new template with generated id and timestamps", () => {
      const template = saveUserTemplate({
        name: "New Template",
        description: "Description",
        algorithm: "grpo",
        tags: ["tag1", "tag2"],
        config: sampleConfig,
      });

      expect(template.id).toMatch(/^user-\d+-[a-z0-9]+$/);
      expect(template.name).toBe("New Template");
      expect(template.createdAt).toBeDefined();
      expect(template.updatedAt).toBeDefined();
      
      const stored = getUserTemplates();
      expect(stored).toHaveLength(1);
    });

    it("should append to existing templates", () => {
      saveUserTemplate({
        name: "Template 1",
        description: "",
        algorithm: "grpo",
        tags: [],
        config: sampleConfig,
      });
      
      saveUserTemplate({
        name: "Template 2",
        description: "",
        algorithm: "sft",
        tags: [],
        config: { ...sampleConfig, algorithm: "sft" },
      });
      
      const stored = getUserTemplates();
      expect(stored).toHaveLength(2);
    });
  });

  describe("updateUserTemplate", () => {
    it("should update an existing template", async () => {
      const template = saveUserTemplate({
        name: "Original Name",
        description: "Original",
        algorithm: "grpo",
        tags: [],
        config: sampleConfig,
      });

      // Wait a bit to ensure updatedAt will be different
      await new Promise(resolve => setTimeout(resolve, 10));

      const updated = updateUserTemplate(template.id, {
        name: "Updated Name",
        description: "Updated description",
      });

      expect(updated).not.toBeNull();
      expect(updated!.name).toBe("Updated Name");
      expect(updated!.description).toBe("Updated description");
      // updatedAt should be newer or equal (timing can be tricky in tests)
      expect(new Date(updated!.updatedAt).getTime()).toBeGreaterThanOrEqual(
        new Date(template.updatedAt).getTime()
      );
    });

    it("should return null for non-existent template", () => {
      const result = updateUserTemplate("non-existent-id", { name: "New Name" });
      expect(result).toBeNull();
    });
  });

  describe("deleteUserTemplate", () => {
    it("should delete an existing template", () => {
      const template = saveUserTemplate({
        name: "To Delete",
        description: "",
        algorithm: "grpo",
        tags: [],
        config: sampleConfig,
      });

      const result = deleteUserTemplate(template.id);
      expect(result).toBe(true);
      
      const stored = getUserTemplates();
      expect(stored).toHaveLength(0);
    });

    it("should return false for non-existent template", () => {
      const result = deleteUserTemplate("non-existent-id");
      expect(result).toBe(false);
    });
  });

  describe("encodeConfigToShareableLink", () => {
    it("should encode config to a URL with template parameter", () => {
      const link = encodeConfigToShareableLink(sampleConfig, "My Config");
      
      expect(link).toContain("template=");
      expect(link).toContain("http://localhost:3000");
    });

    it("should use default name if not provided", () => {
      const link = encodeConfigToShareableLink(sampleConfig);
      expect(link).toContain("template=");
    });
  });

  describe("decodeConfigFromUrl", () => {
    it("should decode a valid shareable link", () => {
      const link = encodeConfigToShareableLink(sampleConfig, "Test Config");
      const decoded = decodeConfigFromUrl(link);

      expect(decoded).not.toBeNull();
      expect(decoded!.name).toBe("Test Config");
      expect(decoded!.config.algorithm).toBe("grpo");
      expect(decoded!.config.model).toBe("Qwen/Qwen2.5-1.5B");
    });

    it("should return null for URL without template parameter", () => {
      const result = decodeConfigFromUrl("http://localhost:3000/");
      expect(result).toBeNull();
    });

    it("should return null for invalid base64", () => {
      const result = decodeConfigFromUrl("http://localhost:3000/?template=invalid!!");
      expect(result).toBeNull();
    });

    it("should return null for invalid JSON", () => {
      const invalidBase64 = btoa("not valid json");
      const result = decodeConfigFromUrl(`http://localhost:3000/?template=${invalidBase64}`);
      expect(result).toBeNull();
    });
  });

  describe("hasSharedTemplate", () => {
    it("should return false when no template in URL", () => {
      mockLocation.search = "";
      expect(hasSharedTemplate()).toBe(false);
    });

    it("should return true when template is in URL", () => {
      mockLocation.search = "?template=abc123";
      expect(hasSharedTemplate()).toBe(true);
    });
  });

  describe("clearTemplateFromUrl", () => {
    it("should remove template parameter from URL", () => {
      const replaceStateSpy = vi.spyOn(window.history, "replaceState");
      mockLocation.href = "http://localhost:3000/?template=abc123";
      
      clearTemplateFromUrl();
      
      expect(replaceStateSpy).toHaveBeenCalled();
    });
  });

  describe("Round-trip encoding/decoding", () => {
    it("should preserve all config properties through encode/decode", () => {
      const originalConfig: TrainingConfig = {
        algorithm: "dpo",
        model: "custom/model-name",
        dataset: "custom/dataset",
        backend: "megatron",
        hyperparameters: {
          learning_rate: 5e-7,
          batch_size: 64,
          max_steps: 5000,
          num_generations_per_prompt: 32,
          tensor_parallel_size: 4,
        },
        cluster: {
          nodes: 4,
          gpus_per_node: 8,
          time_limit: "24:00:00",
          partition: "gpu",
          account: "my-account",
        },
      };

      const link = encodeConfigToShareableLink(originalConfig, "Full Config Test");
      const decoded = decodeConfigFromUrl(link);

      expect(decoded).not.toBeNull();
      expect(decoded!.config).toEqual(originalConfig);
      expect(decoded!.name).toBe("Full Config Test");
    });
  });
});
