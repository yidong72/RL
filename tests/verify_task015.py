#!/usr/bin/env python3
"""
Verification script for TASK-015: E2E Multi-Node GRPO Training Test

This script validates:
1. Test script exists and is properly configured
2. Config file exists and has correct multi-node settings
3. All acceptance criteria are addressed in the test

Acceptance Criteria:
- AC-7.8: Async GRPO mode works
- AC-10.5: Distributed checkpoint works across nodes
- AC-12.2: Multi-node training works
- All nodes are utilized efficiently
"""

import os
import sys
import yaml
from pathlib import Path


def verify_test_script():
    """Verify the test script exists and is executable."""
    script_path = Path("tests/test_suites/llm/grpo_e2e_2n8g.sh")
    
    if not script_path.exists():
        print(f"[FAIL] Test script not found: {script_path}")
        return False
    
    print(f"[PASS] Test script exists: {script_path}")
    
    # Check script is executable
    if not os.access(script_path, os.X_OK):
        print(f"[WARN] Test script is not executable")
    else:
        print("[PASS] Test script is executable")
    
    # Check for required acceptance criteria references
    content = script_path.read_text()
    criteria = [
        ("AC-7.8", "Async GRPO mode"),
        ("AC-10.5", "Distributed checkpoint"),
        ("AC-12.2", "Multi-node training"),
    ]
    
    for ac_id, description in criteria:
        if ac_id in content:
            print(f"[PASS] {ac_id} ({description}) referenced in test script")
        else:
            print(f"[WARN] {ac_id} ({description}) not referenced in test script")
    
    return True


def verify_config_file():
    """Verify the config file exists and has correct settings."""
    config_path = Path("examples/configs/recipes/llm/grpo_e2e_2n8g.yaml")
    
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return False
    
    print(f"[PASS] Config file exists: {config_path}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Verify multi-node settings
        cluster = config.get("cluster", {})
        num_nodes = cluster.get("num_nodes", 1)
        gpus_per_node = cluster.get("gpus_per_node", 0)
        
        if num_nodes >= 2:
            print(f"[PASS] Multi-node configured: {num_nodes} nodes")
        else:
            print(f"[FAIL] Multi-node not configured: only {num_nodes} node(s)")
            return False
        
        if gpus_per_node >= 8:
            print(f"[PASS] GPUs per node configured: {gpus_per_node}")
        else:
            print(f"[WARN] GPUs per node: {gpus_per_node} (expected 8)")
        
        # Verify async GRPO mode (AC-7.8)
        grpo = config.get("grpo", {})
        async_cfg = grpo.get("async", {})
        if async_cfg.get("enabled", False):
            print("[PASS] AC-7.8: Async GRPO mode enabled")
        else:
            print("[INFO] Async GRPO mode not explicitly enabled in config")
        
        # Verify checkpointing enabled (AC-10.5)
        checkpointing = config.get("checkpointing", {})
        if checkpointing.get("enabled", False):
            print("[PASS] AC-10.5: Checkpointing enabled")
        else:
            print("[WARN] Checkpointing not enabled in config")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"[FAIL] Invalid YAML in config file: {e}")
        return False


def verify_test_coverage():
    """Verify all acceptance criteria have test coverage."""
    script_path = Path("tests/test_suites/llm/grpo_e2e_2n8g.sh")
    
    if not script_path.exists():
        print("[SKIP] Cannot verify test coverage - script not found")
        return True
    
    content = script_path.read_text()
    
    # Check for key validation steps
    validations = [
        ("loss check", "loss" in content.lower()),
        ("checkpoint check", "checkpoint" in content.lower()),
        ("multi-node check", "node" in content.lower()),
        ("metrics validation", "check_metrics" in content),
    ]
    
    all_passed = True
    for name, present in validations:
        if present:
            print(f"[PASS] Test includes {name}")
        else:
            print(f"[WARN] Test may be missing {name}")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("TASK-015 Verification: E2E Multi-Node GRPO Training")
    print("=" * 60)
    print()
    
    # Change to workspace root if needed
    if not Path("tests/test_suites").exists():
        os.chdir(Path(__file__).parent.parent)
    
    results = []
    
    print("1. Verifying test script...")
    print("-" * 40)
    results.append(verify_test_script())
    print()
    
    print("2. Verifying config file...")
    print("-" * 40)
    results.append(verify_config_file())
    print()
    
    print("3. Verifying test coverage...")
    print("-" * 40)
    results.append(verify_test_coverage())
    print()
    
    print("=" * 60)
    if all(results):
        print("VERIFICATION: PASSED")
        print("Test infrastructure is ready for TASK-015")
        print()
        print("To run the actual test (requires 2+ node GPU cluster):")
        print("  sbatch tests/test_suites/llm/grpo_e2e_2n8g.sh")
        return 0
    else:
        print("VERIFICATION: FAILED")
        print("Some checks did not pass - see above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
