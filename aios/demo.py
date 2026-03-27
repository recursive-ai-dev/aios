#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIOS Integration Demo

Demonstrates the happy path:
  1. Boot kernel
  2. Attach scheduler
  3. Attach memory subsystem
  4. Run RL operations
  5. Display state summary

This is the canonical integration test for the AIOS framework.
"""

import sys
import time
import random
from typing import Any, Dict, List, Optional

# Import from the aios package
from aios.aios_core import AgentKernel, AgentPriority, agent_method, AgentTrace, AgentRegistry
from aios.aios_memory import MemoryKernel
from aios.aios_scheduler import attach_to_kernel as scheduler_attach
from aios.aios_rl import GridWorld, ReplayBuffer, Transition, AgentRLHarness
from aios.aios_display import Color, Rect, Palette


def run_integration_demo(verbose: bool = True) -> Dict[str, Any]:
    """
    Run the full AIOS integration demo.
    
    Returns a dict with:
      - success: bool
      - kernel: AgentKernel
      - memory: MemoryKernel
      - scheduler_attached: bool
      - rl_transitions: int
      - display_rendered: bool
      - elapsed_seconds: float
    """
    start = time.perf_counter()
    results = {
        "success": False,
        "kernel": None,
        "memory": None,
        "scheduler_attached": False,
        "rl_transitions": 0,
        "display_rendered": False,
        "elapsed_seconds": 0.0,
        "errors": [],
    }
    
    if verbose:
        print("=" * 70)
        print("AIOS Integration Demo")
        print("=" * 70)
    
    # ── Step 1: Boot Kernel ────────────────────────────────────────────────────
    if verbose:
        print("\n[1/5] Booting AgentKernel...")
    
    try:
        kernel = AgentKernel()
        kernel.boot()
        results["kernel"] = kernel
        if verbose:
            print("      ✓ Kernel booted successfully")
    except Exception as e:
        results["errors"].append(f"Kernel boot failed: {e}")
        if verbose:
            print(f"      ✗ Kernel boot failed: {e}")
        return results
    
    # ── Step 2: Attach Scheduler ──────────────────────────────────────────────
    if verbose:
        print("\n[2/5] Attaching Scheduler...")
    
    try:
        scheduler_attached = scheduler_attach(kernel, workers_per_node=2)
        results["scheduler_attached"] = scheduler_attached
        if verbose and scheduler_attached:
            print("      ✓ Scheduler attached")
    except Exception as e:
        results["errors"].append(f"Scheduler attach failed: {e}")
        if verbose:
            print(f"      ✗ Scheduler attach failed: {e}")
    
    # ── Step 3: Attach Memory Subsystem ───────────────────────────────────────
    if verbose:
        print("\n[3/5] Initialising Memory Subsystem...")
    
    try:
        memory = MemoryKernel(d_emb=64)  # Smaller embedding for demo
        results["memory"] = memory
        if verbose:
            print(f"      ✓ Memory subsystem initialised (d_emb={memory.d_emb})")
    except Exception as e:
        results["errors"].append(f"Memory init failed: {e}")
        if verbose:
            print(f"      ✗ Memory init failed: {e}")
    
    # ── Step 4: Run RL Operations ─────────────────────────────────────────────
    if verbose:
        print("\n[4/5] Running RL Operations (GridWorld + ReplayBuffer)...")
    
    try:
        env = GridWorld(grid_size=8, max_steps=50)
        replay = ReplayBuffer(capacity=1000)
        
        # Run a few episodes and store transitions
        episodes = 3
        total_reward = 0.0
        total_transitions = 0
        
        for ep in range(episodes):
            obs = env.reset()
            ep_reward = 0.0
            done = False
            step = 0
            
            while not done and step < 50:
                action = random.randint(0, env.action_space_size - 1)
                next_obs, reward, done, _ = env.step(action)
                ep_reward += reward
                
                # Store transition
                transition = Transition(
                    state=obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                    action=int(action),
                    reward=float(reward),
                    next_state=next_obs.tolist() if hasattr(next_obs, 'tolist') else list(next_obs),
                    done=done,
                )
                replay.push(transition)
                total_transitions += 1
                
                obs = next_obs
                step += 1
            
            total_reward += ep_reward
        
        results["rl_transitions"] = total_transitions
        if verbose:
            print(f"      ✓ Completed {episodes} episodes, {total_transitions} transitions stored")
            print(f"        avg_reward={total_reward/episodes:.2f}, buffer_size={len(replay)}")
            
    except Exception as e:
        results["errors"].append(f"RL operations failed: {e}")
        if verbose:
            print(f"      ✗ RL operations failed: {e}")
    
    # ── Step 5: Display Summary ───────────────────────────────────────────────
    if verbose:
        print("\n[5/5] Rendering State Summary...")
    
    try:
        # Create a text-based summary using available display primitives
        summary_lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║           AIOS Integration State Summary                 ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Kernel:     {'BOOT OK' if results['kernel'] else 'FAILED':<46} ║",
            f"║  Scheduler:  {'ATTACHED' if results['scheduler_attached'] else 'NOT ATTACHED':<46} ║",
            f"║  Memory:     {'INIT OK' if results['memory'] else 'FAILED':<46} ║",
            f"║  RL:         {results['rl_transitions']} transitions stored{'':<23} ║",
            f"║  Elapsed:    {results['elapsed_seconds']:.3f}s{'':<34} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        
        rendered = "\n".join(summary_lines)
        results["display_rendered"] = True
        
        if verbose:
            print("\n" + rendered)
            
    except Exception as e:
        results["errors"].append(f"Display rendering failed: {e}")
        if verbose:
            print(f"      ✗ Display rendering failed: {e}")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start
    results["elapsed_seconds"] = elapsed
    results["success"] = (
        results["kernel"] is not None and
        results["memory"] is not None and
        results["scheduler_attached"] and
        results["rl_transitions"] > 0 and
        results["display_rendered"]
    )
    
    if verbose:
        print("\n" + "=" * 70)
        if results["success"]:
            print("INTEGRATION DEMO: SUCCESS")
        else:
            print("INTEGRATION DEMO: PARTIAL SUCCESS (some components failed)")
            for err in results["errors"]:
                print(f"  Error: {err}")
        print(f"Elapsed: {elapsed:.3f}s")
        print("=" * 70)
    
    return results


def main():
    """Entry point for running the integration demo."""
    verbose = "--quiet" not in sys.argv
    results = run_integration_demo(verbose=verbose)
    
    # Exit code based on success
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
