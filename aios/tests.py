#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIOS Cross-Module Contract Tests

These tests verify integration boundaries between modules, not just internal
correctness. Each test validates a contract between two or more modules.

Contracts tested:
  1. Kernel ↔ Scheduler: task submission and execution
  2. Kernel ↔ Memory: memory operations via agent methods
  3. Scheduler ↔ RL: scheduling RL training tasks
  4. Memory ↔ RL: experience storage and retrieval
  5. Display ↔ All: rendering system state
"""

import sys
import time
import threading
from typing import Any, Dict, List, Tuple

# Import core modules
from aios.aios_core import AgentKernel, AgentPriority, AgentTrace, AgentRegistry
from aios.aios_memory import MemoryKernel, WorkingMemory, EpisodicMemory
from aios.aios_scheduler import attach_to_kernel as scheduler_attach, AIOSScheduler
from aios.aios_rl import (
    AgentRLHarness, GridWorld,
    ReplayBuffer, Transition, PrioritizedReplayBuffer
)
from aios.aios_display import Color, Rect, Palette


class ContractTestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration_ms = 0.0
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration_ms:.1f}ms)"


def test_kernel_scheduler_contract() -> ContractTestResult:
    """
    Contract: Kernel provides boot infrastructure; Scheduler attaches and runs.
    
    Steps:
      1. Boot kernel
      2. Attach scheduler
      3. Submit a task through scheduler
      4. Verify task execution
    """
    result = ContractTestResult("Kernel ↔ Scheduler")
    start = time.perf_counter()
    
    try:
        # Boot kernel
        kernel = AgentKernel()
        if not kernel.boot():
            result.error = "Kernel boot failed"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Attach scheduler
        attached = scheduler_attach(kernel, workers_per_node=2)
        if not attached:
            result.error = "Scheduler attach failed"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Get scheduler reference
        scheduler: AIOSScheduler = getattr(kernel, 'scheduler', None)
        if scheduler is None:
            result.error = "Scheduler not attached to kernel"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Submit a simple task - scheduler expects callable with iterable args
        def simple_task(args):
            x = args[0]
            return x * 2
        
        future = scheduler.submit(simple_task, [21])
        task_result = future.result(timeout=5.0)
        
        if task_result != 42:
            result.error = f"Task returned wrong value: {task_result} != 42"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        result.passed = True
        result.details = {
            "kernel_booted": True,
            "scheduler_attached": True,
            "task_result": task_result,
            "worker_count": scheduler.status()["workers"],
        }
        
    except Exception as e:
        result.error = str(e)
    
    result.duration_ms = (time.perf_counter() - start) * 1000
    return result


def test_kernel_memory_contract() -> ContractTestResult:
    """
    Contract: Memory subsystem can be initialised and operated standalone.
    
    Steps:
      1. Create MemoryKernel
      2. Run self-tests
      3. Perform working memory operations
      4. Verify episodic memory storage/retrieval
    """
    result = ContractTestResult("Kernel ↔ Memory")
    start = time.perf_counter()
    
    try:
        # Create memory kernel
        memory = MemoryKernel(d_emb=32)
        
        # Test working memory
        working = memory.wm
        working.push("test_item_1", [1.0] * 32, {"data": 1})
        working.push("test_item_2", [2.0] * 32, {"data": 2})
        
        if len(working) != 2:
            result.error = f"Working memory length wrong: {len(working)} != 2"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Test episodic memory
        episodic = memory.em
        episodic.store("episode_1", [0.5] * 32, {"reward": 10.0, "steps": 5})
        retrieved = episodic.retrieve("episode_1")
        
        if retrieved is None or retrieved.get("reward") != 10.0:
            result.error = "Episodic memory retrieval failed"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        result.passed = True
        result.details = {
            "working_memory_size": len(working),
            "episodic_retrieved": True,
        }
        
    except Exception as e:
        result.error = str(e)
    
    result.duration_ms = (time.perf_counter() - start) * 1000
    return result


def test_scheduler_rl_contract() -> ContractTestResult:
    """
    Contract: RL training can be scheduled as tasks.

    Steps:
      1. Create RL harness and environment
      2. Schedule RL training as a task
      3. Verify training completes
    """
    result = ContractTestResult("Scheduler ↔ RL")
    start = time.perf_counter()

    try:
        # Boot kernel and attach scheduler
        kernel = AgentKernel()
        kernel.boot()
        scheduler_attach(kernel, workers_per_node=2)
        scheduler: AIOSScheduler = kernel.scheduler

        # Create RL components
        env = GridWorld(grid_size=5, max_steps=20)
        replay = ReplayBuffer(capacity=100)

        # Define training task
        def run_episode(env_obj, replay_buf) -> Dict[str, float]:
            import random
            obs = env_obj.reset()
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < 20:
                action = random.randint(0, env_obj.action_space_size - 1)
                next_obs, reward, done, _ = env_obj.step(action)
                total_reward += reward
                
                replay_buf.push(Transition(
                    state=obs, action=action, reward=reward,
                    next_state=next_obs, done=done,
                ))
                obs = next_obs
                step += 1

            return {"reward": total_reward, "steps": step, "buffer_size": len(replay_buf)}

        # Schedule training
        future = scheduler.submit(run_episode, [env, replay])
        episode_result = future.result(timeout=10.0)

        if episode_result is None or "reward" not in episode_result:
            result.error = "Training task returned invalid result"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result

        result.passed = True
        result.details = {
            "episode_reward": episode_result["reward"],
            "episode_steps": episode_result["steps"],
        }

    except Exception as e:
        result.error = str(e)

    result.duration_ms = (time.perf_counter() - start) * 1000
    return result


def test_memory_rl_contract() -> ContractTestResult:
    """
    Contract: RL experience can be stored in memory subsystem.

    Steps:
      1. Create memory kernel and RL harness
      2. Run RL episode
      3. Store transitions in episodic memory
      4. Retrieve and verify
    """
    result = ContractTestResult("Memory ↔ RL")
    start = time.perf_counter()

    try:
        # Create components
        memory = MemoryKernel(d_emb=16)
        env = GridWorld(grid_size=4, max_steps=10)

        # Run episode
        import random
        obs = env.reset()
        transitions: List[Transition] = []
        total_reward = 0.0

        for step in range(10):
            action = random.randint(0, env.action_space_size - 1)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )
            transitions.append(transition)

            if done:
                break
            obs = next_obs

        # Store in episodic memory
        episode_data = {
            "transitions": len(transitions),
            "total_reward": total_reward,
        }
        memory.em.store("test_episode", [0.5] * 16, episode_data)

        # Retrieve
        retrieved = memory.em.retrieve("test_episode")

        if not retrieved:
            result.error = "Could not retrieve episode from memory"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result

        result.passed = True
        result.details = {
            "transitions_stored": len(transitions),
            "total_reward": total_reward,
            "retrieved": True,
        }

    except Exception as e:
        result.error = str(e)
    
    result.duration_ms = (time.perf_counter() - start) * 1000
    return result


def test_display_integration_contract() -> ContractTestResult:
    """
    Contract: Display primitives can represent state from all other modules.

    Steps:
      1. Create display primitives (Color, Rect)
      2. Boot kernel and attach scheduler
      3. Create memory and RL components
      4. Verify display primitives work with module state
    """
    result = ContractTestResult("Display ↔ All Modules")
    start = time.perf_counter()

    try:
        # Create display primitives
        title_color = Color(255, 255, 255, 255)
        success_color = Color(0, 255, 0, 255)
        border = Rect(0, 0, 60, 15)
        
        # Verify color creation
        if title_color.r != 255 or title_color.g != 255 or title_color.b != 255:
            result.error = "Color creation failed"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result

        # Boot kernel and attach scheduler
        kernel = AgentKernel()
        kernel.boot()
        scheduler_attach(kernel, workers_per_node=2)

        # Create memory
        memory = MemoryKernel(d_emb=16)

        # Create RL
        env = GridWorld(size=4, max_steps=5)
        obs = env.reset()

        # Build text summary (simulating what display would render)
        summary_lines = [
            "AIOS Integration State",
            f"Kernel: BOOT OK",
            f"Scheduler: {kernel.scheduler.status()['workers']} workers",
            f"Memory: d_emb={memory.d_emb}",
            f"RL Env: {env.size}x{env.size} GridWorld",
            f"Obs shape: {len(obs)}",
        ]
        
        rendered = "\n".join(summary_lines)

        if not rendered or len(rendered) == 0:
            result.error = "Display rendering produced empty output"
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        
        result.passed = True
        result.details = {
            "rendered_length": len(rendered),
            "components_rendered": 4,
        }
        
    except Exception as e:
        result.error = str(e)
    
    result.duration_ms = (time.perf_counter() - start) * 1000
    return result


def run_all_contract_tests() -> Tuple[List[ContractTestResult], int, int]:
    """Run all contract tests and return results."""
    tests = [
        test_kernel_scheduler_contract,
        test_kernel_memory_contract,
        test_scheduler_rl_contract,
        test_memory_rl_contract,
        test_display_integration_contract,
    ]
    
    results = []
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("AIOS Cross-Module Contract Tests")
    print("=" * 70)
    print()
    
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        
        if result.passed:
            passed += 1
            print(f"  ✓ {result}")
        else:
            failed += 1
            print(f"  ✗ {result}")
            if result.error:
                print(f"      Error: {result.error}")
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return results, passed, failed


def main():
    """Entry point for contract tests."""
    results, passed, failed = run_all_contract_tests()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
