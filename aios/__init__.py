#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIOS — Agentic Intelligence Operating System

A Python simulation framework for agentic systems primitives.

Core Components:
    - Kernel      : Agent protocol, tracing, registry, context propagation
    - Memory      : Working/episodic/semantic memory, LSH, Hopfield retrieval
    - Scheduler   : Work-stealing, aging, NUMA-aware placement
    - RL          : PPO, DQN, REINFORCE, GAE with self-tests
    - Display     : Compositing, widget system, terminal drivers
    - Neural      : Neural kernel substrate

Usage:
    from aios import Kernel, MemorySubsystem, Scheduler, RLHarness, DisplayManager

    kernel = Kernel()
    memory = MemorySubsystem(d_emb=128)
    scheduler = Scheduler()
    rl = RLHarness()
    display = DisplayManager()

    kernel.attach(memory)
    kernel.attach(scheduler)
    kernel.attach(rl)
    
    # Run integration demo
    from aios.demo import run_integration_demo
    run_integration_demo()
"""

__version__ = "0.1.0"
__codename__ = "GENESIS"

# ── Core Kernel ────────────────────────────────────────────────────────────────
from aios.aios_core import (
    AgentPriority,
    AgentTrace,
    AgentContext,
    AgentRegistry,
    agent_method,
    AgentKernel,
    AgentReasoner,
)

# ── Memory Subsystem ──────────────────────────────────────────────────────────
from aios.aios_memory import (
    MemoryKernel,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    LSHIndex,
    HopfieldMemory,
    PERConsolidator,
)

# ── Scheduler ─────────────────────────────────────────────────────────────────
from aios.aios_scheduler import (
    AIOSScheduler,
    TaskSpec,
    TaskState,
    TaskResult,
    TaskFuture,
    WorkStealingDeque,
    WorkerStats,
    PlacementScorer,
    AdaptiveRebalancer,
)

# ── Reinforcement Learning ────────────────────────────────────────────────────
from aios.aios_rl import (
    AgentRLHarness,
    PPO,
    DQN,
    REINFORCE,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    Transition,
    Episode,
)

# ── Display Manager ───────────────────────────────────────────────────────────
from aios.aios_display import (
    Color,
    Rect,
    Palette,
    DisplayMode,
    PixelFormat,
    EventType,
    KeyCode,
)

# ── Neural Kernel ─────────────────────────────────────────────────────────────
from aios.aios_neural import (
    Tensor,
    Parameter,
    Module,
    Linear,
    Sequential,
    Adam,
    ReLU,
    LayerNorm,
)

# ── Integration Demo ──────────────────────────────────────────────────────────
from aios.demo import run_integration_demo

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # Version
    "__version__",
    "__codename__",
    # Kernel
    "AgentPriority",
    "AgentTrace",
    "AgentContext",
    "AgentRegistry",
    "agent_method",
    "AgentKernel",
    "AgentReasoner",
    # Memory
    "MemoryKernel",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "LSHIndex",
    "HopfieldMemory",
    "PERConsolidator",
    # Scheduler
    "AIOSScheduler",
    "TaskSpec",
    "TaskState",
    "TaskResult",
    "TaskFuture",
    "WorkStealingDeque",
    "WorkerStats",
    "PlacementScorer",
    "AdaptiveRebalancer",
    # RL
    "AgentRLHarness",
    "PPO",
    "DQN",
    "REINFORCE",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "Transition",
    "Episode",
    # Display
    "Color",
    "Rect",
    "Palette",
    "DisplayMode",
    "PixelFormat",
    "EventType",
    "KeyCode",
    # Neural
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "Sequential",
    "Adam",
    "ReLU",
    "LayerNorm",
    # Demo
    "run_integration_demo",
]
