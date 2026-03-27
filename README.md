# AIOS — Agentic Intelligence Operating System

A Python simulation framework for agentic systems primitives.

## Overview

AIOS provides a unified runtime for building agent-based systems with:

- **Kernel**: Agent protocol, tracing, registry, context propagation
- **Memory**: Working/episodic/semantic memory, LSH indexing, Hopfield retrieval
- **Scheduler**: Work-stealing deque, aging, NUMA-aware placement, rebalancing
- **RL**: PPO, DQN, REINFORCE with replay buffers and self-tests
- **Neural**: Tensor operations, modules, optimizers (Adam, SGD)
- **Display**: Color, primitives for rendering system state

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd aios

# Add to Python path
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Quick Start

```python
import aios

# Check version
print(f"AIOS v{aios.__version__} '{aios.__codename__}'")

# Run the integration demo
from aios import run_integration_demo
results = run_integration_demo()

# Or import individual components
from aios import (
    AgentKernel,
    AIOSScheduler,
    MemoryKernel,
    GridWorld,
    ReplayBuffer,
    Color,
    Tensor,
    Linear,
)
```

## Run the Demo

```bash
python3 -m aios.demo
```

Expected output:
```
======================================================================
AIOS Integration Demo
======================================================================

[1/5] Booting AgentKernel...
      ✓ Kernel booted successfully

[2/5] Attaching Scheduler...
      ✓ Scheduler attached

[3/5] Initialising Memory Subsystem...
      ✓ Memory subsystem initialised

[4/5] Running RL Operations...
      ✓ Completed 3 episodes, 150 transitions stored

[5/5] Rendering State Summary...
      ✓ Display rendered

======================================================================
INTEGRATION DEMO: SUCCESS
======================================================================
```

## Run Tests

```bash
python3 -m aios.tests
```

## Package Structure

```
aios/
├── aios/                    # Python package
│   ├── __init__.py          # Public API exports
│   ├── aios_core.py         # Agent kernel, tracing, registry
│   ├── aios_memory.py       # Memory subsystem
│   ├── aios_scheduler.py    # Task scheduler
│   ├── aios_rl.py           # Reinforcement learning
│   ├── aios_neural.py       # Neural network primitives
│   ├── aios_display.py      # Display primitives
│   ├── demo.py              # Integration demo
│   └── tests.py             # Cross-module contract tests
├── research/                # Experimental/draft modules
│   ├── aios-networking.py   # L2/L3 network stack (needs cleanup)
│   ├── aios-binary-compiler.py  # Neural compiler research
│   └── aios-convoboot.py    # Conversational boot research
└── README.md
```

## Public API

The `aios` package exports 49 items:

**Kernel**: `AgentPriority`, `AgentTrace`, `AgentContext`, `AgentRegistry`, `agent_method`, `AgentKernel`, `AgentReasoner`

**Memory**: `MemoryKernel`, `WorkingMemory`, `EpisodicMemory`, `SemanticMemory`, `LSHIndex`, `HopfieldMemory`, `PERConsolidator`

**Scheduler**: `AIOSScheduler`, `TaskSpec`, `TaskState`, `TaskResult`, `TaskFuture`, `WorkStealingDeque`, `WorkerStats`, `PlacementScorer`, `AdaptiveRebalancer`

**RL**: `AgentRLHarness`, `PPO`, `DQN`, `REINFORCE`, `ReplayBuffer`, `PrioritizedReplayBuffer`, `Transition`, `Episode`

**Display**: `Color`, `Rect`, `Palette`, `DisplayMode`, `PixelFormat`, `EventType`, `KeyCode`

**Neural**: `Tensor`, `Parameter`, `Module`, `Linear`, `Sequential`, `Adam`, `ReLU`, `LayerNorm`

**Demo**: `run_integration_demo`

## Design Philosophy

This is **not** a conventional operating system. It is a:

> **Python agent runtime + neural substrate + systems simulation toolkit**

The kernel/memory bus/interrupt/paging material provides a systems-like abstraction for agent reasoning, but runs entirely in userspace Python.

## License

MIT
