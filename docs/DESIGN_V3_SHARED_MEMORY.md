# RegisterGPT v3 — Shared Memory Bank Architecture

## Overview

RegisterGPT v3 models each token position as a core in a multi-core processor. Each core maintains a local register file (the hidden state). Between cores sits a shared memory bank that mediates all cross-position communication. Processing alternates between two program types: **layer-programs** for local computation and **association-programs** for memory-mediated communication.

```
for each step:
    parallel across positions:
        run layer-program(registers)                       # local compute
    parallel across positions:
        run association-program(registers, shared_memory)  # communicate
```

## Program Types

### Layer-Programs (ALU)

Layer-programs operate exclusively on local registers. No cross-position communication. Each position runs its own program independently and in parallel.

```
R[2] = R[0] * R[1]
R[3] = SIGMOID(R[2])
R[0] = R[3] + R[2]
R[1] = RELU(R[0])
```

Each instruction encodes 4-6 values: opcode, source registers, destination register. A 10-instruction program is 40-60 parameters. The program is human-readable — every intermediate result can be inspected.

### Association-Programs (Load/Store Unit)

Association-programs read from and write to shared memory. This is how positions communicate.

```
addr  = HASH(R[0], R[1])            # compute write address from registers
WRITE(shared_memory[addr], R[2])    # store R[2] at that address
query = HASH(R[3], R[0])            # compute read query
R[1]  = READ(shared_memory, query)  # content-addressed read
```

Memory addresses are not fixed integers — they are computed from register contents. A position computes an address and writes; another position computes a query and reads from the best-matching address. This is **content-addressable memory**, similar to what Neural Turing Machines attempted with neural controllers, but implemented here with program controllers.

The cost is O(T) per step — each position performs one write and one read — compared to O(T^2) for attention's pairwise comparison of every position with every other.

## Comparison with Attention

Attention computes a fully connected communication topology at every layer, recomputed from scratch each time. The shared memory model separates writing from reading. Position 3 writes to memory address "subject_noun." Position 12 reads from "subject_noun." The positions never interact directly; the memory mediates.

In a transformer, cross-position information is implicit in attention patterns. Determining that head 4 in layer 7 performs subject-verb agreement requires mechanistic interpretability techniques. In the shared memory model, the memory bank is explicit state. After each association step, the contents at each address are inspectable. If registers operate in vocabulary space, a memory address literally contains the activation pattern of candidate words. The association programs — defining what gets written and what gets read — are the communication protocols between positions, readable directly in the program code.

## Memory Structure Options

### 1. Flat Key-Value Store

Memory is a matrix of K slots, each with a key vector and a value vector.

- **Write:** Position computes a key from registers, stores a value.
- **Read:** Position computes a query, retrieves the value at the closest matching key.
- **Cost:** O(T * K) instead of O(T^2). When K << T, substantially cheaper.
- **Tradeoff:** Functionally similar to attention over a fixed memory bank rather than over all positions.

### 2. Positional Slots with Causal Masking

Memory has T slots, one per position. Position t can only read from slots 0 through t-1.

- **Write:** Each position writes to its own slot (trivial).
- **Read:** A program computes which slots to attend to.
- **Tradeoff:** Most direct analog of causal attention, but mediated through explicit memory rather than implicit QKV products.

### 3. Stack/Queue (Most LGP-Native)

Memory is a stack. Association-programs PUSH and POP.

- **Strength:** Naturally handles nested structure. Opening parentheses push, closing parentheses pop. A stack memory can discover hierarchical structure in language — pushing when encountering a subordinate clause, popping when it closes.
- **Tradeoff:** Limited to strictly hierarchical access patterns. Long-range flat dependencies (e.g., coreference across paragraphs) may not fit a stack discipline.

### 4. Multi-Head Memory (Composite)

Different heads of the association-program access different memory structures simultaneously.

| Head | Structure        | Role                    |
|------|------------------|-------------------------|
| 0    | Key-value store  | Long-range associations |
| 1    | Positional buffer | Local context          |
| 2    | Stack            | Hierarchical structure  |

Each head is a few instructions. Total cost per step remains small. This composite approach allows the model to use the most appropriate memory structure for each type of linguistic relationship.

## Prior Work: External Memory in TPG

Robert Smith's thesis under Heywood added external memory to Tangled Program Graphs (TPG) for ViZDoom navigation. The agent learned to write landmarks to memory and read them back when navigating. The key finding: memory access patterns were interpretable — what the agent remembered and why was directly observable.

For language modeling, the "landmarks" are linguistic features: the subject of the sentence, the current topic, whether the text is inside a quotation, whether a conditional clause has been opened. In a transformer, tracking these features is distributed across attention heads and hidden states. In the shared memory model, tracking is performed by explicit, readable memory operations.

## Modularity and Search

Each layer-program, each association-program, and the memory structure can be evolved independently. This yields three modular search spaces, each vastly smaller than the joint space a transformer operates in.

This mirrors TPG's core insight: modularity enables search. A transformer is a monolithic differentiable function with billions of parameters in a single search space. The shared memory model is a collection of small programs (searchable independently) connected by shared memory (structure searchable independently). The compositional structure of the model mirrors the compositional structure of language — programs compose like morphemes, memory structures compose like syntactic structures.

## Relationship to Current Implementation

The current v3 codebase (`model.py`) already implements the two-phase structure:

- **`FourierRegisterOp`** — the layer-program analog. Within-position register transforms via Fourier basis projections.
- **`AssociativeMemoryStep`** — the association-program analog. Cross-position mixing via a running outer-product memory (Hopfield-style).

The outer-product memory in the current implementation is a specific instance of the flat key-value store option, where keys and values are projected from vocabulary space through Fourier coefficients, and the memory matrix M accumulates key-value outer products with exponential decay.

The design options above generalize this to support alternative memory structures (positional slots, stacks, multi-head composites) and to make the program structure more explicit and inspectable.

## Relationship to v4 Implementation

RegisterGPT v4 (`model_v4.py`) applies parameter-golf optimizations to the v3 architecture, reducing ~329K params to ~101K while preserving the two-phase structure:

- **Shared Q/K projections**: All steps share a single query and key definition ("what is similar"), while keeping V/O unique per step ("what to retrieve" and "how to output"). This reflects the observation that similarity metrics are global while retrieval strategies are local.
- **Multi-head decay (H=4)**: Each head maintains a different temporal decay rate, allowing simultaneous capture of fast (bigram) and slow (clause-level) dependencies within the associative memory.
- **Factored channel_mix**: The dense 128x128 matrix in FourierRegisterOp is replaced with diagonal + rank-8, exploiting the fact that the Fourier read already constrains input rank.
- **Step reuse with per-invocation overrides**: 5 unique step definitions are each invoked twice (10 total depth), with per-invocation scalar overrides for decay, mem_scale, op_scale, and out_scale providing cheap depth specialization.

These optimizations preserve the shared memory bank model: the associative memory still mediates all cross-position communication, and the register ops still perform local computation. The v4 changes are purely about parameter efficiency, not architectural philosophy.
