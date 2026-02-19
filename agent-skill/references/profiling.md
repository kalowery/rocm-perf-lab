# Profiling Internals (v2)

## Runtime Modeling

Runtime derived from:

    SUM(end - start) FROM kernels

From rocpd dispatch table.

---

## Multi-Kernel DAG

1. Extract all kernel dispatches
2. Build execution timeline
3. Aggregate by symbol
4. Compute critical path
5. Compute dominance fraction

---

## Roofline Computation

FLOPs derived from hardware counters.
Bytes derived from TCC_EA0_RDREQ/WRREQ with 32B/64B accounting.

Arithmetic intensity:

    AI = FLOPs / Bytes

---

## ATT Integration

Aggregates:
- ISA-level instruction mix
- Stall cycles
- Idle cycles
- IPC
- Memory latency

Graceful failure supported.

---

## Headroom

Headroom fraction estimates remaining microarchitectural inefficiency.

High headroom → latency hiding potential.
Low headroom → algorithmic bound likely.
