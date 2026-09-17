# Railway Interlocking Safety Verification & Model Checking Suite
**High-Performance Computing Laboratory** — Master's Degree in Computer Engineering  
Department of Information Engineering (DINFO), University of Florence (UNIFI)  
Author: **Jacopo Bruscaglioni** (`jacopo.bruscaglioni@stud.unifi.it`)

---

## 1. Overview
This module implements high-throughput formal verification and state-space exploration for railway interlocking systems. The architecture is divided into two distinct components:
1. **LTS State-Space Generation (`benchmark_generator_railstation.py`):** Uses bitwise bitmask state encoding and multi-core CPU expansion to generate Labeled Transition Systems (LTS) stored as compressed NumPy archives (`.npz`)[cite: 2].
2. **Safety Invariant Verification (`model_checker_2.py`):** Checks mutual exclusion invariants across switch/route configurations comparing sequential CPU execution, multi-core shared-memory parallelism (`multiprocessing.shared_memory`), and SIMT GPU acceleration via Numba CUDA[cite: 3].

---

## 2. Environment Setup & Dependencies

Install required Python dependencies into a dedicated virtual environment:

```bash
python -m venv RailStationVenv
source RailStationVenv/bin/activate  # On Windows: .\RailStationVenv\Scripts\Activate.ps1
pip install numpy pandas numba
```

> **Note on CUDA:** GPU execution requires an NVIDIA GPU with appropriate display drivers and CUDA Toolkit libraries accessible by Numba. If CUDA is unavailable, `model_checker_2.py` automatically falls back to CPU execution modes.

---

## 3. Component 1: State-Space Generation (`benchmark_generator_railstation.py`)

This script expands the interlocking state space using breadth-first search (BFS) over bitwise transitions (switches, routes IDLE $\rightarrow$ REQUESTED $\rightarrow$ RESERVED $\rightarrow$ OCCUPIED). The resulting topology is written to a compressed CSR matrix format inside an `.npz` archive.

### Default Run
Runs the standard configuration ($2$ tracks, $4$ switches/track, $5$ routes/track) using all available CPU cores:
```bash
python benchmark_generator_railstation.py
```
*Output File:* `NEW_interlocking_2tracks_4sw_5routes.npz`.

### Custom Configuration
```bash
python benchmark_generator_railstation.py --tracks 2 --switches 4 --routes 5 --cores 8
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tracks` | `int` | `2` | Number of parallel railway tracks in the yard. |
| `--switches` | `int` | `4` | Number of track switches per line. |
| `--routes` | `int` | `5` | Number of route definitions per line (including crossovers). |
| `--cores` | `int` | `cpu_count()` | Number of parallel worker processes used for frontier expansion. |

---

## 4. Component 2: Safety Model Checking (`model_checker_2.py`)

The verification engine loads generated `.npz` states and verifies conflict-free route allocation and switch alignment rules.

### Standard Mode (Sequential CPU + Multiprocessing + CUDA)
Executes sequential baseline verification, multiprocessing scaling ($1, 2, 4, 8$ processes) using zero-copy POSIX shared memory, and CUDA block-size tuning ($32 \le \text{blockDim.x} \le 1024$):
```bash
python model_checker_2.py --dataset NEW_interlocking_2tracks_4sw_5routes.npz
```
*Output:* Writes benchmarks to `benchmark_results.csv`.

---

### GPU-Only Mode (`--gpu-only`)
Skips the long CPU multiprocessing sweep and benchmarks CUDA kernels exclusively:
```bash
# Standalone run (computes 1 fast sequential pass for speedup baseline)
python model_checker_2.py --dataset NEW_interlocking_2tracks_4sw_5routes.npz --gpu-only

# Providing a known sequential baseline time to bypass all CPU computation
python model_checker_2.py --dataset NEW_interlocking_2tracks_4sw_5routes.npz --gpu-only --t-seq-baseline 1.8420
```
*Output:* Writes benchmarks to `benchmark_results_gpu.csv`.

---

### Custom Worker & Block Sweeps
```bash
python model_checker_2.py \
  --dataset NEW_interlocking_2tracks_4sw_5routes.npz \
  --runs 10 \
  --warmup 3 \
  --threads 1 2 4 8 16 \
  --block-sizes 64 128 256 512
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `NEW_interlocking_2tracks_4sw_5routes.npz` | Path to the compressed input state space archive. |
| `--runs` | `int` | `5` | Number of timed measurement replications per configuration. |
| `--warmup` | `int` | `2` | Number of untimed warmup iterations discarded prior to profiling. |
| `--threads` | `list[int]` | `1 2 4 8` | CPU process worker counts evaluated via shared memory. |
| `--block-sizes` | `list[int]` | `32 64 128 256 512 1024` | CUDA thread block dimensions (`blockDim.x`) evaluated on GPU. |
| `--gpu-only` | `flag` | `False` | Bypasses CPU multi-process benchmarks and evaluates only CUDA kernels. |
| `--t-seq-baseline` | `float` | `None` | Pre-measured baseline execution time in seconds (avoids CPU baseline calculation). |

---

## 5. End-to-End Workflow Example

Run this sequence to execute an end-to-end experiment from generation to model checking:

```bash
# Step 1: Generate state space LTS
python benchmark_generator_railstation.py --tracks 2 --switches 4 --routes 5

# Step 2: Run verification and performance suite
python model_checker_2.py --dataset NEW_interlocking_2tracks_4sw_5routes.npz --runs 5 --warmup 2

# Step 3 (Optional): Plot scaling and tuning curves
python plot_gen.py
```
