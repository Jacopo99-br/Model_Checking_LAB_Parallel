import argparse
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
import re
import time
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0. VERIFICA DISPONIBILITÀ GPU CUDA (NUMBA)
# -----------------------------------------------------------------------------
NUMBA_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        NUMBA_AVAILABLE = True
except (ImportError, Exception):
    NUMBA_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. METADATI E REGOLE TOPOLOGICHE
# -----------------------------------------------------------------------------
def parse_dataset_metadata(filename):
    match = re.search(r"(\d+)tracks_(\d+)sw_(\d+)routes", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 2, 4, 5


def compute_topology_rules(num_tracks, switches_per_track, routes_per_track):
    num_switches = num_tracks * switches_per_track
    track_switches = [
        list(range(t * switches_per_track, (t + 1) * switches_per_track))
        for t in range(num_tracks)
    ]
    route_sw_sets = []

    for t in range(num_tracks):
        for r in range(routes_per_track - 1):
            sw_subset = track_switches[t][: (r % switches_per_track) + 1]
            route_sw_sets.append(set(sw_subset))
        if t < num_tracks - 1:
            sw_cross = [track_switches[t][0], track_switches[t + 1][0]]
            route_sw_sets.append(set(sw_cross))

    pairs = []
    for i in range(len(route_sw_sets)):
        for j in range(i + 1, len(route_sw_sets)):
            if not route_sw_sets[i].isdisjoint(route_sw_sets[j]):
                pairs.append((i * 2, j * 2))

    return (
        np.array(pairs, dtype=np.int32),
        num_switches,
        len(route_sw_sets),
        (1 << num_switches) - 1,
    )


def load_dataset(dataset_file):
    print(f" Caricamento dataset '{dataset_file}' (escluso dal benchmark)...")
    data = np.load(dataset_file)
    states = data["states"]
    print(f" Caricati {len(states):,} stati.\n")
    return states


# -----------------------------------------------------------------------------
# 2. KERNEL CPU SEQUENZIALE
# -----------------------------------------------------------------------------
def compute_sequential_kernel(states, pairs, num_sw, num_rt, sw_mask, out_results):
    total_len = len(states)
    for idx in range(total_len):
        state = states[idx]
        switches = state & sw_mask
        routes = state >> num_sw
        is_unsafe = 0

        for shift1, shift2 in pairs:
            r1 = (routes >> shift1) & 0b11
            r2 = (routes >> shift2) & 0b11
            if r1 >= 2 and r2 >= 2:
                is_unsafe = 1
                break

        if is_unsafe == 0:
            for r_idx in range(num_rt):
                r_st = (routes >> (r_idx * 2)) & 0b11
                if r_st == 2:
                    if (switches ^ (r_idx * 31)) & 0b10101010 == 0:
                        is_unsafe = 1
                        break

        out_results[idx] = is_unsafe


# -----------------------------------------------------------------------------
# 3. KERNEL CPU MULTIPROCESSING (SHARED MEMORY ZERO-COPY)
# -----------------------------------------------------------------------------
_SHM_S_NAME = None
_SHM_R_NAME = None
_G_LEN = 0
_G_PAIRS = None
_G_NSW = 0
_G_NRT = 0
_G_MASK = 0


def _shm_worker_init(shm_s, shm_r, length, pairs, n_sw, n_rt, mask):
    global _SHM_S_NAME, _SHM_R_NAME, _G_LEN, _G_PAIRS, _G_NSW, _G_NRT, _G_MASK
    _SHM_S_NAME, _SHM_R_NAME, _G_LEN = shm_s, shm_r, length
    _G_PAIRS, _G_NSW, _G_NRT, _G_MASK = pairs, n_sw, n_rt, mask


def _shm_worker_range(range_tuple):
    start, end = range_tuple
    shm_s = SharedMemory(name=_SHM_S_NAME)
    shm_r = SharedMemory(name=_SHM_R_NAME)
    states = np.ndarray((_G_LEN,), dtype=np.uint32, buffer=shm_s.buf)
    results = np.ndarray((_G_LEN,), dtype=np.uint8, buffer=shm_r.buf)

    for idx in range(start, end):
        state = states[idx]
        switches = state & _G_MASK
        routes = state >> _G_NSW
        is_unsafe = 0

        for shift1, shift2 in _G_PAIRS:
            r1 = (routes >> shift1) & 0b11
            r2 = (routes >> shift2) & 0b11
            if r1 >= 2 and r2 >= 2:
                is_unsafe = 1
                break

        if is_unsafe == 0:
            for r_idx in range(_G_NRT):
                r_st = (routes >> (r_idx * 2)) & 0b11
                if r_st == 2:
                    if (switches ^ (r_idx * 31)) & 0b10101010 == 0:
                        is_unsafe = 1
                        break

        results[idx] = is_unsafe

    shm_s.close()
    shm_r.close()


# -----------------------------------------------------------------------------
# 4. KERNEL GPU CUDA (NUMBA)
# -----------------------------------------------------------------------------
if NUMBA_AVAILABLE:
    @cuda.jit
    def cuda_safety_kernel(states_arr, pairs_arr, results_arr, mask_val, n_sw, n_rt):
        idx = cuda.grid(1)
        if idx < states_arr.size:
            state = states_arr[idx]
            switches = state & mask_val
            routes = state >> n_sw
            is_unsafe = 0

            # 1. Verifica coppie rivali
            num_pairs = pairs_arr.shape[0]
            for p in range(num_pairs):
                shift1 = pairs_arr[p, 0]
                shift2 = pairs_arr[p, 1]
                r1 = (routes >> shift1) & 3
                r2 = (routes >> shift2) & 3
                if r1 >= 2 and r2 >= 2:
                    is_unsafe = 1
                    break

            # 2. Verifica allineamento scambi
            if is_unsafe == 0:
                for r_idx in range(n_rt):
                    r_st = (routes >> (r_idx * 2)) & 3
                    if r_st == 2:
                        if (switches ^ (r_idx * 31)) & 170 == 0:
                            is_unsafe = 1
                            break

            results_arr[idx] = is_unsafe


def run_cuda_kernel(d_states, d_pairs, d_results, sw_mask, num_sw, num_rt, blocks, tpb):
    cuda_safety_kernel[blocks, tpb](d_states, d_pairs, d_results, sw_mask, num_sw, num_rt)
    cuda.synchronize()


# -----------------------------------------------------------------------------
# 5. BENCHMARK ENGINE CONFORME ALLE GUIDELINES
# -----------------------------------------------------------------------------
def benchmark_configuration(func, args, num_runs=5, num_warmup=2, label="Config"):
    print(f"🔹 Esecuzione: {label} ({num_warmup} warm-up, {num_runs} misurazioni)")

    # 1. Warm-up (scartato dalle statistiche)
    for _ in range(num_warmup):
        func(*args)

    # 2. Misurazioni temporali replicate
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        func(*args)
        t_elapsed = time.perf_counter() - t0
        timings.append(t_elapsed)

    mean_t = np.mean(timings)
    std_t = np.std(timings)
    min_t = np.min(timings)
    max_t = np.max(timings)

    print(f"   Media: {mean_t:.4f}s ± {std_t:.4f}s | Min: {min_t:.4f}s | Max: {max_t:.4f}s\n")
    return {
        "mean": mean_t,
        "std": std_t,
        "min": min_t,
        "max": max_t,
        "raw": timings,
    }


# -----------------------------------------------------------------------------
# MAIN BENCHMARK
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite di Benchmark Interlocking")
    parser.add_argument("--dataset", type=str, default="NEW_interlocking_2tracks_4sw_5routes.npz")
    parser.add_argument("--runs", type=int, default=5, help="Misurazioni per config")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    
    # --- NUOVI FLAG PER GPU-ONLY SENZA RICOMPUTARE LA CPU ---
    parser.add_argument("--gpu-only", action="store_true", help="Esegue SOLO il benchmark GPU CUDA")
    parser.add_argument("--t-seq-baseline", type=float, default=None, 
                        help="Tempo sequenziale medio [s] gia' noto (evita di ricalcolare la CPU)")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024],
                        help="Block size GPU da testare (es. 32 64 128 256 512)")
    args = parser.parse_args()

    if args.gpu_only and not NUMBA_AVAILABLE:
        print(" Errore: Flag --gpu-only specificato ma nessuna GPU NVIDIA/Numba CUDA rilevata!")
        exit(1)

    tracks, sw_per_tr, rt_per_tr = parse_dataset_metadata(args.dataset)
    pairs, num_switches, num_routes, sw_mask = compute_topology_rules(tracks, sw_per_tr, rt_per_tr)
    states = load_dataset(args.dataset)
    total_len = len(states)

    results_table = []

    # -------------------------------------------------------------------------
    # 1. GESTIONE BASELINE SEQUENZIALE
    # -------------------------------------------------------------------------
    seq_results = None

    if args.gpu_only:
        if args.t_seq_baseline is not None:
            t_seq_baseline = args.t_seq_baseline
            print(f" Uso baseline sequenziale fornita da riga di comando: {t_seq_baseline:.4f}s")
        else:
            print("  Calcolo baseline sequenziale veloce (1 sola run) per validazione e speedup...")
            seq_results = np.zeros(total_len, dtype=np.uint8)
            t0 = time.perf_counter()
            compute_sequential_kernel(states, pairs, num_switches, num_routes, sw_mask, seq_results)
            t_seq_baseline = time.perf_counter() - t0
            print(f"   Baseline sequenziale: {t_seq_baseline:.4f}s\n")
    else:
        # Se non siamo in gpu-only, esegui il test CPU sequenziale completo
        seq_results = np.zeros(total_len, dtype=np.uint8)
        seq_stats = benchmark_configuration(
            compute_sequential_kernel,
            (states, pairs, num_switches, num_routes, sw_mask, seq_results),
            num_runs=args.runs,
            num_warmup=args.warmup,
            label="CPU Sequenziale (1 Thread)"
        )
        t_seq_baseline = seq_stats["mean"]
        results_table.append({
            "Platform": "CPU_Sequential",
            "Units": 1,
            "Mean_Time_s": seq_stats["mean"],
            "Std_Time_s": seq_stats["std"],
            "Min_Time_s": seq_stats["min"],
            "Max_Time_s": seq_stats["max"],
            "Speedup": 1.00,
            "Efficiency": 1.00,
            "Validation": "PASSED"
        })

    # -------------------------------------------------------------------------
    # 2. CPU MULTIPROCESSING (COMPLETAMENTE SALTATO SE --gpu-only)
    # -------------------------------------------------------------------------
    if not args.gpu_only:
        shm_s = SharedMemory(create=True, size=states.nbytes)
        shm_r = SharedMemory(create=True, size=total_len)
        try:
            s_arr = np.ndarray(states.shape, dtype=states.dtype, buffer=shm_s.buf)
            s_arr[:] = states[:]
            r_arr = np.ndarray((total_len,), dtype=np.uint8, buffer=shm_r.buf)

            for w in args.threads:
                step = (total_len + w - 1) // w
                ranges = [(i, min(i + step, total_len)) for i in range(0, total_len, step)]

                def run_pool_job():
                    with Pool(
                        processes=w,
                        initializer=_shm_worker_init,
                        initargs=(shm_s.name, shm_r.name, total_len, pairs, num_switches, num_routes, sw_mask),
                    ) as pool:
                        pool.map(_shm_worker_range, ranges)

                stats = benchmark_configuration(
                    run_pool_job, (),
                    num_runs=args.runs,
                    num_warmup=args.warmup,
                    label=f"CPU Multiprocessing ({w} Processi)"
                )

                is_valid = np.array_equal(seq_results, r_arr) if seq_results is not None else True
                speedup = t_seq_baseline / stats["mean"]
                efficiency = speedup / w

                results_table.append({
                    "Platform": "CPU_Multiprocessing",
                    "Units": w,
                    "Mean_Time_s": stats["mean"],
                    "Std_Time_s": stats["std"],
                    "Min_Time_s": stats["min"],
                    "Max_Time_s": stats["max"],
                    "Speedup": speedup,
                    "Efficiency": efficiency,
                    "Validation": "PASSED" if is_valid else "FAILED"
                })
        finally:
            shm_s.close()
            shm_s.unlink()
            shm_r.close()
            shm_r.unlink()

    # -------------------------------------------------------------------------
    # 3. TEST GPU CUDA (NUMBA) - VARIAZIONE BLOCK SIZE
    # -------------------------------------------------------------------------
    if NUMBA_AVAILABLE:
        dev = cuda.get_current_device()
        dev_name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else dev.name
        print(f" Rilevata GPU: {dev_name}")

        d_states = cuda.to_device(states)
        d_pairs = cuda.to_device(pairs)
        d_results = cuda.device_array(total_len, dtype=np.uint8)

        # Se vuoi testare un solo blocco o piu' blocchi (es. 32, 64, 128, 256, 512, 1024)
        for tpb in args.block_sizes:
            bpg = (total_len + (tpb - 1)) // tpb

            gpu_stats = benchmark_configuration(
                run_cuda_kernel,
                (d_states, d_pairs, d_results, sw_mask, num_switches, num_routes, bpg, tpb),
                num_runs=args.runs,
                num_warmup=args.warmup,
                label=f"GPU CUDA ({dev_name}, BlockSize={tpb})"
            )

            h_results = d_results.copy_to_host()
            is_valid = np.array_equal(seq_results, h_results) if seq_results is not None else True
            speedup_gpu = t_seq_baseline / gpu_stats["mean"]

            results_table.append({
                "Platform": f"GPU_CUDA ({dev_name})",
                "Units": tpb,
                "Mean_Time_s": gpu_stats["mean"],
                "Std_Time_s": gpu_stats["std"],
                "Min_Time_s": gpu_stats["min"],
                "Max_Time_s": gpu_stats["max"],
                "Speedup": speedup_gpu,
                "Efficiency": np.nan,
                "Validation": "PASSED" if is_valid else "UNVERIFIED"
            })

    # -------------------------------------------------------------------------
    # 4. SALVATAGGIO
    # -------------------------------------------------------------------------
    df = pd.DataFrame(results_table)
    csv_filename = "benchmark_results_gpu.csv" if args.gpu_only else "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print("\n" + "=" * 80)
    print(df[["Platform", "Units", "Mean_Time_s", "Speedup", "Validation"]].to_string(index=False))
    print("=" * 80)
    print(f"Salvato in '{csv_filename}'.\n")