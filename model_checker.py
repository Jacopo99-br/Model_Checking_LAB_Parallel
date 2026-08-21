import argparse
from multiprocessing import Pool, cpu_count
import re
import time
import numpy as np


# -----------------------------------------------------------------------------
# UTILITY: ESTRAZIONE PARAMETRI E MATRICE RIVALI
# -----------------------------------------------------------------------------
def parse_dataset_metadata(filename):
    """Estrae i parametri topologici dal nome del file .npz"""
    match = re.search(r"(\d+)tracks_(\d+)sw_(\d+)routes", filename)
    if match:
        tracks = int(match.group(1))
        sw_per_track = int(match.group(2))
        rt_per_track = int(match.group(3))
        return tracks, sw_per_track, rt_per_track
    # Fallback predefinito per il vecchio file balanced
    return 2, 4, 5


def compute_topology_rules(num_tracks, switches_per_track, routes_per_track):
    """Ricostruisce le coppie rivali e le maschere bitwise per il checker."""
    num_switches = num_tracks * switches_per_track
    num_routes = num_tracks * routes_per_track

    track_switches = [
        list(
            range(
                t * switches_per_track, (t + 1) * switches_per_track
            )
        )
        for t in range(num_tracks)
    ]
    route_sw_sets = []

    for t in range(num_tracks):
        for r in range(routes_per_track - 1):
            sw_subset = track_switches[t][
                : (r % switches_per_track) + 1
            ]
            route_sw_sets.append(set(sw_subset))

        if t < num_tracks - 1:
            sw_cross = [
                track_switches[t][0],
                track_switches[t + 1][0],
            ]
            route_sw_sets.append(set(sw_cross))

    pairs = []
    for i in range(len(route_sw_sets)):
        for j in range(i + 1, len(route_sw_sets)):
            if not route_sw_sets[i].isdisjoint(route_sw_sets[j]):
                # Memorizza lo shift in bit (i*2, j*2)
                pairs.append((i * 2, j * 2))

    return (
        np.array(pairs, dtype=np.int32),
        num_switches,
        num_routes,
        (1 << num_switches) - 1,
    )


def load_dataset(dataset_file):
    print(f"📦 Caricamento dataset '{dataset_file}'...")
    data = np.load(dataset_file)
    states = data["states"]
    offsets = data["offsets"]
    edges = data["edges"]
    print(
        f"✅ Caricati {len(states):,} stati e {len(edges):,} transizioni.\n"
    )
    return states, offsets, edges


# -----------------------------------------------------------------------------
# 1. CPU WORKERS
# -----------------------------------------------------------------------------
_G_PAIRS = None
_G_NUM_SW = 0
_G_NUM_RT = 0
_G_SW_MASK = 0


def _cpu_worker_init(pairs, num_sw, num_rt, sw_mask):
    global _G_PAIRS, _G_NUM_SW, _G_NUM_RT, _G_SW_MASK
    _G_PAIRS = pairs
    _G_NUM_SW = num_sw
    _G_NUM_RT = num_rt
    _G_SW_MASK = sw_mask


def _cpu_check_chunk(states_chunk):
    global _G_PAIRS, _G_NUM_SW, _G_NUM_RT, _G_SW_MASK
    mask = np.zeros(len(states_chunk), dtype=np.uint8)

    for idx in range(len(states_chunk)):
        state = states_chunk[idx]
        switches = state & _G_SW_MASK
        routes = state >> _G_NUM_SW
        is_unsafe = 0

        # Controllo coppie rivali
        for shift1, shift2 in _G_PAIRS:
            r1 = (routes >> shift1) & 0b11
            r2 = (routes >> shift2) & 0b11
            if r1 >= 2 and r2 >= 2:
                is_unsafe = 1
                break

        # Controllo consistenza scambi
        if is_unsafe == 0:
            for r_idx in range(_G_NUM_RT):
                r_st = (routes >> (r_idx * 2)) & 0b11
                if r_st == 2:
                    if (switches ^ (r_idx * 31)) & 0b10101010 == 0:
                        is_unsafe = 1
                        break

        mask[idx] = is_unsafe
    return mask


def run_sequential(states, pairs, num_sw, num_rt, sw_mask):
    print("🐢 ESECUZIONE: CPU Sequenziale (Single-Thread)...")
    _cpu_worker_init(pairs, num_sw, num_rt, sw_mask)
    t0 = time.perf_counter()
    mask = _cpu_check_chunk(states)
    t_elapsed = time.perf_counter() - t0
    unsafe_ids = np.where(mask == 1)[0].astype(np.int32)
    print(
        f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {len(unsafe_ids)}\n"
    )
    np.save("unsafe_ids_seq.npy", unsafe_ids)
    return t_elapsed


def run_multiprocessing(states, pairs, num_sw, num_rt, sw_mask):
    cores = cpu_count()
    print(f"⚡ ESECUZIONE: CPU Multiprocessing ({cores} Cores)...")
    t0 = time.perf_counter()
    chunk_size = max(1, (len(states) + cores - 1) // cores)
    chunks = [
        states[i : i + chunk_size] for i in range(0, len(states), chunk_size)
    ]

    with Pool(
        processes=cores,
        initializer=_cpu_worker_init,
        initargs=(pairs, num_sw, num_rt, sw_mask),
    ) as pool:
        results = pool.map(_cpu_check_chunk, chunks)

    full_mask = np.concatenate(results)
    t_elapsed = time.perf_counter() - t0
    unsafe_ids = np.where(full_mask == 1)[0].astype(np.int32)
    print(
        f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {len(unsafe_ids)}\n"
    )
    np.save("unsafe_ids_mp.npy", unsafe_ids)
    return t_elapsed


# -----------------------------------------------------------------------------
# 2. GPU CUDA (NUMBA)
# -----------------------------------------------------------------------------
def run_cuda(states, pairs, num_sw, num_rt, sw_mask):
    print("🚀 ESECUZIONE: GPU CUDA (Numba Parallel)...")
    try:
        from numba import cuda
    except ImportError:
        print("❌ Errore: Numba non è installato!")
        return None

    if not cuda.is_available():
        print("❌ Errore: GPU CUDA non rilevata!")
        return None

    @cuda.jit
    def heavy_safety_kernel(
        states_arr, pairs_arr, results_arr, mask_val, n_sw, n_rt
    ):
        idx = cuda.grid(1)
        if idx < states_arr.size:
            state = states_arr[idx]
            switches = state & mask_val
            routes = state >> n_sw
            is_unsafe = 0

            # 1. Ciclo coppie di rivali
            num_pairs = pairs_arr.shape[0]
            for p in range(num_pairs):
                shift1 = pairs_arr[p, 0]
                shift2 = pairs_arr[p, 1]
                r1 = (routes >> shift1) & 3
                r2 = (routes >> shift2) & 3
                if r1 >= 2 and r2 >= 2:
                    is_unsafe = 1
                    break

            # 2. Controllo scambi
            if is_unsafe == 0:
                for r_idx in range(n_rt):
                    r_st = (routes >> (r_idx * 2)) & 3
                    if r_st == 2:
                        if (switches ^ (r_idx * 31)) & 170 == 0:
                            is_unsafe = 1
                            break

            results_arr[idx] = is_unsafe

    t0 = time.perf_counter()

    d_states = cuda.to_device(states)
    d_pairs = cuda.to_device(pairs)
    d_results = cuda.device_array(states.size, dtype=np.int32)

    threads_per_block = 256
    blocks_per_grid = (
        states.size + (threads_per_block - 1)
    ) // threads_per_block

    heavy_safety_kernel[blocks_per_grid, threads_per_block](
        d_states, d_pairs, d_results, sw_mask, num_sw, num_rt
    )

    h_results = d_results.copy_to_host()
    unsafe_ids = np.where(h_results == 1)[0].astype(np.int32)
    t_elapsed = time.perf_counter() - t0

    dev_name = cuda.get_current_device().name
    if isinstance(dev_name, bytes):
        dev_name = dev_name.decode("utf-8")

    print(f"🎮 Dispositivo GPU: {dev_name}")
    print(
        f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {len(unsafe_ids)}\n"
    )
    np.save("unsafe_ids_cuda.npy", unsafe_ids)
    print("💾 Array salvato in 'unsafe_ids_cuda.npy'\n")
    return t_elapsed


# -----------------------------------------------------------------------------
# MAIN CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking Interlocking Model Checker Parametrico"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="interlocking_2tracks_4sw_5routes.npz",
        help="Nome o percorso del file .npz",
    )
    parser.add_argument(
        "--mode",
        choices=["seq", "mp", "cuda", "all"],
        default="all",
        help="Modalità di esecuzione",
    )
    args = parser.parse_args()

    tracks, sw_per_tr, rt_per_tr = parse_dataset_metadata(args.dataset)
    pairs, num_sw, num_rt, sw_mask = compute_topology_rules(
        tracks, sw_per_tr, rt_per_tr
    )

    states, offsets, edges = load_dataset(args.dataset)

    t_seq, t_mp, t_cuda = None, None, None

    if args.mode == "seq":
        run_sequential(states, pairs, num_sw, num_rt, sw_mask)
    elif args.mode == "mp":
        run_multiprocessing(states, pairs, num_sw, num_rt, sw_mask)
    elif args.mode == "cuda":
        run_cuda(states, pairs, num_sw, num_rt, sw_mask)
    elif args.mode == "all":
        t_seq = run_sequential(states, pairs, num_sw, num_rt, sw_mask)
        t_mp = run_multiprocessing(states, pairs, num_sw, num_rt, sw_mask)
        t_cuda = run_cuda(states, pairs, num_sw, num_rt, sw_mask)

        print("=" * 50)
        print("📊 TABELLA COMPARATIVA DELLE PRESTAZIONI")
        print("=" * 50)
        if t_seq:
            print(f"• CPU Sequenziale:     {t_seq:.4f} s  (1.00x Baseline)")
        if t_mp:
            print(
                f"• CPU Multiprocessing: {t_mp:.4f} s  ({t_seq/t_mp:.2f}x Speedup)"
            )
        if t_cuda:
            print(
                f"• GPU CUDA:            {t_cuda:.4f} s  ({t_seq/t_cuda:.2f}x Speedup)"
            )
        print("=" * 50)