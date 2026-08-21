import argparse
from multiprocessing import Pool, cpu_count
import time
import numpy as np

DATASET_FILE = "balanced_interlocking_8sw_9routes.npz"

# Matrice completa delle coppie in conflitto (Rivali)
CONFLICTING_PAIRS = np.array(
    [
        (0, 4),  # R_A vs R_AB1
        (0, 10),  # R_A vs R_AB4
        (2, 6),  # R_B vs R_AB2
        (2, 12),  # R_B vs R_AB5
        (4, 6),  # R_AB1 vs R_AB2
        (4, 10),  # R_AB1 vs R_AB4
        (6, 8),  # R_AB2 vs R_AB3
        (8, 12),  # R_AB3 vs R_AB5
        (10, 14),  # R_AB4 vs R_AB6
        (12, 16),  # R_AB5 vs R_AB7
    ],
    dtype=np.int32,
)


def load_dataset():
    print(f"📦 Caricamento dataset '{DATASET_FILE}'...")
    data = np.load(DATASET_FILE)
    states = data["states"]
    offsets = data["offsets"]
    edges = data["edges"]
    print(
        f"✅ Caricati {len(states):,} stati e {len(edges):,} transizioni (archi).\n"
    )
    return states, offsets, edges


# -----------------------------------------------------------------------------
# 1. VERIFICA INTENSIVA SU CPU
# -----------------------------------------------------------------------------
def heavy_safety_check_chunk(states_chunk):
    chunk_size = len(states_chunk)
    mask = np.zeros(chunk_size, dtype=np.uint8)

    for idx in range(chunk_size):
        state = states_chunk[idx]
        switches = state & 0xFF
        routes = state >> 8
        is_unsafe = 0

        for shift1, shift2 in CONFLICTING_PAIRS:
            r1 = (routes >> shift1) & 0b11
            r2 = (routes >> shift2) & 0b11
            if r1 >= 2 and r2 >= 2:
                is_unsafe = 1
                break

        if is_unsafe == 0:
            for r_idx in range(9):
                r_st = (routes >> (r_idx * 2)) & 0b11
                if r_st == 2:
                    if (switches ^ (r_idx * 31)) & 0b10101010 == 0:
                        is_unsafe = 1
                        break

        mask[idx] = is_unsafe
    return mask


def run_sequential(states):
    print("🐢 ESECUZIONE: CPU Sequenziale (Single-Thread)...")
    t0 = time.perf_counter()
    mask = heavy_safety_check_chunk(states)
    t_elapsed = time.perf_counter() - t0
    unsafe_ids = np.where(mask == 1)[0].astype(np.int32)
    print(f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {len(unsafe_ids)}\n")

    np.save("unsafe_ids_seq.npy", unsafe_ids)
    print("💾 Array salvato in 'unsafe_ids_seq.npy' (~5 MB)\n")
    return t_elapsed


def run_multiprocessing(states):
    cores = cpu_count()
    print(f"⚡ ESECUZIONE: CPU Multiprocessing ({cores} Cores)...")
    t0 = time.perf_counter()

    chunks = np.array_split(states, cores)
    with Pool(processes=cores) as pool:
        results = pool.map(heavy_safety_check_chunk, chunks)

    full_mask = np.concatenate(results)
    t_elapsed = time.perf_counter() - t0
    unsafe_ids = np.where(full_mask == 1)[0].astype(np.int32)
    print(f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {len(unsafe_ids)}\n")

    np.save("unsafe_ids_mp.npy", unsafe_ids)
    print("💾 Array salvato in 'unsafe_ids_mp.npy' (~5 MB)\n")
    return t_elapsed


# -----------------------------------------------------------------------------
# 2. VERIFICA INTENSIVA SU GPU CUDA (NUMBA)
# -----------------------------------------------------------------------------
def run_cuda(states):
    print("🚀 ESECUZIONE: GPU CUDA (Numba Parallel)...")
    try:
        from numba import cuda
    except ImportError:
        print("❌ Errore: Numba non è installato in questo ambiente!")
        return None

    if not cuda.is_available():
        print(
            "❌ Errore: GPU CUDA non rilevata! (Attiva il Runtime GPU su Colab)"
        )
        return None

    @cuda.jit
    def heavy_safety_kernel(states_arr, pairs_arr, results_arr):
        idx = cuda.grid(1)
        if idx < states_arr.size:
            state = states_arr[idx]
            switches = state & 0xFF
            routes = state >> 8

            is_unsafe = 0

            # 1. Ciclo su tutte le coppie di rivali
            num_pairs = pairs_arr.shape[0]
            for p in range(num_pairs):
                shift1 = pairs_arr[p, 0]
                shift2 = pairs_arr[p, 1]

                r1 = (routes >> shift1) & 3
                r2 = (routes >> shift2) & 3

                if r1 >= 2 and r2 >= 2:
                    is_unsafe = 1
                    break

            # 2. Check coerenza scambi
            if is_unsafe == 0:
                for r_idx in range(9):
                    r_st = (routes >> (r_idx * 2)) & 3
                    if r_st == 2:
                        if (switches ^ (r_idx * 31)) & 170 == 0:
                            is_unsafe = 1
                            break

            results_arr[idx] = is_unsafe

    t0 = time.perf_counter()

    # Trasferimento dati in VRAM
    d_states = cuda.to_device(states)
    d_pairs = cuda.to_device(CONFLICTING_PAIRS)
    d_results = cuda.device_array(states.size, dtype=np.int32)

    threads_per_block = 256
    blocks_per_grid = (
        states.size + (threads_per_block - 1)
    ) // threads_per_block

    heavy_safety_kernel[blocks_per_grid, threads_per_block](
        d_states, d_pairs, d_results
    )

    h_results = d_results.copy_to_host()
    unsafe_ids = np.where(h_results == 1)[0].astype(np.int32)
    violations = len(unsafe_ids)


    t_elapsed = time.perf_counter() - t0

    print(f"🎮 Dispositivo GPU: {cuda.get_current_device().name}")
    print(f"⏱️ Tempo impiegato: {t_elapsed:.4f} s | Violazioni: {violations}\n")

    np.save("unsafe_ids_cuda.npy", unsafe_ids)
    print("💾 Array salvato con successo in 'unsafe_ids_cuda.npy' (~5 MB)\n")

    return t_elapsed


# -----------------------------------------------------------------------------
# MAIN CLI CON MENU INTERATTIVO
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking Interlocking Model Checker"
    )
    parser.add_argument(
        "--mode",
        choices=["seq", "mp", "cuda", "all", "interactive"],
        default="interactive",
        help="Modalità di esecuzione",
    )
    args = parser.parse_args()

    mode = args.mode

    # Se l'utente non passa flag da riga di comando, mostra il menu a schermo!
    if mode == "interactive":
        print("==================================================")
        print("  SELEZIONA LA MODALITÀ DI PARALLELIZZAZIONE")
        print("==================================================")
        print(" [1] CPU Sequenziale (Single-Thread)")
        print(" [2] CPU Multiprocessing (Multi-Core)")
        print(" [3] GPU CUDA (Numba)")
        print(" [4] Esegui TUTTI i metodi (Confronto Completo)")
        print("==================================================")

        choice = input("Inserisci il numero dell'opzione (1-4): ").strip()
        mode_map = {"1": "seq", "2": "mp", "3": "cuda", "4": "all"}
        mode = mode_map.get(choice, "seq")

    states, _, _ = load_dataset()

    t_seq, t_mp, t_cuda = None, None, None

    if mode == "seq":
        run_sequential(states)
    elif mode == "mp":
        run_multiprocessing(states)
    elif mode == "cuda":
        run_cuda(states)
    elif mode == "all":
        t_seq = run_sequential(states)
        t_mp = run_multiprocessing(states)
        t_cuda = run_cuda(states)

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