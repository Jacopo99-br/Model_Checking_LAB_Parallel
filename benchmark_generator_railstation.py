import argparse
from collections import deque
from multiprocessing import Pool, cpu_count
import time
import numpy as np


class ParametricInterlockingGenerator:

    def __init__(
        self, num_tracks=2, switches_per_track=4, routes_per_track=5
    ):
        self.num_tracks = num_tracks
        self.switches_per_track = switches_per_track
        self.routes_per_track = routes_per_track

        self.num_switches = num_tracks * switches_per_track
        self.num_routes = num_tracks * routes_per_track

        self.SWITCH_NAMES = [f"s{i+1}" for i in range(self.num_switches)]
        self.ROUTES = {}

        # Generazione itinerari locali e di scambio tra binari adiacenti
        for t in range(num_tracks):
            track_switches = self.SWITCH_NAMES[
                t * switches_per_track : (t + 1) * switches_per_track
            ]

            for r in range(routes_per_track - 1):
                r_name = f"R_T{t+1}_{r+1}"
                sw_subset = track_switches[: (r % switches_per_track) + 1]
                sw_reqs = [(sw, (idx % 2) + 1) for idx, sw in enumerate(sw_subset)]
                self.ROUTES[r_name] = {"switches": sw_reqs}

            if t < num_tracks - 1:
                r_name = f"R_CROSS_{t+1}_{t+2}"
                next_track_switches = self.SWITCH_NAMES[
                    (t + 1) * switches_per_track : (t + 2) * switches_per_track
                ]
                sw_reqs = [
                    (track_switches[0], 2),
                    (next_track_switches[0], 2),
                ]
                self.ROUTES[r_name] = {"switches": sw_reqs}

        self.route_names = list(self.ROUTES.keys())
        self.num_routes = len(self.route_names)

        # Matrice delle rivali: itinerari che condividono almeno uno scambiatore
        self.RIVALS = {r: [] for r in self.route_names}
        for r1 in self.route_names:
            s1_set = {sw[0] for sw in self.ROUTES[r1]["switches"]}
            for r2 in self.route_names:
                if r1 != r2:
                    s2_set = {sw[0] for sw in self.ROUTES[r2]["switches"]}
                    if not s1_set.isdisjoint(s2_set):
                        self.RIVALS[r1].append(r2)

    def get_initial_state(self):
        switches = (0,) * self.num_switches
        routes = (0,) * self.num_routes
        return (switches, routes)


_global_gen = None


def _init_worker(gen_args):
    global _global_gen
    _global_gen = ParametricInterlockingGenerator(*gen_args)


def _expand_state_chunk(state_chunk):
    global _global_gen
    gen = _global_gen
    chunk_transitions = []

    for curr_st in state_chunk:
        switches, routes = curr_st

        # Transizioni scambiatori
        locked_switches = set()
        for idx, r_st in enumerate(routes):
            if r_st == 3:  # OCCUPIED
                r_name = gen.route_names[idx]
                for sw_name, _ in gen.ROUTES[r_name]["switches"]:
                    locked_switches.add(sw_name)

        for sw_idx in range(gen.num_switches):
            sw_name = gen.SWITCH_NAMES[sw_idx]
            if sw_name not in locked_switches:
                new_switches = list(switches)
                new_switches[sw_idx] = 1 - new_switches[sw_idx]
                chunk_transitions.append((curr_st, (tuple(new_switches), routes)))

        # Transizioni ciclo di vita rotte
        for idx, r_st in enumerate(routes):
            r_name = gen.route_names[idx]

            # IDLE -> REQUESTED
            if r_st == 0:
                new_routes = list(routes)
                new_routes[idx] = 1
                chunk_transitions.append((curr_st, (switches, tuple(new_routes))))

            # REQUESTED -> RESERVED
            elif r_st == 1:
                is_conflicting = any(
                    routes[gen.route_names.index(rival)] in (2, 3)
                    for rival in gen.RIVALS[r_name]
                )
                if not is_conflicting:
                    sw_reqs = gen.ROUTES[r_name]["switches"]
                    sw_ok = all(
                        switches[gen.SWITCH_NAMES.index(sw_name)] == (req_pos - 1)
                        for sw_name, req_pos in sw_reqs
                    )
                    if sw_ok:
                        new_routes = list(routes)
                        new_routes[idx] = 2
                        chunk_transitions.append(
                            (curr_st, (switches, tuple(new_routes)))
                        )

                # Annullamento richiesta
                new_routes = list(routes)
                new_routes[idx] = 0
                chunk_transitions.append((curr_st, (switches, tuple(new_routes))))

            # RESERVED -> OCCUPIED
            elif r_st == 2:
                new_routes = list(routes)
                new_routes[idx] = 3
                chunk_transitions.append((curr_st, (switches, tuple(new_routes))))

            # OCCUPIED -> IDLE
            elif r_st == 3:
                new_routes = list(routes)
                new_routes[idx] = 0
                chunk_transitions.append((curr_st, (switches, tuple(new_routes))))

    return chunk_transitions


def generate_lts_parallel(
    num_tracks, switches_per_track, routes_per_track, cores=None
):
    if cores is None:
        cores = cpu_count()

    gen = ParametricInterlockingGenerator(
        num_tracks, switches_per_track, routes_per_track
    )
    init_st = gen.get_initial_state()

    visited = {init_st: 0}
    state_list = [init_st]
    src_indices = []
    dst_indices = []

    current_frontier = [init_st]
    state_counter = 0

    print(f"⚡ AVVIO GENERAZIONE PARALLELA ({cores} Cores CPU)...")
    print(
        f"⚙️ Configurazione: {num_tracks} Binari | {gen.num_switches} Scambi | {gen.num_routes} Itinerari"
    )
    t0 = time.perf_counter()

    gen_args = (num_tracks, switches_per_track, routes_per_track)

    with Pool(processes=cores, initializer=_init_worker, initargs=(gen_args,)) as pool:
        while current_frontier:
            chunks = np.array_split(current_frontier, cores)
            chunks = [c.tolist() for c in chunks if len(c) > 0]

            results = pool.map(_expand_state_chunk, chunks)
            next_frontier = []

            for chunk_res in results:
                for curr_st, nxt in chunk_res:
                    curr_idx = visited[curr_st]

                    if nxt not in visited:
                        state_counter += 1
                        visited[nxt] = state_counter
                        state_list.append(nxt)
                        next_frontier.append(nxt)

                    src_indices.append(curr_idx)
                    dst_indices.append(visited[nxt])

            current_frontier = next_frontier

    t_elapsed = time.perf_counter() - t0

    print("📦 Bitmasking uint32 e costruzione matrici CSR...")
    encoded_states = []
    num_sw = gen.num_switches

    for sw, rt in state_list:
        enc = 0
        for i, val in enumerate(sw):
            enc |= val << i
        for i, val in enumerate(rt):
            enc |= val << (num_sw + i * 2)
        encoded_states.append(enc)

    num_states = len(state_list)
    num_edges = len(src_indices)

    src_arr = np.array(src_indices, dtype=np.int32)
    dst_arr = np.array(dst_indices, dtype=np.int32)

    counts = np.bincount(src_arr, minlength=num_states)
    offsets = np.zeros(num_states + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)

    return (
        num_states,
        num_edges,
        np.array(encoded_states, dtype=np.uint32),
        offsets,
        dst_arr,
        t_elapsed,
        gen,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generatore Parametrico e Parallelo di Benchmark Interlocking"
    )
    parser.add_argument(
        "--tracks", type=int, default=2, help="Numero di binari (default: 2)"
    )
    parser.add_argument(
        "--switches",
        type=int,
        default=4,
        help="Scambiatori per binario (default: 4)",
    )
    parser.add_argument(
        "--routes",
        type=int,
        default=5,
        help="Itinerari per binario (default: 5)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=cpu_count(),
        help="Numero di core CPU (default: tutti i core disponibili)",
    )

    args = parser.parse_args()

    n_states, n_edges, states, offsets, edges, t_gen, gen_instance = (
        generate_lts_parallel(
            num_tracks=args.tracks,
            switches_per_track=args.switches,
            routes_per_track=args.routes,
            cores=args.cores,
        )
    )

    print(f"\n✅ GENERAZIONE COMPLETATA in {t_gen:.2f} s!")
    print(f"   Stati Raggiungibili: {n_states:,}")
    print(f"   Transizioni: {n_edges:,}")

    filename = (
        f"interlocking_{args.tracks}tracks_{args.switches}sw_{args.routes}routes.npz"
    )
    np.savez_compressed(
        filename,
        num_states=n_states,
        num_edges=n_edges,
        states=states,
        offsets=offsets,
        edges=edges,
    )
    print(f"   File salvato: '{filename}'\n")