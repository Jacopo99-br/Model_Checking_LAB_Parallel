import argparse
from multiprocessing import Pool, cpu_count
import time
import numpy as np


class BitwiseInterlockingTopology:

    def __init__(
        self, num_tracks=2, switches_per_track=4, routes_per_track=5
    ):
        self.num_tracks = num_tracks
        self.switches_per_track = switches_per_track
        self.routes_per_track = routes_per_track

        self.num_switches = num_tracks * switches_per_track
        self.num_routes = num_tracks * routes_per_track

        self.sw_mask = (1 << self.num_switches) - 1

        # Mappatura statica di scambi per rotta e requisiti
        # route_reqs[r] = list of (sw_idx, required_bit_val 0/1)
        self.route_sw_reqs = []
        self.route_locked_sw_mask = []

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
            # Rotte locali al binario
            for r in range(routes_per_track - 1):
                sw_subset = track_switches[t][
                    : (r % switches_per_track) + 1
                ]
                reqs = [(sw, (idx % 2)) for idx, sw in enumerate(sw_subset)]
                self.route_sw_reqs.append(reqs)

                mask = 0
                for sw in sw_subset:
                    mask |= 1 << sw
                self.route_locked_sw_mask.append(mask)
                route_sw_sets.append(set(sw_subset))

            # Rotta di crossover tra binari
            if t < num_tracks - 1:
                sw_cross = [
                    track_switches[t][0],
                    track_switches[t + 1][0],
                ]
                reqs = [(sw_cross[0], 1), (sw_cross[1], 1)]
                self.route_sw_reqs.append(reqs)

                mask = (1 << sw_cross[0]) | (1 << sw_cross[1])
                self.route_locked_sw_mask.append(mask)
                route_sw_sets.append(set(sw_cross))

        # Ricostruzione rivali: rival_mask[r] è un bitmask delle rotte in conflitto
        self.rival_indices = []
        for i, s1 in enumerate(route_sw_sets):
            rivals = []
            for j, s2 in enumerate(route_sw_sets):
                if i != j and not s1.isdisjoint(s2):
                    rivals.append(j)
            self.rival_indices.append(rivals)


_TOPOLOGY = None


def _init_worker(args):
    global _TOPOLOGY
    _TOPOLOGY = BitwiseInterlockingTopology(*args)


def _expand_uint32_chunk(chunk_states):
    global _TOPOLOGY
    topo = _TOPOLOGY
    num_sw = topo.num_switches
    num_rt = topo.num_routes
    sw_mask = topo.sw_mask

    transitions = []

    for state in chunk_states:
        switches = state & sw_mask
        routes = state >> num_sw

        # 1. Calcola maschera scambi bloccati da rotte OCCUPIED (stato 3)
        locked_mask = 0
        for r_idx in range(num_rt):
            r_st = (routes >> (r_idx * 2)) & 0b11
            if r_st == 3:  # OCCUPIED
                locked_mask |= topo.route_locked_sw_mask[r_idx]

        # 2. Transizioni Scambi (Toggle scambi non bloccati)
        for sw_idx in range(num_sw):
            if not (locked_mask & (1 << sw_idx)):
                new_sw = switches ^ (1 << sw_idx)
                next_st = (routes << num_sw) | new_sw
                transitions.append((state, next_st))

        # 3. Transizioni Rotte
        for r_idx in range(num_rt):
            shift = r_idx * 2
            r_st = (routes >> shift) & 0b11

            # IDLE (0) -> REQUESTED (1)
            if r_st == 0:
                new_routes = routes | (1 << shift)
                next_st = (new_routes << num_sw) | switches
                transitions.append((state, next_st))

            # REQUESTED (1) -> RESERVED (2) / ANNULLA (0)
            elif r_st == 1:
                # Check rivali attive (stato 2 o 3)
                is_conflicting = False
                for rival_idx in topo.rival_indices[r_idx]:
                    riv_st = (routes >> (rival_idx * 2)) & 0b11
                    if riv_st >= 2:
                        is_conflicting = True
                        break

                if not is_conflicting:
                    # Check posizioni scambi
                    sw_ok = True
                    for sw_id, req_val in topo.route_sw_reqs[r_idx]:
                        curr_val = (switches >> sw_id) & 1
                        if curr_val != req_val:
                            sw_ok = False
                            break

                    if sw_ok:
                        # Clear bit 0,1 e set a 2 (0b10)
                        new_routes = (routes & ~(0b11 << shift)) | (
                            2 << shift
                        )
                        next_st = (new_routes << num_sw) | switches
                        transitions.append((state, next_st))

                # Annulla a IDLE (0)
                new_routes = routes & ~(0b11 << shift)
                next_st = (new_routes << num_sw) | switches
                transitions.append((state, next_st))

            # RESERVED (2) -> OCCUPIED (3)
            elif r_st == 2:
                new_routes = (routes & ~(0b11 << shift)) | (3 << shift)
                next_st = (new_routes << num_sw) | switches
                transitions.append((state, next_st))

            # OCCUPIED (3) -> IDLE (0)
            elif r_st == 3:
                new_routes = routes & ~(0b11 << shift)
                next_st = (new_routes << num_sw) | switches
                transitions.append((state, next_st))

    return transitions


def generate_lts_fast(
    num_tracks, switches_per_track, routes_per_track, cores=None
):
    if cores is None:
        cores = cpu_count()

    topo = BitwiseInterlockingTopology(
        num_tracks, switches_per_track, routes_per_track
    )
    init_st = np.uint32(0)

    visited = {init_st: 0}
    state_list = [init_st]
    src_indices = []
    dst_indices = []

    current_frontier = [init_st]
    state_counter = 0

    print(f"⚡ AVVIO GENERAZIONE ULTRA-COMPATTA ({cores} Cores CPU)...")
    print(
        f"⚙️ Configurazione: {num_tracks} Binari | {topo.num_switches} Scambi | {topo.num_routes} Itinerari"
    )
    t0 = time.perf_counter()

    worker_args = (num_tracks, switches_per_track, routes_per_track)

    with Pool(
        processes=cores, initializer=_init_worker, initargs=(worker_args,)
    ) as pool:
        while current_frontier:
            # Slicing puro Python su interi uint32
            chunk_size = max(1, (len(current_frontier) + cores - 1) // cores)
            chunks = [
                current_frontier[i : i + chunk_size]
                for i in range(0, len(current_frontier), chunk_size)
            ]

            results = pool.map(_expand_uint32_chunk, chunks)
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

    print("📦 Costruzione array NumPy e matrici CSR...")
    states_arr = np.array(state_list, dtype=np.uint32)
    src_arr = np.array(src_indices, dtype=np.int32)
    dst_arr = np.array(dst_indices, dtype=np.int32)

    num_states = len(states_arr)
    num_edges = len(src_indices)

    counts = np.bincount(src_arr, minlength=num_states)
    offsets = np.zeros(num_states + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)

    return num_states, num_edges, states_arr, offsets, dst_arr, t_elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generatore Bitwise Memory-Efficient di Benchmark Interlocking"
    )
    parser.add_argument(
        "--tracks", type=int, default=2, help="Numero di binari"
    )
    parser.add_argument(
        "--switches", type=int, default=4, help="Scambi per binario"
    )
    parser.add_argument(
        "--routes", type=int, default=5, help="Itinerari per binario"
    )
    parser.add_argument(
        "--cores", type=int, default=cpu_count(), help="Core CPU da usare"
    )

    args = parser.parse_args()

    n_states, n_edges, states, offsets, edges, t_gen = generate_lts_fast(
        num_tracks=args.tracks,
        switches_per_track=args.switches,
        routes_per_track=args.routes,
        cores=args.cores,
    )

    print(f"\n✅ GENERAZIONE COMPLETATA in {t_gen:.2f} s!")
    print(f"   Stati Totali: {n_states:,}")
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
    print(f"💾 File salvato con successo in: '{filename}'\n")