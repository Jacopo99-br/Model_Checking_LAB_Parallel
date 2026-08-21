from collections import deque
import time
import numpy as np


class BalancedInterlockingGenerator:

    def __init__(self):
        # 8 Scambi totali: s1..s4 (Binario A), s5..s8 (Binario B)
        self.num_switches = 8
        self.SWITCH_NAMES = [f"s{i}" for i in range(1, 9)]

        # 9 Itinerari bilanciati
        self.ROUTES = {
            # Dedicated
            "R_A": {
                "switches": [("s1", 1), ("s2", 1), ("s3", 1)]
            },  # Solo Binario A
            "R_B": {
                "switches": [("s5", 1), ("s6", 1), ("s7", 1)]
            },  # Solo Binario B
            # Intersecting / Misti
            "R_AB1": {"switches": [("s1", 2), ("s5", 2)]},
            "R_AB2": {"switches": [("s2", 2), ("s5", 1), ("s6", 2)]},
            "R_AB3": {"switches": [("s3", 1), ("s7", 1)]},
            "R_AB4": {"switches": [("s1", 1), ("s2", 2), ("s5", 2)]},
            "R_AB5": {"switches": [("s3", 2), ("s6", 1)]},
            "R_AB6": {"switches": [("s4", 1), ("s8", 1)]},
            "R_AB7": {"switches": [("s4", 2), ("s7", 1), ("s8", 2)]},
        }
        self.route_names = list(self.ROUTES.keys())
        self.num_routes = len(self.route_names)

        # 4 Stati Ciclo di Vita: 0=IDLE, 1=REQUESTED, 2=RESERVED, 3=OCCUPIED
        self.ROUTE_STATES = ["IDLE", "REQUESTED", "RESERVED", "OCCUPIED"]

        # Matrice delle Rivali
        self.RIVALS = {r: [] for r in self.route_names}
        for r1 in self.route_names:
            s1_set = {sw[0] for sw in self.ROUTES[r1]["switches"]}
            for r2 in self.route_names:
                if r1 != r2:
                    s2_set = {sw[0] for sw in self.ROUTES[r2]["switches"]}
                    if not s1_set.isdisjoint(s2_set):
                        self.RIVALS[r1].append(r2)

    def get_initial_state(self):
        switches = (0,) * self.num_switches  # Tutti in NORMAL
        routes = (0,) * self.num_routes  # Tutti in IDLE
        return (switches, routes)

    def is_conflicting_route_active(self, route_idx, routes):
        r_name = self.route_names[route_idx]
        for rival_name in self.RIVALS[r_name]:
            rival_idx = self.route_names.index(rival_name)
            if routes[rival_idx] in (2, 3):  # RESERVED o OCCUPIED
                return True
        return False

    def get_next_states(self, current_state):
        switches, routes = current_state
        next_states = []

        # 1. TRANSIZIONI SCAMBI
        locked_switches = set()
        for idx, r_st in enumerate(routes):
            if r_st == 3:  # OCCUPIED blocca fisicamente gli scambi
                r_name = self.route_names[idx]
                for sw_name, _ in self.ROUTES[r_name]["switches"]:
                    locked_switches.add(sw_name)

        for sw_idx in range(self.num_switches):
            sw_name = self.SWITCH_NAMES[sw_idx]
            if sw_name not in locked_switches:
                new_switches = list(switches)
                new_switches[sw_idx] = 1 - new_switches[sw_idx]
                next_states.append((tuple(new_switches), routes))

        # 2. EVOLUZIONE ROTTE
        for idx, r_st in enumerate(routes):
            r_name = self.route_names[idx]

            # IDLE -> REQUESTED
            if r_st == 0:
                new_routes = list(routes)
                new_routes[idx] = 1
                next_states.append((switches, tuple(new_routes)))

            # REQUESTED -> RESERVED
            elif r_st == 1:
                if not self.is_conflicting_route_active(idx, routes):
                    sw_reqs = self.ROUTES[r_name]["switches"]
                    sw_ok = all(
                        switches[self.SWITCH_NAMES.index(sw_name)]
                        == (req_pos - 1)
                        for sw_name, req_pos in sw_reqs
                    )
                    if sw_ok:
                        new_routes = list(routes)
                        new_routes[idx] = 2
                        next_states.append((switches, tuple(new_routes)))

                # Annullamento
                new_routes = list(routes)
                new_routes[idx] = 0
                next_states.append((switches, tuple(new_routes)))

            # RESERVED -> OCCUPIED
            elif r_st == 2:
                new_routes = list(routes)
                new_routes[idx] = 3
                next_states.append((switches, tuple(new_routes)))

            # OCCUPIED -> IDLE
            elif r_st == 3:
                new_routes = list(routes)
                new_routes[idx] = 0
                next_states.append((switches, tuple(new_routes)))

        return next_states

    def generate_lts(self):
        init_st = self.get_initial_state()
        visited = {init_st: 0}
        queue = deque([init_st])

        state_list = [init_st]
        src_indices = []
        dst_indices = []
        state_counter = 0

        print("Esplorazione BFS completa dello Spazio degli Stati in corso...")
        t0 = time.perf_counter()

        while queue:
            curr_st = queue.popleft()
            curr_idx = visited[curr_st]

            for nxt in self.get_next_states(curr_st):
                if nxt not in visited:
                    state_counter += 1
                    visited[nxt] = state_counter
                    state_list.append(nxt)
                    queue.append(nxt)

                src_indices.append(curr_idx)
                dst_indices.append(visited[nxt])

        t_elapsed = time.perf_counter() - t0

        # Bitmasking uint32: Bit 0..7 (Scambi), Bit 8..25 (9 Rotte x 2 bit = 18 bit)
        encoded_states = []
        for sw, rt in state_list:
            enc = 0
            for i, val in enumerate(sw):
                enc |= val << i
            for i, val in enumerate(rt):
                enc |= val << (8 + i * 2)
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
        )


if __name__ == "__main__":
    gen = BalancedInterlockingGenerator()
    n_states, n_edges, states, offsets, edges, t_gen = gen.generate_lts()

    print(f"\n✅ GENERAZIONE COMPLETATA in {t_gen:.2f} s!")
    print(f"   Stati Totali Unici Raggiungibili: {n_states:,}")
    print(f"   Transizioni Totali (Archi): {n_edges:,}")

    filename = "balanced_interlocking_8sw_9routes.npz"
    np.savez_compressed(
        filename,
        num_states=n_states,
        num_edges=n_edges,
        states=states,
        offsets=offsets,
        edges=edges,
    )
    print(f"   Dataset binario salvato con successo in '{filename}'!\n")