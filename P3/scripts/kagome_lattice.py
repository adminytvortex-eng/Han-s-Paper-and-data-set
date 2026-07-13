"""
Kagome lattice geometry for the Lattice Response Spectroscopy experiment (Paper 3, Stage 1).

*** This module is now a thin wrapper around the user's authoritative reference module,
kagome_vortex_core.py (the corrected lattice, verified there by networkx clique-finding to
have exactly 2*L*L triangles) -- it does not reimplement lattice construction. ***

History, for anyone reading this later: an earlier standalone `build_bonds` (used only for
the H-map / kagome_hmap_miner.py screening pipeline) was mistakenly assumed to be the full
physics lattice in an earlier version of this file. Brute-force triangle enumeration on that
particular bond list finds only L^2 triangles, not 2*L^2 -- it is NOT the lattice used for
drop/vortex physics in Paper 2. kagome_vortex_core.py is: it is imported directly by Paper
2's actual mining script (mine_combined_drop_information_local.py) for build_kagome_lattice,
mc_sweep, detect_vortices, build_plaquette_adjacency, and bfs_distances_from, and its
build_kagome_lattice() asserts len(triangles) == 2*L*L internally. Use *this* file's
KagomeLattice class for anything involving vortices, drop, or plaquette adjacency; the
up-triangle-only H-map path (information density) is unaffected by any of this, since it
only ever used the intra-cell (A,B,C) bonds, which are identical in every version.
"""

import numpy as np

from kagome_vortex_core import build_kagome_lattice, build_plaquette_adjacency


class KagomeLattice:
    def __init__(self, L):
        self.L = L
        self.N = 3 * L * L

        (bonds_arr, x_bonds_arr, triangles_sites,
         plaquette_centers, site_cell_coords) = build_kagome_lattice(L)

        self.bonds = np.asarray(bonds_arr, dtype=np.int64)
        self.x_bonds = np.asarray(x_bonds_arr, dtype=np.int64)
        assert len(self.bonds) == 6 * L * L

        # site -> neighbouring SITE indices (not bond indices), for the numba Metropolis
        # engine in driven_kagome_sim.py. Built directly from self.bonds -- trivially
        # consistent with kagome_vortex_core's own precompute_neighbors (which returns bond
        # indices instead; either representation gives the same physics).
        neighbors = [[] for _ in range(self.N)]
        for i, j in self.bonds.tolist():
            neighbors[i].append(j)
            neighbors[j].append(i)
        neigh_arr = np.array(neighbors, dtype=np.int64)
        assert neigh_arr.shape == (self.N, 4), "expected coordination number 4 everywhere"
        self.neighbors = neigh_arr

        # full plaquette list (2*L^2 triangles, up AND down) -- used for vortex detection
        # (birth) and, via plaq_adj below, for drop's E(r) radial profile. Site-index tuples
        # only (kagome_vortex_core also tags the 'up' ones with (...,ix,iy,'up') internally,
        # but that tag is not preserved in the returned list -- not needed here, since the
        # up-triangle list for the H-map is reconstructed independently below).
        self.plaquettes = np.array([(t[0], t[1], t[2]) for t in triangles_sites], dtype=np.int64)
        self.n_plaq = len(self.plaquettes)
        assert self.n_plaq == 2 * L * L

        self.plaq_adj = build_plaquette_adjacency(triangles_sites)
        deg = [len(a) for a in self.plaq_adj]
        assert all(d == 3 for d in deg), "every kagome plaquette should border exactly 3 others"

        # up-triangles only, indexed by (ix,iy) on a plain square grid -- for the H-map
        # (information density). Site formula A=3*(ix*L+iy), B=A+1, C=A+2 is identical to
        # kagome_vortex_core's own idx(ix,iy,0/1/2) and to the reference kagome_hmap_miner.py,
        # so this is guaranteed consistent with both without depending on triangle ordering.
        up_plaq = []
        for ix in range(L):
            for iy in range(L):
                base = 3 * (ix * L + iy)
                up_plaq.append((base, base + 1, base + 2))
        self.up_plaq = np.array(up_plaq, dtype=np.int64)
        self.n_up = L * L

        # 2D coordinates (plotting only; not used in any physics calculation)
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3) / 2])
        offsets = {0: np.array([0.0, 0.0]), 1: np.array([0.5, 0.0]),
                   2: np.array([0.25, np.sqrt(3) / 4])}
        pos = np.zeros((self.N, 2))
        for s, (ix, iy, sub) in site_cell_coords.items():
            pos[s] = ix * a1 + iy * a2 + offsets[sub]
        self.pos = pos

    def bfs_shells(self, source_plaq, max_r):
        """Return {r: [plaquette indices at graph-distance r]} for r=0..max_r, on the
        corner-sharing plaquette adjacency graph (kagome_vortex_core.build_plaquette_adjacency).
        """
        from collections import deque
        dist = {source_plaq: 0}
        order = deque([source_plaq])
        while order:
            p = order.popleft()
            if dist[p] >= max_r:
                continue
            for q in self.plaq_adj[p]:
                if q not in dist:
                    dist[q] = dist[p] + 1
                    order.append(q)
        shells = {r: [] for r in range(max_r + 1)}
        for p, r in dist.items():
            shells[r].append(p)
        return shells


if __name__ == "__main__":
    lat = KagomeLattice(L=8)
    print(f"N sites = {lat.N}, N plaquettes = {lat.n_plaq} (expect 2*L^2 = {2*8*8})")
    print(f"Coordination number check: neighbors.shape = {lat.neighbors.shape}")
    deg = [len(a) for a in lat.plaq_adj]
    print(f"Plaquette adjacency degree: min={min(deg)}, max={max(deg)} (expect 3 everywhere)")
    shells = lat.bfs_shells(source_plaq=0, max_r=3)
    print("BFS shell sizes from plaquette 0:", {r: len(v) for r, v in shells.items()},
          "(expect 1,3,6,9 for the corner-sharing kagome plaquette graph)")
    assert [len(shells[r]) for r in [0, 1, 2, 3]] == [1, 3, 6, 9]
    print("Geometry self-test passed (using kagome_vortex_core.py directly).")
