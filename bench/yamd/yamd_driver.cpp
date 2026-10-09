// Head-to-head driver for the C++ predecessor yamd (github.com/walidhemati/yamd).
// Compiled against yamd's own sources (lj.cpp, neighbors.cpp, ducastelle.cpp, verlet.cpp, ...)
// by bench/yamd_compare.py. Single thread, no MPI.
//
// usage: yamd_driver <mode> <xyz-binary-in> <nsteps> <out-prefix>
//   mode = lj   : truncated-shifted LJ (eps = sigma = 1, rc = 2.5), yamd neighbor list every step
//   mode = eam  : Gupta/Ducastelle Au (yamd defaults, cutoff 10 A), neighbor list every step
// Input: int64 N, then 3N float64 positions, then 3N float64 velocities.
// Output: <out-prefix>.forces (3N float64 forces at t = 0), and on stdout a JSON line with
// the timing of nsteps velocity-Verlet steps.
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "atoms.h"
#include "ducastelle.h"
#include "lj.h"
#include "neighbors.h"
#include "verlet.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: yamd_driver lj|eam in.bin nsteps out\n";
        return 1;
    }
    std::string mode = argv[1];
    std::ifstream in(argv[2], std::ios::binary);
    int64_t n;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    Positions_t pos(3, n);
    Velocities_t vel(3, n);
    in.read(reinterpret_cast<char *>(pos.data()), 3 * n * sizeof(double));
    in.read(reinterpret_cast<char *>(vel.data()), 3 * n * sizeof(double));
    int nsteps = std::stoi(argv[3]);
    std::string out = argv[4];

    Atoms atoms(pos, vel);
    const double au_mass = 196.96657 * U_PER_MASS_UNIT;
    NeighborList nl;
    auto compute = [&]() -> double {
        if (mode == "lj") return lj_cutoff_direct_summation(atoms, 1.0, 1.0, 2.5);
        nl.update(atoms, 10.0);
        return ducastelle(atoms, nl);  // yamd defaults: Au, cutoff 10 A
    };
    double epot = compute();
    {
        std::ofstream fo(out + ".forces", std::ios::binary);
        fo.write(reinterpret_cast<const char *>(atoms.forces.data()), 3 * n * sizeof(double));
    }
    double dt = mode == "lj" ? 0.005 : 1.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < nsteps; ++s) {
        if (mode == "lj") {
            verlet_step1(atoms.positions, atoms.velocities, atoms.forces, dt);
            compute();
            verlet_step2(atoms.velocities, atoms.forces, dt);
        } else {
            verlet_step1(atoms.positions, atoms.velocities, atoms.forces, dt, au_mass);
            compute();
            verlet_step2(atoms.velocities, atoms.forces, dt, au_mass);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double epot1 = compute();
    std::printf("{\"mode\": \"%s\", \"n\": %lld, \"nsteps\": %d, \"seconds\": %.6f, "
                "\"epot0\": %.12f, \"epot1\": %.12f, \"ekin1\": %.12f}\n",
                mode.c_str(), static_cast<long long>(n), nsteps, secs, epot, epot1,
                mode == "lj" ? 0.5 * atoms.velocities.square().sum()
                             : 0.5 * au_mass * atoms.velocities.square().sum());
    return 0;
}
