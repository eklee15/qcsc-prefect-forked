#include <mpi.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

static constexpr uint32_t BITLEN = 10;
static constexpr uint32_t MAXVAL = (1u << BITLEN);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint32_t total_count = 0;
    std::vector<uint32_t> data;

    if (rank == 0) {
        std::ifstream fin("input.bin", std::ios::binary | std::ios::ate);
        if (!fin) {
            std::cerr << "failed to open input.bin" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        total_count = static_cast<uint32_t>(fin.tellg() / sizeof(uint32_t));
        fin.seekg(0, std::ios::beg);

        data.resize(total_count);
        fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total_count * sizeof(uint32_t)));
        fin.close();
    }

    MPI_Bcast(&total_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    uint32_t local_n = (size > 0) ? total_count / static_cast<uint32_t>(size) : 0;
    std::vector<uint32_t> local_data(local_n);

    MPI_Scatter(
        rank == 0 ? data.data() : nullptr,
        static_cast<int>(local_n),
        MPI_UNSIGNED,
        local_data.data(),
        static_cast<int>(local_n),
        MPI_UNSIGNED,
        0,
        MPI_COMM_WORLD
    );

    std::vector<int> local_hist(MAXVAL, 0);
    for (const auto value : local_data) {
        if (value < MAXVAL) {
            local_hist[value] += 1;
        }
    }

    std::vector<int> global_hist;
    if (rank == 0) {
        global_hist.resize(MAXVAL, 0);
    }

    MPI_Reduce(
        local_hist.data(),
        rank == 0 ? global_hist.data() : nullptr,
        static_cast<int>(MAXVAL),
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::ofstream fout("output.json");
        if (!fout) {
            std::cerr << "failed to open output.json" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        fout << "{";
        bool first = true;
        for (uint32_t i = 0; i < MAXVAL; ++i) {
            if (global_hist[i] > 0) {
                if (!first) {
                    fout << ",";
                }
                fout << "\"" << i << "\":" << global_hist[i];
                first = false;
            }
        }
        fout << "}\n";
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
