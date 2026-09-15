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

    uint64_t total_count = 0;
    std::vector<uint32_t> data;

    if (rank == 0) {
        std::ifstream fin("input.bin", std::ios::binary | std::ios::ate);
        if (!fin) {
            std::cerr << "failed to open input.bin" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        total_count = static_cast<uint64_t>(fin.tellg() / sizeof(uint32_t));
        fin.seekg(0, std::ios::beg);

        data.resize(static_cast<size_t>(total_count));
        fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total_count * sizeof(uint32_t)));
        fin.close();
    }

    MPI_Bcast(&total_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts;
    std::vector<int> displs;
    sendcounts.resize(static_cast<size_t>(size), 0);
    displs.resize(static_cast<size_t>(size), 0);

    const uint64_t base = (size > 0) ? (total_count / static_cast<uint64_t>(size)) : 0;
    const uint64_t rem = (size > 0) ? (total_count % static_cast<uint64_t>(size)) : 0;

    uint64_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const uint64_t n = base + ((static_cast<uint64_t>(r) < rem) ? 1 : 0);
        sendcounts[static_cast<size_t>(r)] = static_cast<int>(n);
        displs[static_cast<size_t>(r)] = static_cast<int>(offset);
        offset += n;
    }

    const int local_n = sendcounts[static_cast<size_t>(rank)];
    std::vector<uint32_t> local_data(static_cast<size_t>(local_n));

    MPI_Scatterv(
        rank == 0 ? data.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_UNSIGNED,
        local_data.data(),
        local_n,
        MPI_UNSIGNED,
        0,
        MPI_COMM_WORLD
    );

    std::vector<uint64_t> local_hist(MAXVAL, 0);
    for (const auto value : local_data) {
        if (value < MAXVAL) {
            local_hist[value] += 1;
        }
    }

    std::vector<uint64_t> global_hist;
    if (rank == 0) {
        global_hist.resize(MAXVAL, 0);
    }

    MPI_Reduce(
        local_hist.data(),
        rank == 0 ? global_hist.data() : nullptr,
        static_cast<int>(MAXVAL),
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::ofstream fout("hist_u64.bin", std::ios::binary);
        if (!fout) {
            std::cerr << "failed to open hist_u64.bin" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        fout.write(
            reinterpret_cast<const char*>(global_hist.data()),
            static_cast<std::streamsize>(global_hist.size() * sizeof(uint64_t))
        );
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
