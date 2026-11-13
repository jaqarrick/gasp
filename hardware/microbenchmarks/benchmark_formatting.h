#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
// this is a general header file that stores functions that every benchmark needs.

// cuda verification and clock helper functions

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

// parsing CLI arguments

struct BenchArgs {
    int reps = 200;
    std::string gpu_override;
    std::string out_dir = "hardware/runs";
};

inline BenchArgs parse_bench_args(int argc, char** argv, int default_reps = 200) {
    BenchArgs args;
    args.reps = default_reps;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) {
            args.reps = std::atoi(argv[++i]);
        } else if (arg == "--gpu" && i + 1 < argc) {
            args.gpu_override = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            args.out_dir = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (args.reps < 100) {
        fprintf(stderr, "--reps must be at least 100\n");
        exit(EXIT_FAILURE);
    }
    return args;
}

// statistics helper functions

struct Stats {
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
}

inline Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    s.median = median(v);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / (v.size() - 1));
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    return s;
}

// helpful for collecting metadata + formatting JSON

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { out += ' '; }
        else { out += c; }
    }
    return out;
}

inline std::string short_gpu_name(const std::string& full_name) {
    std::string lower;
    for (char c : full_name) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* name : {"a100", "h100", "b200", "titan v"}) {
        if (lower.find(name) != std::string::npos) {
            std::string s = name;
            std::replace(s.begin(), s.end(), ' ', '_');
            return s;
        }
    }
    std::string res;
    for (char c : lower) res += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    return res;
}

inline std::string architecture_name(int major) {
    switch (major) {
        case 7: return "volta";
        case 8: return "ampere";
        case 9: return "hopper";
        case 10: return "blackwell";
        default: return "unknown";
    }
}

inline std::string cuda_version_string(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

inline std::string utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buf;
}

inline std::string driver_version() {
    FILE* pipe = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null", "r");
    if (!pipe) return "unknown";
    char buf[128];
    std::string res;
    if (fgets(buf, sizeof(buf), pipe)) {
        res = buf;
        while (!res.empty() && std::isspace(static_cast<unsigned char>(res.back()))) res.pop_back();
    }
    pclose(pipe);
    return res.empty() ? "unknown" : res;
}

// JSON reporting and file I/O

struct BenchmarkReport {
    std::string benchmark;
    std::string parameter;
    std::string unit;
    Stats stats;
    double sm_clock_ghz = 0.0;
    std::vector<std::pair<std::string, std::vector<double>>> arrays;
    std::vector<std::pair<std::string, double>> scalars;
    std::vector<std::pair<std::string, std::string>> measurement_str;
    std::vector<std::pair<std::string, double>> measurement_num;
};

inline void write_report(const BenchArgs& args, const BenchmarkReport& r) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int runtime_ver = 0, driver_ver = 0, shmem_per_sm = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));
    CUDA_CHECK(cudaDriverGetVersion(&driver_ver));
    CUDA_CHECK(cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));

    std::string gpu = args.gpu_override.empty() ? short_gpu_name(prop.name) : args.gpu_override;
    std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);

    json << "{\n"
         << "  \"schema_version\": \"1.1.0\",\n"
         << "  \"benchmark\": \"" << r.benchmark << "\",\n"
         << "  \"benchmark_version\": \"1.1.0\",\n"
         << "  \"parameter\": \"" << r.parameter << "\",\n"
         << "  \"value\": " << r.stats.median << ",\n"
         << "  \"unit\": \"" << r.unit << "\",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture_name(prop.major) << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(prop.name) << "\",\n"
         << "    \"sms\": " << prop.multiProcessorCount << ",\n"
         << "    \"max_warps_per_sm\": " << (prop.maxThreadsPerMultiProcessor / prop.warpSize) << ",\n"
         << "    \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n"
         << "    \"shared_mem_per_sm_bytes\": " << shmem_per_sm << "\n"
         << "  },\n"
         << "  \"stats\": {\n"
         << "    \"median\": " << r.stats.median << ",\n"
         << "    \"mean\": " << r.stats.mean << ",\n"
         << "    \"std\": " << r.stats.stddev << ",\n"
         << "    \"min\": " << r.stats.min << ",\n"
         << "    \"max\": " << r.stats.max << "\n"
         << "  },\n";

    for (const auto& kv : r.arrays) {
        json << "  \"" << kv.first << "\": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            json << (i == 0 ? "" : ", ") << kv.second[i];
        }
        json << "],\n";
    }

    for (const auto& kv : r.scalars) {
        json << "  \"" << kv.first << "\": " << kv.second << ",\n";
    }

    json << "  \"clock\": {\"sm_clock_ghz\": " << r.sm_clock_ghz << "},\n"
         << "  \"measurement\": {\n";

    size_t m_idx = 0, m_total = r.measurement_str.size() + r.measurement_num.size();
    for (const auto& kv : r.measurement_str) {
        json << "    \"" << kv.first << "\": \"" << json_escape(kv.second) << "\""
             << (++m_idx < m_total ? ",\n" : "\n");
    }
    for (const auto& kv : r.measurement_num) {
        json << "    \"" << kv.first << "\": " << kv.second
             << (++m_idx < m_total ? ",\n" : "\n");
    }

    json << "  },\n"
         << "  \"cuda_version\": \"" << cuda_version_string(runtime_ver) << "\",\n"
         << "  \"driver_api_version\": \"" << cuda_version_string(driver_ver) << "\",\n"
         << "  \"driver_version\": \"" << json_escape(driver_version()) << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);
    std::filesystem::path out_path = std::filesystem::path(args.out_dir) / (r.parameter + "." + gpu + ".run.json");

    FILE* f = std::fopen(out_path.string().c_str(), "w");
    if (!f || std::fputs(json.str().c_str(), f) < 0 || std::fclose(f) != 0) {
        fprintf(stderr, "Failed writing results to %s\n", out_path.string().c_str());
        exit(EXIT_FAILURE);
    }

    printf("%s", json.str().c_str());
    fprintf(stderr, "Saved results to %s\n", out_path.string().c_str());
}
