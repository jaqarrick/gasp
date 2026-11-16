// common.h - Common fns used across microbenchmarks
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// SM cycle counter with volatile clock64
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory");
    return c;
}

// Nanoseconds since an arbitrary ~epoch~ independent of SM timer
__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

// Index of the SM the calling thread runs on.
__device__ __forceinline__ uint32_t read_smid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// 32-bit shared store that the compiler cannot remove or merge.
__device__ __forceinline__ void store_shared_volatile(uint32_t address, uint32_t value) {
    asm volatile("st.volatile.shared.u32 [%0], %1;"
                 :
                 : "r"(address), "r"(value)
                 : "memory");
}

// splitmix64 hash
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Fills buf[0..n_words) with splitmix64(seed + i) (grid-stride loop).
__global__ void fill_random_kernel(uint64_t* buf, size_t n_words, uint64_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_words; i += stride) {
        buf[i] = splitmix64(seed + i);
    }
}

// Spins for 2M SM cycles. out[0] = cycles elapsed, out[1] = globaltimer ns
// elapsed, so out[0] / out[1] is the SM clock in GHz.
__global__ void clock_kernel(unsigned long long* out) {
    const uint64_t g0 = read_globaltimer();
    const uint64_t c0 = read_clock64();
    uint64_t c = c0;
    while (c - c0 < 2000000) c = read_clock64();
    const uint64_t g1 = read_globaltimer();
    out[0] = c - c0;
    out[1] = g1 - g0;
}

// STATS:

// Median of v (mean of the middle two for an even count); 0 when v is empty.
static inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static inline double median_of(const double* values, size_t n) {
    return median_of(std::vector<double>(values, values + n));
}

struct Stats {
    double median, mean, std, min, max;
};

// Summary of v; std is the sample standard deviation (n - 1). v needs at
// least two values.
static inline Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    s.median = median_of(v);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.std = std::sqrt(var / (v.size() - 1));
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    return s;
}

static inline Stats compute_stats(const double* values, size_t n) {
    return compute_stats(std::vector<double>(values, values + n));
}

// Output of a shell command, trailing whitespace removed; "" if it can't run.
static inline std::string run_cmd(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    char buf[256];
    std::string s;
    while (fgets(buf, sizeof(buf), pipe)) s += buf;
    pclose(pipe);
    while (!s.empty() && isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static inline std::string driver_version() {
    const std::string v = run_cmd(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1");
    return v.empty() ? "unknown" : v;
}

static inline std::string git_commit() {
    const std::string c = run_cmd("git rev-parse --short HEAD 2>/dev/null");
    return c.empty() ? "unknown" : c;
}

// Get CUDA version e.g. "12.8".
static inline std::string cuda_version_string(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

// Architecture family for a compute capability.
static inline std::string architecture_name(int major, int minor) {
    switch (major) {
        case 7: return minor >= 5 ? "turing" : "volta";
        case 8: return minor == 9 ? "ada" : "ampere";
        case 9: return "hopper";
        case 10: return "blackwell";
        default: return "unknown";
    }
}

// Short key for a GPU name: a known data-center name if one appears in it
// ("NVIDIA A100-SXM4-40GB" -> "a100")
static inline std::string short_gpu_name(const std::string& full) {
    std::string lower;
    for (char c : full) lower += static_cast<char>(tolower(static_cast<unsigned char>(c)));
    for (const char* known : {"a100", "h100", "h200", "b200", "b100", "titan v"}) {
        if (lower.find(known) != std::string::npos) {
            std::string s = known;
            std::replace(s.begin(), s.end(), ' ', '_');
            return s;
        }
    }
    std::string s;
    for (char c : lower) s += isalnum(static_cast<unsigned char>(c)) ? c : '_';
    return s;
}

// Current UTC time, e.g. "2026-11-13T10:00:00Z".
static inline std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buf;
}

// Folder that holds the executable, taken from argv[0].
static inline std::string executable_dir(const char* argv0) {
    const char* slash = strrchr(argv0, '/');
    if (!slash) return ".";
    if (slash == argv0) return "/";
    return std::string(argv0, slash - argv0);
}


// Escapes '"' and '\' for a JSON string; control characters become spaces.
static inline std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += ' ';
        } else {
            out += c;
        }
    }
    return out;
}

// printf into a std::string.
static inline std::string strprintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static inline std::string strprintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int n = vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    std::string out(n > 0 ? n : 0, '\0');
    if (n > 0) vsnprintf(&out[0], n + 1, fmt, args);
    va_end(args);
    return out;
}

// x with a fixed number of decimals, the same text as printf("%.*f")
static inline std::string json_number(double x, int decimals = 4) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", decimals, x);
    return buf;
}

// JSON array of v, e.g. "[1, 2, 3]
// floating-point values with `decimals` decimals
template <class T>
static inline std::string json_array(const std::vector<T>& v, int decimals = 4) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        if constexpr (std::is_integral_v<T>) {
            out += std::to_string(v[i]);
        } else {
            out += json_number(static_cast<double>(v[i]), decimals);
        }
    }
    return out + "]";
}

template <class T>
static inline std::string json_array(const T* values, size_t n, int decimals = 4) {
    return json_array(std::vector<T>(values, values + n), decimals);
}

static inline std::string json_stats(const Stats& s, int decimals = 4) {
    return "  \"stats\": {\n"
           "    \"median\": " + json_number(s.median, decimals) + ",\n"
           "    \"mean\": " + json_number(s.mean, decimals) + ",\n"
           "    \"std\": " + json_number(s.std, decimals) + ",\n"
           "    \"min\": " + json_number(s.min, decimals) + ",\n"
           "    \"max\": " + json_number(s.max, decimals) + "\n"
           "  },\n";
}

// Writes json to <dir>/<parameter>.<gpu>.run.json, creating dir if needed
static inline std::string write_run_json(const std::string& dir, const std::string& parameter,
                                         const std::string& gpu, const std::string& json) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create %s: %s\n", dir.c_str(), ec.message().c_str());
        return "";
    }
    const std::string path = (fs::path(dir) / (parameter + "." + gpu + ".run.json")).string();
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        perror(path.c_str());
        return "";
    }
    const bool ok = fputs(json.c_str(), f) >= 0;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "Failed writing %s\n", path.c_str());
        return "";
    }
    return path;
}

// get Value param from JSON
static inline double read_run_value(const std::string& path, const std::string& parameter = "") {
    std::ifstream f(path);
    if (!f) return -1.0;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    if (!parameter.empty() &&
        s.find("\"parameter\": \"" + parameter + "\"") == std::string::npos)
        return -1.0;
    const size_t k = s.find("\"value\":");
    if (k == std::string::npos) return -1.0;
    return atof(s.c_str() + k + 8);
}
