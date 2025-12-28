#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Args {
    int n = 0;
    int block = 256;
    int p = 1;
    int q = 1;
    int iters = 3;
    int runs = 1;
    double peak_tflops = 0.0;
    std::string hip_cmd = "./build/hip_cholesky --n {n} --iters {iters}";
    std::string roc_cmd = "./build/roc_cholesky --n {n} --iters {iters}";
    std::string scalapack_cmd =
        "mpirun -np {np} ./build/scalapack_cholesky --n {n} --nb {block} --p {p} --q {q} "
        "--iters {iters}";
    std::string out_jsonl = "output/bench_results.jsonl";
    std::string out_csv = "output/bench_results.csv";
};

struct CommandResult {
    int returncode = 0;
    double time_ms = 0.0;
    long memory_kb = -1;
    std::string stdout_text;
    std::string stderr_text;
};

struct Entry {
    std::string timestamp;
    std::string method;
    int n = 0;
    int block = 0;
    int p = 0;
    int q = 0;
    int iters = 0;
    int runs = 0;
    double time_ms = 0.0;
    double memory_usage_kb = -1.0;
    double theoretical_time_ms = -1.0;
    double performance_difference_pct = 0.0;
    bool perf_diff_valid = false;
};

std::string now_iso_utc() {
    std::time_t t = std::time(nullptr);
    std::tm gm = *std::gmtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gm);
    return std::string(buf);
}

bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string replace_all(std::string value, const std::string& key, const std::string& repl) {
    std::string token = "{" + key + "}";
    size_t pos = 0;
    while ((pos = value.find(token, pos)) != std::string::npos) {
        value.replace(pos, token.size(), repl);
        pos += repl.size();
    }
    return value;
}

std::string format_cmd(const std::string& templ, const Args& args) {
    std::string out = templ;
    out = replace_all(out, "n", std::to_string(args.n));
    out = replace_all(out, "block", std::to_string(args.block));
    out = replace_all(out, "p", std::to_string(args.p));
    out = replace_all(out, "q", std::to_string(args.q));
    out = replace_all(out, "iters", std::to_string(args.iters));
    out = replace_all(out, "np", std::to_string(args.p * args.q));
    return out;
}

double parse_time_ms_from_json(const std::string& text) {
    std::regex re("\"time_ms\"\\s*:\\s*([0-9]+(\\.[0-9]+)?)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return std::stod(m[1].str());
    }
    return -1.0;
}

CommandResult run_command(const std::string& command) {
    CommandResult result;

    char stdout_template[] = "/tmp/chol_stdoutXXXXXX";
    char stderr_template[] = "/tmp/chol_stderrXXXXXX";
    int stdout_fd = mkstemp(stdout_template);
    int stderr_fd = mkstemp(stderr_template);
    if (stdout_fd < 0 || stderr_fd < 0) {
        throw std::runtime_error("Failed to create temp files.");
    }

    auto start = std::chrono::steady_clock::now();
    pid_t pid = fork();
    if (pid == 0) {
        dup2(stdout_fd, STDOUT_FILENO);
        dup2(stderr_fd, STDERR_FILENO);
        close(stdout_fd);
        close(stderr_fd);
        execl("/bin/bash", "bash", "-lc", command.c_str(), (char*)nullptr);
        _exit(127);
    }

    int status = 0;
    struct rusage usage;
    if (wait4(pid, &status, 0, &usage) < 0) {
        throw std::runtime_error("wait4 failed.");
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    close(stdout_fd);
    close(stderr_fd);

    result.stdout_text = read_file(stdout_template);
    result.stderr_text = read_file(stderr_template);
    unlink(stdout_template);
    unlink(stderr_template);

    result.returncode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    result.time_ms = elapsed.count();
    result.memory_kb = usage.ru_maxrss;

    double parsed = parse_time_ms_from_json(result.stdout_text);
    if (parsed >= 0.0) {
        result.time_ms = parsed;
    }

    return result;
}

double average(const std::vector<double>& values) {
    if (values.empty()) {
        return -1.0;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

double theoretical_time_ms(int n, double peak_tflops) {
    if (peak_tflops <= 0.0) {
        return -1.0;
    }
    double flops = (static_cast<double>(n) * n * n) / 3.0;
    return (flops / (peak_tflops * 1e12)) * 1000.0;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            args.block = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            args.p = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--q") == 0 && i + 1 < argc) {
            args.q = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            args.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--peak-tflops") == 0 && i + 1 < argc) {
            args.peak_tflops = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--hip-cmd") == 0 && i + 1 < argc) {
            args.hip_cmd = argv[++i];
        } else if (std::strcmp(argv[i], "--roc-cmd") == 0 && i + 1 < argc) {
            args.roc_cmd = argv[++i];
        } else if (std::strcmp(argv[i], "--scalapack-cmd") == 0 && i + 1 < argc) {
            args.scalapack_cmd = argv[++i];
        } else if (std::strcmp(argv[i], "--out-jsonl") == 0 && i + 1 < argc) {
            args.out_jsonl = argv[++i];
        } else if (std::strcmp(argv[i], "--out-csv") == 0 && i + 1 < argc) {
            args.out_csv = argv[++i];
        }
    }
    if (args.n <= 0) {
        throw std::runtime_error("--n is required.");
    }
    return args;
}
}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Argument error: " << ex.what() << "\n";
        return 1;
    }

    std::vector<std::pair<std::string, std::string>> methods = {
        {"hipsolver", args.hip_cmd},
        {"rocsolver", args.roc_cmd},
        {"scalapack", args.scalapack_cmd},
    };

    std::vector<Entry> results;
    for (const auto& method : methods) {
        std::vector<double> run_times;
        std::vector<double> run_memories;
        for (int i = 0; i < args.runs; ++i) {
            std::string command = format_cmd(method.second, args);
            CommandResult outcome = run_command(command);
            if (outcome.returncode != 0) {
                std::cerr << method.first << " failed: " << command << "\n"
                          << outcome.stderr_text << "\n";
                return 2;
            }
            run_times.push_back(outcome.time_ms);
            if (outcome.memory_kb >= 0) {
                run_memories.push_back(static_cast<double>(outcome.memory_kb));
            }
        }

        Entry entry;
        entry.timestamp = now_iso_utc();
        entry.method = method.first;
        entry.n = args.n;
        entry.block = args.block;
        entry.p = args.p;
        entry.q = args.q;
        entry.iters = args.iters;
        entry.runs = args.runs;
        entry.time_ms = average(run_times);
        entry.memory_usage_kb = average(run_memories);
        entry.theoretical_time_ms = theoretical_time_ms(args.n, args.peak_tflops);
        results.push_back(entry);
    }

    Entry* scalapack = nullptr;
    for (auto& entry : results) {
        if (entry.method == "scalapack") {
            scalapack = &entry;
            break;
        }
    }
    if (scalapack) {
        for (auto& entry : results) {
            if (entry.method != "scalapack" && scalapack->time_ms > 0.0) {
                entry.performance_difference_pct =
                    ((entry.time_ms - scalapack->time_ms) / scalapack->time_ms) * 100.0;
                entry.perf_diff_valid = true;
            }
        }
    }

    std::ofstream jsonl(args.out_jsonl, std::ios::out | std::ios::app);
    if (!jsonl.good()) {
        std::cerr << "Failed to open " << args.out_jsonl << "\n";
        return 3;
    }

    for (const auto& entry : results) {
        jsonl << "{";
        jsonl << "\"timestamp\":\"" << entry.timestamp << "\",";
        jsonl << "\"method\":\"" << entry.method << "\",";
        jsonl << "\"n\":" << entry.n << ",";
        jsonl << "\"block\":" << entry.block << ",";
        jsonl << "\"p\":" << entry.p << ",";
        jsonl << "\"q\":" << entry.q << ",";
        jsonl << "\"iters\":" << entry.iters << ",";
        jsonl << "\"runs\":" << entry.runs << ",";
        jsonl << "\"time_ms\":" << entry.time_ms << ",";
        jsonl << "\"memory_usage_kb\":" << entry.memory_usage_kb << ",";
        jsonl << "\"memory_uasge_kb\":" << entry.memory_usage_kb << ",";
        jsonl << "\"theoretical_time_ms\":" << entry.theoretical_time_ms << ",";
        jsonl << "\"theoretical_time\":" << entry.theoretical_time_ms << ",";
        if (entry.perf_diff_valid) {
            jsonl << "\"performance_difference_pct\":" << entry.performance_difference_pct << ",";
            jsonl << "\"performance_difference\":" << entry.performance_difference_pct;
        } else {
            jsonl << "\"performance_difference_pct\":null,";
            jsonl << "\"performance_difference\":null";
        }
        jsonl << "}\n";
    }

    bool csv_exists = file_exists(args.out_csv);
    std::ofstream csv(args.out_csv, std::ios::out | std::ios::app);
    if (!csv.good()) {
        std::cerr << "Failed to open " << args.out_csv << "\n";
        return 4;
    }
    if (!csv_exists) {
        csv << "timestamp,method,n,block,p,q,iters,runs,time_ms,memory_usage_kb,"
               "memory_uasge_kb,theoretical_time_ms,theoretical_time,performance_difference_pct,"
               "performance_difference\n";
    }
    for (const auto& entry : results) {
        csv << entry.timestamp << ",";
        csv << entry.method << ",";
        csv << entry.n << ",";
        csv << entry.block << ",";
        csv << entry.p << ",";
        csv << entry.q << ",";
        csv << entry.iters << ",";
        csv << entry.runs << ",";
        csv << entry.time_ms << ",";
        csv << entry.memory_usage_kb << ",";
        csv << entry.memory_usage_kb << ",";
        csv << entry.theoretical_time_ms << ",";
        csv << entry.theoretical_time_ms << ",";
        if (entry.perf_diff_valid) {
            csv << entry.performance_difference_pct << ",";
            csv << entry.performance_difference_pct << "\n";
        } else {
            csv << ",";
            csv << "\n";
        }
    }

    std::cout << "{\"status\":\"ok\",\"results\":" << results.size() << "}\n";
    return 0;
}
