/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "champsim.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
#include <omp.h>
#include <fmt/chrono.h>
#include <fmt/core.h>

#include "cache.h"
#include "dram_controller.h"
#include "environment.h"
#include "ooo_cpu.h"
#include "operable.h"
#include "phase_info.h"
#include "ptw.h"
#include "tracereader.h"
#include "oracle_l1.h"

constexpr int DEADLOCK_CYCLE{50000};

const auto start_time = std::chrono::steady_clock::now();

std::chrono::seconds elapsed_time() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time); }

// ============================================================================
// Pre-classified operable groups for parallel execution
// ============================================================================
struct parallel_sim_state {
  // per_core_ops[core_id] = list of operables belonging to that core
  std::vector<std::vector<champsim::operable*>> per_core_ops;
  // shared operables (LLC, DRAM) that must run sequentially
  std::vector<champsim::operable*> shared_ops;
  unsigned num_threads = 1;
};

static void classify_operables(champsim::environment& env, parallel_sim_state& state)
{
  auto cpus = env.cpu_view();
  auto caches = env.cache_view();
  auto ptws = env.ptw_view();

  std::size_t num_cpus = cpus.size();
  state.per_core_ops.resize(num_cpus);

  // Classify CPUs
  for (auto& cpu_ref : cpus) {
    auto& cpu = cpu_ref.get();
    cpu.thread_group = static_cast<int>(cpu.cpu);
    state.per_core_ops[cpu.cpu].push_back(&cpu);
  }

  // Classify caches by parsing core id from NAME (e.g. "cpu0_L1D")
  for (auto& cache_ref : caches) {
    auto& cache = cache_ref.get();
    if (cache.NAME.size() >= 4 && cache.NAME.substr(0, 3) == "cpu" &&
        std::isdigit(cache.NAME[3])) {
      int core_id = 0;
      std::size_t i = 3;
      while (i < cache.NAME.size() && std::isdigit(cache.NAME[i])) {
        core_id = core_id * 10 + (cache.NAME[i] - '0');
        i++;
      }
      if (static_cast<std::size_t>(core_id) < num_cpus) {
        cache.thread_group = core_id;
        state.per_core_ops[core_id].push_back(&cache);
      } else {
        cache.thread_group = -1;
        state.shared_ops.push_back(&cache);
      }
    } else {
      cache.thread_group = -1;
      state.shared_ops.push_back(&cache);
    }
  }

  // Classify PTWs by parsing name "cpu0_PTW" -> core 0
  for (auto& ptw_ref : ptws) {
    auto& ptw = ptw_ref.get();
    auto pos = ptw.NAME.find("cpu");
    if (pos != std::string::npos && pos + 3 < ptw.NAME.size() &&
        std::isdigit(ptw.NAME[pos + 3])) {
      int core_id = 0;
      std::size_t i = pos + 3;
      while (i < ptw.NAME.size() && std::isdigit(ptw.NAME[i])) {
        core_id = core_id * 10 + (ptw.NAME[i] - '0');
        i++;
      }
      if (core_id >= 0 && static_cast<std::size_t>(core_id) < num_cpus) {
        ptw.thread_group = core_id;
        state.per_core_ops[core_id].push_back(&ptw);
      } else {
        ptw.thread_group = -1;
        state.shared_ops.push_back(&ptw);
      }
    } else {
      ptw.thread_group = -1;
      state.shared_ops.push_back(&ptw);
    }
  }

  // DRAM is always shared
  auto& dram = env.dram_view();
  dram.thread_group = -1;
  state.shared_ops.push_back(&dram);

  // Thread count
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 4;
  state.num_threads = std::min(hw, static_cast<unsigned>(num_cpus));
  if (const char* env_threads = std::getenv("CHAMPSIM_THREADS")) {
    state.num_threads = static_cast<unsigned>(std::atoi(env_threads));
    if (state.num_threads == 0) state.num_threads = 1;
  }

  // Set OpenMP thread count
  omp_set_num_threads(static_cast<int>(state.num_threads));

  fmt::print("[MT] Parallel simulation: {} simulated cores, {} OpenMP threads, {} shared operables\n",
             num_cpus, state.num_threads, state.shared_ops.size());
}

namespace champsim
{

long do_cycle(environment& env, std::vector<tracereader>& traces, std::vector<std::size_t> trace_index,
              champsim::chrono::clock& global_clock, parallel_sim_state& pstate)
{
  long progress = 0;
  const int num_cores = static_cast<int>(pstate.per_core_ops.size());

  // ---- Phase 1: Run per-core operables in parallel via OpenMP ----
  if (pstate.num_threads > 1) {
    #pragma omp parallel for reduction(+:progress) schedule(dynamic, 1)
    for (int c = 0; c < num_cores; c++) {
      for (auto* op : pstate.per_core_ops[c]) {
        progress += op->operate_on(global_clock);
      }
    }
  } else {
    for (int c = 0; c < num_cores; c++) {
      for (auto* op : pstate.per_core_ops[c]) {
        progress += op->operate_on(global_clock);
      }
    }
  }

  // ---- Phase 2: Run shared operables sequentially (LLC, DRAM) ----
  for (auto* op : pstate.shared_ops) {
    progress += op->operate_on(global_clock);
  }

  // Read from trace
  for (O3_CPU& cpu : env.cpu_view()) {
    auto& trace = traces.at(trace_index.at(cpu.cpu));
    // ORACLE L1 Hook: Feed future instructions to Oracle L1 up to its lookahead distance
    if (!cpu.warmup && g_oracle_l1[cpu.cpu].enabled()) {
      auto lookahead_limit = g_oracle_l1[cpu.cpu].config().lookahead_instructions;
      uint64_t cur_cycle = cpu.current_time.time_since_epoch() / cpu.clock_period;
      uint32_t count = 0;
      
      for (const auto& future_instr : cpu.input_queue) {
        if (count >= lookahead_limit) break;
        if (future_instr.instr_id > cpu.oracle_fed_instr_id) {
          g_oracle_l1[cpu.cpu].feed_future(future_instr, future_instr.instr_id, cur_cycle);
          cpu.oracle_fed_instr_id = future_instr.instr_id;
        }
        count++;
      }
    }
    long max_queue_size = cpu.IN_QUEUE_SIZE;
    if (!cpu.warmup && g_oracle_l1[cpu.cpu].enabled()) {
      long oracle_lookahead = static_cast<long>(g_oracle_l1[cpu.cpu].config().lookahead_instructions);
      if (oracle_lookahead > max_queue_size) {
        max_queue_size = oracle_lookahead;
      }
    }
    for (auto pkt_count = max_queue_size - static_cast<long>(std::size(cpu.input_queue)); !trace.eof() && pkt_count > 0; --pkt_count) {
      cpu.input_queue.push_back(trace());
    }
  }

  return progress;
}

phase_stats do_phase(const phase_info& phase, environment& env, std::vector<tracereader>& traces,
                     champsim::chrono::clock& global_clock, parallel_sim_state& pstate)
{
  auto operables = env.operable_view();
  auto [phase_name, is_warmup, length, trace_index, trace_names] = phase;

  // Initialize phase
  for (champsim::operable& op : operables) {
    op.warmup = is_warmup;
    op.begin_phase();
  }

  const auto time_quantum = std::accumulate(std::cbegin(operables), std::cend(operables), champsim::chrono::clock::duration::max(),
                                            [](const auto acc, const operable& y) { return std::min(acc, y.clock_period); });

  bool livelock_trigger{false};
  uint64_t livelock_period{10000000};
  uint64_t livelock_timer{0};
  //                                   die | critical | warning
  std::vector<double> livelock_threshold{0.01, 0.02, 0.05};
  std::vector<uint64_t> livelock_instr(std::size(env.cpu_view()), 0);

  // Perform phase
  int stalled_cycle{0};
  std::vector<bool> phase_complete(std::size(env.cpu_view()), false);
  while (!std::accumulate(std::begin(phase_complete), std::end(phase_complete), true, std::logical_and{})) {
    auto next_phase_complete = phase_complete;
    global_clock.tick(time_quantum);

    auto progress = do_cycle(env, traces, trace_index, global_clock, pstate);

    if (progress == 0) {
      ++stalled_cycle;
    } else {
      stalled_cycle = 0;
    }

    // Livelock detect, every livelock_period cycles, check progress and alert the user
    // Skip cores whose trace has ended — they are naturally idle with IPC=0.
    livelock_timer++;
    if (livelock_timer >= livelock_period) {
      // for each cpu
      for (O3_CPU& cpu : env.cpu_view()) {
        // Skip livelock check for cores whose trace has ended
        auto& trace = traces.at(trace_index.at(cpu.cpu));
        if (trace.eof()) {
          livelock_instr[cpu.cpu] = cpu.sim_instr();
          continue;
        }
        // for each threshold
        for (auto thres = std::begin(livelock_threshold); thres != std::end(livelock_threshold); thres++) {
          double livelock_ipc = std::ceil(cpu.sim_instr() - livelock_instr[cpu.cpu]) / std::ceil(livelock_period);
          if (livelock_ipc <= *thres) {
            if (std::distance(std::begin(livelock_threshold), thres) == 0) {
              livelock_trigger = true;
              fmt::print("{} CPU {} panic: IPC {:.5g} < {:.5g}\n", phase_name, cpu.cpu, livelock_ipc, *thres);
            } else if (std::distance(std::begin(livelock_threshold), thres) == 1)
              fmt::print("{} CPU {} critical: IPC {:.5g} < {:.5g}\n", phase_name, cpu.cpu, livelock_ipc, *thres);
            else
              fmt::print("{} CPU {} warning: IPC {:.5g} < {:.5g}\n", phase_name, cpu.cpu, livelock_ipc, *thres);

            break;
          }
        }
        livelock_instr[cpu.cpu] = cpu.sim_instr();
      }
      livelock_timer = 0;
    }

    if (stalled_cycle >= DEADLOCK_CYCLE || livelock_trigger) {
      std::for_each(std::begin(operables), std::end(operables), [](champsim::operable& c) { c.print_deadlock(); });
      abort();
    }

    // Individual core EOF handling is done below in the per-core phase_complete check.
    // Do NOT terminate all cores when any single trace reaches EOF — some cores may
    // have shorter traces (e.g., small GEMM shapes with fewer threads than cores).

    // Check for phase finish
    for (O3_CPU& cpu : env.cpu_view()) {
      // Phase complete: either reached instruction target, or trace ended and pipeline drained
      auto& trace = traces.at(trace_index.at(cpu.cpu));
      bool trace_done = trace.eof() && std::empty(cpu.input_queue)
          && std::empty(cpu.IFETCH_BUFFER) && std::empty(cpu.DECODE_BUFFER)
          && std::empty(cpu.DISPATCH_BUFFER) && std::empty(cpu.ROB);
      next_phase_complete[cpu.cpu] = next_phase_complete[cpu.cpu] || (cpu.sim_instr() >= length) || trace_done;
    }

    for (O3_CPU& cpu : env.cpu_view()) {
      if (next_phase_complete[cpu.cpu] != phase_complete[cpu.cpu]) {
        for (champsim::operable& op : operables) {
          op.end_phase(cpu.cpu);
        }

        fmt::print("{} finished CPU {} instructions: {} cycles: {} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", phase_name, cpu.cpu,
                   cpu.sim_instr(), cpu.sim_cycle(), std::ceil(cpu.sim_instr()) / std::ceil(cpu.sim_cycle()), elapsed_time());
      }
    }

    phase_complete = next_phase_complete;
  }

  for (O3_CPU& cpu : env.cpu_view()) {
    fmt::print("{} complete CPU {} instructions: {} cycles: {} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", phase_name, cpu.cpu,
               cpu.sim_instr(), cpu.sim_cycle(), std::ceil(cpu.sim_instr()) / std::ceil(cpu.sim_cycle()), elapsed_time());
  }

  phase_stats stats;
  stats.name = phase.name;

  for (std::size_t i = 0; i < std::size(trace_index); ++i) {
    stats.trace_names.push_back(trace_names.at(trace_index.at(i)));
  }

  auto cpus = env.cpu_view();
  std::transform(std::begin(cpus), std::end(cpus), std::back_inserter(stats.sim_cpu_stats), [](const O3_CPU& cpu) { return cpu.sim_stats; });
  std::transform(std::begin(cpus), std::end(cpus), std::back_inserter(stats.roi_cpu_stats), [](const O3_CPU& cpu) { return cpu.roi_stats; });

  auto caches = env.cache_view();
  std::transform(std::begin(caches), std::end(caches), std::back_inserter(stats.sim_cache_stats), [](const CACHE& cache) { return cache.sim_stats; });
  std::transform(std::begin(caches), std::end(caches), std::back_inserter(stats.roi_cache_stats), [](const CACHE& cache) { return cache.roi_stats; });

  auto dram = env.dram_view();
  std::transform(std::begin(dram.channels), std::end(dram.channels), std::back_inserter(stats.sim_dram_stats),
                 [](const DRAM_CHANNEL& chan) { return chan.sim_stats; });
  std::transform(std::begin(dram.channels), std::end(dram.channels), std::back_inserter(stats.roi_dram_stats),
                 [](const DRAM_CHANNEL& chan) { return chan.roi_stats; });

  return stats;
}

// simulation entry point
std::vector<phase_stats> main(environment& env, std::vector<phase_info>& phases, std::vector<tracereader>& traces)
{
  for (champsim::operable& op : env.operable_view()) {
    op.initialize();
  }

  parallel_sim_state pstate;
  classify_operables(env, pstate);

  champsim::chrono::clock global_clock;
  std::vector<phase_stats> results;
  for (auto phase : phases) {
    auto stats = do_phase(phase, env, traces, global_clock, pstate);
    if (!phase.is_warmup) {
      results.push_back(stats);
    }
  }

  return results;
}
} // namespace champsim
