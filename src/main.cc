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

#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "cache.h" // for CACHE
#include "champsim.h"
#ifndef CHAMPSIM_TEST_BUILD
#include "core_inst.inc"
#endif
#include "defaults.hpp"
#include "environment.h"
#include "ooo_cpu.h" // for O3_CPU
#include "phase_info.h"
#include "stats_printer.h"
#include "tracereader.h"
#include "vmem.h"
#include "ptw.h"
#include "tile_record.h"
#include "oracle_l1.h"

namespace champsim
{
std::vector<phase_stats> main(environment& env, std::vector<phase_info>& phases, std::vector<tracereader>& traces);
}

#ifndef CHAMPSIM_TEST_BUILD
using configured_environment = champsim::configured::generated_environment<CHAMPSIM_BUILD>;

const std::size_t NUM_CPUS = configured_environment::num_cpus;

const unsigned BLOCK_SIZE = configured_environment::block_size;
const unsigned PAGE_SIZE = configured_environment::page_size;
#endif
const unsigned LOG2_BLOCK_SIZE = champsim::lg2(BLOCK_SIZE);
const unsigned LOG2_PAGE_SIZE = champsim::lg2(PAGE_SIZE);

#ifndef CHAMPSIM_TEST_BUILD
int main(int argc, char** argv) // NOLINT(bugprone-exception-escape)
{
  configured_environment gen_environment{};

  CLI::App app{"A microarchitecture simulator for research and education"};

  bool knob_cloudsuite{false};
  long long warmup_instructions = 0;
  long long simulation_instructions = std::numeric_limits<long long>::max();
  std::string json_file_name;
  std::string oracle_json_file;
  std::vector<std::string> trace_names;

  auto set_heartbeat_callback = [&](auto) {
    for (O3_CPU& cpu : gen_environment.cpu_view()) {
      cpu.show_heartbeat = false;
    }
  };

  app.add_flag("-c,--cloudsuite", knob_cloudsuite, "Read all traces using the cloudsuite format");
  app.add_flag("--hide-heartbeat", set_heartbeat_callback, "Hide the heartbeat output");
  auto* warmup_instr_option = app.add_option("-w,--warmup-instructions", warmup_instructions, "The number of instructions in the warmup phase");
  auto* deprec_warmup_instr_option =
      app.add_option("--warmup_instructions", warmup_instructions, "[deprecated] use --warmup-instructions instead")->excludes(warmup_instr_option);
  auto* sim_instr_option = app.add_option("-i,--simulation-instructions", simulation_instructions,
                                          "The number of instructions in the detailed phase. If not specified, run to the end of the trace.");
  auto* deprec_sim_instr_option =
      app.add_option("--simulation_instructions", simulation_instructions, "[deprecated] use --simulation-instructions instead")->excludes(sim_instr_option);

  auto* json_option =
      app.add_option("--json", json_file_name, "The name of the file to receive JSON output. If no name is specified, stdout will be used")->expected(0, 1);

  app.add_option("--oracle-config", oracle_json_file, "Path to a JSON file containing Oracle L1 config settings.");
  app.add_flag("--no-tile-sidecar", g_disable_tile_sidecar, "Disable tile sidecar: AMX tileloads use individual cacheline requests (baseline mode)");
  app.add_flag("--tile-lq-coalesce", g_enable_tile_lq_coalesce, "Coalesce tile loads in LQ: 1 LQ entry per tileload (16 children tracked internally)");
  app.add_flag("--no-skip-prefetch,!--skip-prefetch", g_skip_sw_prefetch, "Control SW prefetch LQ skip (default: skip). Use --no-skip-prefetch to let prefetches enter LQ");
  app.add_flag("--per-child-lfb", g_per_child_lfb, "Each tile child consumes its own MSHR/LFB entry (realistic HW pressure)");
  app.add_flag("--tile-lru-insert", g_tile_lru_insert, "Insert tile data at LRU position in L1D to prevent cache pollution");
  app.add_flag("--warm-scalar-llc", g_warm_scalar_llc, "Force non-tile LLC misses to hit (simulates warm scalar cache from prior GEMM iterations)");
  int tmm_budget_val = 1;
  app.add_option("--tmm-budget", tmm_budget_val, "TMM rename budget (extra physical TMM registers beyond arch 8). Default=1");

  app.add_option("traces", trace_names, "The paths to the traces")->required()->expected(NUM_CPUS)->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  // Apply TMM budget
  O3_CPU::TMM_EXTRA_PHYS = tmm_budget_val;
  O3_CPU::TMM_TOTAL_PHYS = 8 + tmm_budget_val;

  const bool warmup_given = (warmup_instr_option->count() > 0) || (deprec_warmup_instr_option->count() > 0);
  const bool simulation_given = (sim_instr_option->count() > 0) || (deprec_sim_instr_option->count() > 0);

  if (deprec_warmup_instr_option->count() > 0) {
    fmt::print("WARNING: option --warmup_instructions is deprecated. Use --warmup-instructions instead.\n");
  }

  if (deprec_sim_instr_option->count() > 0) {
    fmt::print("WARNING: option --simulation_instructions is deprecated. Use --simulation-instructions instead.\n");
  }

  if (simulation_given && !warmup_given) {
    // Warmup is 20% by default
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    warmup_instructions = simulation_instructions / 5;
  }

  std::vector<champsim::tracereader> traces;
  std::transform(
      std::begin(trace_names), std::end(trace_names), std::back_inserter(traces),
      [knob_cloudsuite, repeat = simulation_given, i = uint8_t(0)](auto name) mutable { return get_tracereader(name, i++, knob_cloudsuite, repeat); });


  // -------------------------------------------------------------------------
  // Initialize Oracle L1 Configuration
  // -------------------------------------------------------------------------
  if (!oracle_json_file.empty()) {
    try {
      std::ifstream ifs(oracle_json_file);
      if (ifs.good()) {
        nlohmann::json j;
        ifs >> j;
        if (j.contains("oracle_l1")) {
          auto oracle_j = j["oracle_l1"];
          OracleConfig ocfg;
          ocfg.enabled = oracle_j.value("enabled", false);
          ocfg.filter  = OracleConfig::filter_from_string(oracle_j.value("filter", "oracle_all_reads"));
          ocfg.lookahead_instructions = oracle_j.value("lookahead_instructions", 4096);
          ocfg.capacity_lines = oracle_j.value("capacity_lines", 0);
          ocfg.hit_latency_cycles = oracle_j.value("hit_latency_cycles", 1);
          ocfg.count_as_true_l1 = oracle_j.value("count_as_true_l1", false);
          
          for (size_t i = 0; i < NUM_CPUS; ++i) {
            g_oracle_l1[i].configure(ocfg);
          }
          fmt::print("Loaded Oracle L1 config from {}: enabled={}\n", oracle_json_file, ocfg.enabled);
        }
      } else {
        fmt::print("WARNING: Could not open --oracle-config file {}\n", oracle_json_file);
      }
    } catch (const std::exception& e) {
      fmt::print("ERROR: Failed to parse oracle JSON: {}\n", e.what());
    }
  }

  std::vector<champsim::phase_info> phases{
      {champsim::phase_info{"Warmup", true, warmup_instructions, std::vector<std::size_t>(std::size(trace_names), 0), trace_names},
       champsim::phase_info{"Simulation", false, simulation_instructions, std::vector<std::size_t>(std::size(trace_names), 0), trace_names}}};

  for (auto& p : phases) {
    std::iota(std::begin(p.trace_index), std::end(p.trace_index), 0);
  }

  fmt::print("\n*** ChampSim Multicore Out-of-Order Simulator ***\nWarmup Instructions: {}\nSimulation Instructions: {}\nNumber of CPUs: {}\nPage size: {}\n",
             phases.at(0).length, phases.at(1).length, std::size(gen_environment.cpu_view()), PAGE_SIZE);
  fmt::print("Tile sidecar: {}\n", g_disable_tile_sidecar ? "DISABLED (baseline cacheline mode)" : "ENABLED (parent tile mode)");
  fmt::print("Tile LQ coalesce: {}\n", g_enable_tile_lq_coalesce ? "ENABLED" : "DISABLED");
  fmt::print("SW prefetch LQ skip: {}\n", g_skip_sw_prefetch ? "ON (prefetches skip LQ)" : "OFF (prefetches enter LQ)");
  if (g_per_child_lfb) fmt::print("Per-child LFB: ENABLED (each tile child uses its own MSHR/LFB entry)\n");
  if (g_tile_lru_insert) fmt::print("Tile LRU insert: ENABLED (tile data inserted at LRU position in L1D)\n");
  if (g_warm_scalar_llc) fmt::print("Warm scalar LLC: ENABLED (non-tile LLC misses forced to hit)\n");
  fmt::print("\n");

  auto phase_stats = champsim::main(gen_environment, phases, traces);

  fmt::print("\nChampSim completed all CPUs\n\n");

  champsim::plain_printer{std::cout}.print(phase_stats);

  for (CACHE& cache : gen_environment.cache_view()) {
    cache.impl_prefetcher_final_stats();
  }

  for (CACHE& cache : gen_environment.cache_view()) {
    cache.impl_replacement_final_stats();
  }

  if (json_option->count() > 0) {
    if (json_file_name.empty()) {
      champsim::json_printer{std::cout}.print(phase_stats);
    } else {
      std::ofstream json_file{json_file_name};
      champsim::json_printer{json_file}.print(phase_stats);
    }
  }

  // Generate AMX tile traces if debug is enabled
  tile_csv_flush();

  // Dump Oracle L1 Statistics
  for (size_t i = 0; i < NUM_CPUS; ++i) {
    g_oracle_l1[i].dump_stats(std::cout);
  }

  return 0;
}
#endif
