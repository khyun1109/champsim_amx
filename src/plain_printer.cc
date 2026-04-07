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

#include <cmath>
#include <numeric>
#include <ratio>
#include <string_view> // for string_view
#include <utility>
#include <vector>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include "stats_printer.h"
#include "tile_record.h"

namespace
{
template <typename N, typename D>
auto print_ratio(N num, D denom)
{
  if (denom > 0) {
    return fmt::format("{:.4g}", std::ceil(num) / std::ceil(denom));
  }
  return std::string{"-"};
}
} // namespace

std::vector<std::string> champsim::plain_printer::format(O3_CPU::stats_type stats)
{
  constexpr std::array types{branch_type::BRANCH_DIRECT_JUMP, branch_type::BRANCH_INDIRECT,      branch_type::BRANCH_CONDITIONAL,
                             branch_type::BRANCH_DIRECT_CALL, branch_type::BRANCH_INDIRECT_CALL, branch_type::BRANCH_RETURN};
  auto total_branch = std::ceil(
      std::accumulate(std::begin(types), std::end(types), 0LL, [tbt = stats.total_branch_types](auto acc, auto next) { return acc + tbt.value_or(next, 0); }));
  auto total_mispredictions = std::ceil(
      std::accumulate(std::begin(types), std::end(types), 0LL, [btm = stats.branch_type_misses](auto acc, auto next) { return acc + btm.value_or(next, 0); }));

  std::vector<std::string> lines{};
  lines.push_back(fmt::format("{} cumulative IPC: {} instructions: {} cycles: {}", stats.name, ::print_ratio(stats.instrs(), stats.cycles()), stats.instrs(),
                              stats.cycles()));

  lines.push_back(fmt::format("{} Branch Prediction Accuracy: {}% MPKI: {} Average ROB Occupancy at Mispredict: {}", stats.name,
                              ::print_ratio(100 * (total_branch - total_mispredictions), total_branch),
                              ::print_ratio(std::kilo::num * total_mispredictions, stats.instrs()),
                              ::print_ratio(stats.total_rob_occupancy_at_branch_mispredict, total_mispredictions)));

  lines.emplace_back("Branch type MPKI");
  for (auto idx : types) {
    lines.push_back(fmt::format("{}: {}", branch_type_names.at(champsim::to_underlying(idx)),
                                ::print_ratio(std::kilo::num * stats.branch_type_misses.value_or(idx, 0), stats.instrs())));
  }

  return lines;
}

std::vector<std::string> champsim::plain_printer::format(CACHE::stats_type stats)
{
  using hits_value_type = typename decltype(stats.hits)::value_type;
  using misses_value_type = typename decltype(stats.misses)::value_type;
  using mshr_merge_value_type = typename decltype(stats.mshr_merge)::value_type;
  using mshr_return_value_type = typename decltype(stats.mshr_return)::value_type;

  std::vector<std::size_t> cpus;

  // build a vector of all existing cpus
  auto stat_keys = {stats.hits.get_keys(), stats.misses.get_keys(), stats.mshr_merge.get_keys(), stats.mshr_return.get_keys()};
  for (auto keys : stat_keys) {
    std::transform(std::begin(keys), std::end(keys), std::back_inserter(cpus), [](auto val) { return val.second; });
  }
  std::sort(std::begin(cpus), std::end(cpus));
  auto uniq_end = std::unique(std::begin(cpus), std::end(cpus));
  cpus.erase(uniq_end, std::end(cpus));

  for (const auto type : {access_type::LOAD, access_type::RFO, access_type::PREFETCH, access_type::WRITE, access_type::TRANSLATION}) {
    for (auto cpu : cpus) {
      stats.hits.allocate(std::pair{type, cpu});
      stats.misses.allocate(std::pair{type, cpu});
      stats.mshr_merge.allocate(std::pair{type, cpu});
      stats.mshr_return.allocate(std::pair{type, cpu});
    }
  }

  std::vector<std::string> lines{};
  for (auto cpu : cpus) {
    hits_value_type total_hits = 0;
    misses_value_type total_misses = 0;
    mshr_merge_value_type total_mshr_merge = 0;
    mshr_return_value_type total_mshr_return = 0;
    for (const auto type : {access_type::LOAD, access_type::RFO, access_type::PREFETCH, access_type::WRITE, access_type::TRANSLATION}) {
      total_hits += stats.hits.value_or(std::pair{type, cpu}, hits_value_type{});
      total_misses += stats.misses.value_or(std::pair{type, cpu}, misses_value_type{});
      total_mshr_merge += stats.mshr_merge.value_or(std::pair{type, cpu}, mshr_merge_value_type{});
      total_mshr_return += stats.mshr_return.value_or(std::pair{type, cpu}, mshr_merge_value_type{});
    }

    fmt::format_string<std::string_view, std::string_view, int, int, int> hitmiss_fmtstr{
        "cpu{}->{} {:<12s} ACCESS: {:10d} HIT: {:10d} MISS: {:10d} MSHR_MERGE: {:10d}"};
    lines.push_back(fmt::format(hitmiss_fmtstr, cpu, stats.name, "TOTAL", total_hits + total_misses, total_hits, total_misses, total_mshr_merge));
    for (const auto type : {access_type::LOAD, access_type::RFO, access_type::PREFETCH, access_type::WRITE, access_type::TRANSLATION}) {
      lines.push_back(
          fmt::format(hitmiss_fmtstr, cpu, stats.name, access_type_names.at(champsim::to_underlying(type)),
                      stats.hits.value_or(std::pair{type, cpu}, hits_value_type{}) + stats.misses.value_or(std::pair{type, cpu}, misses_value_type{}),
                      stats.hits.value_or(std::pair{type, cpu}, hits_value_type{}), stats.misses.value_or(std::pair{type, cpu}, misses_value_type{}),
                      stats.mshr_merge.value_or(std::pair{type, cpu}, mshr_merge_value_type{})));
    }

    lines.push_back(fmt::format("cpu{}->{} PREFETCH REQUESTED: {:10} ISSUED: {:10} USEFUL: {:10} USELESS: {:10}", cpu, stats.name, stats.pf_requested,
                                stats.pf_issued, stats.pf_useful, stats.pf_useless));

    uint64_t total_downstream_demands = total_mshr_return - stats.mshr_return.value_or(std::pair{access_type::PREFETCH, cpu}, mshr_return_value_type{});
    lines.push_back(
        fmt::format("cpu{}->{} AVERAGE MISS LATENCY: {} cycles AVERAGE WAIT QUEUE: {} cycles", cpu, stats.name, ::print_ratio(stats.total_miss_latency_cycles, total_downstream_demands), ::print_ratio(stats.total_miss_queue_cycles, total_downstream_demands)));

    lines.push_back(fmt::format("cpu{}->{} ARBITRATION STALLS MSHR_FULL: {:10} L2_RQ_FULL: {:10}", cpu, stats.name, 
        stats.l1_mshr_full_stall_cycles, stats.l2_rq_full_stall_cycles));

    lines.push_back(fmt::format("cpu{}->{} QUEUE ARBITRATION  RQ_ACCESS: {:10}  RQ_FULL: {:10}  PQ_ACCESS: {:10}  PQ_FULL: {:10}  WQ_ACCESS: {:10}  WQ_FULL: {:10}", cpu, stats.name,
        stats.rq_access, stats.rq_full, stats.pq_access, stats.pq_full, stats.wq_access, stats.wq_full));

    // Queue-Local Tile Sidecar Stats
    if (stats.tile_parent_entry_alloc > 0) {
      lines.push_back(fmt::format("cpu{}->{} TILE SIDECAR ALLOC: {:10} PEAK: {:10} COMPLETIONS: {:10}", cpu, stats.name,
          stats.tile_parent_entry_alloc, stats.tile_parent_entry_peak, stats.tile_completions));

      lines.push_back(fmt::format("cpu{}->{} TILE CHILDREN TOTAL: {:10} HITS: {:10} MISSES: {:10} FILL_COMPLETE: {:10}", cpu, stats.name,
          stats.tile_children_total, stats.tile_children_hits, stats.tile_children_pending, stats.tile_children_issued));

      double hit_rate = stats.tile_children_total > 0 ?
          100.0 * stats.tile_children_hits / stats.tile_children_total : 0.0;
      lines.push_back(fmt::format("cpu{}->{} TILE HIT RATE: {:.1f}% ACTIVE CYCLES: {:10} ATTACHMENT MERGES: {:10}", cpu, stats.name,
          hit_rate, stats.total_cycles_with_active_parents, stats.tile_attachment_merges));

      // Always print sidecar coalescing stats for visibility
      lines.push_back(fmt::format("cpu{}->{} TILE SIDECAR COALESCED: {:10} FILLS_MATCHED: {:10}", cpu, stats.name,
          stats.tile_sidecar_coalesced, stats.tile_sidecar_fills_matched));

      // Per-child-index hit/miss breakdown (A tiles: tmm4/5, B tiles: tmm6/7)
      {
        auto print_child_row = [&](const char* label, const uint64_t* hits, const uint64_t* misses) {
          std::string h_str, m_str;
          uint64_t total_h = 0, total_m = 0;
          for (int i = 0; i < 16; i++) {
            h_str += fmt::format("{:>6}", hits[i]);
            m_str += fmt::format("{:>6}", misses[i]);
            total_h += hits[i]; total_m += misses[i];
          }
          if (total_h + total_m > 0) {
            lines.push_back(fmt::format("cpu{}->{} {} HIT  [{}] total={}", cpu, stats.name, label, h_str, total_h));
            lines.push_back(fmt::format("cpu{}->{} {} MISS [{}] total={}", cpu, stats.name, label, m_str, total_m));
            lines.push_back(fmt::format("cpu{}->{} {} RATE  hit={:.1f}%", cpu, stats.name, label,
                100.0 * total_h / (total_h + total_m)));
          }
        };
        print_child_row("A_TILE_CHILD", stats.a_tile_child_hit, stats.a_tile_child_miss);
        print_child_row("B_TILE_CHILD", stats.b_tile_child_hit, stats.b_tile_child_miss);
      }

    }

    // L1D MSHR occupancy diagnostics (unified LFB model)
    if (stats.lfb_occupancy_cycles > 0) {
        double avg_occ = static_cast<double>(stats.lfb_occupancy_sum) / stats.lfb_occupancy_cycles;
        double avg_nt_mshr = static_cast<double>(stats.non_tile_mshr_occupancy_sum) / stats.lfb_occupancy_cycles;
        double avg_t_mshr = static_cast<double>(stats.tile_mshr_occupancy_sum) / stats.lfb_occupancy_cycles;
        lines.push_back(fmt::format("cpu{}->{} L1D_MSHR OCCUPANCY avg: {:.2f} / {} (non_tile: {:.2f}  tile_parent: {:.2f})",
            cpu, stats.name, avg_occ, stats.lfb_occupancy_max, avg_nt_mshr, avg_t_mshr));
        lines.push_back(fmt::format("cpu{}->{} L1D_MSHR MAX       total: {}  non_tile: {}  tile_parent: {}",
            cpu, stats.name, stats.lfb_occupancy_max, stats.non_tile_mshr_occupancy_max, stats.tile_mshr_occupancy_max));
        lines.push_back(fmt::format("cpu{}->{} TAG CHECK tile_loads: {} scalar_loads: {} (scalar {:.1f}%)",
            cpu, stats.name, stats.tile_loads_tag_check, stats.scalar_loads_tag_check,
            (stats.tile_loads_tag_check + stats.scalar_loads_tag_check > 0)
              ? 100.0 * stats.scalar_loads_tag_check / (stats.tile_loads_tag_check + stats.scalar_loads_tag_check) : 0.0));
        lines.push_back(fmt::format("cpu{}->{} MSHR ALLOCS non_tile: {} tile: {} translation: {} tile_LEAKED: {}",
            cpu, stats.name, stats.non_tile_mshr_allocs, stats.tile_mshr_allocs, stats.translation_mshr_allocs, stats.tile_leaked_to_nontile));
        lines.push_back(fmt::format("cpu{}->{} MSHR NON-TILE BY TYPE  LOAD: {} RFO: {} WRITE: {} PREFETCH: {} OTHER: {}",
            cpu, stats.name, stats.non_tile_mshr_by_load, stats.non_tile_mshr_by_rfo, stats.non_tile_mshr_by_write,
            stats.non_tile_mshr_by_prefetch, stats.non_tile_mshr_by_other));
    }
  }

  return lines;
}

std::vector<std::string> champsim::plain_printer::format(DRAM_CHANNEL::stats_type stats)
{
  std::vector<std::string> lines{};
  lines.push_back(fmt::format("{} RQ ROW_BUFFER_HIT: {:10}", stats.name, stats.RQ_ROW_BUFFER_HIT));
  lines.push_back(fmt::format("  ROW_BUFFER_MISS: {:10}", stats.RQ_ROW_BUFFER_MISS));
  lines.push_back(fmt::format("  AVG DBUS CONGESTED CYCLE: {}", ::print_ratio(stats.dbus_cycle_congested, stats.dbus_count_congested)));
  lines.push_back(fmt::format("{} WQ ROW_BUFFER_HIT: {:10}", stats.name, stats.WQ_ROW_BUFFER_HIT));
  lines.push_back(fmt::format("  ROW_BUFFER_MISS: {:10}", stats.WQ_ROW_BUFFER_MISS));
  lines.push_back(fmt::format("  FULL: {:10}", stats.WQ_FULL));
  lines.push_back(fmt::format("{} TILELOAD RQ ROW_BUFFER_HIT: {:10}", stats.name, stats.RQ_TILELOAD_ROW_BUFFER_HIT));
  lines.push_back(fmt::format("  TILELOAD ROW_BUFFER_MISS: {:10}", stats.RQ_TILELOAD_ROW_BUFFER_MISS));

  if (stats.refresh_cycles > 0)
    lines.push_back(fmt::format("{} REFRESHES ISSUED: {:10}", stats.name, stats.refresh_cycles));
  else
    lines.push_back(fmt::format("{} REFRESHES ISSUED: -", stats.name));

  return lines;
}

void champsim::plain_printer::print(champsim::phase_stats& stats)
{
  auto lines = format(stats);
  std::copy(std::begin(lines), std::end(lines), std::ostream_iterator<std::string>(stream, "\n"));
}

std::vector<std::string> champsim::plain_printer::format(champsim::phase_stats& stats)
{
  std::vector<std::string> lines{};
  lines.push_back(fmt::format("=== {} ===", stats.name));

  int i = 0;
  for (auto tn : stats.trace_names) {
    lines.push_back(fmt::format("CPU {} runs {}", i++, tn));
  }

  if (NUM_CPUS > 1) {
    lines.emplace_back("");
    lines.emplace_back("Total Simulation Statistics (not including warmup)");

    for (const auto& stat : stats.sim_cpu_stats) {
      auto sublines = format(stat);
      lines.emplace_back("");
      std::move(std::begin(sublines), std::end(sublines), std::back_inserter(lines));
      lines.emplace_back("");
    }

    for (const auto& stat : stats.sim_cache_stats) {
      auto sublines = format(stat);
      std::move(std::begin(sublines), std::end(sublines), std::back_inserter(lines));
    }
  }

  lines.emplace_back("");
  lines.emplace_back("Region of Interest Statistics");

  for (const auto& stat : stats.roi_cpu_stats) {
    auto sublines = format(stat);
    lines.emplace_back("");
    std::move(std::begin(sublines), std::end(sublines), std::back_inserter(lines));
    lines.emplace_back("");
  }

  for (const auto& stat : stats.roi_cache_stats) {
    auto sublines = format(stat);
    std::move(std::begin(sublines), std::end(sublines), std::back_inserter(lines));
  }

  lines.emplace_back("");
  lines.emplace_back("DRAM Statistics");
  for (const auto& stat : stats.roi_dram_stats) {
    auto sublines = format(stat);
    lines.emplace_back("");
    std::move(std::begin(sublines), std::end(sublines), std::back_inserter(lines));
  }

  return lines;
}

void champsim::plain_printer::print(std::vector<phase_stats>& stats)
{
  for (auto p : stats) {
    print(p);
  }
  // Print AMX tile summary once after all phases
  tile_print_summary(stream);
}
