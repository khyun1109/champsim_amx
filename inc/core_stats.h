#ifndef CORE_STATS_H
#define CORE_STATS_H

#include <cstdint>
#include <string>

#include "event_counter.h"
#include "instruction.h"

struct cpu_stats {
  std::string name;
  long long begin_instrs = 0;
  long long begin_cycles = 0;
  long long end_instrs = 0;
  uint64_t end_cycles = 0;
  uint64_t total_rob_occupancy_at_branch_mispredict = 0;

  // TAT metrics for Tile Admission Control
  uint64_t num_tile_groups_created = 0;
  uint64_t num_tile_children_buffered = 0;
  uint64_t num_tile_children_issued = 0;
  uint64_t max_tat_occupancy = 0;
  uint64_t cycles_tat_nonempty = 0;
  uint64_t tat_total_issue_span_cycles = 0;

  champsim::stats::event_counter<branch_type> total_branch_types = {};
  champsim::stats::event_counter<branch_type> branch_type_misses = {};

  [[nodiscard]] auto instrs() const { return end_instrs - begin_instrs; }
  [[nodiscard]] auto cycles() const { return end_cycles - begin_cycles; }
};

cpu_stats operator-(cpu_stats lhs, cpu_stats rhs);

#endif
