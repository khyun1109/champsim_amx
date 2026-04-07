#ifndef CACHE_STATS_H
#define CACHE_STATS_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "channel.h"
#include "event_counter.h"

struct cache_stats {
  std::string name;
  // prefetch stats
  uint64_t pf_requested = 0;
  uint64_t pf_issued = 0;
  uint64_t pf_useful = 0;
  uint64_t pf_useless = 0;
  uint64_t pf_fill = 0;

  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> hits = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> misses = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> mshr_merge = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> mshr_return = {};

  long total_miss_latency_cycles{};
  long total_miss_queue_cycles{};
  long l1d_pend_miss_fb_full_cycles{};

  // Event-based stall attribution: accumulate total miss latency per serve_level
  // (L1D only, set in handle_fill when fill completes)
  uint64_t fill_stall_l2_hit = 0;   // serve_level=0: served by L2C
  uint64_t fill_stall_llc_hit = 0;  // serve_level=1: served by LLC
  uint64_t fill_stall_dram = 0;     // serve_level=2: served by DRAM
  uint64_t fill_count_l2_hit = 0;
  uint64_t fill_count_llc_hit = 0;
  uint64_t fill_count_dram = 0;

  // Queue-Local Tile Sidecar Stats (Passive Completion Tracking)
  uint64_t tile_parent_entry_alloc = 0;         // Sidecar allocations (reused name)
  uint64_t tile_parent_entry_peak = 0;          // Peak concurrent sidecars
  uint64_t tile_completions = 0;                // Tiles fully completed
  uint64_t tile_children_total = 0;             // Total children across all tiles
  uint64_t tile_children_hits = 0;              // Children that hit in cache
  uint64_t tile_children_pending = 0;           // Children that missed (entered MSHR path)
  uint64_t tile_children_issued = 0;            // Children completed via fill
  uint64_t tile_admission_throttle_cycles = 0;  // (unused in sidecar mode)
  uint64_t child_issue_deferrals = 0;           // (unused in sidecar mode)
  uint64_t child_issue_deferrals_mshr_full = 0; // (unused in sidecar mode)
  uint64_t child_issue_deferrals_rq_full = 0;   // (unused in sidecar mode)
  uint64_t child_issue_deferrals_parent_window = 0; // (unused in sidecar mode)
  uint64_t child_issue_deferrals_parent_table_full = 0; // (unused in sidecar mode)
  uint64_t parent_driven_issues = 0;            // (unused in sidecar mode)

  // Aggregates
  uint64_t total_pending_children_snapshots = 0;
  uint64_t total_inflight_children_snapshots = 0;
  uint64_t total_lfb_allocs_for_tiles = 0;
  uint64_t total_issue_batches = 0;
  uint64_t total_child_misses_issued_across_cycles = 0;
  uint64_t total_cycles_with_active_parents = 0;        // Cycles with active sidecar entries

  // Legacy stats (deprecated, keep for compatibility)
  uint64_t tile_state_alloc = 0;
  uint64_t tile_state_reuse = 0;
  uint64_t tile_attachment_merges = 0;
  uint64_t tile_sidecar_coalesced = 0;        // Children that bypassed MSHR via sidecar coalescing
  uint64_t tile_sidecar_fills_matched = 0;    // Fills matched against sidecar pending children
  uint64_t tile_children_generated = 0;

  // LFB diagnostic counters
  uint64_t lfb_occupancy_sum = 0;          // Sum of (non-tile MSHR+LFB) each cycle, for average
  uint64_t lfb_occupancy_cycles = 0;       // Number of cycles sampled
  uint64_t lfb_occupancy_max = 0;          // Max observed non-tile MSHR+LFB
  uint64_t sidecar_fills_occupancy_sum = 0; // Sum of sidecar_fills.size() each cycle
  // Per-component breakdown
  uint64_t non_tile_mshr_occupancy_sum = 0; // Sum of non-tile MSHR entries per cycle
  uint64_t tile_mshr_occupancy_sum = 0;     // Sum of tile MSHR entries per cycle
  uint64_t lfb_only_occupancy_sum = 0;      // Sum of LFB entries per cycle
  uint64_t non_tile_mshr_occupancy_max = 0;
  uint64_t tile_mshr_occupancy_max = 0;
  uint64_t lfb_only_occupancy_max = 0;
  uint64_t non_tile_mshr_allocs = 0;        // Total non-tile MSHR allocations
  uint64_t non_tile_mshr_by_load = 0;
  uint64_t non_tile_mshr_by_rfo = 0;
  uint64_t non_tile_mshr_by_write = 0;
  uint64_t non_tile_mshr_by_prefetch = 0;
  uint64_t non_tile_mshr_by_other = 0;
  uint64_t tile_mshr_allocs = 0;            // Total tile MSHR allocations
  uint64_t translation_mshr_allocs = 0;     // Translation MSHR allocations
  uint64_t tile_leaked_to_nontile = 0;      // Tile children that went to non-tile MSHR path
  uint64_t tile_loads_tag_check = 0;        // Tile loads entering tag check
  uint64_t scalar_loads_tag_check = 0;      // Non-tile loads entering tag check
  uint64_t unique_missing_lines_per_tile_sum = 0;
  uint64_t tile_master_alloc = 0;
  uint64_t tile_subreq_merged = 0;
  uint64_t tile_master_occupancy_cycles = 0;

  long l1_mshr_full_stall_cycles{}; // Requested explicit metric
  long l2_rq_full_stall_cycles{};

  // Arbitration / Queue Stats
  uint64_t rq_access = 0;
  uint64_t rq_full = 0;
  uint64_t pq_access = 0;
  uint64_t pq_full = 0;
  uint64_t wq_access = 0;
  uint64_t wq_full = 0;

  // Bandwidth idle tracking
  uint64_t bw_cycles = 0;
  uint64_t bw_tag_total_used = 0;
  uint64_t bw_fill_total_used = 0;
  uint64_t bw_tag_idle_hist[8] = {};
  uint64_t bw_fill_idle_hist[8] = {};

  // inflight_tag_check + MSHR PF occupancy
  uint64_t bw_itc_total = 0;
  uint64_t bw_itc_pf = 0;
  uint64_t bw_mshr_pf = 0;
  uint64_t bw_itc_hist[11] = {};     // ITC size 0-10+
  uint64_t bw_itc_pf_hist[11] = {};  // ITC PF count 0-10+

  // Phase-aware idle: MSHR-active (tile/load in-flight) vs compute (MSHR empty)
  uint64_t bw_tile_phase_cycles = 0;
  uint64_t bw_tile_phase_tag_idle = 0;
  uint64_t bw_tile_phase_fill_idle = 0;
  uint64_t bw_compute_phase_cycles = 0;
  uint64_t bw_compute_phase_tag_idle = 0;
  uint64_t bw_compute_phase_fill_idle = 0;

  // Per-child-index hit/miss tracking for tile children (L1D only)
  // A tiles: tmm4(199), tmm5(200)  |  B tiles: tmm6(201), tmm7(202)
  static constexpr int TILE_CHILDREN_MAX = 16;
  uint64_t a_tile_child_hit[16] = {};
  uint64_t a_tile_child_miss[16] = {};
  uint64_t b_tile_child_hit[16] = {};
  uint64_t b_tile_child_miss[16] = {};
};

cache_stats operator-(cache_stats lhs, cache_stats rhs);

#endif
