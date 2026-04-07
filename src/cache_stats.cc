#include "cache_stats.h"

cache_stats operator-(cache_stats lhs, cache_stats rhs)
{
  cache_stats result;
  result.pf_requested = lhs.pf_requested - rhs.pf_requested;
  result.pf_issued = lhs.pf_issued - rhs.pf_issued;
  result.pf_useful = lhs.pf_useful - rhs.pf_useful;
  result.pf_useless = lhs.pf_useless - rhs.pf_useless;
  result.pf_fill = lhs.pf_fill - rhs.pf_fill;

  result.hits = lhs.hits - rhs.hits;
  result.misses = lhs.misses - rhs.misses;

  result.total_miss_latency_cycles = lhs.total_miss_latency_cycles - rhs.total_miss_latency_cycles;
  result.total_miss_queue_cycles = lhs.total_miss_queue_cycles - rhs.total_miss_queue_cycles;
  result.l1d_pend_miss_fb_full_cycles = lhs.l1d_pend_miss_fb_full_cycles - rhs.l1d_pend_miss_fb_full_cycles;

  // Active parent tile entry stats
  result.tile_parent_entry_alloc = lhs.tile_parent_entry_alloc - rhs.tile_parent_entry_alloc;
  result.tile_parent_entry_peak = lhs.tile_parent_entry_peak; // Peak is absolute, not diffed
  result.tile_completions = lhs.tile_completions - rhs.tile_completions;
  result.tile_children_total = lhs.tile_children_total - rhs.tile_children_total;
  result.tile_children_hits = lhs.tile_children_hits - rhs.tile_children_hits;
  result.tile_children_pending = lhs.tile_children_pending - rhs.tile_children_pending;
  result.tile_children_issued = lhs.tile_children_issued - rhs.tile_children_issued;
  result.tile_admission_throttle_cycles = lhs.tile_admission_throttle_cycles - rhs.tile_admission_throttle_cycles;
  result.child_issue_deferrals = lhs.child_issue_deferrals - rhs.child_issue_deferrals;
  result.child_issue_deferrals_mshr_full = lhs.child_issue_deferrals_mshr_full - rhs.child_issue_deferrals_mshr_full;
  result.child_issue_deferrals_rq_full = lhs.child_issue_deferrals_rq_full - rhs.child_issue_deferrals_rq_full;
  result.child_issue_deferrals_parent_window = lhs.child_issue_deferrals_parent_window - rhs.child_issue_deferrals_parent_window;
  result.child_issue_deferrals_parent_table_full = lhs.child_issue_deferrals_parent_table_full - rhs.child_issue_deferrals_parent_table_full;
  result.parent_driven_issues = lhs.parent_driven_issues - rhs.parent_driven_issues;
  result.total_pending_children_snapshots = lhs.total_pending_children_snapshots - rhs.total_pending_children_snapshots;
  result.total_inflight_children_snapshots = lhs.total_inflight_children_snapshots - rhs.total_inflight_children_snapshots;
  result.total_lfb_allocs_for_tiles = lhs.total_lfb_allocs_for_tiles - rhs.total_lfb_allocs_for_tiles;
  result.total_issue_batches = lhs.total_issue_batches - rhs.total_issue_batches;
  result.total_child_misses_issued_across_cycles = lhs.total_child_misses_issued_across_cycles - rhs.total_child_misses_issued_across_cycles;
  result.total_cycles_with_active_parents = lhs.total_cycles_with_active_parents - rhs.total_cycles_with_active_parents;

  result.tile_attachment_merges = lhs.tile_attachment_merges - rhs.tile_attachment_merges;
  result.tile_sidecar_coalesced = lhs.tile_sidecar_coalesced - rhs.tile_sidecar_coalesced;
  result.tile_sidecar_fills_matched = lhs.tile_sidecar_fills_matched - rhs.tile_sidecar_fills_matched;
  result.lfb_occupancy_sum = lhs.lfb_occupancy_sum - rhs.lfb_occupancy_sum;
  result.lfb_occupancy_cycles = lhs.lfb_occupancy_cycles - rhs.lfb_occupancy_cycles;
  result.lfb_occupancy_max = lhs.lfb_occupancy_max;  // absolute max, not diffed
  result.sidecar_fills_occupancy_sum = lhs.sidecar_fills_occupancy_sum - rhs.sidecar_fills_occupancy_sum;
  result.non_tile_mshr_occupancy_sum = lhs.non_tile_mshr_occupancy_sum - rhs.non_tile_mshr_occupancy_sum;
  result.tile_mshr_occupancy_sum = lhs.tile_mshr_occupancy_sum - rhs.tile_mshr_occupancy_sum;
  result.lfb_only_occupancy_sum = lhs.lfb_only_occupancy_sum - rhs.lfb_only_occupancy_sum;
  result.non_tile_mshr_allocs = lhs.non_tile_mshr_allocs - rhs.non_tile_mshr_allocs;
  result.tile_mshr_allocs = lhs.tile_mshr_allocs - rhs.tile_mshr_allocs;
  result.translation_mshr_allocs = lhs.translation_mshr_allocs - rhs.translation_mshr_allocs;
  result.tile_leaked_to_nontile = lhs.tile_leaked_to_nontile - rhs.tile_leaked_to_nontile;
  result.tile_loads_tag_check = lhs.tile_loads_tag_check - rhs.tile_loads_tag_check;
  result.scalar_loads_tag_check = lhs.scalar_loads_tag_check - rhs.scalar_loads_tag_check;

  // Legacy tile stats
  result.tile_master_alloc = lhs.tile_master_alloc - rhs.tile_master_alloc;
  result.tile_subreq_merged = lhs.tile_subreq_merged - rhs.tile_subreq_merged;
  result.tile_master_occupancy_cycles = lhs.tile_master_occupancy_cycles - rhs.tile_master_occupancy_cycles;
  result.l1_mshr_full_stall_cycles = lhs.l1_mshr_full_stall_cycles - rhs.l1_mshr_full_stall_cycles;
  result.l2_rq_full_stall_cycles = lhs.l2_rq_full_stall_cycles - rhs.l2_rq_full_stall_cycles;

  result.rq_access = lhs.rq_access - rhs.rq_access;
  result.rq_full = lhs.rq_full - rhs.rq_full;
  result.pq_access = lhs.pq_access - rhs.pq_access;
  result.pq_full = lhs.pq_full - rhs.pq_full;
  result.wq_access = lhs.wq_access - rhs.wq_access;
  result.wq_full = lhs.wq_full - rhs.wq_full;

  // Bandwidth idle
  result.bw_cycles = lhs.bw_cycles - rhs.bw_cycles;
  result.bw_tag_total_used = lhs.bw_tag_total_used - rhs.bw_tag_total_used;
  result.bw_fill_total_used = lhs.bw_fill_total_used - rhs.bw_fill_total_used;
  for (int i = 0; i < 8; i++) {
    result.bw_tag_idle_hist[i] = lhs.bw_tag_idle_hist[i] - rhs.bw_tag_idle_hist[i];
    result.bw_fill_idle_hist[i] = lhs.bw_fill_idle_hist[i] - rhs.bw_fill_idle_hist[i];
  }
  result.bw_itc_total = lhs.bw_itc_total - rhs.bw_itc_total;
  result.bw_itc_pf = lhs.bw_itc_pf - rhs.bw_itc_pf;
  result.bw_mshr_pf = lhs.bw_mshr_pf - rhs.bw_mshr_pf;
  for (int i = 0; i < 11; i++) {
    result.bw_itc_hist[i] = lhs.bw_itc_hist[i] - rhs.bw_itc_hist[i];
    result.bw_itc_pf_hist[i] = lhs.bw_itc_pf_hist[i] - rhs.bw_itc_pf_hist[i];
  }
  result.bw_tile_phase_cycles = lhs.bw_tile_phase_cycles - rhs.bw_tile_phase_cycles;
  result.bw_tile_phase_tag_idle = lhs.bw_tile_phase_tag_idle - rhs.bw_tile_phase_tag_idle;
  result.bw_tile_phase_fill_idle = lhs.bw_tile_phase_fill_idle - rhs.bw_tile_phase_fill_idle;
  result.bw_compute_phase_cycles = lhs.bw_compute_phase_cycles - rhs.bw_compute_phase_cycles;
  result.bw_compute_phase_tag_idle = lhs.bw_compute_phase_tag_idle - rhs.bw_compute_phase_tag_idle;
  result.bw_compute_phase_fill_idle = lhs.bw_compute_phase_fill_idle - rhs.bw_compute_phase_fill_idle;

  for (int i = 0; i < 16; i++) {
    result.a_tile_child_hit[i] = lhs.a_tile_child_hit[i] - rhs.a_tile_child_hit[i];
    result.a_tile_child_miss[i] = lhs.a_tile_child_miss[i] - rhs.a_tile_child_miss[i];
    result.b_tile_child_hit[i] = lhs.b_tile_child_hit[i] - rhs.b_tile_child_hit[i];
    result.b_tile_child_miss[i] = lhs.b_tile_child_miss[i] - rhs.b_tile_child_miss[i];
  }

  return result;
}
