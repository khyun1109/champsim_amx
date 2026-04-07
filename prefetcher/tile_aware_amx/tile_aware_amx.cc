#include "tile_aware_amx.h"

#include "cache.h"
#include "tile_record.h"
#include <fmt/core.h>

void tile_aware_amx::prefetcher_initialize()
{
  reset_state();
  fmt::print("[tile_aware_amx] Initialized for cache: {} (enable via --tile-prefetch)\n", intern_->NAME);
}

void tile_aware_amx::reset_state()
{
  ip_table.clear();
  active_tiles.clear();
}

// ============================================================================
// Issue prefetches for next occurrences of a tile IP
// ============================================================================

void tile_aware_amx::issue_tile_prefetch(const IPState& ip_state)
{
  int64_t row_stride = ip_state.row_stride;
  uint8_t num_rows = ip_state.num_children;
  if (row_stride == 0 || num_rows == 0)
    return;

  stats.trigger_count++;

  // L1 Prefetch: +1 stride ahead (next occurrence of this IP)
  uint64_t next_base = ip_state.prev_base_addr + ip_state.stride;
  for (int row = 0; row < num_rows; row++) {
    uint64_t pf_addr = static_cast<uint64_t>(static_cast<int64_t>(next_base) + row * row_stride);
    bool ok = prefetch_line(champsim::address{pf_addr}, true, 0);
    if (ok)
      stats.prefetches_issued_l1++;
    else
      stats.prefetches_dropped_pq++;
  }

  // L2 Prefetch: +2 strides ahead
  uint64_t far_base = ip_state.prev_base_addr + 2 * ip_state.stride;
  for (int row = 0; row < num_rows; row++) {
    uint64_t pf_addr = static_cast<uint64_t>(static_cast<int64_t>(far_base) + row * row_stride);
    bool ok = prefetch_line(champsim::address{pf_addr}, false, 0);
    if (ok)
      stats.prefetches_issued_l2++;
    else
      stats.prefetches_dropped_pq++;
  }
}

// ============================================================================
// Main Prefetcher Hook
// ============================================================================

uint32_t tile_aware_amx::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                                   uint8_t cache_hit, bool useful_prefetch,
                                                   access_type type, uint32_t metadata_in)
{
  return metadata_in; // tile_aware_amx prefetcher disabled

  access_counter++;

  auto& tile_info = intern_->current_tile_info;
  if (!tile_info.is_amx_tileload)
    return metadata_in;

  stats.tile_loads_observed++;

  // Kernel gap detection
  if (last_tile_access > 0 && (access_counter - last_tile_access) > KERNEL_GAP_THRESHOLD) {
    reset_state();
    stats.kernel_resets++;
  }
  last_tile_access = access_counter;

  uint64_t raw_addr = addr.to<uint64_t>();
  uint64_t raw_ip = ip.to<uint64_t>();
  uint64_t group_id = tile_info.tile_group_id;
  uint8_t subidx = tile_info.tile_subidx;
  uint8_t num_children = tile_info.tile_num_children;

  // Get or create active tile entry for this tile_group_id
  auto tile_it = active_tiles.find(group_id);
  if (tile_it == active_tiles.end()) {
    // New tile
    ActiveTile at{};
    at.ip = raw_ip;
    at.num_children = (num_children > 0 && num_children <= MAX_CHILDREN_PER_TILE)
                          ? num_children
                          : static_cast<uint8_t>(MAX_CHILDREN_PER_TILE);
    active_tiles[group_id] = at;
    tile_it = active_tiles.find(group_id);
    stats.tiles_seen++;
  }
  auto& tile = tile_it->second;

  // Child 0: record base address → learn stride → trigger prefetch
  if (subidx == 0 && !tile.base_addr_set) {
    tile.base_addr = raw_addr;
    tile.base_addr_set = true;

    // Stride learning for this IP
    auto& ip_state = ip_table[raw_ip];

    if (ip_state.has_prev) {
      int64_t new_stride = static_cast<int64_t>(raw_addr) - static_cast<int64_t>(ip_state.prev_base_addr);
      if (new_stride != 0) {
        if (ip_state.confidence > 0 && new_stride == ip_state.stride) {
          if (ip_state.confidence < 255)
            ip_state.confidence++;
        } else {
          if (ip_state.confidence > 0)
            stats.stride_relearns++;
          ip_state.stride = new_stride;
          ip_state.confidence = 1;
        }
      }
    }
    ip_state.prev_base_addr = raw_addr;
    ip_state.has_prev = true;
    ip_state.num_children = tile.num_children;

    // Issue prefetch if confident
    if (ip_state.confidence >= CONFIDENCE_THRESHOLD && ip_state.row_stride != 0 && !tile.prefetch_issued) {
      issue_tile_prefetch(ip_state);
      tile.prefetch_issued = true;
    }
  }

  // Child 1: learn row stride
  if (subidx == 1 && tile.base_addr_set && !tile.row_stride_set) {
    tile.row_stride = static_cast<int64_t>(raw_addr) - static_cast<int64_t>(tile.base_addr);
    tile.row_stride_set = true;

    // Update IP's row stride
    auto ip_it = ip_table.find(raw_ip);
    if (ip_it != ip_table.end()) {
      ip_it->second.row_stride = tile.row_stride;
    }
  }

  // Evict completed tiles to bound memory (simple: evict if we've seen too many)
  if (active_tiles.size() > 256) {
    // Remove the oldest entries (crude but effective)
    auto it = active_tiles.begin();
    while (active_tiles.size() > 128 && it != active_tiles.end()) {
      it = active_tiles.erase(it);
    }
  }

  return metadata_in;
}

uint32_t tile_aware_amx::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                                uint8_t prefetch, champsim::address evicted_addr,
                                                uint32_t metadata_in)
{
  return metadata_in;
}

// ============================================================================
// Final Stats
// ============================================================================

void tile_aware_amx::prefetcher_final_stats()
{
  return; // tile_aware_amx prefetcher disabled

  fmt::print("[tile_aware_amx] ======= FINAL STATS =======\n");
  fmt::print("[tile_aware_amx] Cache:                       {}\n", intern_->NAME);
  fmt::print("[tile_aware_amx] tile_loads_observed:          {}\n", stats.tile_loads_observed);
  fmt::print("[tile_aware_amx] tiles_seen:                   {}\n", stats.tiles_seen);
  fmt::print("[tile_aware_amx] trigger_count:                {}\n", stats.trigger_count);
  fmt::print("[tile_aware_amx] prefetches_issued_l1:         {}\n", stats.prefetches_issued_l1);
  fmt::print("[tile_aware_amx] prefetches_issued_l2:         {}\n", stats.prefetches_issued_l2);
  fmt::print("[tile_aware_amx] prefetches_dropped_pq:        {}\n", stats.prefetches_dropped_pq);
  fmt::print("[tile_aware_amx] stride_relearns:              {}\n", stats.stride_relearns);
  fmt::print("[tile_aware_amx] kernel_resets:                {}\n", stats.kernel_resets);
  fmt::print("[tile_aware_amx] ip_table_size:                {}\n", ip_table.size());

  fmt::print("[tile_aware_amx] --- Learned IP Strides ---\n");
  for (auto& [ip_key, s] : ip_table) {
    fmt::print("[tile_aware_amx]   IP: 0x{:x}  stride: {}  row_stride: {}  conf: {}  children: {}\n",
               ip_key, s.stride, s.row_stride, s.confidence, s.num_children);
  }
  fmt::print("[tile_aware_amx] ============================\n");
}
