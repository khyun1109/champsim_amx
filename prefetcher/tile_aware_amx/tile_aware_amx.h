#ifndef TILE_AWARE_AMX_H
#define TILE_AWARE_AMX_H

#include <cstdint>
#include <unordered_map>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct tile_aware_amx : public champsim::modules::prefetcher {

  static constexpr int MAX_CHILDREN_PER_TILE = 16;
  static constexpr int CONFIDENCE_THRESHOLD = 2;
  static constexpr uint64_t KERNEL_GAP_THRESHOLD = 500;

  // ====== Per-IP stride state ======
  struct IPState {
    uint64_t prev_base_addr = 0;   // base_addr from previous occurrence
    int64_t  stride = 0;           // delta between consecutive base_addrs
    int64_t  row_stride = 0;       // addr(child1) - addr(child0)
    uint8_t  num_children = MAX_CHILDREN_PER_TILE;
    uint8_t  confidence = 0;
    bool     has_prev = false;
  };

  // ====== Per active tile (keyed by tile_group_id) ======
  struct ActiveTile {
    uint64_t ip = 0;
    uint64_t base_addr = 0;
    int64_t  row_stride = 0;
    uint8_t  num_children = MAX_CHILDREN_PER_TILE;
    bool     base_addr_set = false;
    bool     row_stride_set = false;
    bool     prefetch_issued = false;
  };

  // ====== Statistics ======
  struct Stats {
    uint64_t tile_loads_observed = 0;
    uint64_t tiles_seen = 0;
    uint64_t prefetches_issued_l1 = 0;
    uint64_t prefetches_issued_l2 = 0;
    uint64_t prefetches_dropped_pq = 0;
    uint64_t stride_relearns = 0;
    uint64_t kernel_resets = 0;
    uint64_t trigger_count = 0;
  };

  // ====== State ======
  std::unordered_map<uint64_t, IPState> ip_table;          // IP → stride state
  std::unordered_map<uint64_t, ActiveTile> active_tiles;   // tile_group_id → tile info
  Stats stats{};
  uint64_t access_counter = 0;
  uint64_t last_tile_access = 0;

  // ====== Interface ======
  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                    uint8_t cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way,
                                 uint8_t prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in);
  void prefetcher_final_stats();

private:
  void issue_tile_prefetch(const IPState& ip_state);
  void reset_state();
};

#endif
