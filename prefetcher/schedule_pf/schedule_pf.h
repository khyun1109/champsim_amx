#ifndef SCHEDULE_PF_H
#define SCHEDULE_PF_H

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct schedule_pf : public champsim::modules::prefetcher {

  static constexpr uint32_t SCHED_MAGIC = 0x50465343;
  static constexpr uint8_t CL_PER_TILE = 16;
  static constexpr int TILES_PER_CYCLE = 2;  // issue up to 2 tiles/cy via prefetch_tile()

  struct pf_entry {
    uint64_t address;
    bool fill_this_level;
  };

  struct schedule_entry {
    std::vector<pf_entry> prefetches;
  };

  // Tile-level pending queue: each entry = 1 tile (up to 16 CL addresses)
  struct pending_tile {
    std::vector<champsim::address> addrs;
    uint64_t tile_group_id;
    bool fill_this_level;
    uint64_t enqueue_cycle = 0;  // for queue delay tracking
  };

  std::unordered_map<uint64_t, schedule_entry> schedule;
  std::deque<pending_tile> pending_tiles;

  uint64_t last_triggered_instr_id = UINT64_MAX;
  uint64_t pf_tile_group_counter = 0x8000000000000000ULL;

  struct Stats {
    uint64_t tiles_seen_roi = 0;
    uint64_t triggers_fired = 0;
    uint64_t tiles_queued = 0;
    uint64_t tiles_issued = 0;
    uint64_t tiles_dropped = 0;
    uint64_t cl_total = 0;
    uint64_t pending_peak = 0;
    uint64_t schedule_entries_total = 0;
    uint64_t schedule_at_roi = 0;
    uint64_t total_queue_delay = 0;
    uint64_t queue_delay_count = 0;
  };
  Stats stats{};

  bool schedule_loaded = false;
  bool was_warmup = true;

  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                    uint8_t cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in);
  void prefetcher_cycle_operate();
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way,
                                 uint8_t prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in);
  void prefetcher_final_stats();

private:
  bool load_schedule(const std::string& path);
};

#endif
