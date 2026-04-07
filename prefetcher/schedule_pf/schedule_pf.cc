#include "schedule_pf.h"

#include "cache.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fmt/core.h>

bool schedule_pf::load_schedule(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    fmt::print(stderr, "[schedule_pf] ERROR: Cannot open {}\n", path);
    return false;
  }

  uint32_t magic = 0, num_entries = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  f.read(reinterpret_cast<char*>(&num_entries), 4);

  if (magic != SCHED_MAGIC) {
    fmt::print(stderr, "[schedule_pf] ERROR: Bad magic 0x{:08x}\n", magic);
    return false;
  }

  for (uint32_t i = 0; i < num_entries; i++) {
    uint64_t trigger_idx = 0;
    uint32_t num_pf = 0;
    f.read(reinterpret_cast<char*>(&trigger_idx), 8);
    f.read(reinterpret_cast<char*>(&num_pf), 4);

    schedule_entry entry;
    entry.prefetches.reserve(num_pf);
    for (uint32_t j = 0; j < num_pf; j++) {
      uint64_t addr = 0;
      uint8_t fill = 0;
      f.read(reinterpret_cast<char*>(&addr), 8);
      f.read(reinterpret_cast<char*>(&fill), 1);
      entry.prefetches.push_back({addr, fill != 0});
    }

    schedule[trigger_idx] = std::move(entry);
  }

  stats.schedule_entries_total = num_entries;
  fmt::print("[schedule_pf] Loaded {} entries from {}\n", num_entries, path);
  return true;
}

void schedule_pf::prefetcher_initialize()
{
  const char* path = std::getenv("CHAMPSIM_PF_SCHEDULE");
  if (path && std::strlen(path) > 0) {
    schedule_loaded = load_schedule(path);
  } else {
    fmt::print("[schedule_pf] No CHAMPSIM_PF_SCHEDULE — disabled\n");
  }
  fmt::print("[schedule_pf] Init cache: {} ({})\n",
             intern_->NAME, schedule_loaded ? "active" : "inactive");
}

uint32_t schedule_pf::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                                uint8_t cache_hit, bool useful_prefetch,
                                                access_type type, uint32_t metadata_in)
{
  if (!schedule_loaded)
    return metadata_in;

  if (was_warmup && !intern_->warmup) {
    was_warmup = false;
    stats = {};
    pending_tiles.clear();
    stats.schedule_entries_total = schedule.size();
    stats.schedule_at_roi = schedule.size();
    fmt::print("[schedule_pf] ROI start: {} schedule entries remaining\n", schedule.size());
  }

  auto& ti = intern_->current_tile_info;
  if (!ti.is_amx_tileload)
    return metadata_in;

  if (!intern_->warmup)
    stats.tiles_seen_roi++;

  uint64_t iid = ti.instr_id;
  if (iid == last_triggered_instr_id)
    return metadata_in;

  auto it = schedule.find(iid);
  if (it != schedule.end()) {
    last_triggered_instr_id = iid;
    // Debug first 3 triggers in ROI
    if (!intern_->warmup && stats.triggers_fired < 3) {
      auto& pfs = it->second.prefetches;
      fmt::print("[SCHED_TRIG] iid={} npf={} pf[0]=0x{:x} demand_vaddr=0x{:x}\n",
          iid, pfs.size(), pfs.size() > 0 ? pfs[0].address : 0,
          addr.to<uint64_t>());
    }

    // Group CL into tiles (every CL_PER_TILE consecutive CLs = 1 tile)
    auto& pfs = it->second.prefetches;
    for (size_t i = 0; i < pfs.size(); i += CL_PER_TILE) {
      pf_tile_group_counter++;
      pending_tile pt;
      pt.tile_group_id = pf_tile_group_counter;
      pt.fill_this_level = pfs[i].fill_this_level;
      pt.enqueue_cycle = intern_->current_time.time_since_epoch() / intern_->clock_period;
      size_t end = std::min(i + (size_t)CL_PER_TILE, pfs.size());
      for (size_t j = i; j < end; j++) {
        pt.addrs.push_back(champsim::address{pfs[j].address});
      }
      pending_tiles.push_back(std::move(pt));
      if (!intern_->warmup) {
        stats.tiles_queued++;
        stats.cl_total += (end - i);
      }
    }
    schedule.erase(it);

    if (!intern_->warmup) {
      stats.triggers_fired++;
      if (pending_tiles.size() > stats.pending_peak)
        stats.pending_peak = pending_tiles.size();
    }
  }

  return metadata_in;
}

void schedule_pf::prefetcher_cycle_operate()
{
  if (!schedule_loaded || pending_tiles.empty())
    return;

  int issued = 0;
  while (!pending_tiles.empty() && issued < TILES_PER_CYCLE) {
    auto& tile = pending_tiles.front();
    bool ok = prefetch_tile(tile.addrs, tile.tile_group_id, tile.fill_this_level);
    if (!ok) {
      if (!intern_->warmup) stats.tiles_dropped++;
      break;  // MSHR full, retry next cycle
    }
    if (!intern_->warmup) {
      stats.tiles_issued++;
      // Queue delay: enqueue → issue
      uint64_t now_cy = intern_->current_time.time_since_epoch() / intern_->clock_period;
      if (tile.enqueue_cycle > 0) {
        stats.total_queue_delay += (now_cy - tile.enqueue_cycle);
        stats.queue_delay_count++;
      }
    }
    pending_tiles.pop_front();
    issued++;
  }
}

uint32_t schedule_pf::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                             uint8_t prefetch, champsim::address evicted_addr,
                                             uint32_t metadata_in)
{
  return metadata_in;
}

void schedule_pf::prefetcher_final_stats()
{
  if (!schedule_loaded) return;

  fmt::print("[schedule_pf] ======= FINAL STATS (ROI) =======\n");
  fmt::print("[schedule_pf] schedule_total:       {}\n", stats.schedule_entries_total);
  fmt::print("[schedule_pf] schedule_at_roi:      {}\n", stats.schedule_at_roi);
  fmt::print("[schedule_pf] schedule_remain:      {}\n", schedule.size());
  fmt::print("[schedule_pf] tiles_seen_roi:       {}\n", stats.tiles_seen_roi);
  fmt::print("[schedule_pf] triggers_fired:       {}\n", stats.triggers_fired);
  fmt::print("[schedule_pf] tiles_queued:         {}\n", stats.tiles_queued);
  fmt::print("[schedule_pf] tiles_issued:         {}\n", stats.tiles_issued);
  fmt::print("[schedule_pf] tiles_dropped:        {}\n", stats.tiles_dropped);
  fmt::print("[schedule_pf] cl_total:             {}\n", stats.cl_total);
  fmt::print("[schedule_pf] pending_peak:         {}\n", stats.pending_peak);
  fmt::print("[schedule_pf] pending_remain:       {}\n", pending_tiles.size());
  if (stats.queue_delay_count > 0) {
    fmt::print("[schedule_pf] avg_queue_delay:      {:.1f}cy (trigger→issue, {} tiles)\n",
        (double)stats.total_queue_delay / stats.queue_delay_count, stats.queue_delay_count);
  }
  fmt::print("[schedule_pf] ====================================\n");
}
