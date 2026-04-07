#ifndef B_STREAM_BUFFER_H
#define B_STREAM_BUFFER_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/core.h>

class VirtualMemory;  // forward declare
// Global vmem pointer for SB VA→PA translation
inline VirtualMemory* g_sb_vmem = nullptr;

// B-Tile Stream Buffer: 4-slot circular, oracle schedule.
//
// 4 slots: 2 for current iteration (consuming), 2 for next iteration (ready/filling).
// Each slot = 1 B tileload = 16 CL.
// On tileload hit: respond immediately, free slot, start filling for iter+2.
// Fill path: L2C RQ (FIFO, 1 CL per request, demand priority).
//
// Iteration flow:
//   iter N: consume slot[cur0], slot[cur1] (tmm6, tmm7)
//   iter N+1: consume slot[next0], slot[next1]
//   freed cur0/cur1 → fill with iter N+2 addresses

struct BTileStreamBuffer {
  static constexpr uint32_t SB_MAGIC = 0x53425346;
  static constexpr int NUM_SLOTS = 4;
  static constexpr int CL_PER_SLOT = 16;
  static constexpr int LOOKAHEAD = 16;        // prefetch this many iterations ahead (covers DRAM via LLC RQ path)
  static constexpr int MAX_PF_INFLIGHT = 12;  // max concurrent SB prefetches (must stay < L2C PF MSHR cap to avoid blocking)

  struct Slot {
    enum State : uint8_t { FREE = 0, FILLING, READY };
    State state = FREE;
    uint64_t iter_id = 0;          // which iteration this slot serves
    uint8_t tmm_id = 0;           // 201=tmm6, 202=tmm7

    uint64_t cl_vaddr[16] = {};   // CL-aligned virtual addresses
    bool line_valid[16] = {};      // fill completed
    int num_cl = 0;
    int lines_filled = 0;
    int pf_cursor = 0;            // next CL to issue prefetch for

    void reset() {
      state = FREE;
      iter_id = 0;
      tmm_id = 0;
      num_cl = 0;
      lines_filled = 0;
      pf_cursor = 0;
      for (int i = 0; i < 16; i++) {
        cl_vaddr[i] = 0;
        line_valid[i] = false;
      }
    }

    void setup(uint64_t iid, uint8_t tmm, const uint64_t* addrs, int n) {
      reset();
      state = FILLING;
      iter_id = iid;
      tmm_id = tmm;
      num_cl = (n > 16) ? 16 : n;
      for (int i = 0; i < num_cl; i++)
        cl_vaddr[i] = addrs[i] & ~63ULL;
    }

    int find(uint64_t vaddr) const {
      uint64_t aligned = vaddr & ~63ULL;
      for (int i = 0; i < num_cl; i++)
        if (cl_vaddr[i] == aligned) return i;
      return -1;
    }

    bool all_valid() const { return lines_filled >= num_cl && num_cl > 0; }
  };

  Slot slots[NUM_SLOTS];

  // Schedule: trigger_instr_id → per-tile addresses
  // Each entry has tmm6 addrs (16) + tmm7 addrs (16)
  struct ScheduleEntry {
    uint64_t tmm6_addrs[16];
    uint64_t tmm7_addrs[16];
    int tmm6_n = 0, tmm7_n = 0;
  };
  std::unordered_map<uint64_t, ScheduleEntry> schedule;

  bool enabled = false;
  bool schedule_loaded = false;

  // Iteration tracking
  uint64_t current_iter = 0;      // current iteration being consumed
  uint64_t next_fill_iter = 0;    // next iteration to request fills for
  uint64_t pf_iter_cursor = UINT64_MAX;  // next iteration to enqueue prefetches for (schedule walker)

  // Slot assignment: which slot indices serve current/next iteration
  // Circular: slots rotate as iterations advance
  int cur_tmm6_slot = 0;   // current iter tmm6
  int cur_tmm7_slot = 1;   // current iter tmm7
  int nxt_tmm6_slot = 2;   // next iter tmm6
  int nxt_tmm7_slot = 3;   // next iter tmm7

  // Map iteration→trigger_iid for looking up schedule
  std::vector<uint64_t> iter_triggers;  // iter_triggers[i] = trigger_iid for iteration i
  std::unordered_map<uint64_t, uint64_t> trigger_to_iter;  // trigger_iid → iteration index

  // Stats
  struct Stats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t miss_filling = 0;  // miss but data in flight
    uint64_t pf_issued = 0;
    uint64_t pf_completed = 0;
    uint64_t slots_consumed = 0;
    uint64_t slots_recycled = 0;
    uint64_t triggers = 0;
  };
  Stats stats{};

  // Load schedule file (same format as gen_sb_schedule.py)
  bool load_schedule(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      fmt::print(stderr, "[SB] Cannot open {}\n", path);
      return false;
    }
    uint32_t magic = 0, num_entries = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&num_entries), 4);
    if (magic != SB_MAGIC) {
      fmt::print(stderr, "[SB] Bad magic 0x{:08x}\n", magic);
      return false;
    }

    // Read entries and build iteration list
    // The schedule file has D=0 (each trigger maps to its OWN iteration's B addrs)
    // We handle the lookahead internally.
    iter_triggers.reserve(num_entries);
    for (uint32_t i = 0; i < num_entries; i++) {
      uint32_t num_cl = 0;
      uint64_t trigger_iid = 0;
      f.read(reinterpret_cast<char*>(&num_cl), 4);
      f.read(reinterpret_cast<char*>(&trigger_iid), 8);

      ScheduleEntry entry{};
      // First 16 CL = tmm6, next 16 = tmm7
      for (uint32_t j = 0; j < num_cl; j++) {
        uint64_t addr = 0;
        f.read(reinterpret_cast<char*>(&addr), 8);
        if (j < 16) {
          entry.tmm6_addrs[j] = addr & ~63ULL;
          entry.tmm6_n = j + 1;
        } else if (j < 32) {
          entry.tmm7_addrs[j - 16] = addr & ~63ULL;
          entry.tmm7_n = j - 16 + 1;
        }
      }
      schedule[trigger_iid] = entry;
      trigger_to_iter[trigger_iid] = iter_triggers.size();
      iter_triggers.push_back(trigger_iid);
    }

    fmt::print("[SB] Loaded {} iterations from {}\n", num_entries, path);
    schedule_loaded = true;
    return true;
  }

  // Called when tileload with subidx==0 arrives at L1D during ROI.
  // Each trigger = start of a new iteration. Advance slot pointers.
  void on_tileload_trigger(uint64_t instr_id) {
    if (!enabled || !schedule_loaded) return;

    auto t2i = trigger_to_iter.find(instr_id);
    if (t2i == trigger_to_iter.end()) return;
    uint64_t iter_idx = t2i->second;

    stats.triggers++;

    if (stats.triggers == 1) {
      // Cold start: fill all 4 slots with current and next iterations
      current_iter = iter_idx;
      next_fill_iter = iter_idx + 2;
      pf_iter_cursor = iter_idx;  // start schedule-walking prefetch from here

      fill_slot(cur_tmm6_slot, iter_idx, 201);
      fill_slot(cur_tmm7_slot, iter_idx, 202);
      if (iter_idx + 1 < iter_triggers.size()) {
        fill_slot(nxt_tmm6_slot, iter_idx + 1, 201);
        fill_slot(nxt_tmm7_slot, iter_idx + 1, 202);
      }
      return;
    }

    // Subsequent triggers: the current iteration has been consumed.
    // Free current slots, advance to next, recycle freed slots for future.
    if (iter_idx > current_iter) {
      // Free current consumption slots
      slots[cur_tmm6_slot].reset();
      slots[cur_tmm7_slot].reset();
      stats.slots_consumed += 2;

      // Advance: next slots become current
      int old_cur6 = cur_tmm6_slot, old_cur7 = cur_tmm7_slot;
      cur_tmm6_slot = nxt_tmm6_slot;
      cur_tmm7_slot = nxt_tmm7_slot;
      nxt_tmm6_slot = old_cur6;
      nxt_tmm7_slot = old_cur7;
      current_iter = iter_idx;

      // Recycle freed slots: fill with iter+1 data (next iteration)
      uint64_t future_iter = iter_idx + 1;
      if (future_iter < iter_triggers.size()) {
        fill_slot(nxt_tmm6_slot, future_iter, 201);
        fill_slot(nxt_tmm7_slot, future_iter, 202);
        stats.slots_recycled += 2;
        next_fill_iter = future_iter + 1;
      }
    }
  }

  // Fill a slot with addresses for a given iteration and tmm register
  void fill_slot(int slot_idx, uint64_t iter_idx, uint8_t tmm) {
    if (iter_idx >= iter_triggers.size()) return;
    uint64_t trig = iter_triggers[iter_idx];
    auto it = schedule.find(trig);
    if (it == schedule.end()) return;

    auto& entry = it->second;
    if (tmm == 201)
      slots[slot_idx].setup(iter_idx, tmm, entry.tmm6_addrs, entry.tmm6_n);
    else
      slots[slot_idx].setup(iter_idx, tmm, entry.tmm7_addrs, entry.tmm7_n);
  }

  // Flat auxiliary lookup: check ALL slots for matching address.
  // Also check a flat valid_lines map for addresses that completed fill.
  std::unordered_set<uint64_t> valid_lines;  // CL-aligned vaddrs with completed fills

  bool lookup(uint64_t vaddr, uint8_t tmm) {
    if (!enabled) return false;
    stats.lookups++;
    uint64_t aligned = vaddr & ~63ULL;

    // Check flat valid_lines first (fastest path)
    if (valid_lines.count(aligned)) {
      stats.hits++;
      valid_lines.erase(aligned);  // consume: single-use
      return true;
    }

    // Check slots
    for (int s = 0; s < NUM_SLOTS; s++) {
      auto& slot = slots[s];
      if (slot.state == Slot::FREE) continue;
      int idx = slot.find(aligned);
      if (idx >= 0 && slot.line_valid[idx]) {
        stats.hits++;
        slot.line_valid[idx] = false;  // consume
        valid_lines.erase(aligned);
        return true;
      }
      if (idx >= 0) { stats.miss_filling++; stats.misses++; return false; }
    }
    stats.misses++;
    return false;
  }

  // Called when ALL 16 children of a B tileload have been served (hit or miss).
  // This frees the slot and starts filling it with iter+2 data.
  void consume_slot(uint8_t tmm) {
    if (!enabled) return;

    int slot_idx = (tmm == 201) ? cur_tmm6_slot : cur_tmm7_slot;
    auto& slot = slots[slot_idx];
    slot.reset();
    stats.slots_consumed++;

    // Recycle: fill with future iteration data
    if (next_fill_iter < iter_triggers.size()) {
      fill_slot(slot_idx, next_fill_iter, tmm);
      stats.slots_recycled++;
    }

    // Check if both current slots are freed → advance to next iteration
    auto& other_slot = slots[(tmm == 201) ? cur_tmm7_slot : cur_tmm6_slot];
    if (other_slot.state == Slot::FREE || other_slot.iter_id < current_iter) {
      // Both consumed → advance iteration
      advance_iteration();
    }
  }

  void advance_iteration() {
    current_iter++;
    next_fill_iter++;

    // Rotate slot assignments
    int old_cur6 = cur_tmm6_slot, old_cur7 = cur_tmm7_slot;
    cur_tmm6_slot = nxt_tmm6_slot;
    cur_tmm7_slot = nxt_tmm7_slot;
    nxt_tmm6_slot = old_cur6;  // these were just freed and refilled
    nxt_tmm7_slot = old_cur7;
  }

  // Receive fill from L2C response (v_address based matching)
  void receive_fill(uint64_t vaddr) {
    uint64_t aligned = vaddr & ~63ULL;

    // Track whether this was an SB prefetch response
    bool was_sb_pf = pf_inflight.erase(aligned) > 0;

    // Check slots (update slot state if matching)
    for (int s = 0; s < NUM_SLOTS; s++) {
      auto& slot = slots[s];
      if (slot.state == Slot::FREE) continue;
      int idx = slot.find(aligned);
      if (idx >= 0 && !slot.line_valid[idx]) {
        slot.line_valid[idx] = true;
        slot.lines_filled++;
        if (slot.all_valid()) slot.state = Slot::READY;
        if (was_sb_pf) stats.pf_completed++;
        valid_lines.insert(aligned);
        return;
      }
    }
    // Address not in any slot — still useful via valid_lines
    if (was_sb_pf) stats.pf_completed++;
    valid_lines.insert(aligned);
  }

  // Pending prefetch queue: (vaddr, paddr) pairs to send to L2C.
  // Completely separate from L1D MSHR — no MSHR slot consumed.
  struct PfEntry { uint64_t vaddr; uint64_t paddr; };
  std::deque<PfEntry> pf_queue;
  std::unordered_set<uint64_t> pf_inflight;  // vaddrs sent to L2C, awaiting response

  // Enqueue prefetch addresses from a specific slot into pf_queue.
  // Requires vmem for VA→PA translation. Implemented in cache.cc.
  void enqueue_slot_prefetches(int slot_idx, VirtualMemory* vmem, uint32_t cpu);

  // Get next prefetch to issue to L2C. Returns {0,0} if none.
  PfEntry get_next_pf() {
    if (pf_queue.empty()) return {0, 0};
    auto entry = pf_queue.front();
    pf_queue.pop_front();
    pf_inflight.insert(entry.vaddr & ~63ULL);
    stats.pf_issued++;
    return entry;
  }

  void begin_phase() {
    stats = {};
    pf_queue.clear();
    pf_inflight.clear();
    pf_iter_cursor = UINT64_MAX;  // reset; will be set on first trigger
    // Don't reset slots or valid_lines — keep prefetched data
  }

  void print_stats() const {
    uint64_t total = stats.hits + stats.misses;
    fmt::print("  lookups: {}  hits: {}  misses: {} (filling: {})  hit_rate: {:.1f}%\n",
        stats.lookups, stats.hits, stats.misses, stats.miss_filling,
        total > 0 ? 100.0 * stats.hits / total : 0.0);
    fmt::print("  pf_issued: {}  pf_completed: {}  fill_rate: {:.1f}%\n",
        stats.pf_issued, stats.pf_completed,
        stats.pf_issued > 0 ? 100.0 * stats.pf_completed / stats.pf_issued : 0.0);
    fmt::print("  triggers: {}  consumed: {}  recycled: {}\n",
        stats.triggers, stats.slots_consumed, stats.slots_recycled);
    // Slot states
    for (int i = 0; i < NUM_SLOTS; i++) {
      fmt::print("  slot[{}]: state={} iter={} tmm={} filled={}/{}\n",
          i, (int)slots[i].state, slots[i].iter_id, slots[i].tmm_id,
          slots[i].lines_filled, slots[i].num_cl);
    }
  }
};

#endif
