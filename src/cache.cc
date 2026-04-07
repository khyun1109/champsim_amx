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

#include "cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <fmt/core.h>

#include "bandwidth.h"
#include "vmem.h"
#include "champsim.h"
#include "chrono.h"
#include "deadlock.h"
#include "instruction.h"
#include "tile_record.h"
#include "util/algorithm.h"
#include "util/bits.h"
#include "util/span.h"

CACHE::CACHE(CACHE&& other)
    : operable(other),

      upper_levels(std::move(other.upper_levels)), lower_level(std::move(other.lower_level)), lower_translate(std::move(other.lower_translate)),

      cpu(other.cpu), NAME(std::move(other.NAME)), NUM_SET(other.NUM_SET), NUM_WAY(other.NUM_WAY), MSHR_SIZE(other.MSHR_SIZE), LFB_SIZE(other.LFB_SIZE),
      PQ_SIZE(other.PQ_SIZE), HIT_LATENCY(other.HIT_LATENCY), FILL_LATENCY(other.FILL_LATENCY), OFFSET_BITS(other.OFFSET_BITS), block(std::move(other.block)),
      MAX_TAG(other.MAX_TAG),
      MAX_FILL(other.MAX_FILL), prefetch_as_load(other.prefetch_as_load), match_offset_bits(other.match_offset_bits), virtual_prefetch(other.virtual_prefetch),
      pref_activate_mask(std::move(other.pref_activate_mask)),

      sim_stats(std::move(other.sim_stats)), roi_stats(std::move(other.roi_stats)),

      pref_module_pimpl(std::move(other.pref_module_pimpl)), repl_module_pimpl(std::move(other.repl_module_pimpl))
{
  pref_module_pimpl->bind(this);
  repl_module_pimpl->bind(this);
}

auto CACHE::operator=(CACHE&& other) -> CACHE&
{
  this->clock_period = other.clock_period;
  this->current_time = other.current_time;
  this->warmup = other.warmup;

  this->upper_levels = std::move(other.upper_levels);
  this->lower_level = std::move(other.lower_level);
  this->lower_translate = std::move(other.lower_translate);

  this->cpu = other.cpu;
  this->NAME = std::move(other.NAME);
  this->NUM_SET = other.NUM_SET;
  this->NUM_WAY = other.NUM_WAY;
  this->MSHR_SIZE = other.MSHR_SIZE;
  this->LFB_SIZE = other.LFB_SIZE;
  this->PQ_SIZE = other.PQ_SIZE;
  this->HIT_LATENCY = other.HIT_LATENCY;
  this->FILL_LATENCY = other.FILL_LATENCY;
  this->OFFSET_BITS = other.OFFSET_BITS;
  ;
  this->block = std::move(other.block);
  this->MAX_TAG = other.MAX_TAG;
  this->MAX_FILL = other.MAX_FILL;
  this->prefetch_as_load = other.prefetch_as_load;
  this->match_offset_bits = other.match_offset_bits;
  this->virtual_prefetch = other.virtual_prefetch;
  this->pref_activate_mask = std::move(other.pref_activate_mask);

  this->sim_stats = std::move(other.sim_stats);
  this->roi_stats = std::move(other.roi_stats);

  this->pref_module_pimpl = std::move(other.pref_module_pimpl);
  this->repl_module_pimpl = std::move(other.repl_module_pimpl);

  pref_module_pimpl->bind(this);
  repl_module_pimpl->bind(this);

  return *this;
}

CACHE::tag_lookup_type::tag_lookup_type(const request_type& req, bool local_pref, bool skip)
    : address(req.address), v_address(req.v_address), data(req.data), ip(req.ip), instr_id(req.instr_id), pf_metadata(req.pf_metadata), cpu(req.cpu),
      type(req.type), prefetch_from_this(local_pref), skip_fill(skip), is_translated(req.is_translated), force_miss(req.force_miss),
      is_amx_tileload(req.is_amx_tileload), tile_burst(req.tile_burst), tile_group_id(req.tile_group_id), tile_subidx(req.tile_subidx),
      tile_num_children(req.tile_num_children), tile_is_store(req.tile_is_store), tile_dest_tmm(req.tile_dest_tmm),
      instr_depend_on_me(req.instr_depend_on_me)
{
}

CACHE::mshr_type::mshr_type(const tag_lookup_type& req, champsim::chrono::clock::time_point _time_enqueued)
    : address(req.address), v_address(req.v_address), ip(req.ip), instr_id(req.instr_id), cpu(req.cpu), type(req.type),
      prefetch_from_this(req.prefetch_from_this), skip_fill(req.skip_fill), time_enqueued(_time_enqueued),
      instr_depend_on_me(req.instr_depend_on_me), to_return(req.to_return), is_amx_tileload(req.is_amx_tileload)
{
  // Tile attachments are added separately after MSHR allocation/merge
}

CACHE::mshr_type CACHE::mshr_type::merge(mshr_type predecessor, mshr_type successor)
{
  std::vector<uint64_t> merged_instr{};
  std::vector<std::deque<response_type>*> merged_return{};

  std::set_union(std::begin(predecessor.instr_depend_on_me), std::end(predecessor.instr_depend_on_me), std::begin(successor.instr_depend_on_me),
                 std::end(successor.instr_depend_on_me), std::back_inserter(merged_instr));
  std::set_union(std::begin(predecessor.to_return), std::end(predecessor.to_return), std::begin(successor.to_return), std::end(successor.to_return),
                 std::back_inserter(merged_return));

  mshr_type retval{(successor.type == access_type::PREFETCH) ? predecessor : successor};

  // set the time enqueued to the predecessor unless its a demand into prefetch, in which case we use the successor
  retval.time_enqueued =
      ((successor.type != access_type::PREFETCH && predecessor.type == access_type::PREFETCH)) ? successor.time_enqueued : predecessor.time_enqueued;
  retval.instr_depend_on_me = merged_instr;
  retval.to_return = merged_return;
  retval.data_promise = predecessor.data_promise;

  // Merge tile attachments (multi-tile support)
  retval.tile_attachments = predecessor.tile_attachments;
  retval.tile_attachments.insert(retval.tile_attachments.end(), successor.tile_attachments.begin(), successor.tile_attachments.end());

  if constexpr (champsim::debug_print) {
    if (successor.type == access_type::PREFETCH) {
      fmt::print("[MSHR] {} address {} type: {} into address {} type: {}\n", __func__, successor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)), predecessor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)));
    } else {
      fmt::print("[MSHR] {} address {} type: {} into address {} type: {}\n", __func__, predecessor.address,
                 access_type_names.at(champsim::to_underlying(predecessor.type)), successor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)));
    }
  }

  return retval;
}

auto CACHE::fill_block(mshr_type mshr, uint32_t metadata) -> BLOCK
{
  CACHE::BLOCK to_fill;
  to_fill.valid = true;
  to_fill.prefetch = mshr.prefetch_from_this;
  to_fill.dirty = (mshr.type == access_type::WRITE);
  to_fill.address = mshr.address;
  to_fill.v_address = mshr.v_address;
  to_fill.data = mshr.data_promise->data;
  to_fill.pf_metadata = metadata;

  return to_fill;
}

auto CACHE::matches_address(champsim::address addr) const
{
  return [match = addr.slice_upper(OFFSET_BITS), shamt = OFFSET_BITS](const auto& entry) {
    return entry.address.slice_upper(shamt) == match;
  };
}

bool CACHE::has_mshr_entry_for(champsim::address addr) const
{
  auto matcher = matches_address(addr);
  for (const auto& entry : MSHR) {
    if (entry.is_tile_entry) {
      for (const auto& child : entry.tile_children) {
        if (matcher(child)) return true;
      }
    } else {
      if (matcher(entry)) return true;
    }
  }
  return false;
}

template <typename T>
champsim::address CACHE::module_address(const T& element) const
{
  auto address = virtual_prefetch ? element.v_address : element.address;
  return champsim::address{address.slice_upper(match_offset_bits ? champsim::data::bits{} : OFFSET_BITS)};
}

bool CACHE::handle_fill(const mshr_type& fill_mshr)
{
  cpu = fill_mshr.cpu;

  // find victim
  auto [set_begin, set_end] = get_set_span(fill_mshr.address);
  auto way = std::find_if_not(set_begin, set_end, [](auto x) { return x.valid; });
  if (way == set_end) {
    way = std::next(set_begin, impl_find_victim(fill_mshr.cpu, fill_mshr.instr_id, get_set_index(fill_mshr.address), &*set_begin, fill_mshr.ip,
                                                fill_mshr.address, fill_mshr.type));
  }
  assert(set_begin <= way);
  assert(way <= set_end);
  assert(way != set_end || fill_mshr.type != access_type::WRITE); // Writes may not bypass
  const auto way_idx = std::distance(set_begin, way);             // cast protected by earlier assertion

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} set: {} way: {} type: {} prefetch_metadata: {} cycle_enqueued: {} cycle: {}\n", NAME, __func__,
               fill_mshr.instr_id, fill_mshr.address, fill_mshr.v_address, get_set_index(fill_mshr.address), way_idx,
               access_type_names.at(champsim::to_underlying(fill_mshr.type)), fill_mshr.data_promise->pf_metadata,
               (fill_mshr.time_enqueued.time_since_epoch()) / clock_period, (current_time.time_since_epoch()) / clock_period);
  }
  // Diagnostic for L2 Return Path
  if (NAME.find("L2C") != std::string::npos) {
       if (fill_mshr.address == champsim::address{0x2a48410c0}) {
//           fmt::print(stderr, "[{}_TRACE] L2 Fill addr: {} to_ret: {}\n", NAME, fill_mshr.address, fill_mshr.to_return.size());
       }
       static thread_local uint64_t fill_count = 0;
       if (++fill_count > 0) {
//           fmt::print(stderr, "[{}_FILL] #{} addr: {} type: {} to_return_sz: {}\n", 
//                       NAME, fill_count, fill_mshr.address, 
//                       access_type_names.at(champsim::to_underlying(fill_mshr.type)), 
//                       fill_mshr.to_return.size());
       }
  }

  if (way != set_end && way->valid && way->dirty) {
    request_type writeback_packet;

    writeback_packet.cpu = fill_mshr.cpu;
    writeback_packet.address = way->address;
    writeback_packet.data = way->data;
    writeback_packet.instr_id = fill_mshr.instr_id;
    writeback_packet.ip = champsim::address{};
    writeback_packet.type = access_type::WRITE;
    writeback_packet.pf_metadata = way->pf_metadata;
    writeback_packet.response_requested = false;

    if constexpr (champsim::debug_print) {
      fmt::print("[{}] {} evict address: {} v_address: {} prefetch_metadata: {}\n", NAME, __func__, writeback_packet.address, writeback_packet.v_address,
                 fill_mshr.data_promise->pf_metadata);
    }

    auto success = lower_level->add_wq(writeback_packet);
    if (!success) {
      return false;
    }
  }

  champsim::address evicting_address{};
  uint32_t metadata_thru = fill_mshr.data_promise->pf_metadata;

  if (way != set_end && way->valid) {
    evicting_address = module_address(*way);
  }

  if (!fill_mshr.skip_fill) {
    metadata_thru = impl_prefetcher_cache_fill(module_address(fill_mshr), get_set_index(fill_mshr.address), way_idx,
                                               (fill_mshr.type == access_type::PREFETCH), evicting_address, fill_mshr.data_promise->pf_metadata);
    // Set tile hint for replacement policy: tile data should be inserted at low LRU priority
    current_fill_is_tile = fill_mshr.is_tile_entry || fill_mshr.is_amx_tileload || (fill_mshr.tile_attachments.size() > 0);
    impl_replacement_cache_fill(fill_mshr.cpu, get_set_index(fill_mshr.address), way_idx, module_address(fill_mshr), fill_mshr.ip, evicting_address,
                                fill_mshr.type);
    current_fill_is_tile = false;
  }

  if (way != set_end && !fill_mshr.skip_fill) {
    if (way->valid && way->prefetch) {
      ++sim_stats.pf_useless;
    }

    if (fill_mshr.type == access_type::PREFETCH) {
      ++sim_stats.pf_fill;
    }

    // ── Prefetch Tracker: fill & eviction (ROI only) ──
    if (!warmup && NAME.find("L1D") != std::string::npos) {
      uint64_t fill_blk = fill_mshr.address.to<uint64_t>() >> 6;
      uint64_t fill_ip = fill_mshr.ip.to<uint64_t>();
      uint64_t now = current_time.time_since_epoch() / clock_period;

      // Check if evicting a prefetch-filled line that was never used by demand
      if (way->valid) {
        uint64_t evict_blk = way->address.to<uint64_t>() >> 6;
        auto ev_it = pf_filled_lines.find(evict_blk);
        if (ev_it != pf_filled_lines.end()) {
          pf_stats.pf_evicted_before_use++;
          uint64_t residency = now - ev_it->second;
          pf_stats.total_pf_residency += residency;
          pf_stats.pf_residency_count++;
          pf_filled_lines.erase(ev_it);
        }
      }

      // Record this fill if it's a prefetch (trace-injected OR HW prefetcher)
      bool is_trace_pf = ((fill_ip >> 32) == 0xDEAD);
      bool is_hw_pf = (fill_mshr.type == access_type::PREFETCH);
      if (is_trace_pf || is_hw_pf) {
        pf_filled_lines[fill_blk] = now;
        pf_stats.pf_fills++;
        // Track PF fill latency (MSHR enqueue → fill complete)
        auto enq_cy = fill_mshr.time_enqueued.time_since_epoch() / clock_period;
        if (enq_cy > 0) {
          auto pf_lat = now - enq_cy;
          pf_stats.pf_fill_latency_total += pf_lat;
          pf_stats.pf_fill_latency_count++;
        }
      }
    }

    *way = fill_block(fill_mshr, metadata_thru);
  }

  // COLLECT STATS
  if (fill_mshr.type != access_type::PREFETCH)
    sim_stats.total_miss_latency_cycles += (current_time - (fill_mshr.time_enqueued + clock_period)) / clock_period;
  sim_stats.mshr_return.increment(std::pair{fill_mshr.type, fill_mshr.cpu});

  // Event-based stall attribution: on fill completion at L1D, accumulate
  // the total miss latency to the level that served the miss.
  if (!warmup && NAME.find("L1D") != std::string::npos) {
    uint64_t miss_latency = (current_time - fill_mshr.time_enqueued) / clock_period;
    switch (fill_mshr.serve_level) {
      case 0: sim_stats.fill_stall_l2_hit += miss_latency; sim_stats.fill_count_l2_hit++; break;
      case 1: sim_stats.fill_stall_llc_hit += miss_latency; sim_stats.fill_count_llc_hit++; break;
      case 2: sim_stats.fill_stall_dram += miss_latency; sim_stats.fill_count_dram++; break;
    }
  }

  response_type response{fill_mshr.address, fill_mshr.v_address, fill_mshr.data_promise->data, metadata_thru, fill_mshr.instr_depend_on_me};
  response.serve_level = fill_mshr.serve_level;  // Propagate source level from lower cache
  for (auto* ret : fill_mshr.to_return) {
    ret->push_back(response);
  }

  // --- Tile instrumentation: fill completion ---
  // L1D fill means this subreq missed L1D; the actual source (L2/LLC/DRAM)
  // was already tagged by L2C/LLC try_hit instrumentation below.
  if (NAME.find("L1D") != std::string::npos) {
    uint64_t la  = fill_mshr.v_address.to<uint64_t>();
    uint64_t cur = current_time.time_since_epoch() / clock_period;
    std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
    std::vector<uint64_t> completed_tids;
    for (auto& [tid, rec] : g_tile_masters) {
      bool all_done = true;
      for (auto& sr : rec.subreqs) {
        if (sr.line_addr == la && sr.fill_cycle == 0 && !sr.l1_hit) {
          sr.fill_cycle = cur;
          sr.l1_hit     = true;  // data now in L1D (via fill from below)
          rec.l1_misses++;       // count as L1D miss
          // l2_hit / llc_hit / dram_miss flags were already set by
          // the downstream cache's try_hit or handle_fill instrumentation.
          // If none were set, classify based on remaining subreq state.
          if (!sr.l2_hit && !sr.llc_hit && !sr.dram_miss) {
            // Fallback: data came from DRAM (no downstream hit was recorded)
            sr.dram_miss = true;
            rec.dram_hits++;
          }
        }
        if (!sr.l1_hit && sr.fill_cycle == 0) all_done = false;
      }
      if (all_done && rec.issue_cycle > 0) {
        completed_tids.push_back(tid);
      }
    }
    for (uint64_t tid : completed_tids) {
      tile_finalize(tid, cur);
    }
  }

  return true;
}

bool CACHE::try_hit(const tag_lookup_type& handle_pkt)
{
  cpu = handle_pkt.cpu;

  // access cache
  auto [set_begin, set_end] = get_set_span(handle_pkt.address);
  auto way = std::find_if(set_begin, set_end, [matcher = matches_address(handle_pkt.address)](const auto& x) { return x.valid && matcher(x); });
  bool hit = (way != set_end);
  if (handle_pkt.force_miss) {
    hit = false;
    way = set_end;
  }

  // TEST: Perfect cache — force hit at specified level via env vars
  // PERFECT_L1D=1, PERFECT_L2C=1, PERFECT_LLC=1
  {
    static bool perf_l1d = std::getenv("PERFECT_L1D") != nullptr;
    static bool perf_l2c = std::getenv("PERFECT_L2C") != nullptr;
    static bool perf_llc = std::getenv("PERFECT_LLC") != nullptr;
    bool force_hit = false;
    if (perf_l1d && NAME.find("L1D") != std::string::npos) force_hit = true;
    if (perf_l2c && NAME.find("L2C") != std::string::npos) force_hit = true;
    if (perf_llc && NAME.find("LLC") != std::string::npos) force_hit = true;
    if (force_hit && !hit) {
      auto [fill_begin, fill_end] = get_set_span(handle_pkt.address);
      auto fill_way = std::find_if_not(fill_begin, fill_end, [](const auto& x) { return x.valid; });
      if (fill_way == fill_end) fill_way = fill_begin;
      fill_way->valid = true;
      fill_way->prefetch = false;
      fill_way->dirty = false;
      fill_way->address = handle_pkt.address;
      fill_way->v_address = handle_pkt.v_address;
      fill_way->data = handle_pkt.data;
      hit = true;
      way = fill_way;
    }
  }

  // Warm scalar LLC: non-tile LLC misses are forced to hit.
  // Simulates warm scalar cache from prior GEMM iterations — in real HW,
  // packing buffers, stack, and runtime data are LLC-resident from repeated execution.
  // Tile loads are NOT affected (they follow the real cache hierarchy).
  if (g_warm_scalar_llc && !warmup && !hit
      && NAME.find("LLC") != std::string::npos
      && !handle_pkt.is_amx_tileload) {
    auto [fill_begin, fill_end] = get_set_span(handle_pkt.address);
    auto fill_way = std::find_if_not(fill_begin, fill_end, [](const auto& x) { return x.valid; });
    if (fill_way == fill_end) fill_way = fill_begin;
    fill_way->valid = true;
    fill_way->prefetch = false;
    fill_way->dirty = false;
    fill_way->address = handle_pkt.address;
    fill_way->v_address = handle_pkt.v_address;
    fill_way->data = handle_pkt.data;
    hit = true;
    way = fill_way;
  }
  const auto useful_prefetch = (hit && way->prefetch && !handle_pkt.prefetch_from_this);

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} data: {} set: {} way: {} ({}) type: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, handle_pkt.data, get_set_index(handle_pkt.address), std::distance(set_begin, way),
               hit ? "HIT" : "MISS", access_type_names.at(champsim::to_underlying(handle_pkt.type)), current_time.time_since_epoch() / clock_period);
  }
  // Diagnostic for L2 Hit/Miss
  if (NAME.find("L2C") != std::string::npos) {
       static uint64_t access_count = 0;
       if (++access_count > 0) {
//           fmt::print(stderr, "[{}_ACCESS] #{} addr: {} type: {} hit: {} to_return_sz: {}\n", 
//                       NAME, access_count, handle_pkt.address, 
//                       access_type_names.at(champsim::to_underlying(handle_pkt.type)), 
//                       hit, handle_pkt.to_return.size());
       }
  }

  // ── SW Prefetch Lifecycle Tracker (L1D only, ROI only, observation only) ──
  // Tile prefetch uses AMX/TILELOADDT1 encoding but with IP=0xDEAD...
  // Distinguish by IP prefix, not by is_amx_tileload flag.
  if (!warmup && NAME.find("L1D") != std::string::npos) {
    uint64_t pkt_ip = handle_pkt.ip.to<uint64_t>();
    bool is_trace_pf = ((pkt_ip >> 32) == 0xDEAD);   // trace-injected prefetch
    bool is_hw_pf = (handle_pkt.type == access_type::PREFETCH); // HW prefetcher
    bool is_tile_pf = is_trace_pf || is_hw_pf;
    bool is_demand_tl = handle_pkt.is_amx_tileload && !is_tile_pf; // real demand tileload
    uint64_t blk_addr = handle_pkt.address.to<uint64_t>() >> 6;
    uint64_t now = current_time.time_since_epoch() / clock_period;

    if (is_tile_pf) {
      pf_stats.pf_requests++;
      if (hit) pf_stats.pf_l1_hit++;
      else     pf_stats.pf_l1_miss++;
      // Debug: first 5 PF requests
      if (pf_stats.pf_requests <= 5)
        fmt::print("[PF_DBG] pf#{} vaddr=0x{:x} paddr=0x{:x} blk=0x{:x} hit={}\n",
            pf_stats.pf_requests, handle_pkt.v_address.to<uint64_t>(),
            handle_pkt.address.to<uint64_t>(), blk_addr, hit);
    } else if (is_demand_tl) {
      // Demand tileload child: overall hit/miss
      if (hit) pf_stats.demand_total_hits++;
      else     pf_stats.demand_total_misses++;
      // Debug: first 5 demand tileload misses
      static int demand_dbg = 0;
      if (!hit && demand_dbg < 5) {
        demand_dbg++;
        fmt::print("[DM_DBG] dm#{} vaddr=0x{:x} paddr=0x{:x} blk=0x{:x} pf_lines_tracked={}\n",
            demand_dbg, handle_pkt.v_address.to<uint64_t>(),
            handle_pkt.address.to<uint64_t>(), blk_addr, pf_filled_lines.size());
      }

      // Check if prefetch brought this specific address
      auto it = pf_filled_lines.find(blk_addr);
      if (it != pf_filled_lines.end()) {
        if (hit) {
          pf_stats.demand_hit_pf_line++;
          pf_stats.total_pf_to_demand_gap += (now - it->second);
          pf_stats.pf_to_demand_count++;
        } else {
          pf_stats.demand_miss_pf_evicted++;
        }
        pf_filled_lines.erase(it);
      } else {
        pf_stats.demand_no_pf++;
      }
    }
  }

  auto metadata_thru = handle_pkt.pf_metadata;
  // Populate tile metadata bridge so prefetcher can read it via intern_->current_tile_info
  current_tile_info = {handle_pkt.is_amx_tileload, handle_pkt.tile_group_id, handle_pkt.instr_id, handle_pkt.tile_subidx, handle_pkt.tile_num_children};
  if (should_activate_prefetcher(handle_pkt)) {
    metadata_thru = impl_prefetcher_cache_operate(module_address(handle_pkt), handle_pkt.ip, hit, useful_prefetch, handle_pkt.type, metadata_thru);
  }

  // update replacement policy
  const auto way_idx = std::distance(set_begin, way);
  impl_update_replacement_state(handle_pkt.cpu, get_set_index(handle_pkt.address), way_idx, module_address(handle_pkt), handle_pkt.ip, {}, handle_pkt.type,
                                hit);

  if (hit) {
    sim_stats.hits.increment(std::pair{handle_pkt.type, handle_pkt.cpu});

    response_type response{handle_pkt.address, handle_pkt.v_address, way->data, metadata_thru, handle_pkt.instr_depend_on_me};
    // Tag serve_level: hit at this cache level
    if (NAME.find("L2C") != std::string::npos) response.serve_level = 0;      // L2 hit
    else if (NAME.find("LLC") != std::string::npos) response.serve_level = 1;  // LLC hit
    for (auto* ret : handle_pkt.to_return) {
      ret->push_back(response);
    }

    way->dirty |= (handle_pkt.type == access_type::WRITE);

    // update prefetch stats and reset prefetch bit
    if (useful_prefetch) {
      ++sim_stats.pf_useful;
      way->prefetch = false;
    }

    // Update sidecar tile state if this is a tile child hit
    // Skip during warmup to avoid wall-clock overhead from map/vector ops
    if (!warmup && handle_pkt.is_amx_tileload && !g_disable_tile_sidecar) {
      auto* sc = find_or_alloc_sidecar(handle_pkt.tile_group_id, handle_pkt.tile_num_children);
      if (sc) {
        uint8_t idx = handle_pkt.tile_subidx;
        if (idx < 16) {
          if (!(sc->observed_mask & (1 << idx))) {
            sc->observed_mask |= (1 << idx);
            sc->observed_children++;
          }
          if (!(sc->completed_mask & (1 << idx))) {
            sc->completed_mask |= (1 << idx);
            sc->hit_mask |= (1 << idx);
            sc->completed_count++;
            sim_stats.tile_children_hits++;
            // Per-child-index hit tracking (A: tmm4=199/tmm5=200, B: tmm6=201/tmm7=202)
            if (idx < 16) {
              uint8_t tmm = handle_pkt.tile_dest_tmm;
              if (tmm == 199 || tmm == 200) sim_stats.a_tile_child_hit[idx]++;
              else if (tmm == 201 || tmm == 202) sim_stats.b_tile_child_hit[idx]++;
            }

            // Check if tile is now fully complete
            if (sc->is_complete()) {
              sc->completion_cycle = current_time.time_since_epoch() / clock_period;
              sim_stats.tile_completions++;
              tile_finalize(sc->tile_group_id, sc->completion_cycle);
              tile_sidecar.erase(sc->tile_group_id);
            }
          }
        }
      }
    }

    // --- Tile instrumentation: L1D hit ---
    if (NAME.find("L1D") != std::string::npos) {
      uint64_t la = handle_pkt.v_address.to<uint64_t>();
      uint64_t cur = current_time.time_since_epoch() / clock_period;
      std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
      std::vector<uint64_t> completed_tids;
      for (auto& [tid, rec] : g_tile_masters) {
        bool all_done = true;
        for (auto& sr : rec.subreqs) {
          if (sr.line_addr == la && !sr.l1_hit) {
            sr.l1_hit          = true;
            sr.fill_cycle      = cur;
            sr.l1d_dispatch_cycle = cur;
            rec.l1_hits++;
          }
          if (!sr.l1_hit && sr.fill_cycle == 0) all_done = false;
        }
        if (all_done && rec.issue_cycle > 0) {
          completed_tids.push_back(tid);
        }
      }
      for (uint64_t tid : completed_tids) {
        tile_finalize(tid, cur);
      }
    }

    // --- Tile instrumentation: L2C hit (tag subreq so L1D fill knows source) ---
    if (NAME.find("L2C") != std::string::npos) {
      uint64_t la = handle_pkt.v_address.to<uint64_t>();
      std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
      for (auto& [tid, rec] : g_tile_masters) {
        for (auto& sr : rec.subreqs) {
          if (sr.line_addr == la && !sr.l1_hit && !sr.l2_hit) {
            sr.l2_hit = true;
            rec.l2_hits++;
            break;  // one match per subreq
          }
        }
      }
    }

    // --- Tile instrumentation: LLC hit ---
    if (NAME.find("LLC") != std::string::npos) {
      uint64_t la = handle_pkt.v_address.to<uint64_t>();
      std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
      for (auto& [tid, rec] : g_tile_masters) {
        for (auto& sr : rec.subreqs) {
          if (sr.line_addr == la && !sr.l1_hit && !sr.l2_hit && !sr.llc_hit) {
            sr.llc_hit = true;
            rec.llc_hits++;
            break;
          }
        }
      }
    }
  }

  return hit;
}

auto CACHE::mshr_and_forward_packet(const tag_lookup_type& handle_pkt) -> std::pair<mshr_type, request_type>
{
  mshr_type to_allocate{handle_pkt, current_time};

  request_type fwd_pkt;

  fwd_pkt.asid[0] = handle_pkt.asid[0];
  fwd_pkt.asid[1] = handle_pkt.asid[1];
  fwd_pkt.type = (handle_pkt.type == access_type::WRITE) ? access_type::RFO : handle_pkt.type;
  fwd_pkt.pf_metadata = handle_pkt.pf_metadata;
  fwd_pkt.cpu = handle_pkt.cpu;

  fwd_pkt.address = handle_pkt.address;
  fwd_pkt.v_address = handle_pkt.v_address;
  fwd_pkt.data = handle_pkt.data;
  fwd_pkt.instr_id = handle_pkt.instr_id;
  fwd_pkt.ip = handle_pkt.ip;

  // Propagate tile metadata to lower levels (paper-faithful semantics)
  // Non-contiguous tiles are now supported via sidecar state.
  fwd_pkt.is_amx_tileload = handle_pkt.is_amx_tileload;
  fwd_pkt.tile_group_id = handle_pkt.tile_group_id;
  fwd_pkt.tile_num_children = handle_pkt.tile_num_children;
  fwd_pkt.tile_subidx = handle_pkt.tile_subidx;
  fwd_pkt.tile_is_store = handle_pkt.tile_is_store;
  fwd_pkt.tile_burst = handle_pkt.tile_burst;  // Legacy flag
  fwd_pkt.tile_dest_tmm = handle_pkt.tile_dest_tmm;



  fwd_pkt.instr_depend_on_me = handle_pkt.instr_depend_on_me;
  // Always request response so the MSHR entry can be freed on fill.
  // Without this, L2-only prefetch (skip_fill=true, prefetch_from_this=true) would
  // set response_requested=false → L2C never responds → L1D MSHR hangs → deadlock.
  fwd_pkt.response_requested = true;

  return std::pair{std::move(to_allocate), std::move(fwd_pkt)};
}

bool CACHE::handle_miss(const tag_lookup_type& handle_pkt)
{
  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} type: {} local_prefetch: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, access_type_names.at(champsim::to_underlying(handle_pkt.type)), handle_pkt.prefetch_from_this,
               current_time.time_since_epoch() / clock_period);
  }

  cpu = handle_pkt.cpu;

  // ============================================================================
  // Queue-Local Tile Sidecar — Observational Tile State Tracking
  // ============================================================================
  if (!warmup && handle_pkt.is_amx_tileload && !g_disable_tile_sidecar) {
    auto* sc = find_or_alloc_sidecar(handle_pkt.tile_group_id, handle_pkt.tile_num_children);
    if (sc) {
      uint8_t idx = handle_pkt.tile_subidx;
      if (idx < 16 && !(sc->observed_mask & (1 << idx))) {
        sc->observed_mask |= (1 << idx);
        sc->observed_children++;
      }
      sim_stats.tile_children_pending++;
      // Per-child-index miss tracking
      if (idx < 16) {
        uint8_t tmm = handle_pkt.tile_dest_tmm;
        if (tmm == 199 || tmm == 200) sim_stats.a_tile_child_miss[idx]++;
        else if (tmm == 201 || tmm == 202) sim_stats.b_tile_child_miss[idx]++;
      }
    }
  }

  // ============================================================================
  // OptAMX Tile-Level MSHR Coalescing
  // ============================================================================
  // For tile children: 1 MSHR entry per tile instruction.
  // First child miss → allocates 1 MSHR entry (the tile entry).
  // Subsequent children → find that MSHR entry by tile_group_id, coalesce INTO it.
  // Each child's request is still sent to L2C individually.
  // Only 1 MSHR slot consumed per tile.
  // ============================================================================

  // Tile-level MSHR coalescing at L1D only (16 CL → 1 MSHR)
  // When g_per_child_lfb is true, skip tile coalescing so each child uses its own LFB entry
  // (realistic HW: each cacheline miss consumes a separate LFB slot).
  if (handle_pkt.is_amx_tileload && !g_disable_tile_sidecar && !g_per_child_lfb && NAME.find("L1D") != std::string::npos) {
    // First: check if same ADDRESS is already in a tile MSHR entry's children
    // (address-level coalescing within tile entry)
    for (auto& entry : MSHR) {
      if (entry.is_tile_entry && entry.tile_group_id == handle_pkt.tile_group_id) {
        // Check if this address already has a child in the tile entry
        auto addr_match = matches_address(handle_pkt.address);
        for (auto& child : entry.tile_children) {
          if (addr_match(child) && !child.fill_installed) {
            // Same address within same tile, not yet installed — merge dependents
            auto& merged_deps = child.instr_depend_on_me;
            for (auto dep : handle_pkt.instr_depend_on_me) {
              if (std::find(merged_deps.begin(), merged_deps.end(), dep) == merged_deps.end())
                merged_deps.push_back(dep);
            }
            for (auto* ret : handle_pkt.to_return) {
              if (std::find(child.to_return.begin(), child.to_return.end(), ret) == child.to_return.end())
                child.to_return.push_back(ret);
            }
            sim_stats.tile_sidecar_coalesced++;
            sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
            return true;
          }
        }

        // Same tile, different address — add new child to existing tile entry
        auto [alloc_mshr, fwd_pkt] = mshr_and_forward_packet(handle_pkt);
        // Always use RQ for tile children (including prefetch tiles) to avoid
        // deadlock from L2C PQ overflow — tile MSHR holds the slot, children
        // must complete to release it.
        bool success = lower_level->add_rq(fwd_pkt);
        if (!success) return false;

        if (handle_pkt.type != access_type::PREFETCH) {
          auto queue_cycles = (current_time - handle_pkt.event_cycle) / clock_period;
          if (queue_cycles > 0) sim_stats.total_miss_queue_cycles += queue_cycles;
        }

        mshr_type::tile_child_entry child{};
        child.address = handle_pkt.address;
        child.v_address = handle_pkt.v_address;
        child.ip = handle_pkt.ip;
        child.instr_id = handle_pkt.instr_id;
        child.cpu = handle_pkt.cpu;
        child.pf_metadata = handle_pkt.pf_metadata;
        child.type = handle_pkt.type;
        child.prefetch_from_this = handle_pkt.prefetch_from_this;
        child.skip_fill = handle_pkt.skip_fill;
        child.child_idx = handle_pkt.tile_subidx;
        child.instr_depend_on_me = handle_pkt.instr_depend_on_me;
        child.to_return = handle_pkt.to_return;
        child.sent_to_lower = true;
        entry.tile_children.push_back(std::move(child));

        sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
        return true;
      }
    }

    // Also check non-tile MSHR entries for address match (normal line merge)
    auto mshr_entry = std::find_if(std::begin(MSHR), std::end(MSHR),
        [&](const auto& entry) { return !entry.is_tile_entry && matches_address(handle_pkt.address)(entry); });
    if (mshr_entry != MSHR.end()) {
      if (mshr_entry->type == access_type::PREFETCH && handle_pkt.type != access_type::PREFETCH) {
        if (mshr_entry->prefetch_from_this) ++sim_stats.pf_useful;
      }
      mshr_type to_merge{handle_pkt, current_time};
      sim_stats.mshr_merge.increment(std::pair{to_merge.type, to_merge.cpu});
      *mshr_entry = mshr_type::merge(*mshr_entry, to_merge);
      mshr_entry->tile_attachments.push_back({handle_pkt.tile_group_id, handle_pkt.tile_subidx});
      sim_stats.tile_attachment_merges++;
      sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
      return true;
    }

    // No existing tile entry and no address match — allocate new tile MSHR entry
    // Unified LFB model: all MSHR entries (tile + non-tile) share LFB_SIZE pool.
    // Tile entries use 1 slot per tile parent.
    if (MSHR.size() >= LFB_SIZE) return false;

    // Prefetch MSHR cap: reserve at least 2 MSHR slots for demand to prevent deadlock.
    // Without this, PF tiles can fill all 12 MSHR slots → demand can't enter → deadlock.
    if (handle_pkt.type == access_type::PREFETCH) {
      auto pf_count = std::count_if(std::begin(MSHR), std::end(MSHR),
          [](const auto& e) { return e.type == access_type::PREFETCH; });
      if (pf_count >= 1) return false;  // Max 1 PF MSHR entry, minimal demand impact
    }

    auto [alloc_mshr, fwd_pkt] = mshr_and_forward_packet(handle_pkt);
    // Always use RQ for tile children to avoid L2C PQ overflow deadlock
    bool success = lower_level->add_rq(fwd_pkt);
    if (!success) return false;

    if (handle_pkt.type != access_type::PREFETCH) {
      auto queue_cycles = (current_time - handle_pkt.event_cycle) / clock_period;
      if (queue_cycles > 0) sim_stats.total_miss_queue_cycles += queue_cycles;
    }

    // Create tile MSHR entry
    alloc_mshr.is_tile_entry = true;
    alloc_mshr.tile_group_id = handle_pkt.tile_group_id;
    alloc_mshr.tile_num_children = handle_pkt.tile_num_children;

    mshr_type::tile_child_entry first_child{};
    first_child.address = handle_pkt.address;
    first_child.v_address = handle_pkt.v_address;
    first_child.ip = handle_pkt.ip;
    first_child.instr_id = handle_pkt.instr_id;
    first_child.cpu = handle_pkt.cpu;
    first_child.pf_metadata = handle_pkt.pf_metadata;
    first_child.type = handle_pkt.type;
    first_child.prefetch_from_this = handle_pkt.prefetch_from_this;
    first_child.skip_fill = handle_pkt.skip_fill;
    first_child.child_idx = handle_pkt.tile_subidx;
    first_child.instr_depend_on_me = handle_pkt.instr_depend_on_me;
    first_child.to_return = handle_pkt.to_return;
    first_child.sent_to_lower = true;
    alloc_mshr.tile_children.push_back(std::move(first_child));

    // Clear the parent-level dependents (children own them now)
    alloc_mshr.instr_depend_on_me.clear();
    alloc_mshr.to_return.clear();

    MSHR.emplace_back(std::move(alloc_mshr));
    sim_stats.tile_mshr_allocs++;
    sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
    return true;
  }

  // ============================================================================
  // Non-tile path: standard MSHR Address Match merge
  // ============================================================================

  mshr_type to_allocate{handle_pkt, current_time};
  auto mshr_pkt = mshr_and_forward_packet(handle_pkt);

  // Check non-tile MSHR entries for address match
  auto mshr_entry = std::find_if(std::begin(MSHR), std::end(MSHR),
      [&](const auto& entry) { return !entry.is_tile_entry && matches_address(handle_pkt.address)(entry); });

  if (mshr_entry != MSHR.end())
  {
    if (mshr_entry->type == access_type::PREFETCH && handle_pkt.type != access_type::PREFETCH) {
      if (mshr_entry->prefetch_from_this) {
        ++sim_stats.pf_useful;
      }
    }
    sim_stats.mshr_merge.increment(std::pair{to_allocate.type, to_allocate.cpu});
    *mshr_entry = mshr_type::merge(*mshr_entry, to_allocate);

    sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
    return true;
  }

  // Also check if any tile entry has a child with the same address (already inflight)
  {
    auto addr_match = matches_address(handle_pkt.address);
    for (auto& entry : MSHR) {
      if (!entry.is_tile_entry) continue;
      for (auto& child : entry.tile_children) {
        if (addr_match(child) && !child.fill_installed) {
          // Address already inflight as tile child, not yet installed — merge dependents
          for (auto dep : handle_pkt.instr_depend_on_me) {
            if (std::find(child.instr_depend_on_me.begin(), child.instr_depend_on_me.end(), dep) == child.instr_depend_on_me.end())
              child.instr_depend_on_me.push_back(dep);
          }
          for (auto* ret : to_allocate.to_return) {
            if (std::find(child.to_return.begin(), child.to_return.end(), ret) == child.to_return.end())
              child.to_return.push_back(ret);
          }
          sim_stats.mshr_merge.increment(std::pair{to_allocate.type, to_allocate.cpu});
          sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
          return true;
        }
      }
    }
  }

  // No MSHR match — new cacheline miss, allocate normal MSHR entry
  // Unified LFB model: all MSHR entries (tile + non-tile) share LFB_SIZE pool.
  // Tile entries use 1 slot per tile parent.
  if (MSHR.size() >= LFB_SIZE) {
    return false;
  }

  // Prefetch MSHR cap: reserve at least 2 slots for demand to prevent deadlock.
  if (handle_pkt.type == access_type::PREFETCH) {
    auto pf_count = std::count_if(std::begin(MSHR), std::end(MSHR),
        [](const auto& e) { return e.type == access_type::PREFETCH; });
    if (pf_count >= static_cast<long>(LFB_SIZE) / 2) return false;
  }

  // Tile children (demand and prefetch) use RQ to avoid L2C PQ overflow deadlock.
  const bool is_tile_child = handle_pkt.is_amx_tileload;
  const bool send_to_rq = is_tile_child || prefetch_as_load || handle_pkt.type != access_type::PREFETCH;
  bool success = send_to_rq ? lower_level->add_rq(mshr_pkt.second) : lower_level->add_pq(mshr_pkt.second);

  if (!success) {
    return false;
  }

  if (mshr_pkt.second.response_requested) {
    if (handle_pkt.type != access_type::PREFETCH) {
      auto queue_cycles = (current_time - handle_pkt.event_cycle) / clock_period;
      if (queue_cycles > 0) {
        sim_stats.total_miss_queue_cycles += queue_cycles;
      }
    }

    MSHR.emplace_back(std::move(mshr_pkt.first));
    if (handle_pkt.type == access_type::TRANSLATION)
      sim_stats.translation_mshr_allocs++;
    else if (handle_pkt.is_amx_tileload)
      sim_stats.tile_leaked_to_nontile++;  // BUG: tile child in non-tile path!
    else {
      sim_stats.non_tile_mshr_allocs++;
      switch (handle_pkt.type) {
        case access_type::LOAD:     sim_stats.non_tile_mshr_by_load++; break;
        case access_type::RFO:      sim_stats.non_tile_mshr_by_rfo++; break;
        case access_type::WRITE:    sim_stats.non_tile_mshr_by_write++; break;
        case access_type::PREFETCH: sim_stats.non_tile_mshr_by_prefetch++; break;
        default:                    sim_stats.non_tile_mshr_by_other++; break;
      }
    }
  }

  sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
  return true;
}

bool CACHE::handle_write(const tag_lookup_type& handle_pkt)
{
  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} type: {} local_prefetch: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, access_type_names.at(champsim::to_underlying(handle_pkt.type)), handle_pkt.prefetch_from_this,
               current_time.time_since_epoch() / clock_period);
  }

  mshr_type to_allocate{handle_pkt, current_time};
  to_allocate.data_promise.ready_at(current_time + (warmup ? champsim::chrono::clock::duration{} : FILL_LATENCY));
  inflight_writes.push_back(to_allocate);

  sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});

  return true;
}


template <bool UpdateRequest>
auto CACHE::initiate_tag_check(champsim::channel* ul)
{
  return [time = current_time + (warmup ? champsim::chrono::clock::duration{} : HIT_LATENCY), ul, name = NAME](const auto& entry) {
    CACHE::tag_lookup_type retval{entry};
    retval.event_cycle = time;

    if constexpr (UpdateRequest) {
      if (entry.response_requested) {
        retval.to_return = {&ul->returned};
      }
    } else {
      (void)ul; // supress warning about ul being unused
    }

    if constexpr (champsim::debug_print) {
      fmt::print("[TAG] initiate_tag_check instr_id: {} address: {} v_address: {} type: {} response_requested: {}\n", retval.instr_id, retval.address,
                 retval.v_address, access_type_names.at(champsim::to_underlying(retval.type)), !std::empty(retval.to_return));
    }

    return retval;
  };
}

long CACHE::operate()
{
  long progress{0};

  auto is_ready = [time = current_time](const auto& entry) {
    return entry.event_cycle <= time;
  };
  auto is_translated = [](const auto& entry) {
    return entry.is_translated;
  };

  if (NAME.find("L1D") != std::string::npos && MSHR.size() >= MSHR_SIZE) {
      static thread_local uint64_t dump_counter = 0;
      if (dump_counter++ % 1000 == 0) {
//          fmt::print(stderr, "[{}_MSHR_DUMP] cycle: {} MSHR Full. Count: {}\n", NAME, current_time.time_since_epoch()/clock_period, dump_counter);
          for (const auto& entry : MSHR) {
//              fmt::print(stderr, " - Addr: {} Type: {} Time: {} TM: {}\n", entry.address, 
//                          access_type_names.at(champsim::to_underlying(entry.type)), 
//                          entry.time_enqueued.time_since_epoch()/clock_period, entry.is_tile_master);
          }
      }
  }

  for (auto* ul : upper_levels) {
    ul->check_collision();
  }

  // Finish returns
  std::for_each(std::cbegin(lower_level->returned), std::cend(lower_level->returned), [this](const auto& pkt) { this->finish_packet(pkt); });
  progress += std::distance(std::cbegin(lower_level->returned), std::cend(lower_level->returned));
  lower_level->returned.clear();

  // Finish translations
  if (lower_translate != nullptr) {
    std::for_each(std::cbegin(lower_translate->returned), std::cend(lower_translate->returned), [this](const auto& pkt) { this->finish_translation(pkt); });
    progress += std::distance(std::cbegin(lower_translate->returned), std::cend(lower_translate->returned));
    lower_translate->returned.clear();
  }



  // Perform fills from MSHR (unified LFB model — non-tile entries with data ready)
  champsim::bandwidth fill_bw{MAX_FILL};
  {
    // Scan non-tile MSHR entries whose data_promise is ready for fill installation
    auto is_ready_nontile = [time = current_time](const auto& x) {
      return !x.is_tile_entry && x.data_promise.is_ready_at(time);
    };
    // Partition ready non-tile entries to the front
    auto ready_end = std::stable_partition(std::begin(MSHR), std::end(MSHR), is_ready_nontile);
    // Consume up to fill bandwidth
    auto fill_end = ready_end;
    if (std::distance(std::begin(MSHR), ready_end) > static_cast<long>(fill_bw.amount_remaining())) {
      fill_end = std::next(std::begin(MSHR), static_cast<long>(fill_bw.amount_remaining()));
    }
    auto complete_end = std::find_if_not(std::begin(MSHR), fill_end, [this](const auto& x) { return this->handle_fill(x); });
    fill_bw.consume(std::distance(std::begin(MSHR), complete_end));
    MSHR.erase(std::begin(MSHR), complete_end);
  }
  
  // Also handle inflight_writes
  {
    auto [fill_begin, fill_end] = champsim::get_span_p(std::cbegin(inflight_writes), std::cend(inflight_writes), fill_bw,
                                                       [time = current_time](const auto& x) { return x.data_promise.is_ready_at(time); });
    auto complete_end = std::find_if_not(fill_begin, fill_end, [this](const auto& x) { return this->handle_fill(x); });
    fill_bw.consume(std::distance(fill_begin, complete_end));
    inflight_writes.erase(fill_begin, complete_end);
  }

  // ============================================================================
  // Tile MSHR entry: process ready children fills
  // ============================================================================
  // Each tile child whose fill_ready_at has arrived gets installed into cache
  // (via handle_fill) and its dependents are notified. The tile MSHR entry is
  // removed when ALL children are installed.
  // ============================================================================
  {
    for (auto& entry : MSHR) {
      if (!entry.is_tile_entry) continue;
      for (auto& child : entry.tile_children) {
        if (child.fill_received && !child.fill_installed && child.fill_ready_at <= current_time
            && fill_bw.amount_remaining() > 0) {
          mshr_type temp_fill{};
          temp_fill.address = child.address;
          temp_fill.v_address = child.v_address;
          temp_fill.ip = child.ip;
          temp_fill.instr_id = child.instr_id;
          temp_fill.cpu = child.cpu;
          temp_fill.type = child.type;
          temp_fill.prefetch_from_this = child.prefetch_from_this;
          temp_fill.skip_fill = child.skip_fill;
          temp_fill.time_enqueued = entry.time_enqueued;
          temp_fill.serve_level = child.serve_level;
          temp_fill.instr_depend_on_me = child.instr_depend_on_me;
          temp_fill.to_return = child.to_return;

          mshr_type::returned_value rv{child.data, child.pf_metadata};
          temp_fill.data_promise = champsim::waitable{rv, child.fill_ready_at};

          current_fill_is_tile = true;  // hint for replacement policy
          bool fill_ok = handle_fill(temp_fill);
          current_fill_is_tile = false;
          if (fill_ok) {
            child.fill_installed = true;
            fill_bw.consume(1);
            sidecar_mark_child_complete(entry.tile_group_id, child.child_idx);
          }
        }
      }
    }

    // Remove completed tile entries using erase-remove idiom
    // (std::deque::erase invalidates all iterators, so stored iterators can't be used)
    MSHR.erase(std::remove_if(MSHR.begin(), MSHR.end(), [](const mshr_type& e) {
      if (!e.is_tile_entry || e.tile_children.empty()) return false;
      return std::all_of(e.tile_children.begin(), e.tile_children.end(),
          [](const auto& c) { return c.fill_installed; });
    }), MSHR.end());
  }

  // Initiate tag checks
  const champsim::bandwidth::maximum_type bandwidth_from_tag_checks{champsim::to_underlying(MAX_TAG) * (long)(HIT_LATENCY / clock_period)
                                                                    - (long)std::size(inflight_tag_check)};
  champsim::bandwidth initiate_tag_bw{std::clamp(bandwidth_from_tag_checks, champsim::bandwidth::maximum_type{0}, MAX_TAG)};
  auto can_translate = [avail = (std::size(translation_stash) < static_cast<std::size_t>(MSHR_SIZE))](const auto& entry) {
    return avail || entry.is_translated;
  };
  auto stash_bandwidth_consumed =
      champsim::transform_while_n(translation_stash, std::back_inserter(inflight_tag_check), initiate_tag_bw, is_translated, initiate_tag_check<false>());
  initiate_tag_bw.consume(stash_bandwidth_consumed);
  std::vector<long long> channels_bandwidth_consumed{};

  if (std::size(upper_levels) > 1) {
    std::rotate(upper_levels.begin(), upper_levels.begin() + 1, upper_levels.end());
  }

  // upper levels get an equal portion of the remaining bandwidth
  champsim::bandwidth::maximum_type per_upper_bandwidth =
      std::size(upper_levels) >= 1
          ? (champsim::bandwidth::maximum_type)std::max((size_t)initiate_tag_bw.amount_remaining() / std::size(upper_levels), size_t{1})
          : champsim::bandwidth::maximum_type{};

  for (auto* ul : upper_levels) {
    // Diagnostic moved inside loop
    for (auto q : {std::ref(ul->WQ), std::ref(ul->RQ), std::ref(ul->PQ)}) {
      // this needs to be in this loop, we need to ensure that for cases where bandwidth doesn't divide nicely across upstreams,
      // we don't accidentally consume more bandwidth than expected
      champsim::bandwidth per_upper_tag_bw{std::min(per_upper_bandwidth, champsim::bandwidth::maximum_type{initiate_tag_bw.amount_remaining()})};
      
      bool is_rq = (&q.get() == &ul->RQ);
      bool is_pq = (&q.get() == &ul->PQ);
      if (NAME.find("L2C") != std::string::npos && (is_rq || is_pq) && ul->rq_occupancy() > 0) {
          if (!q.get().empty()) {
              auto& head = q.get().front();
              if (head.address == champsim::address{0x2a48410c0}) {
//                  fmt::print(stderr, "[{}_TRACE] L2 OP LOOP addr: {} RR: {} Trans: {}\n", NAME, head.address, head.response_requested, head.is_translated);
              }
              bool can_trans = can_translate(head);
//              fmt::print(stderr, "[{}_OP_LOOP] cycle: {} Q: {} Sz: {} Limit: {} CanTrans: {} Trans: {} Cond: {}/{}/{} RR: {}\n", 
//                          NAME, current_time.time_since_epoch()/clock_period, (is_rq ? "RQ" : "PQ"), q.get().size(), 
//                          per_upper_tag_bw.amount_remaining(), can_trans, head.is_translated,
//                          !q.get().empty(), per_upper_tag_bw.amount_remaining() > 0, can_trans, head.response_requested);
          }
      }
      
      long bandwidth_consumed = 0;
      auto tag_check_func = initiate_tag_check<true>(ul);
      while (!q.get().empty() && per_upper_tag_bw.amount_remaining() > 0 && can_translate(q.get().front())) {
          auto pkt = tag_check_func(q.get().front());
          inflight_tag_check.push_back(std::move(pkt));
          q.get().pop_front();
          per_upper_tag_bw.consume(1);
          bandwidth_consumed++;
          if (NAME.find("L2C") != std::string::npos) {
//             fmt::print(stderr, "[{}_MANUAL_MOVE] Moved RQ entry. Inflight: {}\n", NAME, inflight_tag_check.size());
          }
      }
      // auto bandwidth_consumed =
      //    champsim::transform_while_n(q.get(), std::back_inserter(inflight_tag_check), per_upper_tag_bw, can_translate, initiate_tag_check<true>(ul));
      channels_bandwidth_consumed.push_back(bandwidth_consumed);
      initiate_tag_bw.consume(bandwidth_consumed);
    }
  }

  // internal_PQ processing moved below tile_PQ expansion (dedicated PF bandwidth)

  // ============================================================================
  // Idle-Aware Tile Prefetch: use ONLY leftover TAG bandwidth
  // ============================================================================
  // 1. Expand tile_PQ → internal_PQ (up to 1 tile/cy, only if idle TAG exists)
  // 2. Drain internal_PQ using leftover TAG bandwidth (no dedicated/guaranteed)
  // 3. PF uses sidecar MSHR coalescing (1 MSHR per tile, same as demand)
  // 4. Back off when L2C RQ is busy (>50% full)
  // ============================================================================
  {
    long leftover = initiate_tag_bw.amount_remaining();

    // Check L2C pressure
    bool l2c_busy = false;
    if (lower_level != nullptr) {
      auto ll_rq_occ = lower_level->rq_occupancy();
      auto ll_rq_sz = lower_level->rq_size();
      l2c_busy = (ll_rq_occ * 2 > ll_rq_sz);
    }

    // Idle-aware PF: leftover TAG + L2C not busy. No fill check needed
    // (PF goes through handle_miss → L2C, fill comes back later when fill is idle)
    if (leftover > 0 && !l2c_busy) {
      // Expand 1 tile from tile_PQ into internal_PQ
      if (!tile_PQ.empty()) {
        auto& tile = tile_PQ.front();
        for (uint8_t i = 0; i < tile.addrs.size(); i++) {
          request_type pf_packet;
          pf_packet.type = access_type::PREFETCH;
          pf_packet.pf_metadata = 0;
          pf_packet.cpu = cpu;
          pf_packet.address = tile.addrs[i];
          pf_packet.v_address = virtual_prefetch ? tile.addrs[i] : champsim::address{};
          pf_packet.is_translated = !virtual_prefetch;
          pf_packet.is_amx_tileload = true;
          pf_packet.tile_group_id = tile.tile_group_id;
          pf_packet.tile_subidx = i;
          pf_packet.tile_num_children = static_cast<uint8_t>(tile.addrs.size());
          internal_PQ.emplace_back(pf_packet, true, !tile.fill_this_level);
        }
        tile_PQ.pop_front();
      }

      // Drain internal_PQ using ONLY leftover bandwidth (pure idle-aware)
      champsim::bandwidth pf_tag_bw{champsim::bandwidth::maximum_type{leftover}};
      auto pf_bw_consumed =
          champsim::transform_while_n(internal_PQ, std::back_inserter(inflight_tag_check), pf_tag_bw, can_translate, initiate_tag_check<false>());
      initiate_tag_bw.consume(pf_bw_consumed);
    }
  }

#ifdef ENABLE_STREAM_ENGINE
  // Stream Engine: Process internal_STREAM_PQ -> translation_stash (or direct issue)
  // We reuse 'can_translate' to check MSHR/Stash space availability.
  // BUT we want to bypass inflight_tag_check. 
  // Mod: transform_while_n to 'translation_stash' directly if not translated?
  // Actually, initiate_tag_check puts them in retval.
  // We want to route them to issue_translation, then finisht_translation handles them.
  // So adding them to inflight_tag_check is fine IF we filter them out of do_handle_miss.
  
  // Wait, if we add to inflight_tag_check, they consume MAX_TAG bandwidth.
  // AND they will be processed by do_handle_miss.
  // We want to avoid Tag Check entirely.
  
  // Strategy:
  // 1. Issue translation directly here.
  // 2. If valid, move to a separate 'inflight_stream_translation' or just issue to lower_translate.
  
  // Simpler Strategy for now (fitting existing structure):
  // Let them go through inflight_tag_check but flag them to SKIP handle_miss.
  // We'll update do_handle_miss to ignore is_stream.
  
  // Decoupled Stream Translation Issue (DISABLED for Parallel SB Mode)
  // In parallel SB mode, demand loads are routed in the SB extraction loop below.
#endif

  // Issue translations
  std::for_each(std::begin(inflight_tag_check), std::end(inflight_tag_check), [this](auto& x) { this->issue_translation(x); });
  std::for_each(std::begin(translation_stash), std::end(translation_stash), [this](auto& x) { this->issue_translation(x); });

  // Find entries that would be ready except that they have not finished translation, move them to the stash
  auto [last_not_missed, stash_end] = champsim::extract_if(std::begin(inflight_tag_check), std::end(inflight_tag_check), std::back_inserter(translation_stash),
                                                           [is_ready, is_translated](const auto& x) { return is_ready(x) && !is_translated(x); });
  progress += std::distance(last_not_missed, std::end(inflight_tag_check));
  inflight_tag_check.erase(last_not_missed, std::end(inflight_tag_check));

  // Perform tag checks
  auto do_handle_miss = [this](const auto& pkt) {
    if (pkt.type == access_type::WRITE && !this->match_offset_bits) {
      return this->handle_write(pkt); // Treat writes (that is, writebacks) like fills
    }
    return this->handle_miss(pkt); // Treat writes (that is, stores) like reads
  };
  champsim::bandwidth tag_check_bw{MAX_TAG};
  auto [tag_check_ready_begin, tag_check_ready_end] =
      champsim::get_span_p(std::begin(inflight_tag_check), std::end(inflight_tag_check), tag_check_bw,
                           [is_ready, is_translated](const auto& pkt) { return is_ready(pkt) && is_translated(pkt); });
  // Count tile vs scalar loads entering tag check (L1D only)
  if (!warmup && NAME.find("L1D") != std::string::npos) {
    for (auto it = tag_check_ready_begin; it != tag_check_ready_end; ++it) {
      if (it->type == access_type::LOAD || it->type == access_type::RFO) {
        if (it->is_amx_tileload)
          sim_stats.tile_loads_tag_check++;
        else
          sim_stats.scalar_loads_tag_check++;
      }
    }
  }
  auto hits_end = std::stable_partition(tag_check_ready_begin, tag_check_ready_end, [this](const auto& pkt) { return this->try_hit(pkt); });

  // Normal miss handling: PF tile misses go through handle_miss → sidecar MSHR → L1D fill.
  // With dedicated PF bandwidth, PF arrives D iterations early → no MSHR competition.
  auto finish_tag_check_end = std::stable_partition(hits_end, tag_check_ready_end, do_handle_miss);
  tag_check_bw.consume(std::distance(tag_check_ready_begin, finish_tag_check_end));
  inflight_tag_check.erase(tag_check_ready_begin, finish_tag_check_end);

  impl_prefetcher_cycle_operate();

  // Sidecar tile state is purely passive — no per-cycle scheduling needed.
  // Track active sidecar count for stats.
  if (!warmup && !tile_sidecar.empty()) {
    sim_stats.total_cycles_with_active_parents++;
  }

  if (!warmup && LFB_SIZE > 0) {
    // Unified LFB model: all MSHR entries share the pool
    auto total_lfb_pressure = std::size(MSHR);
    auto non_tile_mshr = static_cast<std::size_t>(std::count_if(std::begin(MSHR), std::end(MSHR),
        [](const auto& e) { return !e.is_tile_entry; }));
    auto demand_outstanding = std::count_if(std::begin(MSHR), std::end(MSHR),
        [](const auto& entry) { return entry.type != access_type::PREFETCH; });
    if (demand_outstanding > 0 && total_lfb_pressure >= LFB_SIZE) {
      ++sim_stats.l1d_pend_miss_fb_full_cycles;
    }
    // LFB occupancy diagnostics (L1D only)
    if (NAME.find("L1D") != std::string::npos) {
      auto tile_mshr = std::size(MSHR) - non_tile_mshr;
      sim_stats.lfb_occupancy_sum += total_lfb_pressure;
      sim_stats.lfb_occupancy_cycles++;
      if (total_lfb_pressure > sim_stats.lfb_occupancy_max) {
        sim_stats.lfb_occupancy_max = total_lfb_pressure;
      }
      sim_stats.non_tile_mshr_occupancy_sum += non_tile_mshr;
      sim_stats.tile_mshr_occupancy_sum += tile_mshr;
      sim_stats.lfb_only_occupancy_sum += 0;  // No separate LFB in unified model
      if (non_tile_mshr > sim_stats.non_tile_mshr_occupancy_max)
        sim_stats.non_tile_mshr_occupancy_max = non_tile_mshr;
      if (tile_mshr > sim_stats.tile_mshr_occupancy_max)
        sim_stats.tile_mshr_occupancy_max = tile_mshr;

      // Snapshot when MSHR is completely full (DISABLED for sweep runs)
      static int mshr_full_snapshots = 0;
      static uint64_t last_snapshot_cycle = 0;
      auto snap_cycle = current_time.time_since_epoch() / clock_period;
      if (false && std::size(MSHR) >= LFB_SIZE && mshr_full_snapshots < 30
          && (snap_cycle - last_snapshot_cycle) >= 2000) {
        mshr_full_snapshots++;
        last_snapshot_cycle = snap_cycle;
        auto cycle = current_time.time_since_epoch() / clock_period;
        fmt::print("\n[MSHR_FULL_SNAPSHOT #{} cycle={}] MSHR={}/{} non_tile={} tile_parent={}\n",
          mshr_full_snapshots, cycle, std::size(MSHR), LFB_SIZE, non_tile_mshr, tile_mshr);
        for (size_t i = 0; i < MSHR.size(); i++) {
          auto& e = MSHR[i];
          if (e.is_tile_entry) {
            uint16_t children_received = 0, children_installed = 0;
            for (auto& c : e.tile_children) {
              if (c.fill_received) children_received++;
              if (c.fill_installed) children_installed++;
            }
            fmt::print("  [{}] TILE  grp=0x{:x} children={} recv={} installed={} addr={}\n",
              i, e.tile_group_id, e.tile_children.size(), children_received, children_installed, e.address);
          } else {
            fmt::print("  [{}] {:5} instr={} addr={} type={}\n",
              i, (e.type == access_type::LOAD ? "LOAD" :
                  e.type == access_type::WRITE ? "WRITE" :
                  e.type == access_type::TRANSLATION ? "TRANS" : "OTHER"),
              e.instr_id, e.address, static_cast<int>(e.type));
          }
        }
      }
      sim_stats.lfb_only_occupancy_max = 0;  // No separate LFB in unified model
      sim_stats.sidecar_fills_occupancy_sum += 0;
    }
  }

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} cycle completed: {} tags checked: {} remaining: {} stash consumed: {} remaining: {} channel consumed: {} pq consumed {} unused consume "
               "bw {}\n",
               NAME, __func__, current_time.time_since_epoch() / clock_period, tag_check_bw.amount_consumed(), std::size(inflight_tag_check),
               stash_bandwidth_consumed, std::size(translation_stash), channels_bandwidth_consumed, 0L, initiate_tag_bw.amount_remaining());
  }



  // Bandwidth idle tracking (ROI only)
  if (!warmup) {
    long tag_used = initiate_tag_bw.amount_consumed();
    long tag_idle = champsim::to_underlying(MAX_TAG) - tag_used;
    if (tag_idle < 0) tag_idle = 0;
    long fill_used = fill_bw.amount_consumed();
    long fill_max = champsim::to_underlying(MAX_FILL);
    long fill_idle = fill_max - fill_used;
    if (fill_idle < 0) fill_idle = 0;

    sim_stats.bw_cycles++;
    sim_stats.bw_tag_total_used += tag_used;
    sim_stats.bw_fill_total_used += fill_used;
    if (tag_idle < 8) sim_stats.bw_tag_idle_hist[tag_idle]++;
    if (fill_idle < 8) sim_stats.bw_fill_idle_hist[fill_idle]++;

    // inflight_tag_check occupancy: total, PF, demand
    long itc_total = inflight_tag_check.size();
    long itc_pf = std::count_if(inflight_tag_check.begin(), inflight_tag_check.end(),
        [](const auto& e) { return e.type == access_type::PREFETCH; });
    sim_stats.bw_itc_total += itc_total;
    sim_stats.bw_itc_pf += itc_pf;
    // ITC histogram (0-10+)
    long itc_bin = std::min(itc_total, 10L);
    sim_stats.bw_itc_hist[itc_bin]++;
    long itc_pf_bin = std::min(itc_pf, 10L);
    sim_stats.bw_itc_pf_hist[itc_pf_bin]++;
    // MSHR PF occupancy
    long mshr_pf = std::count_if(MSHR.begin(), MSHR.end(),
        [](const auto& e) { return e.type == access_type::PREFETCH; });
    sim_stats.bw_mshr_pf += mshr_pf;

    // Cycle trace: dump first 500 ROI cycles for L1D (disabled for batch runs)
    // Enable by compiling with -DAMX_TILE_DEBUG for detailed per-cycle analysis
#ifdef AMX_TILE_DEBUG
    if (NAME.find("L1D") != std::string::npos && sim_stats.bw_cycles > 50000 && sim_stats.bw_cycles <= 50500) {
      long rq_occ = 0;
      for (auto* ul : upper_levels) rq_occ += ul->rq_occupancy();
      long tile_mshr_cnt = std::count_if(MSHR.begin(), MSHR.end(), [](const auto& e) { return e.is_tile_entry; });
      long nontile_mshr_cnt = MSHR.size() - tile_mshr_cnt;
      long pf_in_pq = internal_PQ.size();
      long itc = inflight_tag_check.size();
      long itc_tile = std::count_if(inflight_tag_check.begin(), inflight_tag_check.end(),
          [](const auto& e) { return e.is_amx_tileload; });
      fmt::print("[L1D_CY] cy={} tag_used={} fill_used={} rq={} itc={}/{} mshr={}/{}+{} pq={}\n",
          sim_stats.bw_cycles, tag_used, fill_used, rq_occ, itc_tile, itc, tile_mshr_cnt, nontile_mshr_cnt, (long)MSHR.size(), pf_in_pq);
    }
#endif

    // Phase-aware: tile-active vs compute-only
    bool has_tile_mshr = false;
    for (const auto& e : MSHR) {
      if (e.is_tile_entry || (e.type == access_type::LOAD && NAME.find("L1D") != std::string::npos)) {
        // Check if any tile sidecar entry is active
        has_tile_mshr = true;
        break;
      }
    }
    // Simpler: check if any MSHR entry exists (demand is in-flight)
    bool mshr_active = !MSHR.empty();
    bool inflight_tags_busy = !inflight_tag_check.empty();

    if (mshr_active) {
      sim_stats.bw_tile_phase_cycles++;
      sim_stats.bw_tile_phase_tag_idle += tag_idle;
      sim_stats.bw_tile_phase_fill_idle += fill_idle;
    } else {
      sim_stats.bw_compute_phase_cycles++;
      sim_stats.bw_compute_phase_tag_idle += tag_idle;
      sim_stats.bw_compute_phase_fill_idle += fill_idle;
    }
  }

  return progress + fill_bw.amount_consumed() + initiate_tag_bw.amount_consumed() + tag_check_bw.amount_consumed();
}

// LCOV_EXCL_START exclude deprecated function
uint64_t CACHE::get_set(uint64_t address) const { return static_cast<uint64_t>(get_set_index(champsim::address{address})); }
// LCOV_EXCL_STOP

long CACHE::get_set_index(champsim::address address) const { return address.slice(champsim::dynamic_extent{OFFSET_BITS, champsim::lg2(NUM_SET)}).to<long>(); }

template <typename It>
std::pair<It, It> get_span(It anchor, typename std::iterator_traits<It>::difference_type set_idx, typename std::iterator_traits<It>::difference_type num_way)
{
  auto begin = std::next(anchor, set_idx * num_way);
  return {std::move(begin), std::next(begin, num_way)};
}

auto CACHE::get_set_span(champsim::address address) -> std::pair<set_type::iterator, set_type::iterator>
{
  const auto set_idx = get_set_index(address);
  assert(set_idx < NUM_SET);
  return get_span(std::begin(block), static_cast<set_type::difference_type>(set_idx), NUM_WAY); // safe cast because of prior assert
}

auto CACHE::get_set_span(champsim::address address) const -> std::pair<set_type::const_iterator, set_type::const_iterator>
{
  const auto set_idx = get_set_index(address);
  assert(set_idx < NUM_SET);
  return get_span(std::cbegin(block), static_cast<set_type::difference_type>(set_idx), NUM_WAY); // safe cast because of prior assert
}

// LCOV_EXCL_START exclude deprecated function
uint64_t CACHE::get_way(uint64_t address, uint64_t /*unused set index*/) const
{
  champsim::address intern_addr{address};
  auto [begin, end] = get_set_span(intern_addr);
  return static_cast<uint64_t>(std::distance(begin, std::find_if(begin, end, matches_address(champsim::address{address}))));
}
// LCOV_EXCL_STOP

long CACHE::invalidate_entry(champsim::address inval_addr)
{
  auto [begin, end] = get_set_span(inval_addr);
  auto inv_way = std::find_if(begin, end, matches_address(inval_addr));

  if (inv_way != end) {
    inv_way->valid = false;
  }

  return std::distance(begin, inv_way);
}

bool CACHE::prefetch_line(champsim::address pf_addr, bool fill_this_level, uint32_t prefetch_metadata)
{
  ++sim_stats.pf_requested;

  if (std::size(internal_PQ) >= PQ_SIZE) {
    return false;
  }

  request_type pf_packet;
  pf_packet.type = access_type::PREFETCH;
  pf_packet.pf_metadata = prefetch_metadata;
  pf_packet.cpu = cpu;
  pf_packet.address = pf_addr;
  pf_packet.v_address = virtual_prefetch ? pf_addr : champsim::address{};
  pf_packet.is_translated = !virtual_prefetch;

  internal_PQ.emplace_back(pf_packet, true, !fill_this_level);
  ++sim_stats.pf_issued;

  return true;
}

bool CACHE::prefetch_tile(const std::vector<champsim::address>& addrs,
                          uint64_t tile_group_id, bool fill_this_level)
{
  ++sim_stats.pf_requested;

  // Add 1 tile entry to tile_PQ (16 CL = 1 PQ slot, like Berti's PQ model).
  // Processed with leftover demand bandwidth in operate().
  if (tile_PQ.size() >= TILE_PQ_SIZE) {
    return false;
  }

  tile_PQ.push_back({tile_group_id, addrs, fill_this_level});
  ++sim_stats.pf_issued;
  return true;
}

bool CACHE::prefetch_tile_line(champsim::address pf_addr, bool fill_this_level,
                               uint64_t tile_group_id, uint8_t tile_subidx, uint8_t tile_num_children)
{
  ++sim_stats.pf_requested;

  if (std::size(internal_PQ) >= PQ_SIZE) {
    return false;
  }

  request_type pf_packet;
  pf_packet.type = access_type::PREFETCH;
  pf_packet.pf_metadata = 0;
  pf_packet.cpu = cpu;
  pf_packet.address = pf_addr;
  pf_packet.v_address = virtual_prefetch ? pf_addr : champsim::address{};
  pf_packet.is_translated = !virtual_prefetch;

  // Tile metadata — enables sidecar/tile MSHR coalescing
  pf_packet.is_amx_tileload = true;
  pf_packet.tile_group_id = tile_group_id;
  pf_packet.tile_subidx = tile_subidx;
  pf_packet.tile_num_children = tile_num_children;

  internal_PQ.emplace_back(pf_packet, true, !fill_this_level);
  ++sim_stats.pf_issued;

  return true;
}

// LCOV_EXCL_START exclude deprecated function
bool CACHE::prefetch_line(uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata)
{
  return prefetch_line(champsim::address{pf_addr}, fill_this_level, prefetch_metadata);
}

bool CACHE::prefetch_line(uint64_t /*deprecated*/, uint64_t /*deprecated*/, uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata)
{
  return prefetch_line(champsim::address{pf_addr}, fill_this_level, prefetch_metadata);
}
// LCOV_EXCL_STOP

void CACHE::finish_packet(const response_type& packet)
{
  // ============================================================================
  // Line Fill Processing (Unified LFB Model)
  // ============================================================================
  // Process ALL matching entries for this address:
  // 1. Normal (non-tile) MSHR entries → set data_promise in-place (stays in MSHR)
  // 2. Tile MSHR entries' children → mark child filled
  // Both paths run so that address overlap between tile and non-tile is handled.
  // Entry removal happens later in operate() after handle_fill succeeds.
  // ============================================================================


  auto addr_match = matches_address(packet.address);

  // --- Path 1: Normal (non-tile) MSHR entry — set data_promise in-place ---
  auto mshr_entry = std::find_if(std::begin(MSHR), std::end(MSHR),
      [&](const auto& entry) { return !entry.is_tile_entry && addr_match(entry); });

  if (mshr_entry != MSHR.end()) {
    mshr_entry->serve_level = packet.serve_level;  // Record where data came from
    mshr_type::returned_value finished_value{packet.data, packet.pf_metadata};
    mshr_entry->data_promise = champsim::waitable{finished_value, current_time + (warmup ? champsim::chrono::clock::duration{} : FILL_LATENCY)};

    if constexpr (champsim::debug_print) {
      fmt::print("[{}_ULFB] finish_packet data_promise set instr_id: {} address: {} type: {} current: {}\n",
                 this->NAME, mshr_entry->instr_id, mshr_entry->address,
                 access_type_names.at(champsim::to_underlying(mshr_entry->type)),
                 current_time.time_since_epoch() / clock_period);
    }

    // Process tile attachments (sidecar notifications)
    for (const auto& attach : mshr_entry->tile_attachments) {
      sidecar_mark_child_complete(attach.tile_group_id, attach.tile_child_idx);
    }
  }

  // --- Path 2: Tile MSHR entries — match against children's addresses ---
  // (Always runs, even if Path 1 matched, to handle address overlap)
  for (auto& entry : MSHR) {
    if (!entry.is_tile_entry) continue;
    for (auto& child : entry.tile_children) {
      if (!child.fill_received && addr_match(child)) {
        child.fill_received = true;
        child.data = packet.data;
        child.pf_metadata = packet.pf_metadata;
        child.serve_level = packet.serve_level;  // Record where data came from
        child.fill_ready_at = current_time + (warmup ? champsim::chrono::clock::duration{} : FILL_LATENCY);
        sim_stats.tile_sidecar_fills_matched++;
        // Don't break — multiple tile entries could have children at same address
      }
    }
  }
}



void CACHE::finish_translation(const response_type& packet)
{


  auto matches_vpage = [page_num = champsim::page_number{packet.v_address}](const auto& entry) {
    return (champsim::page_number{entry.v_address} == page_num) && !entry.is_translated;
  };
  auto mark_translated = [p_page = champsim::page_number{packet.data}, this](auto& entry) {
    [[maybe_unused]] auto old_address = entry.address;
    entry.address = champsim::address{champsim::splice(p_page, champsim::page_offset{entry.v_address})}; // translated address
    entry.is_translated = true;                                                                          // This entry is now translated

    if constexpr (champsim::debug_print) {
      fmt::print("[{}_TRANSLATE] finish_translation old: {} paddr: {} vaddr: {} type: {} cycle: {}\n", this->NAME, old_address, entry.address, entry.v_address,
                 access_type_names.at(champsim::to_underlying(entry.type)), this->current_time.time_since_epoch() / this->clock_period);
    }
  };

  // Restart stashed translations
  auto finish_begin = std::find_if_not(std::begin(translation_stash), std::end(translation_stash), [](const auto& x) { return x.is_translated; });
  auto finish_end = std::stable_partition(finish_begin, std::end(translation_stash), matches_vpage);
  std::for_each(finish_begin, finish_end, mark_translated);

  // Find all packets that match the page of the returned packet
  for (auto& entry : inflight_tag_check) {
    if (matches_vpage(entry)) {
      mark_translated(entry);
    }
  }
}

void CACHE::issue_translation(tag_lookup_type& q_entry) const
{
  if (!q_entry.translate_issued && !q_entry.is_translated) {
    request_type fwd_pkt;
    fwd_pkt.asid[0] = q_entry.asid[0];
    fwd_pkt.asid[1] = q_entry.asid[1];
    fwd_pkt.type = access_type::LOAD;
    fwd_pkt.cpu = q_entry.cpu;

    fwd_pkt.address = q_entry.address;
    fwd_pkt.v_address = q_entry.v_address;
    fwd_pkt.data = q_entry.data;
    fwd_pkt.instr_id = q_entry.instr_id;
    fwd_pkt.ip = q_entry.ip;

    fwd_pkt.instr_depend_on_me = q_entry.instr_depend_on_me;
    fwd_pkt.is_translated = true;


    q_entry.translate_issued = lower_translate->add_rq(fwd_pkt);
    if constexpr (champsim::debug_print) {
      if (q_entry.translate_issued) {
        fmt::print("[TRANSLATE] do_issue_translation instr_id: {} paddr: {} vaddr: {} type: {}\n", q_entry.instr_id, q_entry.address, q_entry.v_address,
                   access_type_names.at(champsim::to_underlying(q_entry.type)));
      }
    }
  }
}

std::size_t CACHE::get_mshr_occupancy() const { return std::size(MSHR); }

std::size_t CACHE::get_lfb_occupancy() const {
  // Unified LFB model: all MSHR entries (tile + non-tile) share the same pool.
  return std::size(MSHR);
}

std::size_t CACHE::get_demand_mshr_occupancy() const
{
  return std::count_if(std::begin(MSHR), std::end(MSHR), [](const auto& entry) { return entry.type != access_type::PREFETCH; });
}

std::vector<std::size_t> CACHE::get_rq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->rq_occupancy(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_wq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->wq_occupancy(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_pq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->pq_occupancy(); });
  retval.push_back(std::size(internal_PQ));
  return retval;
}

// LCOV_EXCL_START exclude deprecated function
std::size_t CACHE::get_occupancy(uint8_t queue_type, uint64_t /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_occupancy();
  }
  return 0;
}

std::size_t CACHE::get_occupancy(uint8_t queue_type, champsim::address /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_occupancy();
  }
  return 0;
}
// LCOV_EXCL_STOP

std::size_t CACHE::get_mshr_size() const { return MSHR_SIZE; }
std::size_t CACHE::get_lfb_size() const { return LFB_SIZE; }
std::vector<std::size_t> CACHE::get_rq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->rq_size(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_wq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->wq_size(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_pq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->pq_size(); });
  retval.push_back(PQ_SIZE);
  return retval;
}

// LCOV_EXCL_START exclude deprecated function
std::size_t CACHE::get_size(uint8_t queue_type, champsim::address /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_size();
  }
  return 0;
}

std::size_t CACHE::get_size(uint8_t queue_type, uint64_t /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_size();
  }
  return 0;
}
// LCOV_EXCL_STOP

namespace
{
double occupancy_ratio(std::size_t occ, std::size_t sz) { return std::ceil(occ) / std::ceil(sz); }

std::vector<double> occupancy_ratio_vec(std::vector<std::size_t> occ, std::vector<std::size_t> sz)
{
  std::vector<double> retval;
  std::transform(std::begin(occ), std::end(occ), std::begin(sz), std::back_inserter(retval), occupancy_ratio);
  return retval;
}
} // namespace

double CACHE::get_mshr_occupancy_ratio() const { return ::occupancy_ratio(get_mshr_occupancy(), get_mshr_size()); }

std::vector<double> CACHE::get_rq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_rq_occupancy(), get_rq_size()); }

std::vector<double> CACHE::get_wq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_wq_occupancy(), get_wq_size()); }

std::vector<double> CACHE::get_pq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_pq_occupancy(), get_pq_size()); }

void CACHE::impl_prefetcher_initialize() const { pref_module_pimpl->impl_prefetcher_initialize(); }

uint32_t CACHE::impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                              uint32_t metadata_in) const
{
  return pref_module_pimpl->impl_prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in);
}

uint32_t CACHE::impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr,
                                           uint32_t metadata_in) const
{
  return pref_module_pimpl->impl_prefetcher_cache_fill(addr, set, way, prefetch, evicted_addr, metadata_in);
}

void CACHE::impl_prefetcher_cycle_operate() const { pref_module_pimpl->impl_prefetcher_cycle_operate(); }

void CACHE::impl_prefetcher_final_stats() const { pref_module_pimpl->impl_prefetcher_final_stats(); }

void CACHE::impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) const
{
  pref_module_pimpl->impl_prefetcher_branch_operate(ip, branch_type, branch_target);
}

void CACHE::impl_initialize_replacement() const { repl_module_pimpl->impl_initialize_replacement(); }

long CACHE::impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set, champsim::address ip, champsim::address full_addr,
                             access_type type) const
{
  return repl_module_pimpl->impl_find_victim(triggering_cpu, instr_id, set, current_set, ip, full_addr, type);
}

void CACHE::impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                          champsim::address victim_addr, access_type type, bool hit) const
{
  repl_module_pimpl->impl_update_replacement_state(triggering_cpu, set, way, full_addr, ip, victim_addr, type, hit);
}

void CACHE::impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                        champsim::address victim_addr, access_type type) const
{
  repl_module_pimpl->impl_replacement_cache_fill(triggering_cpu, set, way, full_addr, ip, victim_addr, type);
}

void CACHE::impl_replacement_final_stats() const { repl_module_pimpl->impl_replacement_final_stats(); }

// ============================================================================
// Active Parent Tile Entry Helper Methods (Parent-Owned Miss Management)
// ============================================================================

// ============================================================================
// Queue-Local Tile Sidecar Helper Methods
// ============================================================================

CACHE::tile_sidecar_state* CACHE::find_or_alloc_sidecar(uint64_t tile_group_id, uint8_t num_children)
{
  auto it = tile_sidecar.find(tile_group_id);
  if (it != tile_sidecar.end()) {
    return &it->second;
  }

  // Allocate new sidecar entry
  tile_sidecar_state sc{};
  sc.valid = true;
  sc.tile_group_id = tile_group_id;
  sc.num_children = num_children;
  sc.first_seen_cycle = current_time.time_since_epoch() / clock_period;
  sim_stats.tile_parent_entry_alloc++;  // reuse stat name for sidecar alloc count

  auto [ins_it, inserted] = tile_sidecar.emplace(tile_group_id, sc);

  // Track peak concurrent sidecars
  if (tile_sidecar.size() > sim_stats.tile_parent_entry_peak) {
    sim_stats.tile_parent_entry_peak = tile_sidecar.size();
  }

  // Count total tile children on first allocation
  sim_stats.tile_children_total += num_children;

  return &ins_it->second;
}

CACHE::tile_sidecar_state* CACHE::find_sidecar(uint64_t tile_group_id)
{
  auto it = tile_sidecar.find(tile_group_id);
  return (it != tile_sidecar.end()) ? &it->second : nullptr;
}

void CACHE::sidecar_mark_child_complete(uint64_t tile_group_id, uint8_t child_idx)
{
  auto it = tile_sidecar.find(tile_group_id);
  if (it == tile_sidecar.end()) return;

  auto& sc = it->second;
  if (child_idx < 16 && !(sc.completed_mask & (1 << child_idx))) {
    sc.completed_mask |= (1 << child_idx);
    sc.completed_count++;
    sim_stats.tile_children_issued++;  // reuse: counts children that completed via fill

    // Check if tile is now fully complete
    if (sc.is_complete()) {
      sc.completion_cycle = current_time.time_since_epoch() / clock_period;
      sim_stats.tile_completions++;
      tile_finalize(sc.tile_group_id, sc.completion_cycle);
      tile_sidecar.erase(it);
    }
  }
}

// ============================================================================

void CACHE::initialize()
{
  impl_prefetcher_initialize();
  impl_initialize_replacement();
}

void CACHE::begin_phase()
{
  // Clear tile_sidecar tracking state across phase boundaries.
  tile_sidecar.clear();
  // Clear prefetch lifecycle tracker for ROI-only measurement
  pf_filled_lines.clear();
  pf_stats = {};
  // Clear PF queues — warmup PF entries must not pollute ROI
  tile_PQ.clear();
  internal_PQ.clear();
  // Reset stream buffer stats for ROI

  stats_type new_roi_stats;
  stats_type new_sim_stats;

  new_roi_stats.name = NAME;
  new_sim_stats.name = NAME;

  roi_stats = new_roi_stats;
  sim_stats = new_sim_stats;

  for (auto* ul : upper_levels) {
    channel_type::stats_type ul_new_roi_stats;
    channel_type::stats_type ul_new_sim_stats;
    ul->roi_stats = ul_new_roi_stats;
    ul->sim_stats = ul_new_sim_stats;
  }
}

void CACHE::end_phase(unsigned finished_cpu)
{
  finished_cpu = finished_cpu;
  roi_stats.total_miss_latency_cycles = sim_stats.total_miss_latency_cycles;
  roi_stats.total_miss_queue_cycles = sim_stats.total_miss_queue_cycles;
  roi_stats.l1d_pend_miss_fb_full_cycles = sim_stats.l1d_pend_miss_fb_full_cycles;

  roi_stats.hits = sim_stats.hits;
  roi_stats.misses = sim_stats.misses;
  roi_stats.mshr_merge = sim_stats.mshr_merge;
  roi_stats.mshr_return = sim_stats.mshr_return;

  roi_stats.pf_requested = sim_stats.pf_requested;
  roi_stats.pf_issued = sim_stats.pf_issued;
  roi_stats.pf_useful = sim_stats.pf_useful;
  roi_stats.pf_useless = sim_stats.pf_useless;
  roi_stats.pf_fill = sim_stats.pf_fill;

  roi_stats.tile_master_alloc = sim_stats.tile_master_alloc;
  roi_stats.tile_subreq_merged = sim_stats.tile_subreq_merged;
  roi_stats.tile_master_occupancy_cycles = sim_stats.tile_master_occupancy_cycles;
  roi_stats.l1_mshr_full_stall_cycles = sim_stats.l1_mshr_full_stall_cycles;
  roi_stats.l2_rq_full_stall_cycles = sim_stats.l2_rq_full_stall_cycles;

  // Queue-local tile sidecar stats
  roi_stats.tile_parent_entry_alloc = sim_stats.tile_parent_entry_alloc;
  roi_stats.tile_parent_entry_peak = sim_stats.tile_parent_entry_peak;
  roi_stats.tile_completions = sim_stats.tile_completions;
  roi_stats.tile_children_total = sim_stats.tile_children_total;
  roi_stats.tile_children_hits = sim_stats.tile_children_hits;
  roi_stats.tile_children_pending = sim_stats.tile_children_pending;
  roi_stats.tile_children_issued = sim_stats.tile_children_issued;
  roi_stats.tile_admission_throttle_cycles = sim_stats.tile_admission_throttle_cycles;
  roi_stats.child_issue_deferrals = sim_stats.child_issue_deferrals;
  roi_stats.child_issue_deferrals_mshr_full = sim_stats.child_issue_deferrals_mshr_full;
  roi_stats.child_issue_deferrals_rq_full = sim_stats.child_issue_deferrals_rq_full;
  roi_stats.child_issue_deferrals_parent_window = sim_stats.child_issue_deferrals_parent_window;
  roi_stats.child_issue_deferrals_parent_table_full = sim_stats.child_issue_deferrals_parent_table_full;
  roi_stats.parent_driven_issues = sim_stats.parent_driven_issues;
  roi_stats.total_pending_children_snapshots = sim_stats.total_pending_children_snapshots;
  roi_stats.total_inflight_children_snapshots = sim_stats.total_inflight_children_snapshots;
  roi_stats.total_lfb_allocs_for_tiles = sim_stats.total_lfb_allocs_for_tiles;
  roi_stats.total_issue_batches = sim_stats.total_issue_batches;
  roi_stats.total_child_misses_issued_across_cycles = sim_stats.total_child_misses_issued_across_cycles;
  roi_stats.total_cycles_with_active_parents = sim_stats.total_cycles_with_active_parents;
  roi_stats.tile_attachment_merges = sim_stats.tile_attachment_merges;
  roi_stats.tile_sidecar_coalesced = sim_stats.tile_sidecar_coalesced;
  roi_stats.tile_sidecar_fills_matched = sim_stats.tile_sidecar_fills_matched;
  roi_stats.lfb_occupancy_sum = sim_stats.lfb_occupancy_sum;
  roi_stats.lfb_occupancy_cycles = sim_stats.lfb_occupancy_cycles;
  roi_stats.lfb_occupancy_max = sim_stats.lfb_occupancy_max;
  roi_stats.sidecar_fills_occupancy_sum = sim_stats.sidecar_fills_occupancy_sum;
  roi_stats.non_tile_mshr_occupancy_sum = sim_stats.non_tile_mshr_occupancy_sum;
  roi_stats.tile_mshr_occupancy_sum = sim_stats.tile_mshr_occupancy_sum;
  roi_stats.lfb_only_occupancy_sum = sim_stats.lfb_only_occupancy_sum;
  roi_stats.non_tile_mshr_occupancy_max = sim_stats.non_tile_mshr_occupancy_max;
  roi_stats.tile_mshr_occupancy_max = sim_stats.tile_mshr_occupancy_max;
  roi_stats.lfb_only_occupancy_max = sim_stats.lfb_only_occupancy_max;
  roi_stats.non_tile_mshr_allocs = sim_stats.non_tile_mshr_allocs;
  roi_stats.non_tile_mshr_by_load = sim_stats.non_tile_mshr_by_load;
  roi_stats.non_tile_mshr_by_rfo = sim_stats.non_tile_mshr_by_rfo;
  roi_stats.non_tile_mshr_by_write = sim_stats.non_tile_mshr_by_write;
  roi_stats.non_tile_mshr_by_prefetch = sim_stats.non_tile_mshr_by_prefetch;
  roi_stats.non_tile_mshr_by_other = sim_stats.non_tile_mshr_by_other;
  roi_stats.tile_mshr_allocs = sim_stats.tile_mshr_allocs;
  roi_stats.translation_mshr_allocs = sim_stats.translation_mshr_allocs;
  roi_stats.tile_leaked_to_nontile = sim_stats.tile_leaked_to_nontile;
  roi_stats.tile_loads_tag_check = sim_stats.tile_loads_tag_check;
  roi_stats.scalar_loads_tag_check = sim_stats.scalar_loads_tag_check;

  // Bandwidth idle
  roi_stats.bw_cycles = sim_stats.bw_cycles;
  roi_stats.bw_tag_total_used = sim_stats.bw_tag_total_used;
  roi_stats.bw_fill_total_used = sim_stats.bw_fill_total_used;
  for (int i = 0; i < 8; i++) {
    roi_stats.bw_tag_idle_hist[i] = sim_stats.bw_tag_idle_hist[i];
    roi_stats.bw_fill_idle_hist[i] = sim_stats.bw_fill_idle_hist[i];
  }
  roi_stats.bw_itc_total = sim_stats.bw_itc_total;
  roi_stats.bw_itc_pf = sim_stats.bw_itc_pf;
  roi_stats.bw_mshr_pf = sim_stats.bw_mshr_pf;
  for (int i = 0; i < 11; i++) {
    roi_stats.bw_itc_hist[i] = sim_stats.bw_itc_hist[i];
    roi_stats.bw_itc_pf_hist[i] = sim_stats.bw_itc_pf_hist[i];
  }
  roi_stats.bw_tile_phase_cycles = sim_stats.bw_tile_phase_cycles;
  roi_stats.bw_tile_phase_tag_idle = sim_stats.bw_tile_phase_tag_idle;
  roi_stats.bw_tile_phase_fill_idle = sim_stats.bw_tile_phase_fill_idle;
  roi_stats.bw_compute_phase_cycles = sim_stats.bw_compute_phase_cycles;
  roi_stats.bw_compute_phase_tag_idle = sim_stats.bw_compute_phase_tag_idle;
  roi_stats.bw_compute_phase_fill_idle = sim_stats.bw_compute_phase_fill_idle;

  // Event-based fill stall attribution
  roi_stats.fill_stall_l2_hit = sim_stats.fill_stall_l2_hit;
  roi_stats.fill_stall_llc_hit = sim_stats.fill_stall_llc_hit;
  roi_stats.fill_stall_dram = sim_stats.fill_stall_dram;
  roi_stats.fill_count_l2_hit = sim_stats.fill_count_l2_hit;
  roi_stats.fill_count_llc_hit = sim_stats.fill_count_llc_hit;
  roi_stats.fill_count_dram = sim_stats.fill_count_dram;

  for (int i = 0; i < 16; i++) {
    roi_stats.a_tile_child_hit[i] = sim_stats.a_tile_child_hit[i];
    roi_stats.a_tile_child_miss[i] = sim_stats.a_tile_child_miss[i];
    roi_stats.b_tile_child_hit[i] = sim_stats.b_tile_child_hit[i];
    roi_stats.b_tile_child_miss[i] = sim_stats.b_tile_child_miss[i];
  }

  for (auto* ul : upper_levels) {
    ul->roi_stats.RQ_ACCESS = ul->sim_stats.RQ_ACCESS;
    ul->roi_stats.RQ_MERGED = ul->sim_stats.RQ_MERGED;
    ul->roi_stats.RQ_FULL = ul->sim_stats.RQ_FULL;
    ul->roi_stats.RQ_TO_CACHE = ul->sim_stats.RQ_TO_CACHE;

    ul->roi_stats.PQ_ACCESS = ul->sim_stats.PQ_ACCESS;
    ul->roi_stats.PQ_MERGED = ul->sim_stats.PQ_MERGED;
    ul->roi_stats.PQ_FULL = ul->sim_stats.PQ_FULL;
    ul->roi_stats.PQ_TO_CACHE = ul->sim_stats.PQ_TO_CACHE;

    ul->roi_stats.WQ_ACCESS = ul->sim_stats.WQ_ACCESS;
    ul->roi_stats.WQ_MERGED = ul->sim_stats.WQ_MERGED;
    ul->roi_stats.WQ_FULL = ul->sim_stats.WQ_FULL;
    ul->roi_stats.WQ_TO_CACHE = ul->sim_stats.WQ_TO_CACHE;
    ul->roi_stats.WQ_FORWARD = ul->sim_stats.WQ_FORWARD;

    roi_stats.rq_access += ul->roi_stats.RQ_ACCESS;
    roi_stats.rq_full += ul->roi_stats.RQ_FULL;
    roi_stats.pq_access += ul->roi_stats.PQ_ACCESS;
    roi_stats.pq_full += ul->roi_stats.PQ_FULL;
    roi_stats.wq_access += ul->roi_stats.WQ_ACCESS;
    roi_stats.wq_full += ul->roi_stats.WQ_FULL;

    sim_stats.rq_access += ul->sim_stats.RQ_ACCESS;
    sim_stats.rq_full += ul->sim_stats.RQ_FULL;
    sim_stats.pq_access += ul->sim_stats.PQ_ACCESS;
    sim_stats.pq_full += ul->sim_stats.PQ_FULL;
    sim_stats.wq_access += ul->sim_stats.WQ_ACCESS;
    sim_stats.wq_full += ul->sim_stats.WQ_FULL;
  }

  // ── Event-Based Miss Stall Attribution (L1D only) ──
  if (NAME.find("L1D") != std::string::npos) {
    uint64_t total_fill_stall = roi_stats.fill_stall_l2_hit + roi_stats.fill_stall_llc_hit + roi_stats.fill_stall_dram;
    uint64_t total_fill_count = roi_stats.fill_count_l2_hit + roi_stats.fill_count_llc_hit + roi_stats.fill_count_dram;
    fmt::print("cpu{}->{} FILL_STALL_ATTRIBUTION (event-based, ROI):\n", finished_cpu, NAME);
    fmt::print("  total_fills: {}  total_stall_cycles: {}\n", total_fill_count, total_fill_stall);
    fmt::print("  l2_hit:  fills={:>8}  stall_cy={:>12}  ({:.1f}%)  avg={:.1f}cy\n",
        roi_stats.fill_count_l2_hit, roi_stats.fill_stall_l2_hit,
        total_fill_stall > 0 ? 100.0 * roi_stats.fill_stall_l2_hit / total_fill_stall : 0.0,
        roi_stats.fill_count_l2_hit > 0 ? (double)roi_stats.fill_stall_l2_hit / roi_stats.fill_count_l2_hit : 0.0);
    fmt::print("  llc_hit: fills={:>8}  stall_cy={:>12}  ({:.1f}%)  avg={:.1f}cy\n",
        roi_stats.fill_count_llc_hit, roi_stats.fill_stall_llc_hit,
        total_fill_stall > 0 ? 100.0 * roi_stats.fill_stall_llc_hit / total_fill_stall : 0.0,
        roi_stats.fill_count_llc_hit > 0 ? (double)roi_stats.fill_stall_llc_hit / roi_stats.fill_count_llc_hit : 0.0);
    fmt::print("  dram:    fills={:>8}  stall_cy={:>12}  ({:.1f}%)  avg={:.1f}cy\n",
        roi_stats.fill_count_dram, roi_stats.fill_stall_dram,
        total_fill_stall > 0 ? 100.0 * roi_stats.fill_stall_dram / total_fill_stall : 0.0,
        roi_stats.fill_count_dram > 0 ? (double)roi_stats.fill_stall_dram / roi_stats.fill_count_dram : 0.0);
  }

  // ── SW Prefetch Lifecycle Stats (L1D only) ──
  if (NAME.find("L1D") != std::string::npos && pf_stats.pf_requests > 0) {
    fmt::print("cpu{}->{} SW_PREFETCH_LIFECYCLE:\n", finished_cpu, NAME);
    fmt::print("  pf_requests:           {:>10}  (L1_hit: {} L1_miss: {})\n",
        pf_stats.pf_requests, pf_stats.pf_l1_hit, pf_stats.pf_l1_miss);
    fmt::print("  pf_fills_installed:    {:>10}\n", pf_stats.pf_fills);
    fmt::print("  pf_evicted_before_use: {:>10}", pf_stats.pf_evicted_before_use);
    if (pf_stats.pf_residency_count > 0)
      fmt::print("  (avg_residency: {:.1f} cy)", (double)pf_stats.total_pf_residency / pf_stats.pf_residency_count);
    fmt::print("\n");
    uint64_t demand_total = pf_stats.demand_total_hits + pf_stats.demand_total_misses;
    fmt::print("  demand_tileload_L1D: total={} hit={} miss={} (hit_rate={:.1f}%)\n",
        demand_total, pf_stats.demand_total_hits, pf_stats.demand_total_misses,
        demand_total > 0 ? 100.0 * pf_stats.demand_total_hits / demand_total : 0.0);
    fmt::print("    hit_from_pf:         {:>10}  ({:.1f}% of all demand hits)\n",
        pf_stats.demand_hit_pf_line,
        pf_stats.demand_total_hits > 0 ? 100.0 * pf_stats.demand_hit_pf_line / pf_stats.demand_total_hits : 0.0);
    if (pf_stats.pf_to_demand_count > 0)
      fmt::print("      avg_pf_to_demand_gap: {:.1f} cy\n",
          (double)pf_stats.total_pf_to_demand_gap / pf_stats.pf_to_demand_count);
    fmt::print("    miss_pf_evicted:     {:>10}  (prefetch filled but L1 evicted before demand!)\n", pf_stats.demand_miss_pf_evicted);
    fmt::print("    no_prior_pf:         {:>10}  (no prefetch for this address)\n", pf_stats.demand_no_pf);
    if (pf_stats.pf_fills > 0) {
      double useful_pct = 100.0 * pf_stats.demand_hit_pf_line / pf_stats.pf_fills;
      double evicted_pct = 100.0 * pf_stats.pf_evicted_before_use / pf_stats.pf_fills;
      fmt::print("  pf_efficiency: {:.1f}% useful, {:.1f}% evicted_unused\n", useful_pct, evicted_pct);
    }
    if (pf_stats.pf_fill_latency_count > 0) {
      fmt::print("  pf_fill_latency: avg={:.1f}cy (MSHR_enqueue→fill, {} samples)\n",
          (double)pf_stats.pf_fill_latency_total / pf_stats.pf_fill_latency_count,
          pf_stats.pf_fill_latency_count);
    }
  }


  // ── Bandwidth Idle Stats ──
  if (roi_stats.bw_cycles > 0) {
    double avg_tag = (double)roi_stats.bw_tag_total_used / roi_stats.bw_cycles;
    double avg_fill = (double)roi_stats.bw_fill_total_used / roi_stats.bw_cycles;
    fmt::print("cpu{}->{} BANDWIDTH_IDLE (ROI, {} cycles):\n", finished_cpu, NAME, roi_stats.bw_cycles);
    fmt::print("  TAG:  avg_used={:.2f}/{} idle_dist=[", avg_tag, champsim::to_underlying(MAX_TAG));
    for (int i = 0; i <= champsim::to_underlying(MAX_TAG) && i < 8; i++) {
      if (i > 0) fmt::print(",");
      fmt::print("{}={:.1f}%", i, 100.0 * roi_stats.bw_tag_idle_hist[i] / roi_stats.bw_cycles);
    }
    fmt::print("]\n");
    fmt::print("  FILL: avg_used={:.2f}/{} idle_dist=[", avg_fill, champsim::to_underlying(MAX_FILL));
    for (int i = 0; i <= champsim::to_underlying(MAX_FILL) && i < 8; i++) {
      if (i > 0) fmt::print(",");
      fmt::print("{}={:.1f}%", i, 100.0 * roi_stats.bw_fill_idle_hist[i] / roi_stats.bw_cycles);
    }
    fmt::print("]\n");
    // inflight_tag_check & MSHR PF occupancy
    if (roi_stats.bw_cycles > 0) {
      fmt::print("  ITC: avg_occ={:.2f} avg_pf={:.2f} ({:.1f}% PF)\n",
          (double)roi_stats.bw_itc_total / roi_stats.bw_cycles,
          (double)roi_stats.bw_itc_pf / roi_stats.bw_cycles,
          roi_stats.bw_itc_total > 0 ? 100.0 * roi_stats.bw_itc_pf / roi_stats.bw_itc_total : 0.0);
      fmt::print("  MSHR_PF: avg_occ={:.2f}\n",
          (double)roi_stats.bw_mshr_pf / roi_stats.bw_cycles);
      // ITC histogram
      fmt::print("  ITC_hist: [");
      for (int i = 0; i <= 10; i++) {
        if (i > 0) fmt::print(",");
        fmt::print("{}={:.1f}%", i, 100.0 * roi_stats.bw_itc_hist[i] / roi_stats.bw_cycles);
      }
      fmt::print("]\n");
      fmt::print("  ITC_PF_hist: [");
      for (int i = 0; i <= 10; i++) {
        if (i > 0) fmt::print(",");
        fmt::print("{}={:.1f}%", i, 100.0 * roi_stats.bw_itc_pf_hist[i] / roi_stats.bw_cycles);
      }
      fmt::print("]\n");
    }
    // Phase breakdown
    if (roi_stats.bw_tile_phase_cycles > 0 || roi_stats.bw_compute_phase_cycles > 0) {
      uint64_t tp = roi_stats.bw_tile_phase_cycles;
      uint64_t cp = roi_stats.bw_compute_phase_cycles;
      fmt::print("  MSHR-active phase: {:>8} cy ({:.1f}%)  avg_tag_idle={:.2f}  avg_fill_idle={:.2f}\n",
          tp, 100.0 * tp / roi_stats.bw_cycles,
          tp > 0 ? (double)roi_stats.bw_tile_phase_tag_idle / tp : 0.0,
          tp > 0 ? (double)roi_stats.bw_tile_phase_fill_idle / tp : 0.0);
      fmt::print("  Compute phase:     {:>8} cy ({:.1f}%)  avg_tag_idle={:.2f}  avg_fill_idle={:.2f}\n",
          cp, 100.0 * cp / roi_stats.bw_cycles,
          cp > 0 ? (double)roi_stats.bw_compute_phase_tag_idle / cp : 0.0,
          cp > 0 ? (double)roi_stats.bw_compute_phase_fill_idle / cp : 0.0);
    }
  }

}

template <typename T>
bool CACHE::should_activate_prefetcher(const T& pkt) const
{
  return !pkt.prefetch_from_this && std::count(std::begin(pref_activate_mask), std::end(pref_activate_mask), pkt.type) > 0;
}

// LCOV_EXCL_START Exclude the following function from LCOV
void CACHE::print_deadlock()
{
  std::string_view mshr_write{"instr_id: {} address: {} v_addr: {} type: {} ready: {}"};
  auto mshr_pack = [time = current_time](const auto& entry) {
    return std::tuple{entry.instr_id, entry.address, entry.v_address, access_type_names.at(champsim::to_underlying(entry.type)),
                      entry.data_promise.is_ready_at(time)};
  };

  std::string_view tag_check_write{"instr_id: {} address: {} v_addr: {} is_translated: {} translate_issued: {} event_cycle: {}"};
  auto tag_check_pack = [period = clock_period](const auto& entry) {
    return std::tuple{entry.instr_id,      entry.address,          entry.v_address,
                      entry.is_translated, entry.translate_issued, entry.event_cycle.time_since_epoch() / period};
  };

  champsim::range_print_deadlock(MSHR, NAME + "_MSHR", mshr_write, mshr_pack);
  champsim::range_print_deadlock(inflight_tag_check, NAME + "_tags", tag_check_write, tag_check_pack);
  champsim::range_print_deadlock(translation_stash, NAME + "_translation", tag_check_write, tag_check_pack);

  std::string_view q_writer{"instr_id: {} address: {} v_addr: {} type: {} translated: {}"};
  auto q_entry_pack = [](const auto& entry) {
    return std::tuple{entry.instr_id, entry.address, entry.v_address, access_type_names.at(champsim::to_underlying(entry.type)), entry.is_translated};
  };

  for (auto* ul : upper_levels) {
    champsim::range_print_deadlock(ul->RQ, NAME + "_RQ", q_writer, q_entry_pack);
    champsim::range_print_deadlock(ul->WQ, NAME + "_WQ", q_writer, q_entry_pack);
    champsim::range_print_deadlock(ul->PQ, NAME + "_PQ", q_writer, q_entry_pack);
  }
}
// LCOV_EXCL_STOP
