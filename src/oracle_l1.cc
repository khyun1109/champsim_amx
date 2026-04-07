// =============================================================================
// Oracle L1 Residency Mechanism — implementations
// =============================================================================

#define ORACLE_L1_IMPL    // define global storage here
#include "oracle_l1.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

// ---------------------------------------------------------------------------
// insert_line: add one cache line to OLRT, evict oldest if capacity exceeded
// ---------------------------------------------------------------------------
void OracleL1::insert_line(uint64_t line_addr, const OracleLineInfo& info)
{
  olrt_[line_addr].push_back(info);

  if (cfg_.capacity_lines > 0) {
    insertion_order_.push_back(line_addr);
    
    // Evict oldest TOKEN if total tokens exceed capacity
    if (static_cast<int64_t>(insertion_order_.size()) > cfg_.capacity_lines) {
      uint64_t victim = insertion_order_.front();
      insertion_order_.pop_front();
      auto it = olrt_.find(victim);
      if (it != olrt_.end() && !it->second.empty()) {
        if (!it->second.front().used) ++stats_.oracle_lines_unused;
        ++stats_.oracle_lines_evicted;
        it->second.pop_front();
        if (it->second.empty()) {
          olrt_.erase(it);
        }
      }
    }
  }

  ++stats_.oracle_lines_inserted;
}

// ---------------------------------------------------------------------------
// feed_future: called at initialize_instruction time with a future instr
// ---------------------------------------------------------------------------
void OracleL1::feed_future(const ooo_model_instr& future_instr,
                           uint64_t cur_instr_id,
                           uint64_t cur_cycle)
{
  if (!cfg_.enabled) return;

  bool is_amx = (future_instr.instr_class ==
                 static_cast<uint8_t>(trace_instr_class::AMX)) &&
                (future_instr.amx_op ==
                     static_cast<uint8_t>(trace_amx_op::TILELOADD) ||
                 future_instr.amx_op ==
                     static_cast<uint8_t>(trace_amx_op::TILELOADDT1) ||
                 future_instr.amx_op == 
                     static_cast<uint8_t>(trace_amx_op::TILESTORED));

  if (!passes_filter(is_amx)) return;

  // Expand source_mem_ops (AMX multi-line) into cache lines
  constexpr uint64_t LINE  = 64;
  // Helper to expand a base address and size into cache lines correctly
  auto expand_mem_op = [&](uint64_t raw_addr, uint32_t size) {
    if (size == 0) {
      uint64_t la = raw_addr & ~(LINE - 1);
      OracleLineInfo info;
      info.inserting_instr_id = future_instr.instr_id;
      info.inserting_cycle    = cur_cycle;
      info.is_amx             = is_amx;
      insert_line(la, info);
      return;
    }
    
    uint64_t start_line = raw_addr / LINE;
    uint64_t end_line   = (raw_addr + size - 1) / LINE;
    uint32_t nlines     = end_line - start_line + 1;

    for (uint32_t i = 0; i < nlines; ++i) {
      uint64_t la = (start_line + i) * LINE;
      OracleLineInfo info;
      info.inserting_instr_id = future_instr.instr_id;
      info.inserting_cycle    = cur_cycle;
      info.is_amx             = is_amx;
      insert_line(la, info);
    }
  };

  if (!future_instr.source_mem_ops.empty()) {
    // AMX path — expand each row into cache lines
    for (const auto& op : future_instr.source_mem_ops) {
      expand_mem_op(op.address.to<uint64_t>(), op.size);
    }
  } else {
    // Standard path — one line per source_memory address
    for (const auto& addr : future_instr.source_memory) {
      expand_mem_op(addr.to<uint64_t>(), 0);
    }
  }

  // Also expand destination memory operations (for writes to be bypassed)
  if (!future_instr.destination_mem_ops.empty()) {
    for (const auto& op : future_instr.destination_mem_ops) {
      expand_mem_op(op.address.to<uint64_t>(), op.size);
    }
  } else {
    for (const auto& addr : future_instr.destination_memory) {
      expand_mem_op(addr.to<uint64_t>(), 0);
    }
  }
}

// ---------------------------------------------------------------------------
// check_hit: check OLRT before L1D issue; update demand-read stats
// ---------------------------------------------------------------------------
bool OracleL1::check_hit(uint64_t line_addr, bool is_amx,
                         uint64_t demand_instr_id, uint64_t demand_cycle, bool is_retry, bool is_write)
{
  if (!is_retry && !is_write) {
    // Only count the original first attempt for hit-rate denominator
    ++stats_.total_demand_reads;
    if (is_amx) ++stats_.total_amx_reads;
    else        ++stats_.total_non_amx_reads;
  } else if (!is_retry && is_write) {
    ++stats_.total_demand_writes;
    if (is_amx) ++stats_.total_amx_writes;
    else        ++stats_.total_non_amx_writes;
  }

  if (!cfg_.enabled) return false;
  if (!passes_filter(is_amx)) return false;

  uint64_t la = align_to_line(line_addr);
  auto it = olrt_.find(la);
  if (it == olrt_.end() || it->second.empty()) return false;

  // Oracle hit
  OracleLineInfo info = it->second.front();

  if (!is_write) {
    ++stats_.oracle_l1_hits;
    if (is_amx) ++stats_.oracle_l1_hits_amx;
    else        ++stats_.oracle_l1_hits_non_amx;
  } else {
    ++stats_.oracle_l1_hits_write;
    if (is_amx) ++stats_.oracle_l1_hits_amx_write;
    else        ++stats_.oracle_l1_hits_non_amx_write;
  }

  // Lead-distance tracking
  if (demand_instr_id >= info.inserting_instr_id) {
    uint64_t di = demand_instr_id - info.inserting_instr_id;
    stats_.lead_distance_insts_sum  += di;
    if (demand_cycle >= info.inserting_cycle) {
      stats_.lead_distance_cycles_sum += (demand_cycle - info.inserting_cycle);
    }
    ++stats_.lead_distance_count;
  }

  // Remove used entry token so we don't double-count
  it->second.pop_front();
  if (it->second.empty()) {
    olrt_.erase(it);
  }
  return true;
}

// ---------------------------------------------------------------------------
// retire_past: called on every retired instruction; count unused entries
// ---------------------------------------------------------------------------
void OracleL1::retire_past(uint64_t retired_instr_id)
{
  if (!cfg_.enabled) return;

  std::vector<uint64_t> to_remove;
  for (auto& [addr, dq] : olrt_) {
    while (!dq.empty() && dq.front().inserting_instr_id <= retired_instr_id) {
      // It wasn't consumed by check_hit, so it went unused
      ++stats_.oracle_lines_unused;
      dq.pop_front();
    }
    if (dq.empty()) {
      to_remove.push_back(addr);
    }
  }
  for (uint64_t addr : to_remove) {
    olrt_.erase(addr);
  }
}

// ---------------------------------------------------------------------------
// dump_stats: print the [ORACLE_L1] summary block
// ---------------------------------------------------------------------------
void OracleL1::dump_stats(std::ostream& out) const
{
  out << "\n[ORACLE_L1] CPU " << cpu_ << "\n";
  out << "  enabled:                          " << (cfg_.enabled ? 1 : 0) << "\n";
  out << "  mode:                             oracle_l1_hit\n";
  out << "  filter:                           " << OracleConfig::filter_to_string(cfg_.filter) << "\n";
  out << "  lookahead_instructions:           " << cfg_.lookahead_instructions << "\n";
  out << "  capacity_lines:                   ";
  if (cfg_.capacity_lines <= 0) out << "unlimited\n";
  else                          out << cfg_.capacity_lines << "\n";
  out << "  hit_latency_cycles:               " << cfg_.hit_latency_cycles << "\n";
  out << "  count_as_true_l1:                 " << (cfg_.count_as_true_l1 ? "true" : "false") << "\n";

  out << "  oracle_l1_hits:                   " << stats_.oracle_l1_hits << "\n";
  out << "  oracle_l1_hits_amx:               " << stats_.oracle_l1_hits_amx << "\n";
  out << "  oracle_l1_hits_non_amx:           " << stats_.oracle_l1_hits_non_amx << "\n";
  if (stats_.total_demand_writes > 0) {
    out << "  oracle_l1_hits_write:             " << stats_.oracle_l1_hits_write << "\n";
    out << "  oracle_l1_hits_amx_write:         " << stats_.oracle_l1_hits_amx_write << "\n";
    out << "  oracle_l1_hits_non_amx_write:     " << stats_.oracle_l1_hits_non_amx_write << "\n";
  }
  out << "  oracle_lines_inserted:            " << stats_.oracle_lines_inserted << "\n";
  out << "  oracle_lines_evicted:             " << stats_.oracle_lines_evicted << "\n";
  out << "  oracle_lines_unused:              " << stats_.oracle_lines_unused << "\n";

  // Hit rates
  auto pct = [](uint64_t num, uint64_t den) -> double {
    return (den > 0) ? (100.0 * num / den) : 0.0;
  };
  out << std::fixed << std::setprecision(2);
  out << "  oracle_hit_rate_on_demand_reads:  "
      << pct(stats_.oracle_l1_hits, stats_.total_demand_reads) << "%\n";
  out << "  oracle_hit_rate_on_amx_reads:     "
      << pct(stats_.oracle_l1_hits_amx, stats_.total_amx_reads) << "%\n";
  out << "  oracle_hit_rate_on_non_amx_reads: "
      << pct(stats_.oracle_l1_hits_non_amx, stats_.total_non_amx_reads) << "%\n";

  if (stats_.total_demand_writes > 0) {
    out << "  oracle_hit_rate_on_demand_writes: "
        << pct(stats_.oracle_l1_hits_write, stats_.total_demand_writes) << "%\n";
    out << "  oracle_hit_rate_on_amx_writes:    "
        << pct(stats_.oracle_l1_hits_amx_write, stats_.total_amx_writes) << "%\n";
    out << "  oracle_hit_rate_on_non_amx_writes:"
        << pct(stats_.oracle_l1_hits_non_amx_write, stats_.total_non_amx_writes) << "%\n";
  }

  // Lead distances
  double avg_di = (stats_.lead_distance_count > 0)
                  ? static_cast<double>(stats_.lead_distance_insts_sum) / stats_.lead_distance_count
                  : 0.0;
  double avg_dc = (stats_.lead_distance_count > 0)
                  ? static_cast<double>(stats_.lead_distance_cycles_sum) / stats_.lead_distance_count
                  : 0.0;
  out << "  oracle_avg_lead_distance_insts:   " << avg_di << "\n";
  out << "  oracle_avg_lead_distance_cycles:  " << avg_dc << "\n";
  out << std::flush;
}
