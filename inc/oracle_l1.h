#ifndef ORACLE_L1_H
#define ORACLE_L1_H

// =============================================================================
// Oracle L1 Residency Mechanism — upper-bound evaluation for LFB-free
// ideal prepositioning. NOT a realistic prefetcher.
//
// When disabled:  zero overhead, baseline behavior unchanged.
// When enabled:   future demand reads are pre-marked as "in L1"; on demand,
//                 execute_load() short-circuits before L1D_bus.issue_read(),
//                 so NO LFB/MSHR/RQ resources are consumed.
// =============================================================================

#include <cstdint>
#include <deque>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "instruction.h"   // ooo_model_instr, memory_operation

// ---------------------------------------------------------------------------
// Filter modes
// ---------------------------------------------------------------------------
enum class OracleFilter {
  ALL_READS,       // every future demand read
  AMX_ONLY,        // only AMX TILELOADD/TILELOADDT1
  NON_AMX_ONLY     // everything that is NOT an AMX tile-load
};

// ---------------------------------------------------------------------------
// Config knobs (populated from JSON or hard-coded defaults)
// ---------------------------------------------------------------------------
struct OracleConfig {
  bool        enabled               = false;
  OracleFilter filter               = OracleFilter::ALL_READS;
  int64_t     lookahead_instructions = 4096;
  int64_t     capacity_lines        = 0;     // 0 = unlimited
  int32_t     hit_latency_cycles    = 1;
  bool        count_as_true_l1      = false;

  // Helpers
  static OracleFilter filter_from_string(const std::string& s)
  {
    if (s == "oracle_amx_only")     return OracleFilter::AMX_ONLY;
    if (s == "oracle_non_amx_only") return OracleFilter::NON_AMX_ONLY;
    return OracleFilter::ALL_READS;
  }
  static std::string filter_to_string(OracleFilter f)
  {
    switch (f) {
      case OracleFilter::AMX_ONLY:     return "oracle_amx_only";
      case OracleFilter::NON_AMX_ONLY: return "oracle_non_amx_only";
      default:                         return "oracle_all_reads";
    }
  }
};

// ---------------------------------------------------------------------------
// Per-line metadata stored in OLRT
// ---------------------------------------------------------------------------
struct OracleLineInfo {
  uint64_t inserting_instr_id = 0;   // instr_id of the future instruction
  uint64_t inserting_cycle    = 0;   // simulator cycle when inserted
  bool     used               = false;
  bool     is_amx             = false;
};

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
struct OracleStats {
  uint64_t oracle_l1_hits              = 0;
  uint64_t oracle_l1_hits_amx          = 0;
  uint64_t oracle_l1_hits_non_amx      = 0;
  uint64_t oracle_lines_inserted       = 0;
  uint64_t oracle_lines_evicted        = 0;
  uint64_t oracle_lines_unused         = 0;

  uint64_t oracle_l1_hits_write        = 0;
  uint64_t oracle_l1_hits_amx_write    = 0;
  uint64_t oracle_l1_hits_non_amx_write= 0;

  uint64_t total_demand_reads          = 0;   // for hit-rate calculation
  uint64_t total_amx_reads             = 0;
  uint64_t total_non_amx_reads         = 0;

  uint64_t total_demand_writes         = 0;   // for write hit-rate calculation
  uint64_t total_amx_writes            = 0;
  uint64_t total_non_amx_writes        = 0;

  // Lead distance accumulators
  uint64_t lead_distance_insts_sum     = 0;
  uint64_t lead_distance_cycles_sum    = 0;
  uint64_t lead_distance_count         = 0;
};

// ---------------------------------------------------------------------------
// Oracle L1 Residency Table (OLRT) — main class
// ---------------------------------------------------------------------------
class OracleL1
{
public:
  // ctor: one instance per CPU, configured before simulation starts
  explicit OracleL1(uint32_t cpu_idx = 0) : cpu_(cpu_idx) {}

  // Configure from a filled OracleConfig struct
  void configure(const OracleConfig& cfg) { cfg_ = cfg; }
  [[nodiscard]] bool enabled() const { return cfg_.enabled; }
  [[nodiscard]] const OracleConfig& config() const { return cfg_; }

  // -----------------------------------------------------------------------
  // feed_future: called during initialize_instruction with a future instruction
  //   cur_instr_id: instr_id of the instruction currently being initialized
  //   cur_cycle:    current simulator cycle
  // -----------------------------------------------------------------------
  void feed_future(const ooo_model_instr& future_instr,
                   uint64_t              cur_instr_id,
                   uint64_t              cur_cycle);

  // -----------------------------------------------------------------------
  // check_hit: called in execute_load before L1D issue
  //   line_addr:     cache-line-aligned virtual address of the demand load
  //   is_amx:        whether the load is tagged as AMX tileload
  //   demand_instr_id: instr_id of the load instruction (for lead distance)
  //   demand_cycle:   current cycle
  // Returns true if oracle hit (caller should bypass L1D), false otherwise.
  // -----------------------------------------------------------------------
  bool check_hit(uint64_t line_addr, bool is_amx,
                 uint64_t demand_instr_id, uint64_t demand_cycle, bool is_retry = false, bool is_write = false);

  // -----------------------------------------------------------------------
  // retire_past: called when an instruction retires
  //   retired_instr_id: the instr_id of the retiring instruction
  // Remove stale OLRT entries and count unused ones.
  // -----------------------------------------------------------------------
  void retire_past(uint64_t retired_instr_id);

  // -----------------------------------------------------------------------
  // dump_stats: print the [ORACLE_L1] summary block
  // -----------------------------------------------------------------------
  void dump_stats(std::ostream& out) const;

  // Access stats (for cross-cpu aggregation if needed)
  [[nodiscard]] const OracleStats& stats() const { return stats_; }

private:
  uint32_t    cpu_   = 0;
  OracleConfig cfg_  {};
  OracleStats  stats_{};

  // OLRT: line_addr (cache-line-aligned) → queue of tokens
  std::unordered_map<uint64_t, std::deque<OracleLineInfo>> olrt_;

  // Insertion order for FIFO eviction (only used when capacity > 0)
  std::deque<uint64_t> insertion_order_;

  // Helper: cache-line align (assume 64-byte lines)
  static constexpr uint64_t CACHE_LINE_SIZE = 64;
  static uint64_t align_to_line(uint64_t addr)
  {
    return addr & ~(CACHE_LINE_SIZE - 1);
  }

  // Helper: should this future read be inserted given the current filter?
  [[nodiscard]] bool passes_filter(bool is_amx) const
  {
    switch (cfg_.filter) {
      case OracleFilter::AMX_ONLY:     return is_amx;
      case OracleFilter::NON_AMX_ONLY: return !is_amx;
      default:                         return true;
    }
  }

  // Insert one line into OLRT (handles capacity eviction)
  void insert_line(uint64_t line_addr, const OracleLineInfo& info);
};

// ---------------------------------------------------------------------------
// Global per-cpu oracle instances (defined once in oracle_l1.cc)
// ---------------------------------------------------------------------------
#ifndef ORACLE_L1_MAX_CPUS
#define ORACLE_L1_MAX_CPUS 64
#endif

#ifdef ORACLE_L1_IMPL
OracleL1 g_oracle_l1[ORACLE_L1_MAX_CPUS];
#else
extern OracleL1 g_oracle_l1[ORACLE_L1_MAX_CPUS];
#endif

#endif // ORACLE_L1_H
