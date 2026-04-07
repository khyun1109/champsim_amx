#ifndef TILE_RECORD_H
#define TILE_RECORD_H

// =============================================================================
// AMX TILELOADD Instrumentation: per-tile and per-subrequest tracking
//
// Enable by compiling with -DAMX_TILE_DEBUG
// CSV files: tile_master_trace.csv, tile_subreq_trace.csv
// =============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-subrequest record
// ---------------------------------------------------------------------------
struct tile_subreq_record {
  uint64_t tile_id      = 0;
  uint64_t line_addr    = 0;
  uint32_t row_idx      = 0;

  uint64_t gen_cycle           = 0;
  uint64_t l1d_enqueue_cycle   = 0;
  uint64_t l1d_dispatch_cycle  = 0;
  uint64_t l2_enqueue_cycle    = 0;
  uint64_t l2_dispatch_cycle   = 0;
  uint64_t fill_cycle          = 0;

  bool merged   = false;
  bool l1_hit   = false;
  bool l2_hit   = false;
  bool llc_hit  = false;
  bool dram_miss = false;
};

// ---------------------------------------------------------------------------
// Per-tile master record
// ---------------------------------------------------------------------------
struct tile_master_record {
  uint64_t id            = 0;
  uint64_t issue_cycle   = 0;
  uint64_t complete_cycle = 0;
  uint64_t retire_cycle  = 0;

  uint64_t instr_pc      = 0;
  uint64_t base_addr     = 0;
  int64_t  stride        = 0;

  uint32_t rows          = 0;
  uint32_t colsb         = 0;
  uint32_t payload_bytes = 0;

  bool is_temporal    = false;  // TILELOADD=true, TILELOADDT1=false
  bool page_crossing  = false;

  uint32_t row_count           = 0;
  uint32_t touched_lines_total = 0;
  uint32_t unique_lines        = 0;
  uint32_t merged_subreqs      = 0;

  uint32_t l1_hits   = 0;
  uint32_t l1_misses = 0;
  uint32_t l2_hits   = 0;
  uint32_t llc_hits  = 0;
  uint32_t dram_hits = 0;

  uint32_t lfb_allocs       = 0;
  uint32_t lfb_hits         = 0;
  uint32_t mshr_allocs      = 0;
  uint32_t l1d_rq_enqueues  = 0;
  uint32_t l2_rq_enqueues   = 0;

  uint64_t l1d_queue_wait_cycles = 0;
  uint64_t l2_queue_wait_cycles  = 0;
  uint64_t service_cycles        = 0;

  // "live" subreq list — used during simulation; NOT serialised to CSV
  std::vector<tile_subreq_record> subreqs;

  // completion latency (filled by tile_finalize)
  uint64_t completion_latency = 0;
};

// ---------------------------------------------------------------------------
// Global instrumentation state (defined once in tile_record.cc / a .cc file
// that includes this header and defines AMX_TILE_IMPL)
// ---------------------------------------------------------------------------
#ifdef AMX_TILE_IMPL

// Primary storage
std::map<uint64_t, tile_master_record> g_tile_masters;
std::recursive_mutex g_tile_mtx; // thread safety for parallel simulation

// Completed masters kept for summary / CSV output
std::vector<tile_master_record> g_completed_tiles;

// Global counters
uint64_t g_total_tileloadd   = 0;
uint64_t g_total_tileloaddt1 = 0;
uint64_t g_total_amx_compute = 0;  // TDPBF16PS etc.

// Arrival time ring for inter-arrival and sliding-window calculations
std::deque<uint64_t> g_arrival_cycles;   // cycle of each tileloadd issue

// Max active tile masters / subreqs seen
uint32_t g_max_active_masters  = 0;
uint32_t g_max_active_subreqs  = 0;

// Sliding-window max (10 / 20 cycle)
uint32_t g_max_tileloads_10cy  = 0;
uint32_t g_max_tileloads_20cy  = 0;
uint32_t g_max_subreqs_10cy    = 0;
uint32_t g_max_subreqs_20cy    = 0;

// When true, tile sidecar is disabled: AMX tileloads are sent as
// individual cacheline requests (baseline mode).
bool g_disable_tile_sidecar = false;
bool g_enable_tile_lq_coalesce = false;
bool g_skip_sw_prefetch = false;  // When false, SW prefetches enter LQ like scalar loads (matches real HW)
bool g_per_child_lfb = false;   // When true, each tile child consumes its own MSHR/LFB entry (realistic HW pressure)
bool g_tile_lru_insert = false; // When true, tile data is inserted at LRU position in L1D (anti-pollution)
bool g_warm_scalar_llc = false; // When true, non-tile LLC misses are forced to hit (simulates warm scalar cache from prior GEMM iterations)

#else

extern std::map<uint64_t, tile_master_record> g_tile_masters;
extern std::vector<tile_master_record>         g_completed_tiles;
extern uint64_t g_total_tileloadd;
extern uint64_t g_total_tileloaddt1;
extern uint64_t g_total_amx_compute;
extern std::deque<uint64_t> g_arrival_cycles;
extern uint32_t g_max_active_masters;
extern uint32_t g_max_active_subreqs;
extern uint32_t g_max_tileloads_10cy;
extern uint32_t g_max_tileloads_20cy;
extern uint32_t g_max_subreqs_10cy;
extern uint32_t g_max_subreqs_20cy;
extern bool g_disable_tile_sidecar;
extern bool g_enable_tile_lq_coalesce;
extern bool g_skip_sw_prefetch;
extern bool g_per_child_lfb;
extern bool g_tile_lru_insert;
extern bool g_warm_scalar_llc;
extern std::recursive_mutex g_tile_mtx;

#endif  // AMX_TILE_IMPL

// ---------------------------------------------------------------------------
// Helper: get or create a tile_master_record for a given tile_id
// ---------------------------------------------------------------------------
inline tile_master_record& tile_get_or_create(uint64_t tile_id)
{
  std::lock_guard<std::recursive_mutex> lk(g_tile_mtx);
  return g_tile_masters[tile_id];
}

// ---------------------------------------------------------------------------
// Helper: record arrival, update sliding-window max counts
// ---------------------------------------------------------------------------
inline void tile_record_arrival(uint64_t cycle, uint32_t num_subreqs)
{
  std::lock_guard<std::recursive_mutex> lk(g_tile_mtx);
  g_arrival_cycles.push_back(cycle);

  // Evict entries older than 20 cycles
  while (!g_arrival_cycles.empty() && cycle - g_arrival_cycles.front() > 20) {
    g_arrival_cycles.pop_front();
  }

  // Count how many are within 10 vs 20 cycles
  uint32_t cnt10 = 0, cnt20 = 0;
  for (uint64_t t : g_arrival_cycles) {
    if (cycle - t <= 10) ++cnt10;
    ++cnt20;  // all within 20 by construction above
  }
  if (cnt10 > g_max_tileloads_10cy) g_max_tileloads_10cy = cnt10;
  if (cnt20 > g_max_tileloads_20cy) g_max_tileloads_20cy = cnt20;

  // Track subreq window maxima (approximate: multiply by unique_lines)
  uint32_t sr10 = cnt10 * num_subreqs;
  uint32_t sr20 = cnt20 * num_subreqs;
  if (sr10 > g_max_subreqs_10cy) g_max_subreqs_10cy = sr10;
  if (sr20 > g_max_subreqs_20cy) g_max_subreqs_20cy = sr20;
}

// ---------------------------------------------------------------------------
// Helper: finalize a tile master record when all subreqs are complete
// ---------------------------------------------------------------------------
inline void tile_finalize(uint64_t tile_id, uint64_t complete_cycle)
{
  std::lock_guard<std::recursive_mutex> lk(g_tile_mtx);
  auto it = g_tile_masters.find(tile_id);
  if (it == g_tile_masters.end()) return;

  tile_master_record& rec = it->second;
  rec.complete_cycle      = complete_cycle;
  rec.completion_latency  = (complete_cycle > rec.issue_cycle) ? (complete_cycle - rec.issue_cycle) : 0;

  // Accumulate subreq service cycles (fill_cycle - gen_cycle for each)
  rec.service_cycles = 0;
  for (const auto& sr : rec.subreqs) {
    if (sr.fill_cycle > sr.gen_cycle) {
      rec.service_cycles += sr.fill_cycle - sr.gen_cycle;
    }
  }

  g_completed_tiles.push_back(rec);
  g_tile_masters.erase(it);
}

// ---------------------------------------------------------------------------
// percentile helper (operates on sorted copy)
// ---------------------------------------------------------------------------
template <typename T>
static inline double percentile(std::vector<T> v, double pct)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  std::size_t idx = static_cast<std::size_t>(pct / 100.0 * (v.size() - 1));
  return static_cast<double>(v[idx]);
}

// ---------------------------------------------------------------------------
// CSV flush – writes two CSV files; only compiled in when AMX_TILE_DEBUG is set
// ---------------------------------------------------------------------------
inline void tile_csv_flush()
{
#ifdef AMX_TILE_DEBUG
  // tile_master_trace.csv
  {
    std::ofstream f("tile_master_trace.csv");
    f << "tile_id,issue_cycle,complete_cycle,retire_cycle,pc,base_addr,stride,"
         "rows,colsb,payload_bytes,temporal,page_crossing,touched_lines_total,"
         "unique_lines,merged_subreqs,l1_hits,l1_misses,l2_hits,llc_hits,dram_hits,"
         "lfb_allocs,mshr_allocs,l1d_rq_enqueues,l2_rq_enqueues,"
         "l1d_queue_wait_cycles,l2_queue_wait_cycles,service_cycles,completion_latency\n";
    for (const auto& r : g_completed_tiles) {
      f << r.id              << ',' << r.issue_cycle    << ',' << r.complete_cycle << ','
        << r.retire_cycle    << ',' << r.instr_pc       << ',' << r.base_addr      << ','
        << r.stride          << ',' << r.rows           << ',' << r.colsb          << ','
        << r.payload_bytes   << ',' << (int)r.is_temporal << ',' << (int)r.page_crossing << ','
        << r.touched_lines_total << ',' << r.unique_lines << ',' << r.merged_subreqs << ','
        << r.l1_hits         << ',' << r.l1_misses      << ',' << r.l2_hits        << ','
        << r.llc_hits        << ',' << r.dram_hits      << ',' << r.lfb_allocs     << ','
        << r.mshr_allocs     << ',' << r.l1d_rq_enqueues << ',' << r.l2_rq_enqueues << ','
        << r.l1d_queue_wait_cycles << ',' << r.l2_queue_wait_cycles << ','
        << r.service_cycles  << ',' << r.completion_latency << '\n';
    }
  }

  // tile_subreq_trace.csv
  {
    std::ofstream f("tile_subreq_trace.csv");
    f << "tile_id,row_idx,line_addr,gen_cycle,l1d_enqueue_cycle,l1d_dispatch_cycle,"
         "l2_enqueue_cycle,l2_dispatch_cycle,fill_cycle,merged,l1_hit,l2_hit,llc_hit,dram_miss\n";
    for (const auto& r : g_completed_tiles) {
      for (const auto& sr : r.subreqs) {
        f << sr.tile_id       << ',' << sr.row_idx          << ',' << sr.line_addr      << ','
          << sr.gen_cycle     << ',' << sr.l1d_enqueue_cycle << ',' << sr.l1d_dispatch_cycle << ','
          << sr.l2_enqueue_cycle << ',' << sr.l2_dispatch_cycle << ',' << sr.fill_cycle  << ','
          << (int)sr.merged   << ',' << (int)sr.l1_hit      << ',' << (int)sr.l2_hit    << ','
          << (int)sr.llc_hit  << ',' << (int)sr.dram_miss   << '\n';
      }
    }
  }
#endif  // AMX_TILE_DEBUG
}

// ---------------------------------------------------------------------------
// Print AMX summary – always compiled in (guards on g_completed_tiles.empty)
// ---------------------------------------------------------------------------
inline void tile_print_summary(std::ostream& out)
{
  uint64_t total_loads = g_total_tileloadd + g_total_tileloaddt1;

  // Build vectors for percentile calculations
  std::vector<uint64_t> completion_latencies;
  std::vector<uint64_t> inter_arrivals;
  std::vector<uint32_t> unique_lines_vec;
  completion_latencies.reserve(g_completed_tiles.size());
  unique_lines_vec.reserve(g_completed_tiles.size());

  double sum_payload = 0, sum_unique = 0, sum_merged = 0;
  double sum_lat = 0;
  double sum_lfb = 0, sum_lfbh = 0, sum_l2rq = 0;
  double sum_l1h = 0, sum_l2h = 0, sum_llch = 0, sum_dram = 0;

  for (const auto& r : g_completed_tiles) {
    completion_latencies.push_back(r.completion_latency);
    unique_lines_vec.push_back(r.unique_lines);
    sum_payload  += r.payload_bytes;
    sum_unique   += r.unique_lines;
    sum_merged   += r.merged_subreqs;
    sum_lat      += r.completion_latency;
    sum_lfb      += r.lfb_allocs;
    sum_lfbh     += r.lfb_hits;
    sum_l2rq     += r.l2_rq_enqueues;
    sum_l1h      += r.l1_hits;
    sum_l2h      += r.l2_hits;
    sum_llch     += r.llc_hits;
    sum_dram     += r.dram_hits;
  }

  // Inter-arrival
  auto& arr = g_arrival_cycles;  // NOTE: already flushed during sim; this is an approximation
  // Rebuild from completed tiles by sorting issue_cycles
  std::vector<uint64_t> issue_sorted;
  issue_sorted.reserve(g_completed_tiles.size());
  for (const auto& r : g_completed_tiles) issue_sorted.push_back(r.issue_cycle);
  std::sort(issue_sorted.begin(), issue_sorted.end());
  for (std::size_t i = 1; i < issue_sorted.size(); ++i) {
    inter_arrivals.push_back(issue_sorted[i] - issue_sorted[i - 1]);
  }
  double avg_inter = inter_arrivals.empty() ? 0.0 :
      static_cast<double>(std::accumulate(inter_arrivals.begin(), inter_arrivals.end(), 0ULL)) / inter_arrivals.size();

  std::size_t N = g_completed_tiles.size();
  auto avg = [&](double s) { return N ? s / N : 0.0; };

  out << "\n=== AMX Tile Load Summary ===\n";
  out << "  total tileloads:                       " << total_loads << "\n";
  out << "  total TILELOADD / TILELOADDT1:         " << g_total_tileloadd << " / " << g_total_tileloaddt1 << "\n";
  out << "  total AMX compute (TDPBF16PS etc.):    " << g_total_amx_compute << "\n";
  out << "  avg payload bytes / tileload:          " << avg(sum_payload) << "\n";
  out << "  avg unique cache lines / tileload:     " << avg(sum_unique) << "\n";
  out << "  p95 unique cache lines / tileload:     " << percentile(unique_lines_vec, 95) << "\n";
  out << "  avg merged subreqs / tileload:         " << avg(sum_merged) << "\n";
  out << "  avg completion latency:                " << avg(sum_lat) << " cycles\n";
  out << "  p95 completion latency:                " << percentile(completion_latencies, 95) << " cycles\n";
  out << "  avg inter-arrival cycles:              " << avg_inter << "\n";
  out << "  p95 inter-arrival cycles:              " << percentile(inter_arrivals, 95) << "\n";
  out << "  max tileloads / 10 cycles:             " << g_max_tileloads_10cy << "\n";
  out << "  max tileloads / 20 cycles:             " << g_max_tileloads_20cy << "\n";
  out << "  max subreqs / 10 cycles (approx):      " << g_max_subreqs_10cy << "\n";
  out << "  max subreqs / 20 cycles (approx):      " << g_max_subreqs_20cy << "\n";
  out << "  max active tile masters:               " << g_max_active_masters << "\n";
  out << "  avg LFB allocs / tile:                 " << avg(sum_lfb) << "\n";
  out << "  avg LFB child hits / tile (MSHR):      " << avg(sum_lfbh) << "\n";
  out << "  avg L2 RQ enqueues / tile:             " << avg(sum_l2rq) << "\n";
  out << "  avg L1-hit lines / tile:               " << avg(sum_l1h) << "\n";
  out << "  avg L2-hit lines / tile:               " << avg(sum_l2h) << "\n";
  out << "  avg LLC-hit lines / tile:              " << avg(sum_llch) << "\n";
  out << "  avg DRAM lines / tile:                 " << avg(sum_dram) << "\n";
  out << std::flush;
}

#endif  // TILE_RECORD_H
