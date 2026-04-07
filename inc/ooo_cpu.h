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

#ifndef OOO_CPU_H
#define OOO_CPU_H

#ifdef CHAMPSIM_MODULE
#define SET_ASIDE_CHAMPSIM_MODULE
#undef CHAMPSIM_MODULE
#endif

#include <array>
#include <bitset>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "bandwidth.h"
#include "champsim.h"
#include "channel.h"
#include "core_builder.h"
#include "core_stats.h"
#include "instruction.h"
#include "modules.h"
#include "operable.h"
#include "register_allocator.h"
#include "util/lru_table.h"
#include "util/to_underlying.h"

class CACHE;
class CacheBus
{
  using channel_type = champsim::channel;
  using request_type = typename channel_type::request_type;
  using response_type = typename channel_type::response_type;

  channel_type* lower_level;
  uint32_t cpu;

  friend class O3_CPU;

public:
  CacheBus(uint32_t cpu_idx, champsim::channel* ll) : lower_level(ll), cpu(cpu_idx) {}
  bool issue_read(request_type packet);
  bool issue_write(request_type packet);
};

struct LSQ_ENTRY : champsim::program_ordered<LSQ_ENTRY> {
  champsim::address virtual_address{};
  champsim::address ip{};
  champsim::chrono::clock::time_point ready_time{champsim::chrono::clock::time_point::max()};

  std::array<uint8_t, 2> asid = {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::max()};
  bool fetch_issued = false;
  bool amx_tileload = false;
  bool is_tile_instruction = false;  // Always true for tile loads, regardless of sidecar mode
  uint64_t tile_group_id = 0;
  uint8_t tile_subidx = 0;
  uint8_t tile_num_children = 0;  // Total children in this tile parent
  bool tile_burst = false;
  bool oracle_checked = false;
  bool amx_tilestore = false;
  bool amx_non_temporal = false;  // true for TILELOADDT1 (non-temporal hint → skip L1D fill)
  bool oracle_hit_status = false;
  uint64_t lq_creation_cycle = 0;  // cycle when this LQ entry was created
  uint64_t lq_issue_cycle = 0;     // cycle when fetch_issued became true
  bool amx_in_tat = false;
  uint8_t tile_dest_tmm = 0;  // destination TMM register (199-202 = tmm4-7)

  // Tile LQ coalescing: single LQ entry holds all children
  bool is_tile_lq_parent = false;
  std::vector<champsim::address> tile_child_addrs;       // child cache line addresses
  uint16_t tile_children_total_lq = 0;                   // total children (e.g. 16)
  uint16_t tile_children_issued_lq = 0;                  // children sent to L1D
  uint16_t tile_children_completed_lq = 0;               // children with L1D response
  uint32_t tile_child_completed_mask = 0;                // per-child completion bitmap

  uint64_t producer_id = std::numeric_limits<uint64_t>::max();
  std::vector<std::reference_wrapper<std::optional<LSQ_ENTRY>>> lq_depend_on_me{};

  LSQ_ENTRY(champsim::address addr, champsim::program_ordered<LSQ_ENTRY>::id_type id, champsim::address ip, std::array<uint8_t, 2> asid);
  void finish(ooo_model_instr& rob_entry) const;
  void finish(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end) const;
};

// cpu
class O3_CPU : public champsim::operable
{
public:
  uint32_t cpu = 0;

  // cycle
  champsim::chrono::clock::time_point begin_phase_time{};
  long long begin_phase_instr = 0;
  champsim::chrono::clock::time_point finish_phase_time{};
  long long finish_phase_instr = 0;
  champsim::chrono::clock::time_point last_heartbeat_time{};
  long long last_heartbeat_instr = 0;

  // instruction
  long long num_retired = 0;
  long long num_retired_prefetch = 0;  // SW prefetch instructions retired (excluded from sim_instr)
  uint64_t oracle_fed_instr_id = 0;

  bool show_heartbeat = true;

  using stats_type = cpu_stats;

  stats_type roi_stats{}, sim_stats{};

  struct stall_breakdown {
    uint64_t total_cycles = 0;
    uint64_t retire_cycles = 0;
    uint64_t front_end = 0;
    uint64_t rob_full = 0;
    uint64_t lq_full = 0;
    uint64_t sq_full = 0;
    uint64_t l1d_miss = 0;
    uint64_t l2_bound = 0;   // l1d_miss sub: miss pending at L2
    uint64_t l3_bound = 0;   // l1d_miss sub: miss pending at L3
    uint64_t dram_bound = 0; // l1d_miss sub: miss pending at DRAM
    uint64_t lfb_full = 0;
    // lfb_full sub: where is the ROB-head miss actually pending?
    uint64_t lfb_full_l2_pending = 0;
    uint64_t lfb_full_l3_pending = 0;
    uint64_t lfb_full_dram_pending = 0;
    uint64_t lfb_full_no_miss = 0;     // lfb_full but ROB head has no pending miss
    uint64_t other = 0;
    // "other" breakdown
    uint64_t other_amx_compute = 0;   // ROB head is TDPBF16PS waiting for latency
    uint64_t other_exec_wait = 0;     // ROB head executed but not completed (non-AMX)
    uint64_t other_not_executed = 0;  // ROB head not yet executed
    uint64_t other_dispatch_empty = 0; // dispatch buffer empty, nothing to dispatch
    uint64_t other_misc = 0;          // everything else
    // ROB-head source classification: when stall is l2_bound/dram_bound/mshr_full,
    // what type of instruction is the ROB head? (observation only, no behavioral change)
    uint64_t l2_bound_tileload = 0;
    uint64_t l2_bound_prefetch = 0;
    uint64_t l2_bound_scalar = 0;
    uint64_t dram_bound_tileload = 0;
    uint64_t dram_bound_prefetch = 0;
    uint64_t dram_bound_scalar = 0;
    uint64_t mshr_full_tileload = 0;
    uint64_t mshr_full_prefetch = 0;
    uint64_t mshr_full_scalar = 0;
    // MSHR originator: who INITIATED the DRAM/L2 request that the ROB head is waiting on?
    uint64_t l2_orig_prefetch = 0;     // L2 miss was prefetch-initiated
    uint64_t l2_orig_demand = 0;       // L2 miss was demand-initiated
    uint64_t dram_orig_prefetch = 0;   // DRAM miss was prefetch-initiated
    uint64_t dram_orig_demand = 0;     // DRAM miss was demand-initiated
    // Overlappable raw counters (each measured independently every cycle, sum > 100%)
    uint64_t raw_lfb_full = 0;
    uint64_t raw_l1d_miss_pending = 0;  // ROB head waiting for L1D miss
    uint64_t raw_l2_miss_pending = 0;   // any L1D miss is in L2 MSHR
    uint64_t raw_llc_miss_pending = 0;  // any L1D miss is in LLC MSHR
    uint64_t raw_dram_pending = 0;      // any L1D miss went to DRAM
    uint64_t raw_amx_compute_all = 0;   // ROB head is TDPBF16PS waiting (entire, including overlap)
    uint64_t raw_lq_full = 0;
    uint64_t raw_rob_full = 0;
    uint64_t raw_sq_full = 0;
    uint64_t raw_no_retire = 0;
  };

  stall_breakdown roi_stall_stats{};

  // instruction buffer
  struct dib_shift {
    champsim::data::bits shamt;
    auto operator()(champsim::address val) const { return val.slice_upper(shamt); }
  };
  using dib_type = champsim::lru_table<champsim::address, dib_shift, dib_shift>;
  dib_type DIB;

  // reorder buffer, load/store queue, register file
  std::deque<ooo_model_instr> IFETCH_BUFFER;
  std::deque<ooo_model_instr> DISPATCH_BUFFER;
  std::deque<ooo_model_instr> DECODE_BUFFER;
  std::deque<ooo_model_instr> ROB;
  std::deque<ooo_model_instr> DIB_HIT_BUFFER;

  std::vector<std::optional<LSQ_ENTRY>> LQ;
  std::deque<LSQ_ENTRY> SQ;

  // Constants
  const std::size_t IFETCH_BUFFER_SIZE, DISPATCH_BUFFER_SIZE, DECODE_BUFFER_SIZE, REGISTER_FILE_SIZE, ROB_SIZE, SQ_SIZE, DIB_HIT_BUFFER_SIZE;
  champsim::bandwidth::maximum_type FETCH_WIDTH, DECODE_WIDTH, DISPATCH_WIDTH, SCHEDULER_SIZE, EXEC_WIDTH, DIB_INORDER_WIDTH;
  champsim::bandwidth::maximum_type LQ_WIDTH, SQ_WIDTH;
  champsim::bandwidth::maximum_type RETIRE_WIDTH;
  champsim::chrono::clock::duration BRANCH_MISPREDICT_PENALTY;
  champsim::chrono::clock::duration DISPATCH_LATENCY;
  champsim::chrono::clock::duration DECODE_LATENCY;
  champsim::chrono::clock::duration SCHEDULING_LATENCY;
  champsim::chrono::clock::duration EXEC_LATENCY;
  champsim::chrono::clock::duration DIB_HIT_LATENCY;

  champsim::bandwidth::maximum_type L1I_BANDWIDTH, L1D_BANDWIDTH;

  RegisterAllocator reg_allocator{REGISTER_FILE_SIZE};

  // branch
  champsim::chrono::clock::time_point fetch_resume_time{};

  const long IN_QUEUE_SIZE;
  std::deque<ooo_model_instr> input_queue;

  CacheBus L1I_bus, L1D_bus;
  
  // Tile Admission Table (TAT)
  struct tile_admission_entry {
    bool valid = false;
    uint64_t tile_group_id = 0;
    std::deque<CacheBus::request_type> pending_children{};
    uint32_t in_flight_count = 0;
    uint64_t creation_cycle = 0;
  };
  std::array<tile_admission_entry, 16> TAT{};
  void service_l1d_tile_admission();

  CACHE* l1i;
  CACHE* l1d;
  CACHE* l2c;
  CACHE* llc;

  // AMX execution unit throughput model
  // Real Xeon AMX: latency=52 cycles, throughput=16 cycles (initiation interval)
  // Max inflight = 52/16 ≈ 3 pipelined TDPBFs
  static constexpr long long AMX_THROUGHPUT_CYCLES = 16;  // initiation interval
  champsim::chrono::clock::time_point amx_unit_busy_until{};  // earliest cycle AMX unit accepts next TDPBF16PS

  // ── TMM Version-Based WAR Dependency Tracking ──
  // Replaces MAX_INFLIGHT_TILELOADS. A tileload to tmmX can issue only when
  // all older TDP instructions that READ tmmX have issued (completed their read).
  //
  // tmm0-7 → index 0-7 (arch reg 195-202 → index = reg - 195)
  static constexpr int TMM_REG_BASE = 195;  // tmm0 = arch reg 195
  static constexpr int NUM_TMM = 8;

  struct TmmVersion {
    uint64_t version = 0;         // current version ID
    int pending_readers = 0;      // TDPs that have dispatched but not yet issued, reading this version
  };

  // Version map: version_id → pending reader count
  std::unordered_map<uint64_t, int> tmm_version_readers;
  // Current version per TMM register
  std::array<uint64_t, NUM_TMM> tmm_current_version{};
  uint64_t tmm_version_counter = 0;  // monotonic version ID generator

  // Stats
  uint64_t tmm_war_stalls = 0;  // tileload stalled due to WAR dependency

  // Tileload latency tracking (ROI)
  uint64_t tileload_total_latency = 0;
  uint64_t tileload_count = 0;

  // Legacy TMM rename state (kept for compatibility, not actively used)
  static inline int TMM_EXTRA_PHYS = 64;
  static inline int TMM_TOTAL_PHYS = 8 + TMM_EXTRA_PHYS;
  std::array<int8_t, 256> tmm_rat_table{};
  std::queue<int8_t> tmm_phys_free_list{};
  int8_t tmm_next_arch_phys = 0;
  uint64_t tmm_rename_stall_count = 0;
  uint64_t tmm_rename_alloc_count = 0;
  uint64_t tmm_rename_free_count = 0;

  // LQ occupancy + lifetime stats (ROI)
  uint64_t lq_stat_samples = 0;
  uint64_t lq_stat_tile_entries = 0;
  uint64_t lq_stat_scalar_entries = 0;
  uint64_t lq_stat_total_occupied = 0;
  uint64_t lq_stat_max_occupied = 0;
  uint64_t lq_stat_max_tile = 0;
  uint64_t lq_stat_max_scalar = 0;
  // LQ entry lifetime (creation → reset)
  uint64_t lq_lifetime_tile_total = 0;     // sum of tile child lifetimes (cycles)
  uint64_t lq_lifetime_tile_count = 0;     // number of tile child entries freed
  uint64_t lq_lifetime_scalar_total = 0;   // sum of scalar load lifetimes (cycles)
  uint64_t lq_lifetime_scalar_count = 0;   // number of scalar load entries freed
  uint64_t lq_lifetime_tile_max = 0;
  uint64_t lq_lifetime_scalar_max = 0;
  // LQ issued→return latency (fetch_issued → reset)
  uint64_t lq_issue_tile_total = 0;
  uint64_t lq_issue_tile_count = 0;
  uint64_t lq_issue_scalar_total = 0;
  uint64_t lq_issue_scalar_count = 0;

  void initialize() final;
  long operate() final;
  void begin_phase() final;
  void end_phase(unsigned cpu) final;

  void initialize_instruction();
  long check_dib();
  long fetch_instruction();
  long promote_to_decode();
  long decode_instruction();
  long dispatch_instruction();
  long schedule_instruction();
  long execute_instruction();
  long operate_lsq();
  long complete_inflight_instruction();
  long handle_memory_return();
  long retire_rob();

  void update_stall_stats(long retire_count);
  void print_stall_stats() const;

  bool do_init_instruction(ooo_model_instr& instr);
  bool do_predict_branch(ooo_model_instr& instr);
  void do_check_dib(ooo_model_instr& instr);
  bool do_fetch_instruction(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end);
  void do_dib_update(const ooo_model_instr& instr);
  void do_scheduling(ooo_model_instr& instr);
  void do_execution(ooo_model_instr& instr);
  void do_memory_scheduling(ooo_model_instr& instr);
  void do_complete_execution(ooo_model_instr& instr);
  void do_sq_forward_to_lq(LSQ_ENTRY& sq_entry, LSQ_ENTRY& lq_entry);

  void do_finish_store(const LSQ_ENTRY& sq_entry);
  bool do_complete_store(const LSQ_ENTRY& sq_entry);
  bool execute_load(const LSQ_ENTRY& lq_entry);

  [[nodiscard]] auto roi_instr() const { return roi_stats.instrs(); }
  [[nodiscard]] auto roi_cycle() const { return roi_stats.cycles(); }
  [[nodiscard]] auto sim_instr() const { return (num_retired - num_retired_prefetch) - begin_phase_instr; }
  [[nodiscard]] auto sim_cycle() const { return (current_time.time_since_epoch() / clock_period) - sim_stats.begin_cycles; }

  void print_deadlock() final;

#include "module_decl.inc"

  struct branch_module_concept {
    virtual ~branch_module_concept() = default;

    virtual void impl_initialize_branch_predictor() = 0;
    virtual void impl_last_branch_result(champsim::address ip, champsim::address target, bool taken, uint8_t branch_type) = 0;
    virtual bool impl_predict_branch(champsim::address ip, champsim::address predicted_target, bool always_taken, uint8_t branch_type) = 0;
  };

  struct btb_module_concept {
    virtual ~btb_module_concept() = default;

    virtual void impl_initialize_btb() = 0;
    virtual void impl_update_btb(champsim::address ip, champsim::address predicted_target, bool taken, uint8_t branch_type) = 0;
    virtual std::pair<champsim::address, bool> impl_btb_prediction(champsim::address ip, uint8_t branch_type) = 0;
  };

  template <typename... Bs>
  struct branch_module_model final : branch_module_concept {
    std::tuple<Bs...> intern_;
    explicit branch_module_model(O3_CPU* cpu) : intern_(Bs{cpu}...) { (void)cpu; /* silence -Wunused-but-set-parameter when sizeof...(Bs) == 0 */ }

    void impl_initialize_branch_predictor() final;
    void impl_last_branch_result(champsim::address ip, champsim::address target, bool taken, uint8_t branch_type) final;
    [[nodiscard]] bool impl_predict_branch(champsim::address ip, champsim::address predicted_target, bool always_taken, uint8_t branch_type) final;
  };

  template <typename... Ts>
  struct btb_module_model final : btb_module_concept {
    std::tuple<Ts...> intern_;
    explicit btb_module_model(O3_CPU* cpu) : intern_(Ts{cpu}...) { (void)cpu; /* silence -Wunused-but-set-parameter when sizeof...(Ts) == 0 */ }

    void impl_initialize_btb() final;
    void impl_update_btb(champsim::address ip, champsim::address predicted_target, bool taken, uint8_t branch_type) final;
    [[nodiscard]] std::pair<champsim::address, bool> impl_btb_prediction(champsim::address ip, uint8_t branch_type) final;
  };

  std::unique_ptr<branch_module_concept> branch_module_pimpl;
  std::unique_ptr<btb_module_concept> btb_module_pimpl;

  // NOLINTBEGIN(readability-make-member-function-const): legacy modules use non-const hooks
  void impl_initialize_branch_predictor() const;
  void impl_last_branch_result(champsim::address ip, champsim::address target, bool taken, uint8_t branch_type) const;
  [[nodiscard]] bool impl_predict_branch(champsim::address ip, champsim::address predicted_target, bool always_taken, uint8_t branch_type) const;

  void impl_initialize_btb() const;
  void impl_update_btb(champsim::address ip, champsim::address predicted_target, bool taken, uint8_t branch_type) const;
  [[nodiscard]] std::pair<champsim::address, bool> impl_btb_prediction(champsim::address ip, uint8_t branch_type) const;
  // NOLINTEND(readability-make-member-function-const)

  template <typename... Bs, typename... Ts>
  explicit O3_CPU(champsim::core_builder<champsim::core_builder_module_type_holder<Bs...>, champsim::core_builder_module_type_holder<Ts...>> b)
      : champsim::operable(b.m_clock_period), cpu(b.m_cpu),
        DIB(b.m_dib_set, b.m_dib_way, {champsim::data::bits{champsim::lg2(b.m_dib_window)}}, {champsim::data::bits{champsim::lg2(b.m_dib_window)}}),
        LQ(b.m_lq_size), IFETCH_BUFFER_SIZE(b.m_ifetch_buffer_size), DISPATCH_BUFFER_SIZE(b.m_dispatch_buffer_size), DECODE_BUFFER_SIZE(b.m_decode_buffer_size),
        REGISTER_FILE_SIZE(b.m_register_file_size), ROB_SIZE(b.m_rob_size), SQ_SIZE(b.m_sq_size), DIB_HIT_BUFFER_SIZE(b.m_dib_hit_buffer_size),
        FETCH_WIDTH(b.m_fetch_width), DECODE_WIDTH(b.m_decode_width), DISPATCH_WIDTH(b.m_dispatch_width), SCHEDULER_SIZE(b.m_schedule_width),
        EXEC_WIDTH(b.m_execute_width), DIB_INORDER_WIDTH(b.m_dib_inorder_width), LQ_WIDTH(b.m_lq_width), SQ_WIDTH(b.m_sq_width), RETIRE_WIDTH(b.m_retire_width),
        BRANCH_MISPREDICT_PENALTY(b.m_mispredict_penalty * b.m_clock_period), DISPATCH_LATENCY(b.m_dispatch_latency * b.m_clock_period),
        DECODE_LATENCY(b.m_decode_latency * b.m_clock_period), SCHEDULING_LATENCY(b.m_schedule_latency * b.m_clock_period),
        EXEC_LATENCY(b.m_execute_latency * b.m_clock_period), DIB_HIT_LATENCY(b.m_dib_hit_latency * b.m_clock_period), L1I_BANDWIDTH(b.m_l1i_bw),
        L1D_BANDWIDTH(b.m_l1d_bw), IN_QUEUE_SIZE(2 * champsim::to_underlying(b.m_fetch_width)), L1I_bus(b.m_cpu, b.m_fetch_queues),
        L1D_bus(b.m_cpu, b.m_data_queues), l1i(b.m_l1i), l1d(b.m_l1d), l2c(b.m_l2c), llc(b.m_llc),
        branch_module_pimpl(std::make_unique<branch_module_model<Bs...>>(this)),
        btb_module_pimpl(std::make_unique<btb_module_model<Ts...>>(this))
  {
  }
};

template <typename... Bs>
void O3_CPU::branch_module_model<Bs...>::impl_initialize_branch_predictor()
{
  [[maybe_unused]] auto process_one = [&](auto& b) {
    using namespace champsim::modules;
    if constexpr (branch_predictor::has_initialize<decltype(b)>)
      b.initialize_branch_predictor();
  };

  std::apply([&](auto&... b) { (..., process_one(b)); }, intern_);
}

template <typename... Bs>
void O3_CPU::branch_module_model<Bs...>::impl_last_branch_result(champsim::address ip, champsim::address target, bool taken, uint8_t branch_type)
{
  [[maybe_unused]] auto process_one = [&](auto& b) {
    using namespace champsim::modules;
    if constexpr (branch_predictor::has_last_branch_result<decltype(b), uint64_t, uint64_t, bool, uint8_t>)
      b.last_branch_result(ip.to<uint64_t>(), target.to<uint64_t>(), taken, branch_type);
    if constexpr (branch_predictor::has_last_branch_result<decltype(b), champsim::address, champsim::address, bool, uint8_t>)
      b.last_branch_result(ip, target, taken, branch_type);
  };

  std::apply([&](auto&... b) { (..., process_one(b)); }, intern_);
}

template <typename... Bs>
bool O3_CPU::branch_module_model<Bs...>::impl_predict_branch(champsim::address ip, champsim::address predicted_target, bool always_taken, uint8_t branch_type)
{
  using return_type = bool;
  [[maybe_unused]] auto process_one = [&](auto& b) {
    using namespace champsim::modules;
    /* Strong addresses, full size */
    if constexpr (branch_predictor::has_predict_branch<decltype(b), champsim::address, champsim::address, bool, uint8_t>)
      return return_type{b.predict_branch(ip, predicted_target, always_taken, branch_type)};

    /* Raw integer addresses, full size */
    if constexpr (branch_predictor::has_predict_branch<decltype(b), uint64_t, uint64_t, bool, uint8_t>)
      return return_type{b.predict_branch(ip.to<uint64_t>(), predicted_target.to<uint64_t>(), always_taken, branch_type)};

    /* Strong addresses, short size */
    if constexpr (branch_predictor::has_predict_branch<decltype(b), champsim::address>)
      return return_type{b.predict_branch(ip)};

    /* Raw integer addresses, short size */
    if constexpr (branch_predictor::has_predict_branch<decltype(b), uint64_t>)
      return return_type{b.predict_branch(ip.to<uint64_t>())};

    return return_type{};
  };

  if constexpr (sizeof...(Bs)) {
    return std::apply([&](auto&... b) { return (..., process_one(b)); }, intern_);
  }
  return return_type{};
}

template <typename... Ts>
void O3_CPU::btb_module_model<Ts...>::impl_initialize_btb()
{
  [[maybe_unused]] auto process_one = [&](auto& t) {
    using namespace champsim::modules;
    if constexpr (btb::has_initialize<decltype(t)>)
      t.initialize_btb();
  };

  std::apply([&](auto&... t) { (..., process_one(t)); }, intern_);
}

template <typename... Ts>
void O3_CPU::btb_module_model<Ts...>::impl_update_btb(champsim::address ip, champsim::address predicted_target, bool taken, uint8_t branch_type)
{
  [[maybe_unused]] auto process_one = [&](auto& t) {
    using namespace champsim::modules;
    if constexpr (btb::has_update_btb<decltype(t), champsim::address, champsim::address, bool, uint8_t>)
      t.update_btb(ip, predicted_target, taken, branch_type);
    if constexpr (btb::has_update_btb<decltype(t), uint64_t, uint64_t, bool, uint8_t>)
      t.update_btb(ip.to<uint64_t>(), predicted_target.to<uint64_t>(), taken, branch_type);
  };

  std::apply([&](auto&... t) { (..., process_one(t)); }, intern_);
}

template <typename... Ts>
std::pair<champsim::address, bool> O3_CPU::btb_module_model<Ts...>::impl_btb_prediction(champsim::address ip, uint8_t branch_type)
{
  using return_type = std::pair<champsim::address, bool>;
  [[maybe_unused]] auto process_one = [&](auto& t) {
    using namespace champsim::modules;

    /* Strong addresses, full size */
    if constexpr (btb::has_btb_prediction<decltype(t), champsim::address, uint8_t>)
      return return_type{t.btb_prediction(ip, branch_type)};

    /* Strong addresses, short size */
    if constexpr (btb::has_btb_prediction<decltype(t), champsim::address>)
      return return_type{t.btb_prediction(ip)};

    /* Raw integer addresses, full size */
    if constexpr (btb::has_btb_prediction<decltype(t), uint64_t, uint8_t>)
      return return_type{t.btb_prediction(ip.to<uint64_t>(), branch_type)};

    /* Raw integer addresses, short size */
    if constexpr (btb::has_btb_prediction<decltype(t), uint64_t>)
      return return_type{t.btb_prediction(ip.to<uint64_t>())};

    return return_type{};
  };

  if constexpr (sizeof...(Ts) > 0) {
    return std::apply([&](auto&... t) { return (..., process_one(t)); }, intern_);
  }
  return return_type{};
}

#ifdef SET_ASIDE_CHAMPSIM_MODULE
#undef SET_ASIDE_CHAMPSIM_MODULE
#define CHAMPSIM_MODULE
#endif

#endif
