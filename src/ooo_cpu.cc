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

#include "ooo_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "cache.h"
#include "champsim.h"
#include "deadlock.h"
#include "instruction.h"
#include "util/span.h"

// AMX tile instrumentation (global storage defined here, extern in other TUs)
#define AMX_TILE_IMPL
#include "tile_record.h"
#include "oracle_l1.h"
#include <set>

std::chrono::seconds elapsed_time();

constexpr long long STAT_PRINTING_PERIOD = 10000000;

long O3_CPU::operate()
{
  long progress{0};
  auto retired = retire_rob();                // retire
  progress += retired;
  progress += complete_inflight_instruction(); // finalize execution
  progress += execute_instruction();           // execute instructions
  progress += schedule_instruction();          // schedule instructions
  progress += handle_memory_return();          // finalize memory transactions
  progress += operate_lsq();                   // execute memory transactions
  // Tile children now flow directly to L1D — no TAT draining needed.

  progress += dispatch_instruction(); // dispatch
  progress += decode_instruction();   // decode
  progress += promote_to_decode();

  progress += fetch_instruction(); // fetch
  progress += check_dib();
  initialize_instruction();

  // heartbeat
  if (show_heartbeat && (num_retired >= (last_heartbeat_instr + STAT_PRINTING_PERIOD))) {
    using double_duration = std::chrono::duration<double, typename champsim::chrono::picoseconds::period>;
    auto heartbeat_instr{std::ceil(num_retired - last_heartbeat_instr)};
    auto heartbeat_cycle{double_duration{current_time - last_heartbeat_time} / clock_period};

    auto phase_instr{std::ceil(num_retired - begin_phase_instr)};
    auto phase_cycle{double_duration{current_time - begin_phase_time} / clock_period};

    fmt::print("Heartbeat CPU {} instructions: {} cycles: {} heartbeat IPC: {:.4g} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", cpu,
               num_retired, current_time.time_since_epoch() / clock_period, heartbeat_instr / heartbeat_cycle, phase_instr / phase_cycle, elapsed_time());

    last_heartbeat_instr = num_retired;
    last_heartbeat_time = current_time;
  }

  update_stall_stats(retired);

  return progress;
}

// TMM rename debug logging (set to false to suppress per-instruction logs)
static constexpr bool TMM_RENAME_DEBUG = false;

void O3_CPU::initialize()
{
  // BRANCH PREDICTOR & BTB
  impl_initialize_branch_predictor();
  impl_initialize_btb();

  // Initialize TMM physical register rename state
  tmm_rat_table.fill(-1);
  while (!tmm_phys_free_list.empty()) tmm_phys_free_list.pop();
  for (int8_t i = 8; i < static_cast<int8_t>(TMM_TOTAL_PHYS); i++) {
    tmm_phys_free_list.push(i);
  }
  tmm_next_arch_phys = 0;
  tmm_rename_stall_count = 0;
  tmm_rename_alloc_count = 0;
  tmm_rename_free_count = 0;
  fmt::print("[TMM_RENAME] CPU {} initialized: arch=8 extra={} total={} free_list_size={}\n",
             cpu, TMM_EXTRA_PHYS, TMM_TOTAL_PHYS, tmm_phys_free_list.size());
}

void O3_CPU::begin_phase()
{
  begin_phase_instr = num_retired - num_retired_prefetch;
  begin_phase_time = current_time;
  roi_stall_stats = {};
  tileload_total_latency = 0;
  tileload_count = 0;

  // Record where the next phase begins
  stats_type stats;
  stats.name = "CPU " + std::to_string(cpu);
  stats.begin_instrs = num_retired;
  stats.begin_cycles = begin_phase_time.time_since_epoch() / clock_period;
  sim_stats = stats;
}

void O3_CPU::end_phase(unsigned finished_cpu)
{
  // Record where the phase ended (overwrite if this is later)
  sim_stats.end_instrs = num_retired;
  sim_stats.end_cycles = current_time.time_since_epoch() / clock_period;

  if (finished_cpu == this->cpu) {
    finish_phase_instr = num_retired;
    finish_phase_time = current_time;

    roi_stats = sim_stats;
    if (!warmup) {
      print_stall_stats();
    }
  }
}

void O3_CPU::initialize_instruction()
{
  champsim::bandwidth instrs_to_read_this_cycle{
      std::min(FETCH_WIDTH, champsim::bandwidth::maximum_type{static_cast<long>(IFETCH_BUFFER_SIZE - std::size(IFETCH_BUFFER))})};

  bool stop_fetch = false;
  while (current_time >= fetch_resume_time && instrs_to_read_this_cycle.has_remaining() && !stop_fetch && !std::empty(input_queue)) {
    instrs_to_read_this_cycle.consume();

    stop_fetch = do_init_instruction(input_queue.front());

    // Removed Oracle L1 Hook trace lookahead from here; now in champsim.cc
    // Add to IFETCH_BUFFER
    IFETCH_BUFFER.push_back(input_queue.front());
    input_queue.pop_front();

    IFETCH_BUFFER.back().ready_time = current_time;
  }
}

namespace
{
void do_stack_pointer_folding(ooo_model_instr& arch_instr)
{
  // The exact, true value of the stack pointer for any given instruction can usually be determined immediately after the instruction is decoded without
  // waiting for the stack pointer's dependency chain to be resolved.
  bool writes_sp = (std::count(std::begin(arch_instr.destination_registers), std::end(arch_instr.destination_registers), champsim::REG_STACK_POINTER) > 0);
  if (writes_sp) {
    // Avoid creating register dependencies on the stack pointer for calls, returns, pushes, and pops, but not for variable-sized changes in the
    // stack pointer position. reads_other indicates that the stack pointer is being changed by a variable amount, which can't be determined before
    // execution.
    bool reads_other =
        (std::count_if(std::begin(arch_instr.source_registers), std::end(arch_instr.source_registers),
                       [](auto r) { return r != champsim::REG_STACK_POINTER && r != champsim::REG_FLAGS && r != champsim::REG_INSTRUCTION_POINTER; })
         > 0);
    if ((arch_instr.is_branch) || !(std::empty(arch_instr.destination_memory) && std::empty(arch_instr.source_memory)) || (!reads_other)) {
      auto nonsp_end = std::remove(std::begin(arch_instr.destination_registers), std::end(arch_instr.destination_registers), champsim::REG_STACK_POINTER);
      arch_instr.destination_registers.erase(nonsp_end, std::end(arch_instr.destination_registers));
    }
  }
}
} // namespace

bool O3_CPU::do_predict_branch(ooo_model_instr& arch_instr)
{
  bool stop_fetch = false;

  // handle branch prediction for all instructions as at this point we do not know if the instruction is a branch
  sim_stats.total_branch_types.increment(arch_instr.branch);
  auto [predicted_branch_target, always_taken] = impl_btb_prediction(arch_instr.ip, arch_instr.branch);
  arch_instr.branch_prediction = impl_predict_branch(arch_instr.ip, predicted_branch_target, always_taken, arch_instr.branch) || always_taken;
  if (!arch_instr.branch_prediction) {
    predicted_branch_target = champsim::address{};
  }

  if (arch_instr.is_branch) {
    if constexpr (champsim::debug_print) {
      fmt::print("[BRANCH] instr_id: {} ip: {} taken: {}\n", arch_instr.instr_id, arch_instr.ip, arch_instr.branch_taken);
    }

    // call code prefetcher every time the branch predictor is used
    l1i->impl_prefetcher_branch_operate(arch_instr.ip, arch_instr.branch, predicted_branch_target);

    if (predicted_branch_target != arch_instr.branch_target
        || (((arch_instr.branch == BRANCH_CONDITIONAL) || (arch_instr.branch == BRANCH_OTHER))
            && arch_instr.branch_taken != arch_instr.branch_prediction)) { // conditional branches are re-evaluated at decode when the target is computed
      sim_stats.total_rob_occupancy_at_branch_mispredict += std::size(ROB);
      sim_stats.branch_type_misses.increment(arch_instr.branch);
      if (!warmup) {
        fetch_resume_time = champsim::chrono::clock::time_point::max();
        stop_fetch = true;
        arch_instr.branch_mispredicted = true;
      }
    } else {
      stop_fetch = arch_instr.branch_taken; // if correctly predicted taken, then we can't fetch anymore instructions this cycle
    }

    impl_update_btb(arch_instr.ip, arch_instr.branch_target, arch_instr.branch_taken, arch_instr.branch);
    impl_last_branch_result(arch_instr.ip, arch_instr.branch_target, arch_instr.branch_taken, arch_instr.branch);
  }

  return stop_fetch;
}

bool O3_CPU::do_init_instruction(ooo_model_instr& arch_instr)
{
  // fast warmup eliminates register dependencies between instructions branch predictor, cache contents, and prefetchers are still warmed up
  if (warmup) {
    arch_instr.source_registers.clear();
    arch_instr.destination_registers.clear();
  }

  ::do_stack_pointer_folding(arch_instr);
  return do_predict_branch(arch_instr);
}

long O3_CPU::check_dib()
{
  // scan through IFETCH_BUFFER to find instructions that hit in the decoded instruction buffer
  auto begin = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), [](const ooo_model_instr& x) { return !x.dib_checked; });
  auto [window_begin, window_end] = champsim::get_span(begin, std::end(IFETCH_BUFFER), champsim::bandwidth{FETCH_WIDTH});
  std::for_each(window_begin, window_end, [this](auto& ifetch_entry) { this->do_check_dib(ifetch_entry); });
  return std::distance(window_begin, window_end);
}

void O3_CPU::do_check_dib(ooo_model_instr& instr)
{
  // Check DIB to see if we recently fetched this line
  auto dib_result = DIB.check_hit(instr.ip);
  if (dib_result) {
    // The cache line is in the L0, so we can mark this as complete
    instr.fetch_completed = true;

    // Also mark it as decoded
    instr.decoded = true;

    // It can be acted on immediately
    instr.ready_time = current_time;
  }

  instr.dib_checked = true;

  if constexpr (champsim::debug_print) {
    fmt::print("[DIB] {} instr_id: {} ip: {} hit: {} cycle: {}\n", __func__, instr.instr_id, instr.ip, dib_result.has_value(),
               current_time.time_since_epoch() / clock_period);
  }
}

long O3_CPU::fetch_instruction()
{
  long progress{0};

  // Fetch a single cache line
  auto fetch_ready = [](const ooo_model_instr& x) {
    return x.dib_checked && !x.fetch_issued;
  };

  // Find the chunk of instructions in the block
  auto no_match_ip = [](const auto& lhs, const auto& rhs) {
    return champsim::block_number{lhs.ip} != champsim::block_number{rhs.ip};
  };

  auto l1i_req_begin = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), fetch_ready);
  for (champsim::bandwidth l1i_bw{L1I_BANDWIDTH}; l1i_bw.has_remaining() && l1i_req_begin != std::end(IFETCH_BUFFER); l1i_bw.consume()) {
    auto l1i_req_end = std::adjacent_find(l1i_req_begin, std::end(IFETCH_BUFFER), no_match_ip);
    if (l1i_req_end != std::end(IFETCH_BUFFER)) {
      l1i_req_end = std::next(l1i_req_end); // adjacent_find returns the first of the non-equal elements
    }

    // Issue to L1I
    auto success = do_fetch_instruction(l1i_req_begin, l1i_req_end);
    if (success) {
      std::for_each(l1i_req_begin, l1i_req_end, [](auto& x) { x.fetch_issued = true; });
      ++progress;
    }

    l1i_req_begin = std::find_if(l1i_req_end, std::end(IFETCH_BUFFER), fetch_ready);
  }

  return progress;
}

bool O3_CPU::do_fetch_instruction(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end)
{
  CacheBus::request_type fetch_packet;
  fetch_packet.v_address = begin->ip;
  fetch_packet.instr_id = begin->instr_id;
  fetch_packet.ip = begin->ip;

  std::transform(begin, end, std::back_inserter(fetch_packet.instr_depend_on_me), [](const auto& instr) { return instr.instr_id; });

  if constexpr (champsim::debug_print) {
    fmt::print("[IFETCH] {} instr_id: {} ip: {} dependents: {} event_cycle: {}\n", __func__, begin->instr_id, begin->ip,
               std::size(fetch_packet.instr_depend_on_me), begin->ready_time.time_since_epoch() / clock_period);
  }

  return L1I_bus.issue_read(fetch_packet);
}

long O3_CPU::promote_to_decode()
{
  auto is_decoded = [](const ooo_model_instr& x) {
    return x.decoded;
  };

  auto fetch_complete_and_ready = [time = current_time](const auto& x) {
    return x.fetch_completed && x.ready_time <= time;
  };

  champsim::bandwidth available_fetch_bandwidth{
      std::min(FETCH_WIDTH, std::min(champsim::bandwidth::maximum_type{static_cast<long>(DIB_HIT_BUFFER_SIZE - std::size(DIB_HIT_BUFFER))},
                                     champsim::bandwidth::maximum_type{static_cast<long>(DECODE_BUFFER_SIZE - std::size(DECODE_BUFFER))}))};

  auto fetched_check_end = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), [](const ooo_model_instr& x) { return !x.fetch_completed; });
  // find the first not fetch completed
  auto [window_begin, window_end] = champsim::get_span_p(std::begin(IFETCH_BUFFER), fetched_check_end, available_fetch_bandwidth, fetch_complete_and_ready);
  auto decoded_window_end = std::stable_partition(window_begin, window_end, is_decoded); // reorder instructions
  auto mark_for_decode = [time = current_time, lat = DECODE_LATENCY, warmup = warmup](auto& x) {
    return x.ready_time = time + (warmup ? champsim::chrono::clock::duration{} : lat);
  };
  // to DIB_HIT_BUFFER
  auto mark_for_dib = [time = current_time, lat = DIB_HIT_LATENCY, warmup = warmup](auto& x) {
    return x.ready_time = time + lat;
  };

  std::for_each(window_begin, decoded_window_end, mark_for_dib); // assume DECODE_LATENCY = DIB_HIT_LATENCY
  std::move(window_begin, decoded_window_end, std::back_inserter(DIB_HIT_BUFFER));
  // to DECODE_BUFFER

  std::for_each(decoded_window_end, window_end, mark_for_decode);
  std::move(decoded_window_end, window_end, std::back_inserter(DECODE_BUFFER));

  long progress{std::distance(window_begin, window_end)};
  IFETCH_BUFFER.erase(window_begin, window_end);
  return progress;
}
long O3_CPU::decode_instruction()
{
  auto is_ready = [time = current_time](const auto& x) {
    return x.ready_time <= time;
  };

  auto dib_hit_buffer_begin = std::begin(DIB_HIT_BUFFER);
  auto dib_hit_buffer_end = dib_hit_buffer_begin;
  auto decode_buffer_begin = std::begin(DECODE_BUFFER);
  auto decode_buffer_end = decode_buffer_begin;

  champsim::bandwidth available_decode_bandwidth{DECODE_WIDTH};

  // bw move instructions to dispatch_buffer
  champsim::bandwidth available_dib_inorder_bandwidth{
      std::min(DIB_INORDER_WIDTH, champsim::bandwidth::maximum_type{static_cast<long>(DISPATCH_BUFFER_SIZE - std::size(DISPATCH_BUFFER))})};

  // conditions choose how many instructions sent to dispatch_buffer
  while (dib_hit_buffer_end != std::end(DIB_HIT_BUFFER) && decode_buffer_end != std::end(DECODE_BUFFER) && available_dib_inorder_bandwidth.has_remaining()
         && available_decode_bandwidth.has_remaining() && is_ready(std::min(*dib_hit_buffer_end, *decode_buffer_end, ooo_model_instr::program_order))) {
    if (ooo_model_instr::program_order(*dib_hit_buffer_end, *decode_buffer_end)) {
      dib_hit_buffer_end++;
      available_dib_inorder_bandwidth.consume();
    } else {
      decode_buffer_end++;
      available_dib_inorder_bandwidth.consume();
      available_decode_bandwidth.consume();
    }
  }
  while (dib_hit_buffer_end != std::end(DIB_HIT_BUFFER) && available_dib_inorder_bandwidth.has_remaining() && is_ready(*dib_hit_buffer_end)
         && (decode_buffer_end == std::end(DECODE_BUFFER) || ooo_model_instr::program_order(*dib_hit_buffer_end, *decode_buffer_end))) {
    dib_hit_buffer_end++;
    available_dib_inorder_bandwidth.consume();
  }
  while (decode_buffer_end != std::end(DECODE_BUFFER) && available_dib_inorder_bandwidth.has_remaining() && available_decode_bandwidth.has_remaining()
         && is_ready(*decode_buffer_end)
         && (dib_hit_buffer_end == std::end(DIB_HIT_BUFFER) || ooo_model_instr::program_order(*decode_buffer_end, *dib_hit_buffer_end))) {
    decode_buffer_end++;
    available_dib_inorder_bandwidth.consume();
    available_decode_bandwidth.consume();
  }

  // decode instructions have not decoded, merge instructions with dib_hit_buffer then send to dispatch_buffer
  auto do_decode = [&, this](auto& db_entry) {
    this->do_dib_update(db_entry);

    // Resume fetch
    if (db_entry.branch_mispredicted) {
      // These branches detect the misprediction at decode
      if ((db_entry.branch == BRANCH_DIRECT_JUMP) || (db_entry.branch == BRANCH_DIRECT_CALL)
          || (((db_entry.branch == BRANCH_CONDITIONAL) || (db_entry.branch == BRANCH_OTHER)) && db_entry.branch_taken == db_entry.branch_prediction)) {
        // clear the branch_mispredicted bit so we don't attempt to resume fetch again at execute
        db_entry.branch_mispredicted = 0;
        // pay misprediction penalty
        this->fetch_resume_time = this->current_time + BRANCH_MISPREDICT_PENALTY;
      }
    }
    // Add to dispatch
    db_entry.ready_time = this->current_time + (this->warmup ? champsim::chrono::clock::duration{} : this->DISPATCH_LATENCY);

    if constexpr (champsim::debug_print) {
      fmt::print("[DECODE] do_decode instr_id: {} time: {}\n", db_entry.instr_id, this->current_time.time_since_epoch() / this->clock_period);
    }
  };

  auto do_dib_hit = [&, this](auto& dib_entry) {
    dib_entry.ready_time = this->current_time + (this->warmup ? champsim::chrono::clock::duration{} : this->DISPATCH_LATENCY);
  };

  std::for_each(decode_buffer_begin, decode_buffer_end, do_decode);
  std::for_each(dib_hit_buffer_begin, dib_hit_buffer_end, do_dib_hit);

  long progress{std::distance(dib_hit_buffer_begin, dib_hit_buffer_end) + std::distance(decode_buffer_begin, decode_buffer_end)};

  std::merge(dib_hit_buffer_begin, dib_hit_buffer_end, decode_buffer_begin, decode_buffer_end, std::back_inserter(DISPATCH_BUFFER),
             ooo_model_instr::program_order);
  DECODE_BUFFER.erase(decode_buffer_begin, decode_buffer_end);
  DIB_HIT_BUFFER.erase(dib_hit_buffer_begin, dib_hit_buffer_end);

  return progress;
}

void O3_CPU::do_dib_update(const ooo_model_instr& instr) { DIB.fill(instr.ip); }

long O3_CPU::dispatch_instruction()
{
  champsim::bandwidth available_dispatch_bandwidth{DISPATCH_WIDTH};

  // dispatch DISPATCH_WIDTH instructions into the ROB
  while (available_dispatch_bandwidth.has_remaining() && !std::empty(DISPATCH_BUFFER) && DISPATCH_BUFFER.front().ready_time <= current_time
         && std::size(ROB) != ROB_SIZE) {
    // LQ/SQ capacity check
    const auto& front_instr = DISPATCH_BUFFER.front();
    bool is_tileload_lq = front_instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
        && (front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
            || front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
    // Tileload children drip into LQ one-by-one in operate_lsq, not all-at-once here
    auto num_lq = is_tileload_lq ? (std::size_t)0 : std::size(front_instr.source_memory);
    auto num_sq = std::size(front_instr.destination_memory);

    auto free_lq_slots = (std::size_t)std::count_if(std::begin(LQ), std::end(LQ),
        [](const auto& lq_entry) { return !lq_entry.has_value(); });
    if (free_lq_slots < num_lq) break;
    if ((num_sq + std::size(SQ)) > SQ_SIZE) break;

    // ── AMX instruction classification ──
    bool is_tileload_dispatch = front_instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
        && (front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
            || front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
    bool is_tdp_dispatch = front_instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
        && front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
    bool is_tilezero_dispatch = front_instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
        && front_instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILEZERO);
    // TILEZERO writes a TMM register (same WAR constraint as tileload)
    bool is_tmm_writer = is_tileload_dispatch || is_tilezero_dispatch;

    // ── TMM WAR dependency check: tileload/tilezero to tmmX can only dispatch if
    //    all older TDPs that read tmmX have already ISSUED (read complete).
    //    This models the real HW constraint: you can't overwrite a tile register
    //    while an AMX compute unit is still reading from it.
    //    Note: register renaming handles WAR for scalar regs, but TMM tiles are
    //    large (1KB) and NOT renamed in real HW — physical tile reg = arch tile reg.
    if (is_tmm_writer && !front_instr.destination_registers.empty()) {
      int16_t arch_reg = front_instr.destination_registers[0];
      if (arch_reg >= TMM_REG_BASE && arch_reg < TMM_REG_BASE + NUM_TMM) {
        int tmm_idx = arch_reg - TMM_REG_BASE;
        uint64_t cur_ver = tmm_current_version[tmm_idx];
        auto it = tmm_version_readers.find(cur_ver);
        if (it != tmm_version_readers.end() && it->second > 0) {
          ++tmm_war_stalls;
          break;  // WAR dependency — older TDP still reading this TMM
        }
      }
    }

    ROB.push_back(std::move(DISPATCH_BUFFER.front()));
    DISPATCH_BUFFER.pop_front();
    auto& instr_ref = ROB.back();

    if (is_tmm_writer) {
      instr_ref.dispatch_time = current_time;
      // Save architectural TMM register BEFORE rename
      if (!instr_ref.destination_registers.empty()) {
        int16_t arch_reg = instr_ref.destination_registers[0];
        instr_ref.tmm_rename_arch_reg = arch_reg;

        if (arch_reg >= TMM_REG_BASE && arch_reg < TMM_REG_BASE + NUM_TMM) {
          int tmm_idx = arch_reg - TMM_REG_BASE;
          // Record old version (for WAR check was already done above)
          instr_ref.tmm_old_version = tmm_current_version[tmm_idx];
          // Bump to new version
          uint64_t new_ver = ++tmm_version_counter;
          tmm_current_version[tmm_idx] = new_ver;
          tmm_version_readers[new_ver] = 0;  // no readers yet for new version
        }
      }
    }

    if (is_tdp_dispatch) {
      // Bind to source TMM versions. Increment then immediately decrement
      // pending_readers — the WAR dependency is resolved at dispatch time.
      // Rationale: once a TDP is dispatched, it has committed to reading its
      // source TMM versions. The actual tile register read happens within 1-2
      // cycles of issue to the pipelined AMX unit, so subsequent tileloads to
      // the same register can proceed. The WAR check above (on the current
      // version's readers) ensures in-order WAR correctness within a single
      // dispatch cycle.
      for (auto src_reg : instr_ref.source_registers) {
        if (src_reg >= TMM_REG_BASE && src_reg < TMM_REG_BASE + NUM_TMM) {
          int tmm_idx = src_reg - TMM_REG_BASE;
          uint64_t ver = tmm_current_version[tmm_idx];
          instr_ref.tmm_src_versions.push_back(ver);
          // No net increment: dispatch-time WAR resolution
        }
      }
    }

    do_memory_scheduling(instr_ref);

    available_dispatch_bandwidth.consume();
    ROB.back().ready_time = current_time + (warmup ? champsim::chrono::clock::duration{} : SCHEDULING_LATENCY);
  }

  return available_dispatch_bandwidth.amount_consumed();
}

long O3_CPU::schedule_instruction()
{
  champsim::bandwidth search_bw{SCHEDULER_SIZE};
  int progress{0};
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && search_bw.has_remaining(); ++rob_it) {
    // TMM rename stall is now in dispatch_instruction() — before LQ entries are created.

    // if there aren't enough physical registers available for the next instruction, stop scheduling
    unsigned long sources_to_allocate = std::count_if(rob_it->source_registers.begin(), rob_it->source_registers.end(),
                                                      [&alloc = std::as_const(reg_allocator)](auto srcreg) { return !alloc.isAllocated(srcreg); });
    if (reg_allocator.count_free_registers() < (sources_to_allocate + rob_it->destination_registers.size())) {
      break;
    }
    if (!rob_it->scheduled && rob_it->ready_time <= current_time) {
      do_scheduling(*rob_it);
      ++progress;
    }

    if (!rob_it->executed) {
      search_bw.consume();
    }
  }

  return progress;
}

void O3_CPU::do_scheduling(ooo_model_instr& instr)
{
  // Save original source register IDs for TMM consumer logging
  [[maybe_unused]] auto orig_src_regs = instr.source_registers;

  // Mark register dependencies - rename source registers
  for (auto& src_reg : instr.source_registers) {
    src_reg = reg_allocator.rename_src_register(src_reg);
  }

  // Rename destination registers
  for (auto& dreg : instr.destination_registers) {
    dreg = reg_allocator.rename_dest_register(dreg, instr.instr_id);
  }

  // NOTE: TMM physical register rename is now done at dispatch time
  // (dispatch_instruction), BEFORE LQ entries are created.
  // This ensures tileload dispatch stalls also prevent LQ flooding.

  // ── Log consumer TMM source capture (tdpbf16ps) ──
  if constexpr (TMM_RENAME_DEBUG) {
    bool is_compute = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
        && instr.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
    if (is_compute) {
      for (auto src_reg : orig_src_regs) {
        if (src_reg > champsim::REG_INSTRUCTION_POINTER) {
          uint8_t reg_idx = static_cast<uint8_t>(src_reg);
          int8_t phys = tmm_rat_table[reg_idx];
          fmt::print("[TMM_RENAME] CONSUMER PC: {:#x} id: {} tdpbf16ps "
                     "src_arch_reg: {} captured_phys: P{}\n",
                     instr.ip.to<uint64_t>(), instr.instr_id, src_reg, phys);
        }
      }
    }
  }

  instr.scheduled = true;
}

long O3_CPU::execute_instruction()
{
  champsim::bandwidth exec_bw{EXEC_WIDTH};
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && exec_bw.has_remaining(); ++rob_it) {
    if (rob_it->scheduled && !rob_it->executed && rob_it->ready_time <= current_time) {
      bool ready = std::all_of(std::begin(rob_it->source_registers), std::end(rob_it->source_registers),
                               [&alloc = std::as_const(reg_allocator)](auto srcreg) { return alloc.isValid(srcreg); });
      if (ready) {
        // AMX throughput constraint: AMX unit accepts 1 TDPBF16PS every 16 cycles
        bool is_amx_compute = rob_it->instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
            && rob_it->amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
        if (is_amx_compute && current_time < amx_unit_busy_until) {
          continue; // AMX unit busy — skip, try other instructions
        }

        do_execution(*rob_it);
        exec_bw.consume();

        // TMM WAR: pending_readers now decremented at schedule time (do_scheduling),
        // not at execute time. This models that the TDP has committed to reading the
        // TMM version once scheduled, so subsequent tileloads can proceed.

        // Reserve AMX unit for throughput interval
        if (is_amx_compute && !warmup) {
          auto next_available = current_time + AMX_THROUGHPUT_CYCLES * clock_period;
          if (next_available > amx_unit_busy_until) {
            amx_unit_busy_until = next_available;
          }
        }
      }
    }
  }

  return exec_bw.amount_consumed();
}

void O3_CPU::do_execution(ooo_model_instr& instr)
{
  instr.executed = true;

  // AMX TDPBF16PS has a fixed 52-cycle execution latency
  constexpr long long AMX_TDPBF16PS_LATENCY_CYCLES = 52;
  // TILEZERO: register-only zero-fill, ~2 cycle latency (no memory access)
  constexpr long long AMX_TILEZERO_LATENCY_CYCLES = 2;
  bool is_amx_compute = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && instr.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
  bool is_tilezero = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILEZERO);

  if (warmup) {
    instr.ready_time = current_time;
  } else if (is_amx_compute) {
    instr.ready_time = current_time + AMX_TDPBF16PS_LATENCY_CYCLES * clock_period;
  } else if (is_tilezero) {
    instr.ready_time = current_time + AMX_TILEZERO_LATENCY_CYCLES * clock_period;
  } else {
    instr.ready_time = current_time + EXEC_LATENCY;
  }

  // Mark LQ entries as ready to translate
  for (auto& lq_entry : LQ) {
    if (lq_entry.has_value() && lq_entry->instr_id == instr.instr_id) {
      lq_entry->ready_time = current_time + (warmup ? champsim::chrono::clock::duration{} : EXEC_LATENCY);
    }
  }

  // Mark SQ entries as ready to translate
  for (auto& sq_entry : SQ) {
    if (sq_entry.instr_id == instr.instr_id) {
      sq_entry.ready_time = current_time + (warmup ? champsim::chrono::clock::duration{} : EXEC_LATENCY);
    }
  }

  if constexpr (champsim::debug_print) {
    fmt::print("[ROB] {} instr_id: {} ready_time: {}\n", __func__, instr.instr_id, instr.ready_time.time_since_epoch() / clock_period);
  }
}

void O3_CPU::do_memory_scheduling(ooo_model_instr& instr)
{
  constexpr uint32_t CACHE_LINE_SIZE = 64;
  std::size_t total_loads = 0;
  std::size_t total_stores = 0;
  bool is_amx_tileload = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && (instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
          || instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));

  bool is_amx_tilestore = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILESTORED);

  bool is_amx_compute = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && instr.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);

  // Count AMX compute instructions
  if (is_amx_compute) {
    ++g_total_amx_compute;
  }

  // ── AMX tile register dependencies ──
  // Trace v2 already captures real tmm registers (195-202 = tmm0-tmm7):
  //   TILELOADDT1: dst=[tmm4/5], src=[GP regs]
  //   TDPBF16PS:   dst=[tmm0-3], src=[tmm_acc, tmm_A, tmm_B]
  // RAW/WAR dependencies are naturally enforced by the register renamer,
  // limiting concurrent tile loads to 4 (tmm4,5,6,7) per the real ISA.

  // Helper lambda to add a load to LQ
  auto add_load = [&](champsim::address addr, uint8_t subidx = 0, bool is_burst = false, uint8_t num_children = 0) {
    auto q_entry = std::find_if_not(std::begin(LQ), std::end(LQ), [](const auto& lq_entry) { return lq_entry.has_value(); });
    assert(q_entry != std::end(LQ));
    q_entry->emplace(addr, instr.instr_id, instr.ip, instr.asid);
    (*q_entry)->lq_creation_cycle = current_time.time_since_epoch() / clock_period;
    (*q_entry)->is_tile_instruction = is_amx_tileload;  // always track, regardless of sidecar
    if (is_amx_tileload && !instr.destination_registers.empty())
      (*q_entry)->tile_dest_tmm = static_cast<uint8_t>(instr.destination_registers[0]);

    // Verify or Detect AMX Tile Load
    // When tile sidecar is disabled, treat AMX tileloads as plain cacheline requests (baseline)
    (*q_entry)->amx_tileload = is_amx_tileload && !g_disable_tile_sidecar;
    if (is_amx_tileload && !g_disable_tile_sidecar) {
        (*q_entry)->tile_group_id = (static_cast<uint64_t>(this->cpu) << 48) | instr.instr_id;
        (*q_entry)->tile_subidx = subidx;
        (*q_entry)->tile_num_children = num_children;
        (*q_entry)->tile_burst = is_burst;
    }
    
    // Check for forwarding
    auto sq_it = std::max_element(std::begin(SQ), std::end(SQ), [addr](const auto& lhs, const auto& rhs) {
      return lhs.virtual_address != addr || (rhs.virtual_address == addr && LSQ_ENTRY::program_order(lhs, rhs));
    });
    if (sq_it != std::end(SQ) && sq_it->virtual_address == addr) {
      if (sq_it->fetch_issued) {
        (*q_entry)->finish(instr);
        q_entry->reset();
      } else {
        assert(sq_it->instr_id < instr.instr_id);
        sq_it->lq_depend_on_me.emplace_back(*q_entry);
        (*q_entry)->producer_id = sq_it->instr_id;

        if constexpr (champsim::debug_print) {
          fmt::print("[DISPATCH] {} instr_id: {} waits on: {}\n", __func__, instr.instr_id, sq_it->instr_id);
        }
      }
    }
    total_loads++;
  };

  // Helper lambda to add a store to SQ
  uint32_t tilestore_child_counter = 0;
  auto add_store = [&](champsim::address addr) {
    SQ.emplace_back(addr, instr.instr_id, instr.ip, instr.asid);
    SQ.back().amx_tilestore = is_amx_tilestore;
    if (is_amx_tilestore) {
      SQ.back().is_tile_instruction = true;
      SQ.back().tile_group_id = (static_cast<uint64_t>(this->cpu) << 48) | instr.instr_id;
      SQ.back().tile_subidx = static_cast<uint8_t>(tilestore_child_counter);
      // Total children computed from destination_mem_ops expansion
      SQ.back().tile_num_children = 0;  // will be patched after all stores added
      tilestore_child_counter++;
    }
    total_stores++;
  };

  // -----------------------------------------------------------------
  // AMX TILELOADD instrumentation: build tile_master_record at dispatch
  // -----------------------------------------------------------------
  if (is_amx_tileload && !instr.source_mem_ops.empty()) {
    uint64_t tile_id = (static_cast<uint64_t>(this->cpu) << 48) | instr.instr_id;
    uint64_t cur_cycle = current_time.time_since_epoch() / clock_period;

    // Infer tile geometry from source_mem_ops
    // Each mem_op represents one row: size=colsb, addr=row_start
    uint32_t rows  = static_cast<uint32_t>(instr.source_mem_ops.size());
    uint32_t colsb = (rows > 0) ? instr.source_mem_ops[0].size : 0;
    if (colsb == 0 && rows > 0) colsb = CACHE_LINE_SIZE; // fallback
    int64_t  stride = 0;
    if (rows >= 2) {
      stride = static_cast<int64_t>(
          instr.source_mem_ops[1].address.to<uint64_t>()) -
        static_cast<int64_t>(
          instr.source_mem_ops[0].address.to<uint64_t>());
    }
    uint64_t base_addr = rows > 0 ? instr.source_mem_ops[0].address.to<uint64_t>() : 0;

    // Collect all unique cache-line addresses + touched total
    std::set<uint64_t> unique_lines_set;
    uint32_t touched_total = 0;
    bool page_crossing = false;
    constexpr uint64_t PAGE_MASK = ~((uint64_t)0xFFF);
    uint64_t first_page = base_addr & PAGE_MASK;
    for (const auto& op : instr.source_mem_ops) {
      uint32_t nlines = op.num_cache_lines(CACHE_LINE_SIZE);
      touched_total += nlines;
      for (uint32_t li = 0; li < nlines; ++li) {
        uint64_t la = op.get_cache_line_addr(li, CACHE_LINE_SIZE).to<uint64_t>();
        unique_lines_set.insert(la);
        if ((la & PAGE_MASK) != first_page) page_crossing = true;
      }
    }

    tile_master_record& rec = tile_get_or_create(tile_id);
    rec.id               = tile_id;
    rec.issue_cycle      = cur_cycle;
    rec.instr_pc         = instr.ip.to<uint64_t>();
    rec.base_addr        = base_addr;
    rec.stride           = stride;
    rec.rows             = rows;
    rec.colsb            = colsb;
    rec.payload_bytes    = rows * colsb;
    rec.is_temporal      = (instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD));
    rec.page_crossing    = page_crossing;
    rec.row_count        = rows;
    rec.touched_lines_total = touched_total;
    rec.unique_lines     = static_cast<uint32_t>(unique_lines_set.size());

    // Build subreq records (one per unique cache line, row_idx = row index)
    uint32_t row_idx = 0;
    for (const auto& op : instr.source_mem_ops) {
      uint32_t nlines = op.num_cache_lines(CACHE_LINE_SIZE);
      for (uint32_t li = 0; li < nlines; ++li) {
        tile_subreq_record sr;
        sr.tile_id   = tile_id;
        sr.line_addr = op.get_cache_line_addr(li, CACHE_LINE_SIZE).to<uint64_t>();
        sr.row_idx   = row_idx;
        sr.gen_cycle = cur_cycle;
        rec.subreqs.push_back(sr);
      }
      ++row_idx;
    }

    // Global counters
    if (instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD))
      ++g_total_tileloadd;
    else
      ++g_total_tileloaddt1;

    // Sliding-window arrival tracking
    tile_record_arrival(cur_cycle, rec.unique_lines);

    // Active master watermark
    { std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
      uint32_t active = static_cast<uint32_t>(g_tile_masters.size());
      if (active > g_max_active_masters) g_max_active_masters = active;
    }
  }

  // Detect SW prefetch: GENERIC instruction with READ memops but no destination registers.
  // Real loads always write to a register; prefetch (prefetcht0/t1/t2/nta) does not.
  bool is_sw_prefetch = !is_amx_tileload && !is_amx_tilestore && !is_amx_compute
      && instr.destination_registers.empty()
      && (!instr.source_mem_ops.empty() || !instr.source_memory.empty());

  // Process loads - check if we have AMX-style memory operations with size info
  if (is_sw_prefetch && g_skip_sw_prefetch) {
    // SW prefetch: skip load queue — these don't consume MSHR/LFB entries.
    // Clear source mem ops so num_cache_line_accesses() returns 0 for reads,
    // allowing the instruction to retire without waiting for memory.
    instr.source_mem_ops.clear();
    instr.source_memory.clear();
  } else if (is_amx_tileload && !instr.source_mem_ops.empty()) {
    // Tileload: precompute child cache-line addresses but do NOT create LQ entries.
    // Children drip into LQ one-by-one in operate_lsq(), same rate as scalar loads.
    for (const auto& mem_op : instr.source_mem_ops) {
      uint32_t num_lines = mem_op.num_cache_lines(CACHE_LINE_SIZE);
      for (uint32_t i = 0; i < num_lines; i++) {
        instr.tile_child_addrs.push_back(mem_op.get_cache_line_addr(i, CACHE_LINE_SIZE));
      }
    }

    // tile_children_in_lq = 0 → operate_lsq will allocate them progressively
  } else if (!instr.source_mem_ops.empty()) {
    // Non-tile path: expand each memory operation to cache line LQ entries immediately
    uint32_t child_counter = 0;
    for (const auto& mem_op : instr.source_mem_ops) {
      uint32_t num_lines = mem_op.num_cache_lines(CACHE_LINE_SIZE);
      for (uint32_t i = 0; i < num_lines; i++) {
        champsim::address line_addr = mem_op.get_cache_line_addr(i, CACHE_LINE_SIZE);
        add_load(line_addr, child_counter, false, 0);
        child_counter++;
      }
    }
  } else {
    // Legacy path: single cache line per memory address
    for (auto& smem : instr.source_memory) {
      add_load(smem);
    }
  }

  // Process stores - check if we have AMX-style memory operations with size info
  if (!instr.destination_mem_ops.empty()) {
    uint32_t store_child_counter = 0;
    for (const auto& mem_op : instr.destination_mem_ops) {
      uint32_t num_lines = mem_op.num_cache_lines(CACHE_LINE_SIZE);
      for (uint32_t i = 0; i < num_lines; i++) {
        champsim::address line_addr = mem_op.get_cache_line_addr(i, CACHE_LINE_SIZE);
        add_store(line_addr);
        store_child_counter++;
      }
    }
  } else {
    // Legacy path: single cache line per memory address
    for (auto& dmem : instr.destination_memory) {
      add_store(dmem);
    }
  }

  // Patch tile_num_children for all tilestore SQ entries now that we know total count
  if (is_amx_tilestore && tilestore_child_counter > 0) {
    uint8_t total_children = static_cast<uint8_t>(tilestore_child_counter);
    for (auto it = SQ.rbegin(); it != SQ.rend() && it->instr_id == instr.instr_id; ++it) {
      if (it->amx_tilestore)
        it->tile_num_children = total_children;
    }
  }

  if constexpr (champsim::debug_print) {
    fmt::print("[DISPATCH] {} instr_id: {} loads: {} stores: {} (AMX expanded) cycle: {}\n", __func__, instr.instr_id, total_loads,
               total_stores, current_time.time_since_epoch() / clock_period);
  }
}

long O3_CPU::operate_lsq()
{
  champsim::bandwidth store_bw{SQ_WIDTH};

  const auto complete_id = std::empty(ROB) ? std::numeric_limits<uint64_t>::max() : ROB.front().instr_id;
  auto do_complete = [time = current_time, finished = LSQ_ENTRY::precedes(complete_id), this](const auto& x) {
    return finished(x) && x.ready_time <= time && this->do_complete_store(x);
  };

  auto unfetched_begin = std::partition_point(std::begin(SQ), std::end(SQ), [](const auto& x) { return x.fetch_issued; });
  auto [fetch_begin, fetch_end] =
      champsim::get_span_p(unfetched_begin, std::end(SQ), store_bw, [time = current_time](const auto& x) { return !x.fetch_issued && x.ready_time <= time; });
  store_bw.consume(std::distance(fetch_begin, fetch_end));
  std::for_each(fetch_begin, fetch_end, [time = current_time, this](auto& sq_entry) {
    // Oracle L1 Hook for Writes: Check and consume token BEFORE retirement drops it
    if (!this->warmup && g_oracle_l1[this->cpu].enabled()) {
      auto cur_cycle = current_time.time_since_epoch() / clock_period;
      sq_entry.oracle_hit_status = g_oracle_l1[this->cpu].check_hit(sq_entry.virtual_address.template to<uint64_t>(), sq_entry.amx_tilestore, sq_entry.instr_id, cur_cycle, false, true);
    }
    
    this->do_finish_store(sq_entry);
    sq_entry.fetch_issued = true;
    sq_entry.ready_time = time;
  });

  auto [complete_begin, complete_end] = champsim::get_span_p(std::cbegin(SQ), std::cend(SQ), store_bw, do_complete);
  store_bw.consume(std::distance(complete_begin, complete_end));
  SQ.erase(complete_begin, complete_end);

  // ── Tile child drip-feed: create LQ entries one-by-one, same rate as scalar loads ──
  // Instead of 16 LQ entries at dispatch, tile children enter LQ progressively here,
  // sharing the same LQ_WIDTH budget as scalar loads.
  champsim::bandwidth load_bw{LQ_WIDTH};

  for (auto& rob_entry : ROB) {
    if (!load_bw.has_remaining()) break;
    if (rob_entry.tile_child_addrs.empty()) continue;
    if (!rob_entry.executed) continue;
    if (rob_entry.tile_children_in_lq >= rob_entry.tile_child_addrs.size()) continue;

    // Inner loop: create multiple children from the same tileload up to LQ_WIDTH
    while (load_bw.has_remaining() && rob_entry.tile_children_in_lq < rob_entry.tile_child_addrs.size()) {
      // Find a free LQ slot
      auto q_entry = std::find_if_not(std::begin(LQ), std::end(LQ),
          [](const auto& e) { return e.has_value(); });
      if (q_entry == std::end(LQ)) break; // LQ full — backpressure

      auto addr = rob_entry.tile_child_addrs[rob_entry.tile_children_in_lq];
      q_entry->emplace(addr, rob_entry.instr_id, rob_entry.ip, rob_entry.asid);
      (*q_entry)->lq_creation_cycle = current_time.time_since_epoch() / clock_period;
      (*q_entry)->is_tile_instruction = true;
      (*q_entry)->amx_non_temporal = (rob_entry.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));

      // Tile metadata for cache hierarchy (sidecar, instrumentation)
      if (!rob_entry.destination_registers.empty())
        // Use tmm_rename_arch_reg (architectural TMM ID, set before rename).
        // If not set, fall back to 0 (won't match tmm6/7 filter).
        (*q_entry)->tile_dest_tmm = (rob_entry.tmm_rename_arch_reg >= 0)
            ? static_cast<uint8_t>(rob_entry.tmm_rename_arch_reg) : 0;
      bool sidecar_active = !g_disable_tile_sidecar;
      (*q_entry)->amx_tileload = sidecar_active;
      if (sidecar_active) {
        (*q_entry)->tile_group_id = (static_cast<uint64_t>(this->cpu) << 48) | rob_entry.instr_id;
        (*q_entry)->tile_subidx = static_cast<uint8_t>(rob_entry.tile_children_in_lq);
        (*q_entry)->tile_num_children = static_cast<uint8_t>(rob_entry.tile_child_addrs.size());
      }

      // Store-to-load forwarding check
      auto sq_it = std::max_element(std::begin(SQ), std::end(SQ), [addr](const auto& lhs, const auto& rhs) {
        return lhs.virtual_address != addr || (rhs.virtual_address == addr && LSQ_ENTRY::program_order(lhs, rhs));
      });
      if (sq_it != std::end(SQ) && sq_it->virtual_address == addr) {
        if (sq_it->fetch_issued) {
          (*q_entry)->finish(rob_entry);
          q_entry->reset();
        } else {
          sq_it->lq_depend_on_me.emplace_back(*q_entry);
          (*q_entry)->producer_id = sq_it->instr_id;
        }
        rob_entry.tile_children_in_lq++;
        load_bw.consume();
        continue; // next child of same tileload
      }

      // Issue to L1D immediately
      auto success = execute_load(*(*q_entry));
      if (success) {
        (*q_entry)->fetch_issued = true;
        (*q_entry)->lq_issue_cycle = current_time.time_since_epoch() / clock_period;
      }
      (*q_entry)->ready_time = current_time; // ready for retry if !success

      rob_entry.tile_children_in_lq++;
      load_bw.consume();
    }
  }

  // ── Normal load issue: scalar loads + tile children retrying after L1D backpressure ──
  for (auto& lq_entry : LQ) {
    if (load_bw.has_remaining() && lq_entry.has_value() && lq_entry->producer_id == std::numeric_limits<uint64_t>::max() && !lq_entry->fetch_issued
        && lq_entry->ready_time < current_time) {
      // Oracle L1 Hook (applies to both tile and scalar loads)
      bool oracle_hit = false;
      if (!this->warmup && g_oracle_l1[this->cpu].enabled()) {
        auto cur_cycle = current_time.time_since_epoch() / clock_period;
        oracle_hit = g_oracle_l1[this->cpu].check_hit(lq_entry->virtual_address.to<uint64_t>(), lq_entry->amx_tileload, lq_entry->instr_id, cur_cycle, lq_entry->oracle_checked);
        lq_entry->oracle_checked = true;
      }
      if (oracle_hit) {
        // Tile subreq tracking for instrumentation
        if (lq_entry->amx_tileload) {
          std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
          uint64_t la = lq_entry->virtual_address.to<uint64_t>();
          auto it = g_tile_masters.find(lq_entry->tile_group_id);
          if (it != g_tile_masters.end()) {
            bool all_done = true;
            for (auto& sr : it->second.subreqs) {
              if (sr.line_addr == la && !sr.l1_hit) {
                sr.l1_hit = true;
                sr.fill_cycle = current_time.time_since_epoch() / clock_period;
              }
              if (!sr.l1_hit && sr.fill_cycle == 0) all_done = false;
            }
            if (all_done && it->second.issue_cycle > 0) {
              tile_finalize(it->first, current_time.time_since_epoch() / clock_period);
            }
          }
        }
        lq_entry->finish(std::begin(ROB), std::end(ROB));
        lq_entry.reset();
        load_bw.consume();
        continue;
      }

      auto success = execute_load(*lq_entry);
      if (success) {
        load_bw.consume();
        lq_entry->fetch_issued = true;
        lq_entry->lq_issue_cycle = current_time.time_since_epoch() / clock_period;
      }
    }
  }

  return store_bw.amount_consumed() + load_bw.amount_consumed();
}

void O3_CPU::do_finish_store(const LSQ_ENTRY& sq_entry)
{
  if constexpr (champsim::debug_print) {
    fmt::print("[SQ] {} instr_id: {} vaddr: {}\n", __func__, sq_entry.instr_id, sq_entry.virtual_address);
  }

  sq_entry.finish(std::begin(ROB), std::end(ROB));

  // Release dependent loads
  for (std::optional<LSQ_ENTRY>& dependent : sq_entry.lq_depend_on_me) {
    assert(dependent.has_value()); // LQ entry is still allocated
    assert(dependent->producer_id == sq_entry.instr_id);

    if (dependent->amx_tileload) {
      std::lock_guard<std::recursive_mutex> _tile_lk(g_tile_mtx);
      uint64_t la = dependent->virtual_address.to<uint64_t>();
      auto it = g_tile_masters.find(dependent->tile_group_id);
      if (it != g_tile_masters.end()) {
        bool all_done = true;
        for (auto& sr : it->second.subreqs) {
          if (sr.line_addr == la && !sr.l1_hit) {
            sr.l1_hit = true;
            sr.fill_cycle = current_time.time_since_epoch() / clock_period;
          }
          if (!sr.l1_hit && sr.fill_cycle == 0) all_done = false;
        }
        if (all_done && it->second.issue_cycle > 0) {
          tile_finalize(it->first, current_time.time_since_epoch() / clock_period);
        }
      }
    }

    dependent->finish(std::begin(ROB), std::end(ROB));
    dependent.reset();
  }
}

bool O3_CPU::do_complete_store(const LSQ_ENTRY& sq_entry)
{
  CacheBus::request_type data_packet;
  data_packet.v_address = sq_entry.virtual_address;
  data_packet.instr_id = sq_entry.instr_id;
  data_packet.ip = sq_entry.ip;

  // Propagate tile store metadata so write misses use tile MSHR path
  if (sq_entry.amx_tilestore && !g_disable_tile_sidecar) {
    data_packet.is_amx_tileload = true;  // reuse flag for tile coalescing path
    data_packet.tile_group_id = sq_entry.tile_group_id;
    data_packet.tile_subidx = sq_entry.tile_subidx;
    data_packet.tile_num_children = sq_entry.tile_num_children;
    data_packet.tile_is_store = true;
  }

  if constexpr (champsim::debug_print) {
    fmt::print("[SQ] {} instr_id: {} vaddr: {}\n", __func__, data_packet.instr_id, data_packet.v_address);
  }

  // Oracle L1 Hook for Writes (Token was already consumed and cached before retirement)
  if (!this->warmup && g_oracle_l1[this->cpu].enabled()) {
    if (sq_entry.oracle_hit_status) {
      return true; // Bypass L1D WQ entirely
    }
  }

  return L1D_bus.issue_write(data_packet);
}

bool O3_CPU::execute_load(const LSQ_ENTRY& lq_entry)
{
  CacheBus::request_type data_packet;
  data_packet.v_address = lq_entry.virtual_address;
  data_packet.instr_id = lq_entry.instr_id;
  data_packet.ip = lq_entry.ip;
  // Propagate tile metadata (auxiliary, doesn't replace line address)
  // Use is_tile_instruction for DRAM stats tracking (independent of sidecar mode)
  data_packet.is_amx_tileload = lq_entry.amx_tileload || lq_entry.is_tile_instruction;
  data_packet.tile_group_id = lq_entry.tile_group_id;
  data_packet.tile_subidx = lq_entry.tile_subidx;
  data_packet.tile_num_children = lq_entry.tile_num_children;
  data_packet.tile_burst = lq_entry.tile_burst; // Legacy flag
  data_packet.tile_is_store = lq_entry.amx_tilestore;
  data_packet.tile_dest_tmm = lq_entry.tile_dest_tmm;
  // Tile loads use normal cache semantics — hits are fine, sidecar tracks completion
  // No force_miss needed.

  if constexpr (champsim::debug_print) {
    fmt::print("[LQ] {} instr_id: {} vaddr: {}\n", __func__, data_packet.instr_id, data_packet.v_address);
  }

  // L2-only SW prefetch: IP=0xDEAD00000002 → skip L1D fill (data stays in L2)
  // L2-only tile prefetch: IP=0xDEAD00000002 → skip L1D fill (data stays in L2)
  if (data_packet.ip.to<uint64_t>() == 0xDEAD00000002ULL) {
    data_packet.skip_l1_fill = true;
  }

  // TILELOADDT1 non-temporal hint: bypass L1D fill (data stays in L2 only).
  // Real Intel HW: T1 hint means data is not expected to be reused soon,
  // so it should not pollute L1D cache.
  if (lq_entry.amx_non_temporal) {
    data_packet.skip_l1_fill = true;
  }


  // Paper-faithful: tile children flow directly to L1D as normal line requests.
  // No CPU-side buffering or throttling — the cache hierarchy handles them normally.
  // The L1D RQ provides natural backpressure if it is full.

  return L1D_bus.issue_read(data_packet);
}

void O3_CPU::do_complete_execution(ooo_model_instr& instr)
{
  for (auto dreg : instr.destination_registers) {
    // mark physical register's data as valid
    reg_allocator.complete_dest_register(dreg);
  }

  instr.completed = true;

  // Tileload complete (all 16 CL filled) — release inflight slot.
  // This allows next iteration's tileloads to dispatch while TMUL executes,
  // modeling the pipeline overlap seen on real hardware.
  bool is_tileload = instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && (instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
          || instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
  if (is_tileload) {
    // Track tileload latency (dispatch → complete)
    if (!warmup && instr.dispatch_time.time_since_epoch().count() > 0) {
      auto lat = (current_time - instr.dispatch_time) / clock_period;
      tileload_total_latency += lat;
      tileload_count++;
    }
    // Clean up old version reader map entry (garbage collect)
    if (instr.tmm_old_version > 0) {
      tmm_version_readers.erase(instr.tmm_old_version);
    }
  }

  if (instr.branch_mispredicted) {
    fetch_resume_time = current_time + BRANCH_MISPREDICT_PENALTY;
  }
}

long O3_CPU::complete_inflight_instruction()
{
  // update ROB entries with completed executions
  champsim::bandwidth complete_bw{EXEC_WIDTH};
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && complete_bw.has_remaining(); ++rob_it) {
    if (rob_it->executed && !rob_it->completed && (rob_it->ready_time <= current_time) && rob_it->completed_mem_ops == rob_it->num_cache_line_accesses()) {
      do_complete_execution(*rob_it);
      complete_bw.consume();
    }
  }

  return complete_bw.amount_consumed();
}

long O3_CPU::handle_memory_return()
{
  long progress{0};

  for (champsim::bandwidth fetch_bw{FETCH_WIDTH}, l1i_bw{L1I_BANDWIDTH};
       fetch_bw.has_remaining() && l1i_bw.has_remaining() && !L1I_bus.lower_level->returned.empty(); l1i_bw.consume()) {
    auto& l1i_entry = L1I_bus.lower_level->returned.front();

    while (fetch_bw.has_remaining() && !l1i_entry.instr_depend_on_me.empty()) {
      auto fetched = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), ooo_model_instr::matches_id(l1i_entry.instr_depend_on_me.front()));
      if (fetched != std::end(IFETCH_BUFFER) && champsim::block_number{fetched->ip} == champsim::block_number{l1i_entry.v_address} && fetched->fetch_issued) {
        fetched->fetch_completed = true;
        fetch_bw.consume();
        ++progress;

        if constexpr (champsim::debug_print) {
          fmt::print("[IFETCH] {} instr_id: {} fetch completed\n", __func__, fetched->instr_id);
        }
      }

      l1i_entry.instr_depend_on_me.erase(std::begin(l1i_entry.instr_depend_on_me));
    }

    // remove this entry if we have serviced all of its instructions
    if (l1i_entry.instr_depend_on_me.empty()) {
      L1I_bus.lower_level->returned.pop_front();
      ++progress;
    }
  }

  auto l1d_it = std::begin(L1D_bus.lower_level->returned);
  for (champsim::bandwidth l1d_bw{L1D_BANDWIDTH}; l1d_bw.has_remaining() && l1d_it != std::end(L1D_bus.lower_level->returned); l1d_bw.consume(), ++l1d_it) {
    for (auto& lq_entry : LQ) {
      if (lq_entry.has_value() && lq_entry->fetch_issued && champsim::block_number{lq_entry->virtual_address} == champsim::block_number{l1d_it->v_address}) {
        // LQ lifetime measurement (ROI only)
        if (!warmup && lq_entry->lq_creation_cycle > 0) {
          uint64_t now = current_time.time_since_epoch() / clock_period;
          uint64_t lifetime = now - lq_entry->lq_creation_cycle;
          uint64_t issue_lat = (lq_entry->lq_issue_cycle > 0) ? (now - lq_entry->lq_issue_cycle) : 0;
          if (lq_entry->is_tile_instruction) {
            lq_lifetime_tile_total += lifetime;
            lq_lifetime_tile_count++;
            if (lifetime > lq_lifetime_tile_max) lq_lifetime_tile_max = lifetime;
            lq_issue_tile_total += issue_lat;
            lq_issue_tile_count++;
          } else {
            lq_lifetime_scalar_total += lifetime;
            lq_lifetime_scalar_count++;
            if (lifetime > lq_lifetime_scalar_max) lq_lifetime_scalar_max = lifetime;
            lq_issue_scalar_total += issue_lat;
            lq_issue_scalar_count++;
          }
        }
        lq_entry->finish(std::begin(ROB), std::end(ROB));
        lq_entry.reset();
        ++progress;
      }
    }
    ++progress;
  }
  L1D_bus.lower_level->returned.erase(std::begin(L1D_bus.lower_level->returned), l1d_it);

  return progress;
}

long O3_CPU::retire_rob()
{
  auto [retire_begin, retire_end] =
      champsim::get_span_p(std::cbegin(ROB), std::cend(ROB), champsim::bandwidth{RETIRE_WIDTH}, [](const auto& x) { return x.completed; });
  assert(std::distance(retire_begin, retire_end) >= 0); // end succeeds begin
  if constexpr (champsim::debug_print) {
    std::for_each(retire_begin, retire_end, [cycle = current_time.time_since_epoch() / clock_period](const auto& x) {
      fmt::print("[ROB] retire_rob instr_id: {} is retired cycle: {}\n", x.instr_id, cycle);
    });
  }

  // commit register writes to backend RAT
  // and recycle the old physical registers
  for (auto rob_it = retire_begin; rob_it != retire_end; ++rob_it) {
    for (auto dreg : rob_it->destination_registers) {
      reg_allocator.retire_dest_register(dreg);
    }

    // TMM rename handled by main reg_allocator.retire_dest_register() above

    g_oracle_l1[this->cpu].retire_past(rob_it->instr_id);
  }

  auto retire_count = std::distance(retire_begin, retire_end);
  // Count SW prefetch retires separately (excluded from simulation instruction count)
  for (auto it = retire_begin; it != retire_end; ++it) {
    uint64_t rip = it->ip.to<uint64_t>();
    if ((rip >> 32) == 0xDEAD) {
      ++num_retired_prefetch;
    }
  }
  num_retired += retire_count;
  ROB.erase(retire_begin, retire_end);

  return retire_count;
}

void O3_CPU::update_stall_stats(long retire_count)
{
  if (warmup) {
    return;
  }

  // Per-cycle LQ occupancy sampling
  {
    std::size_t tile = 0, scalar = 0;
    for (const auto& lq_entry : LQ) {
      if (lq_entry.has_value()) {
        if (lq_entry->is_tile_instruction) ++tile;
        else ++scalar;
      }
    }
    ++lq_stat_samples;
    lq_stat_tile_entries += tile;
    lq_stat_scalar_entries += scalar;
    lq_stat_total_occupied += (tile + scalar);
    if (tile + scalar > lq_stat_max_occupied) lq_stat_max_occupied = tile + scalar;
    if (tile > lq_stat_max_tile) lq_stat_max_tile = tile;
    if (scalar > lq_stat_max_scalar) lq_stat_max_scalar = scalar;
  }

  ++roi_stall_stats.total_cycles;

  // ── Overlappable raw counters: measured EVERY cycle independently (including retire cycles) ──
  {
    // LFB full
    if (l1d != nullptr && l1d->get_lfb_size() > 0 && l1d->get_lfb_occupancy() >= l1d->get_lfb_size())
      ++roi_stall_stats.raw_lfb_full;

    // L1D miss pending (ROB head waiting for memory) + sub-classify by level
    if (!std::empty(ROB)) {
      const auto& head = ROB.front();
      if (head.executed && head.num_cache_line_accesses() > 0 && head.completed_mem_ops < head.num_cache_line_accesses()) {
        ++roi_stall_stats.raw_l1d_miss_pending;
        // Sub-classify: same logic as exclusive breakdown (check L2/LLC directly)
        bool found_dram = false, found_llc = false;
        auto check_addr = [&](champsim::address addr) {
          if (l2c != nullptr && l2c->has_mshr_entry_for(addr)) {
            if (llc != nullptr && llc->has_mshr_entry_for(addr)) {
              found_dram = true;
            } else {
              found_llc = true;
            }
          }
        };
        for (const auto& addr : head.source_memory) check_addr(addr);
        for (const auto& addr : head.destination_memory) check_addr(addr);
        if (found_dram) ++roi_stall_stats.raw_dram_pending;
        else if (found_llc) ++roi_stall_stats.raw_llc_miss_pending;
        else ++roi_stall_stats.raw_l2_miss_pending;  // in L1D MSHR but not L2 → L2 servicing
      }

      // AMX compute (ROB head is TDPBF16PS, executed but not completed)
      bool head_amx = head.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
          && head.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
      if (head_amx && head.executed && !head.completed)
        ++roi_stall_stats.raw_amx_compute_all;
    }

    // LQ full (dispatch would stall)
    if (!std::empty(DISPATCH_BUFFER) && DISPATCH_BUFFER.front().ready_time <= current_time) {
      auto free_lq_raw = std::count_if(std::begin(LQ), std::end(LQ), [](const auto& e) { return !e.has_value(); });
      bool front_tile = DISPATCH_BUFFER.front().instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
          && (DISPATCH_BUFFER.front().amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
              || DISPATCH_BUFFER.front().amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
      auto needed = front_tile ? (std::size_t)0 : std::size(DISPATCH_BUFFER.front().source_memory);
      if (static_cast<std::size_t>(free_lq_raw) < needed) ++roi_stall_stats.raw_lq_full;
      if ((std::size(DISPATCH_BUFFER.front().destination_memory) + std::size(SQ)) > SQ_SIZE) ++roi_stall_stats.raw_sq_full;
      if (std::size(ROB) >= ROB_SIZE) ++roi_stall_stats.raw_rob_full;
    }

    if (retire_count == 0) ++roi_stall_stats.raw_no_retire;
  }

  if (retire_count > 0) {
    ++roi_stall_stats.retire_cycles;
    return;
  }

  bool dispatch_ready = !std::empty(DISPATCH_BUFFER) && DISPATCH_BUFFER.front().ready_time <= current_time;
  bool rob_full = dispatch_ready && std::size(ROB) >= ROB_SIZE;

  std::size_t free_lq = 0;
  if (dispatch_ready) {
    free_lq = std::count_if(std::begin(LQ), std::end(LQ), [](const auto& lq_entry) { return !lq_entry.has_value(); });
  }
  // Tileloads use drip-feed LQ allocation (0 LQ slots needed at dispatch)
  bool front_is_tileload = dispatch_ready
      && DISPATCH_BUFFER.front().instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
      && (DISPATCH_BUFFER.front().amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
          || DISPATCH_BUFFER.front().amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
  std::size_t needed_lq_stall = front_is_tileload ? 0 : std::size(DISPATCH_BUFFER.front().source_memory);
  std::size_t needed_sq_stall = std::size(DISPATCH_BUFFER.front().destination_memory);
  bool lq_full = dispatch_ready && free_lq < needed_lq_stall;
  bool sq_full = dispatch_ready && (needed_sq_stall + std::size(SQ)) > SQ_SIZE;

  bool l1d_miss = false;
  if (!std::empty(ROB)) {
    const auto& head = ROB.front();
    l1d_miss = head.executed && head.num_cache_line_accesses() > 0 && head.completed_mem_ops < head.num_cache_line_accesses();
  }

  // Check LFB full independently - this models HW where LFB pressure blocks new loads
  bool lfb_full = false;
  if (l1d != nullptr) {
    auto lfb_size = l1d->get_lfb_size();
    auto lfb_occupancy = l1d->get_lfb_occupancy();
    if (lfb_size > 0 && lfb_occupancy >= lfb_size) {
      // LFB is full - check if there are pending loads that would be blocked
      auto pending_loads = std::count_if(std::begin(LQ), std::end(LQ), [](const auto& lq_entry) {
        return lq_entry.has_value() && !lq_entry->fetch_issued;
      });
      if (pending_loads > 0 || l1d_miss) {
        lfb_full = true;
      }
    }
  }

  bool frontend_stall = false;
  if (current_time < fetch_resume_time) {
    frontend_stall = true;
  } else if (std::empty(DISPATCH_BUFFER) && std::empty(DECODE_BUFFER) && std::empty(DIB_HIT_BUFFER)) {
    auto any_fetch_ready = std::any_of(std::cbegin(IFETCH_BUFFER), std::cend(IFETCH_BUFFER), [time = current_time](const auto& x) {
      return x.fetch_completed && x.ready_time <= time;
    });
    frontend_stall = !any_fetch_ready;
  }

  // Helper: classify miss level by L1D MSHR serve_level (event-based).
  // serve_level is set by finish_packet when the fill response arrives from lower level.
  // 0=L2_HIT, 1=LLC_HIT, 2=DRAM. Default 0 before response arrives.
  auto classify_miss_level = [&](bool& out_dram, bool& out_l3) {
    if (l1d == nullptr || std::empty(ROB)) return;
    const auto& head = ROB.front();
    auto check_addr = [&](champsim::address addr) {
      for (const auto& entry : l1d->MSHR) {
        bool match = false;
        uint8_t sl = 0;
        if (!entry.is_tile_entry) {
          if (entry.address == addr) { match = true; sl = entry.serve_level; }
        } else {
          for (const auto& c : entry.tile_children) {
            if (c.address == addr) { match = true; sl = c.serve_level; break; }
          }
        }
        if (match) {
          if (sl >= 2) out_dram = true;
          else if (sl == 1) out_l3 = true;
          return;
        }
      }
    };
    for (const auto& addr : head.source_memory) check_addr(addr);
    for (const auto& addr : head.destination_memory) check_addr(addr);
  };

  // Helper: classify the MSHR originator for the miss that the ROB head is waiting on.
  // Checks L1D MSHR entries for the ROB head's addresses and returns the originator type.
  // 0=scalar, 1=tileload, 2=prefetch (based on MSHR IP, not ROB head IP)
  auto classify_miss_originator = [&]() -> int {
    if (l1d == nullptr || std::empty(ROB)) return 0;
    const auto& head = ROB.front();
    // Collect addresses the ROB head is waiting on
    auto check_mshr_ip = [&](champsim::address addr) -> int {
      for (const auto& mshr : l1d->MSHR) {
        if (mshr.address == addr || (mshr.is_tile_entry && [&]() {
              for (const auto& c : mshr.tile_children)
                if (c.address == addr) return true;
              return false;
            }())) {
          uint64_t mshr_ip = mshr.ip.to<uint64_t>();
          if ((mshr_ip >> 32) == 0xDEAD) return 2;  // prefetch-initiated
          return 1;  // demand (tileload or scalar — MSHR IP is original requester)
        }
      }
      return 0;
    };
    for (const auto& addr : head.source_memory) {
      int r = check_mshr_ip(addr);
      if (r > 0) return r;
    }
    // Also check tile child addrs if this is a tileload
    for (const auto& addr : head.tile_child_addrs) {
      int r = check_mshr_ip(addr);
      if (r > 0) return r;
    }
    return 0;  // no MSHR match → scalar/unknown
  };

  // Helper: classify ROB head instruction type (what IS the stalled instruction)
  auto classify_rob_head_type = [&]() -> int {
    // 0=scalar, 1=tileload, 2=prefetch
    if (!std::empty(ROB)) {
      const auto& head = ROB.front();
      uint64_t head_ip = head.ip.to<uint64_t>();
      if ((head_ip >> 32) == 0xDEAD) return 2;
      bool is_tile = head.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
          && (head.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
              || head.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
      if (is_tile) return 1;
    }
    return 0;
  };

  if (lfb_full) {
    ++roi_stall_stats.lfb_full;
    { int src = classify_rob_head_type();
      if (src == 1) ++roi_stall_stats.mshr_full_tileload;
      else if (src == 2) ++roi_stall_stats.mshr_full_prefetch;
      else ++roi_stall_stats.mshr_full_scalar; }
    if (l1d_miss) {
      bool found_dram = false, found_l3 = false;
      classify_miss_level(found_dram, found_l3);
      if (found_dram) ++roi_stall_stats.lfb_full_dram_pending;
      else if (found_l3) ++roi_stall_stats.lfb_full_l3_pending;
      else ++roi_stall_stats.lfb_full_l2_pending;
    } else {
      ++roi_stall_stats.lfb_full_no_miss;
    }
  } else if (l1d_miss) {
    ++roi_stall_stats.l1d_miss;
    bool found_dram = false, found_l3 = false;
    classify_miss_level(found_dram, found_l3);
    { int src = classify_rob_head_type();
      int orig = classify_miss_originator();
      if (found_dram) {
        ++roi_stall_stats.dram_bound;
        if (src == 1) ++roi_stall_stats.dram_bound_tileload;
        else if (src == 2) ++roi_stall_stats.dram_bound_prefetch;
        else ++roi_stall_stats.dram_bound_scalar;
        if (orig == 2) ++roi_stall_stats.dram_orig_prefetch;
        else ++roi_stall_stats.dram_orig_demand;
      } else if (found_l3) {
        ++roi_stall_stats.l3_bound;
      } else {
        ++roi_stall_stats.l2_bound;
        if (src == 1) ++roi_stall_stats.l2_bound_tileload;
        else if (src == 2) ++roi_stall_stats.l2_bound_prefetch;
        else ++roi_stall_stats.l2_bound_scalar;
        if (orig == 2) ++roi_stall_stats.l2_orig_prefetch;
        else ++roi_stall_stats.l2_orig_demand;
      }
    }
  } else if (rob_full) {
    ++roi_stall_stats.rob_full;
  } else if (lq_full) {
    ++roi_stall_stats.lq_full;
  } else if (sq_full) {
    ++roi_stall_stats.sq_full;
  } else if (frontend_stall) {
    ++roi_stall_stats.front_end;
  } else {
    ++roi_stall_stats.other;
    // Sub-classify "other"
    if (!std::empty(ROB)) {
      const auto& head = ROB.front();
      bool head_is_amx_compute = head.instr_class == static_cast<uint8_t>(trace_instr_class::AMX)
          && head.amx_op == static_cast<uint8_t>(trace_amx_op::TDPBF16PS);
      if (head_is_amx_compute && head.executed && !head.completed) {
        ++roi_stall_stats.other_amx_compute;
      } else if (head.executed && !head.completed) {
        ++roi_stall_stats.other_exec_wait;
      } else if (!head.executed) {
        ++roi_stall_stats.other_not_executed;
      } else {
        ++roi_stall_stats.other_misc;
      }
    } else if (!dispatch_ready) {
      ++roi_stall_stats.other_dispatch_empty;
    } else {
      ++roi_stall_stats.other_misc;
    }
  }
}

void O3_CPU::print_stall_stats() const
{
  if (roi_stall_stats.total_cycles == 0) {
    return;
  }

  double total = static_cast<double>(roi_stall_stats.total_cycles);
  auto pct = [total](uint64_t value) { return (100.0 * static_cast<double>(value)) / total; };

  fmt::print("CPU {} Stall Breakdown (ROI cycles: {})\n", cpu, roi_stall_stats.total_cycles);
  fmt::print("  retire:      {:>10} ({:>5.1f}%)\n", roi_stall_stats.retire_cycles, pct(roi_stall_stats.retire_cycles));
  fmt::print("  mshr_full:   {:>10} ({:>5.1f}%)\n", roi_stall_stats.lfb_full, pct(roi_stall_stats.lfb_full));
  fmt::print("    by_tileload:{:>9} ({:>5.1f}%) by_prefetch:{:>9} ({:>5.1f}%) by_scalar:{:>9} ({:>5.1f}%)\n",
    roi_stall_stats.mshr_full_tileload, pct(roi_stall_stats.mshr_full_tileload),
    roi_stall_stats.mshr_full_prefetch, pct(roi_stall_stats.mshr_full_prefetch),
    roi_stall_stats.mshr_full_scalar, pct(roi_stall_stats.mshr_full_scalar));
  fmt::print("    at_l2:     {:>10} ({:>5.1f}%)\n", roi_stall_stats.lfb_full_l2_pending, pct(roi_stall_stats.lfb_full_l2_pending));
  fmt::print("    at_l3:     {:>10} ({:>5.1f}%)\n", roi_stall_stats.lfb_full_l3_pending, pct(roi_stall_stats.lfb_full_l3_pending));
  fmt::print("    at_dram:   {:>10} ({:>5.1f}%)\n", roi_stall_stats.lfb_full_dram_pending, pct(roi_stall_stats.lfb_full_dram_pending));
  fmt::print("    no_miss:   {:>10} ({:>5.1f}%)\n", roi_stall_stats.lfb_full_no_miss, pct(roi_stall_stats.lfb_full_no_miss));
  fmt::print("  l2_bound:    {:>10} ({:>5.1f}%)\n", roi_stall_stats.l2_bound, pct(roi_stall_stats.l2_bound));
  fmt::print("    rob_head:  tileload:{:>9} ({:>5.1f}%) prefetch:{:>9} ({:>5.1f}%) scalar:{:>9} ({:>5.1f}%)\n",
    roi_stall_stats.l2_bound_tileload, pct(roi_stall_stats.l2_bound_tileload),
    roi_stall_stats.l2_bound_prefetch, pct(roi_stall_stats.l2_bound_prefetch),
    roi_stall_stats.l2_bound_scalar, pct(roi_stall_stats.l2_bound_scalar));
  fmt::print("    initiated: pf_orig:{:>10} ({:>5.1f}%) demand_orig:{:>7} ({:>5.1f}%)\n",
    roi_stall_stats.l2_orig_prefetch, pct(roi_stall_stats.l2_orig_prefetch),
    roi_stall_stats.l2_orig_demand, pct(roi_stall_stats.l2_orig_demand));
  fmt::print("  l3_bound:    {:>10} ({:>5.1f}%)\n", roi_stall_stats.l3_bound, pct(roi_stall_stats.l3_bound));
  fmt::print("  dram_bound:  {:>10} ({:>5.1f}%)\n", roi_stall_stats.dram_bound, pct(roi_stall_stats.dram_bound));
  fmt::print("    rob_head:  tileload:{:>9} ({:>5.1f}%) prefetch:{:>9} ({:>5.1f}%) scalar:{:>9} ({:>5.1f}%)\n",
    roi_stall_stats.dram_bound_tileload, pct(roi_stall_stats.dram_bound_tileload),
    roi_stall_stats.dram_bound_prefetch, pct(roi_stall_stats.dram_bound_prefetch),
    roi_stall_stats.dram_bound_scalar, pct(roi_stall_stats.dram_bound_scalar));
  fmt::print("    initiated: pf_orig:{:>10} ({:>5.1f}%) demand_orig:{:>7} ({:>5.1f}%)\n",
    roi_stall_stats.dram_orig_prefetch, pct(roi_stall_stats.dram_orig_prefetch),
    roi_stall_stats.dram_orig_demand, pct(roi_stall_stats.dram_orig_demand));
  fmt::print("  rob_full:    {:>10} ({:>5.1f}%)\n", roi_stall_stats.rob_full, pct(roi_stall_stats.rob_full));
  fmt::print("  lq_full:     {:>10} ({:>5.1f}%)\n", roi_stall_stats.lq_full, pct(roi_stall_stats.lq_full));
  fmt::print("  sq_full:     {:>10} ({:>5.1f}%)\n", roi_stall_stats.sq_full, pct(roi_stall_stats.sq_full));
  fmt::print("  front_end:   {:>10} ({:>5.1f}%)\n", roi_stall_stats.front_end, pct(roi_stall_stats.front_end));
  fmt::print("  other:       {:>10} ({:>5.1f}%)\n", roi_stall_stats.other, pct(roi_stall_stats.other));
  if (roi_stall_stats.other > 0) {
    fmt::print("    amx_TMUL:     {:>10} ({:>5.1f}%)\n", roi_stall_stats.other_amx_compute, pct(roi_stall_stats.other_amx_compute));
    fmt::print("    exec_wait:    {:>10} ({:>5.1f}%)\n", roi_stall_stats.other_exec_wait, pct(roi_stall_stats.other_exec_wait));
    fmt::print("    not_executed: {:>10} ({:>5.1f}%)\n", roi_stall_stats.other_not_executed, pct(roi_stall_stats.other_not_executed));
    fmt::print("    disp_empty:   {:>10} ({:>5.1f}%)\n", roi_stall_stats.other_dispatch_empty, pct(roi_stall_stats.other_dispatch_empty));
    fmt::print("    misc:         {:>10} ({:>5.1f}%)\n", roi_stall_stats.other_misc, pct(roi_stall_stats.other_misc));
  }

  // Raw overlappable ratios (each independently measured, sum > 100% expected)
  fmt::print("\nCPU {} Raw Overlappable Ratios (/ ROI cycles: {})\n", cpu, roi_stall_stats.total_cycles);
  fmt::print("  mshr_full:           {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_lfb_full, pct(roi_stall_stats.raw_lfb_full));
  fmt::print("  l1d_miss_pending:    {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_l1d_miss_pending, pct(roi_stall_stats.raw_l1d_miss_pending));
  fmt::print("    at_l2:             {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_l2_miss_pending, pct(roi_stall_stats.raw_l2_miss_pending));
  fmt::print("    at_llc:            {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_llc_miss_pending, pct(roi_stall_stats.raw_llc_miss_pending));
  fmt::print("    at_dram:           {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_dram_pending, pct(roi_stall_stats.raw_dram_pending));
  fmt::print("  amx_TMUL:            {:>10} ({:>5.1f}%)  <- pure compute, no mem overlap\n", roi_stall_stats.other_amx_compute, pct(roi_stall_stats.other_amx_compute));
  fmt::print("  amx_compute_entire:  {:>10} ({:>5.1f}%)  <- includes overlap w/ mem stalls\n", roi_stall_stats.raw_amx_compute_all, pct(roi_stall_stats.raw_amx_compute_all));
  fmt::print("  lq_full:             {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_lq_full, pct(roi_stall_stats.raw_lq_full));
  fmt::print("  rob_full:            {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_rob_full, pct(roi_stall_stats.raw_rob_full));
  fmt::print("  sq_full:             {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_sq_full, pct(roi_stall_stats.raw_sq_full));
  fmt::print("  no_retire:           {:>10} ({:>5.1f}%)\n", roi_stall_stats.raw_no_retire, pct(roi_stall_stats.raw_no_retire));

  // Tileload latency stats
  if (tileload_count > 0) {
    fmt::print("\nCPU {} Tileload Latency (ROI): count={} avg={:.1f}cy total={}cy\n",
        cpu, tileload_count, (double)tileload_total_latency / tileload_count, tileload_total_latency);
  }

  // TMM rename stats
  fmt::print("\nCPU {} TMM Rename Stats (extra_phys={})\n", cpu, TMM_EXTRA_PHYS);
  fmt::print("  tmm_rename_stalls:  {:>10}\n", tmm_rename_stall_count);
  fmt::print("  tmm_rename_allocs:  {:>10}\n", tmm_rename_alloc_count);
  fmt::print("  tmm_rename_frees:   {:>10}\n", tmm_rename_free_count);
  fmt::print("  tmm_war_stalls:     {:>10}\n", tmm_war_stalls);
  fmt::print("  tmm_free_list_now:  {:>10}\n", tmm_phys_free_list.size());

  // LQ occupancy + lifetime diagnostics
  if (lq_stat_samples > 0) {
    double avg_total = static_cast<double>(lq_stat_total_occupied) / lq_stat_samples;
    double avg_tile = static_cast<double>(lq_stat_tile_entries) / lq_stat_samples;
    double avg_scalar = static_cast<double>(lq_stat_scalar_entries) / lq_stat_samples;
    fmt::print("\nCPU {} LQ Occupancy (per-cycle avg, LQ_SIZE={})\n", cpu, LQ.size());
    fmt::print("  avg_occupied:  {:.1f} / {} ({:.1f}%)\n", avg_total, LQ.size(), 100.0 * avg_total / LQ.size());
    fmt::print("  avg_tile:      {:.1f}  (max {})\n", avg_tile, lq_stat_max_tile);
    fmt::print("  avg_scalar:    {:.1f}  (max {})\n", avg_scalar, lq_stat_max_scalar);
    fmt::print("  max_occupied:  {}\n", lq_stat_max_occupied);
  }
  fmt::print("\nCPU {} LQ Entry Lifetime (creation→free, cycles)\n", cpu);
  if (lq_lifetime_tile_count > 0) {
    fmt::print("  tile_child:  avg={:.1f}  max={}  count={}\n",
               static_cast<double>(lq_lifetime_tile_total) / lq_lifetime_tile_count,
               lq_lifetime_tile_max, lq_lifetime_tile_count);
  }
  if (lq_lifetime_scalar_count > 0) {
    fmt::print("  scalar:      avg={:.1f}  max={}  count={}\n",
               static_cast<double>(lq_lifetime_scalar_total) / lq_lifetime_scalar_count,
               lq_lifetime_scalar_max, lq_lifetime_scalar_count);
  }
  fmt::print("\nCPU {} LQ Issue Latency (issued→free, cycles)\n", cpu);
  if (lq_issue_tile_count > 0) {
    fmt::print("  tile_child:  avg={:.1f}  count={}\n",
               static_cast<double>(lq_issue_tile_total) / lq_issue_tile_count, lq_issue_tile_count);
  }
  if (lq_issue_scalar_count > 0) {
    fmt::print("  scalar:      avg={:.1f}  count={}\n",
               static_cast<double>(lq_issue_scalar_total) / lq_issue_scalar_count, lq_issue_scalar_count);
  }
}

void O3_CPU::impl_initialize_branch_predictor() const { branch_module_pimpl->impl_initialize_branch_predictor(); }

void O3_CPU::impl_last_branch_result(champsim::address ip, champsim::address target, bool taken, uint8_t branch_type) const
{
  branch_module_pimpl->impl_last_branch_result(ip, target, taken, branch_type);
}

bool O3_CPU::impl_predict_branch(champsim::address ip, champsim::address predicted_target, bool always_taken, uint8_t branch_type) const
{
  return branch_module_pimpl->impl_predict_branch(ip, predicted_target, always_taken, branch_type);
}

void O3_CPU::impl_initialize_btb() const { btb_module_pimpl->impl_initialize_btb(); }

void O3_CPU::impl_update_btb(champsim::address ip, champsim::address predicted_target, bool taken, uint8_t branch_type) const
{
  btb_module_pimpl->impl_update_btb(ip, predicted_target, taken, branch_type);
}

std::pair<champsim::address, bool> O3_CPU::impl_btb_prediction(champsim::address ip, uint8_t branch_type) const
{
  return btb_module_pimpl->impl_btb_prediction(ip, branch_type);
}

// LCOV_EXCL_START Exclude the following function from LCOV
void O3_CPU::print_deadlock()
{
  fmt::print("DEADLOCK! CPU {} cycle {}\n", cpu, current_time.time_since_epoch() / clock_period);

  auto instr_pack = [period = clock_period, this](const auto& entry) {
    return std::tuple{entry.instr_id,
                      entry.fetch_issued,
                      entry.fetch_completed,
                      entry.scheduled,
                      entry.executed,
                      entry.completed,
                      reg_allocator.count_reg_dependencies(entry),
                      entry.num_cache_line_accesses() - entry.completed_mem_ops,
                      entry.ready_time.time_since_epoch() / period};
  };
  std::string_view instr_fmt{
      "instr_id: {} fetch_issued: {} fetch_completed: {} scheduled: {} executed: {} completed: {} num_reg_dependent: {} num_mem_ops: {} event: {}"};
  champsim::range_print_deadlock(IFETCH_BUFFER, "cpu" + std::to_string(cpu) + "_IFETCH", instr_fmt, instr_pack);
  champsim::range_print_deadlock(DECODE_BUFFER, "cpu" + std::to_string(cpu) + "_DECODE", instr_fmt, instr_pack);
  champsim::range_print_deadlock(DISPATCH_BUFFER, "cpu" + std::to_string(cpu) + "_DISPATCH", instr_fmt, instr_pack);
  champsim::range_print_deadlock(ROB, "cpu" + std::to_string(cpu) + "_ROB", instr_fmt, instr_pack);

  // print occupied physical registers
  reg_allocator.print_deadlock();

  // print LQ entry
  auto lq_pack = [period = clock_period](const auto& entry) {
    std::string depend_id{"-"};
    if (entry->producer_id != std::numeric_limits<uint64_t>::max()) {
      depend_id = std::to_string(entry->producer_id);
    }
    return std::tuple{entry->instr_id, entry->virtual_address, entry->fetch_issued, entry->ready_time.time_since_epoch() / period, depend_id};
  };
  std::string_view lq_fmt{"instr_id: {} address: {} fetch_issued: {} event_cycle: {} waits on {}"};

  auto sq_pack = [period = clock_period](const auto& entry) {
    std::vector<uint64_t> depend_ids;
    std::transform(std::begin(entry.lq_depend_on_me), std::end(entry.lq_depend_on_me), std::back_inserter(depend_ids),
                   [](const std::optional<LSQ_ENTRY>& lq_entry) { return lq_entry->producer_id; });
    return std::tuple{entry.instr_id, entry.virtual_address, entry.fetch_issued, entry.ready_time.time_since_epoch() / period, depend_ids};
  };
  std::string_view sq_fmt{"instr_id: {} address: {} fetch_issued: {} event_cycle: {} LQ waiting: {}"};
  champsim::range_print_deadlock(LQ, "cpu" + std::to_string(cpu) + "_LQ", lq_fmt, lq_pack);
  champsim::range_print_deadlock(SQ, "cpu" + std::to_string(cpu) + "_SQ", sq_fmt, sq_pack);
}
// LCOV_EXCL_STOP

LSQ_ENTRY::LSQ_ENTRY(champsim::address addr, champsim::program_ordered<LSQ_ENTRY>::id_type id, champsim::address local_ip, std::array<uint8_t, 2> local_asid)
    : champsim::program_ordered<LSQ_ENTRY>{id}, virtual_address(addr), ip(local_ip), asid(local_asid)
{
}

void LSQ_ENTRY::finish(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end) const
{
  auto rob_entry = std::partition_point(begin, end, ooo_model_instr::precedes(this->instr_id));
  assert(rob_entry != end);
  finish(*rob_entry);
}

void LSQ_ENTRY::finish(ooo_model_instr& rob_entry) const
{
  assert(rob_entry.instr_id == this->instr_id);

  ++rob_entry.completed_mem_ops;
  assert(rob_entry.completed_mem_ops <= rob_entry.num_cache_line_accesses());

  if constexpr (champsim::debug_print) {
    fmt::print("[LSQ] {} instr_id: {} full_address: {} remain_mem_ops: {}\n", __func__, instr_id, virtual_address,
               rob_entry.num_cache_line_accesses() - rob_entry.completed_mem_ops);
  }
}

bool CacheBus::issue_read(request_type data_packet)
{
  data_packet.address = data_packet.v_address;
  data_packet.is_translated = false;
  data_packet.cpu = cpu;
  data_packet.type = access_type::LOAD;

  return lower_level->add_rq(data_packet);
}

bool CacheBus::issue_write(request_type data_packet)
{
  data_packet.address = data_packet.v_address;
  data_packet.is_translated = false;
  data_packet.cpu = cpu;
  data_packet.type = access_type::WRITE;
  data_packet.response_requested = false;

  return lower_level->add_wq(data_packet);
}

void O3_CPU::service_l1d_tile_admission()
{
  // No-op: tile children now flow directly to L1D in execute_load().
  // TAT buffering removed — paper-faithful passive sidecar approach.
}
