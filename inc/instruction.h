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

#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "chrono.h"
#include "trace_instruction.h"

// Memory operation with size information for AMX support
struct memory_operation {
  champsim::address address{};
  uint32_t size = 0;  // Size in bytes (0 means default cache line size)

  memory_operation() = default;
  memory_operation(champsim::address addr, uint32_t sz = 0) : address(addr), size(sz) {}
  explicit memory_operation(champsim::address addr) : address(addr), size(0) {}

  // Calculate number of cache lines this operation spans
  [[nodiscard]] uint32_t num_cache_lines(uint32_t cache_line_size = 64) const {
    if (size == 0) return 1;  // Default: single cache line
    uint64_t start_line = address.to<uint64_t>() / cache_line_size;
    uint64_t end_line = (address.to<uint64_t>() + size - 1) / cache_line_size;
    return static_cast<uint32_t>(end_line - start_line + 1);
  }

  // Get the i-th cache line address for this operation
  [[nodiscard]] champsim::address get_cache_line_addr(uint32_t index, uint32_t cache_line_size = 64) const {
    uint64_t base = (address.to<uint64_t>() / cache_line_size) * cache_line_size;
    return champsim::address{base + index * cache_line_size};
  }
};

// branch types
enum branch_type {
  BRANCH_DIRECT_JUMP = 0,
  BRANCH_INDIRECT,
  BRANCH_CONDITIONAL,
  BRANCH_DIRECT_CALL,
  BRANCH_INDIRECT_CALL,
  BRANCH_RETURN,
  BRANCH_OTHER,
  NOT_BRANCH
};

using PHYSICAL_REGISTER_ID = int16_t; // signed to use -1 to indicate no physical register

using namespace std::literals::string_view_literals;
inline constexpr std::array branch_type_names{"BRANCH_DIRECT_JUMP"sv, "BRANCH_INDIRECT"sv,      "BRANCH_CONDITIONAL"sv,
                                              "BRANCH_DIRECT_CALL"sv, "BRANCH_INDIRECT_CALL"sv, "BRANCH_RETURN"sv};

namespace champsim
{
template <typename T>
struct program_ordered {
  using id_type = uint64_t;
  id_type instr_id = 0;

  /**
   * Return a functor that matches this element's ID.
   * \overload
   */
  static auto matches_id(id_type id)
  {
    return [id](const T& instr) {
      return instr.instr_id == id;
    };
  }

  /**
   * Return a functor that matches this element's ID.
   */
  static auto matches_id(const T& instr) { return precedes(instr.instr_id); }

  /**
   * Order two elements of this type in the program.
   */
  static bool program_order(const T& lhs, const T& rhs) { return lhs.instr_id < rhs.instr_id; }

  /**
   * Return a functor that tests whether an instruction precededes the given instruction ID.
   * \overload
   */
  static auto precedes(id_type id)
  {
    return [id](const T& instr) {
      return instr.instr_id < id;
    };
  }

  /**
   * Return a functor that tests whether an instruction precededes the given instruction.
   */
  static auto precedes(const T& instr) { return precedes(instr.instr_id); }
};
} // namespace champsim

struct ooo_model_instr : champsim::program_ordered<ooo_model_instr> {
  champsim::address ip{};
  champsim::chrono::clock::time_point ready_time{};
  champsim::chrono::clock::time_point dispatch_time{};  // tileload latency: dispatch → complete

  bool is_branch = false;
  bool branch_taken = false;
  bool branch_prediction = false;
  bool branch_mispredicted = false; // A branch can be mispredicted even if the direction prediction is correct when the predicted target is not correct

  std::array<uint8_t, 2> asid = {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::max()};

  uint8_t instr_class = static_cast<uint8_t>(trace_instr_class::GENERIC);
  uint8_t amx_op = static_cast<uint8_t>(trace_amx_op::NONE);

  branch_type branch{NOT_BRANCH};
  champsim::address branch_target{};

  bool dib_checked = false;
  bool fetch_issued = false;
  bool fetch_completed = false;
  bool decoded = false;
  bool scheduled = false;
  bool executed = false;
  bool completed = false;

  unsigned completed_mem_ops = 0;
  int num_reg_dependent = 0;

  // TMM physical register rename tracking (valid only for tileload with TMM dest)
  bool tmm_rename_consumed = false;    // consumed a TMM physical register slot
  int8_t tmm_rename_old_phys = -1;     // old physical TMM ID (freed at retire)
  int8_t tmm_rename_new_phys = -1;     // new physical TMM ID assigned
  int16_t tmm_rename_arch_reg = -1;    // architectural register ID of TMM destination

  // TMM WAR dependency: tileload stores old version, TDP stores source versions
  uint64_t tmm_old_version = 0;        // tileload: version of dest TMM before overwrite
  std::vector<uint64_t> tmm_src_versions{};  // TDP: versions of source TMM regs at dispatch

  // Lazy tile child LQ allocation: children created at issue time, not dispatch
  std::vector<champsim::address> tile_child_addrs{};  // precomputed cache line addresses
  uint32_t tile_children_in_lq = 0;                   // how many children allocated to LQ so far

  std::vector<PHYSICAL_REGISTER_ID> destination_registers = {}; // output registers
  std::vector<PHYSICAL_REGISTER_ID> source_registers = {};      // input registers

  std::vector<champsim::address> destination_memory = {};
  std::vector<champsim::address> source_memory = {};

  // Memory operations with size information (for AMX support)
  std::vector<memory_operation> destination_mem_ops = {};
  std::vector<memory_operation> source_mem_ops = {};

  // these are indices of instructions in the ROB that depend on me
  std::vector<std::reference_wrapper<ooo_model_instr>> registers_instrs_depend_on_me;

private:
  void classify_branch_type(bool trace_branch_taken)
  {
    bool writes_sp = std::count(std::begin(destination_registers), std::end(destination_registers), champsim::REG_STACK_POINTER);
    bool writes_ip = std::count(std::begin(destination_registers), std::end(destination_registers), champsim::REG_INSTRUCTION_POINTER);
    bool reads_sp = std::count(std::begin(source_registers), std::end(source_registers), champsim::REG_STACK_POINTER);
    bool reads_flags = std::count(std::begin(source_registers), std::end(source_registers), champsim::REG_FLAGS);
    bool reads_ip = std::count(std::begin(source_registers), std::end(source_registers), champsim::REG_INSTRUCTION_POINTER);
    bool reads_other = std::count_if(std::begin(source_registers), std::end(source_registers), [](PHYSICAL_REGISTER_ID r) {
      return r != champsim::REG_STACK_POINTER && r != champsim::REG_FLAGS && r != champsim::REG_INSTRUCTION_POINTER;
    });

    // determine what kind of branch this is, if any
    if (!reads_sp && !reads_flags && writes_ip && !reads_other) {
      // direct jump
      is_branch = true;
      branch_taken = true;
      branch = BRANCH_DIRECT_JUMP;
    } else if (!reads_sp && !reads_ip && !reads_flags && writes_ip && reads_other) {
      // indirect branch
      is_branch = true;
      branch_taken = true;
      branch = BRANCH_INDIRECT;
    } else if (!reads_sp && reads_ip && !writes_sp && writes_ip && (reads_flags || reads_other)) {
      // conditional branch
      is_branch = true;
      branch_taken = trace_branch_taken;
      branch = BRANCH_CONDITIONAL;
    } else if (reads_sp && reads_ip && writes_sp && writes_ip && !reads_flags && !reads_other) {
      // direct call
      is_branch = true;
      branch_taken = true;
      branch = BRANCH_DIRECT_CALL;
    } else if (reads_sp && reads_ip && writes_sp && writes_ip && !reads_flags && reads_other) {
      // indirect call
      is_branch = true;
      branch_taken = true;
      branch = BRANCH_INDIRECT_CALL;
    } else if (reads_sp && !reads_ip && writes_sp && writes_ip) {
      // return
      is_branch = true;
      branch_taken = true;
      branch = BRANCH_RETURN;
    } else if (writes_ip) {
      // some other branch type that doesn't fit the above categories
      is_branch = true;
      branch_taken = trace_branch_taken;
      branch = BRANCH_OTHER;
    } else {
      branch_taken = false;
    }
  }

  template <typename T>
  ooo_model_instr(T instr, std::array<uint8_t, 2> local_asid) : ip(instr.ip), is_branch(instr.is_branch), branch_taken(instr.branch_taken), asid(local_asid)
  {
    std::remove_copy(std::begin(instr.destination_registers), std::end(instr.destination_registers), std::back_inserter(this->destination_registers), 0);
    std::remove_copy(std::begin(instr.source_registers), std::end(instr.source_registers), std::back_inserter(this->source_registers), 0);

    auto dmem_end = std::remove(std::begin(instr.destination_memory), std::end(instr.destination_memory), uint64_t{0});
    std::transform(std::begin(instr.destination_memory), dmem_end, std::back_inserter(this->destination_memory), [](auto x) { return champsim::address{x}; });

    auto smem_end = std::remove(std::begin(instr.source_memory), std::end(instr.source_memory), uint64_t{0});
    std::transform(std::begin(instr.source_memory), smem_end, std::back_inserter(this->source_memory), [](auto x) { return champsim::address{x}; });

    classify_branch_type(instr.branch_taken);
  }

public:
  ooo_model_instr() = default;

  ooo_model_instr(uint8_t cpu, input_instr instr) : ooo_model_instr(instr, {cpu, cpu}) {}
  ooo_model_instr(uint8_t /*cpu*/, cloudsuite_instr instr) : ooo_model_instr(instr, {instr.asid[0], instr.asid[1]}) {}

  ooo_model_instr(uint8_t cpu, const trace_record_v2& header, std::vector<uint8_t>&& dst_regs, std::vector<uint8_t>&& src_regs,
                  std::vector<champsim::address>&& dst_mem, std::vector<champsim::address>&& src_mem)
      : ip(header.ip), is_branch(header.is_branch), branch_taken(header.branch_taken), asid({cpu, cpu}), instr_class(header.instr_class), amx_op(header.amx_op)
  {
    destination_registers.reserve(dst_regs.size());
    std::transform(dst_regs.begin(), dst_regs.end(), std::back_inserter(destination_registers),
                   [](auto r) { return static_cast<PHYSICAL_REGISTER_ID>(r); });
    source_registers.reserve(src_regs.size());
    std::transform(src_regs.begin(), src_regs.end(), std::back_inserter(source_registers),
                   [](auto r) { return static_cast<PHYSICAL_REGISTER_ID>(r); });

    destination_memory = std::move(dst_mem);
    source_memory = std::move(src_mem);

    classify_branch_type(header.branch_taken);
  }

  // Constructor with memory operation sizes (for AMX support)
  ooo_model_instr(uint8_t cpu, const trace_record_v2& header, std::vector<uint8_t>&& dst_regs, std::vector<uint8_t>&& src_regs,
                  std::vector<memory_operation>&& dst_mem_ops, std::vector<memory_operation>&& src_mem_ops, bool /*use_mem_ops*/)
      : ip(header.ip), is_branch(header.is_branch), branch_taken(header.branch_taken), asid({cpu, cpu}), instr_class(header.instr_class), amx_op(header.amx_op)
  {
    destination_registers.reserve(dst_regs.size());
    std::transform(dst_regs.begin(), dst_regs.end(), std::back_inserter(destination_registers),
                   [](auto r) { return static_cast<PHYSICAL_REGISTER_ID>(r); });
    source_registers.reserve(src_regs.size());
    std::transform(src_regs.begin(), src_regs.end(), std::back_inserter(source_registers),
                   [](auto r) { return static_cast<PHYSICAL_REGISTER_ID>(r); });

    destination_mem_ops = std::move(dst_mem_ops);
    source_mem_ops = std::move(src_mem_ops);

    // Also populate legacy destination_memory/source_memory for backward compatibility
    for (const auto& op : destination_mem_ops) {
      destination_memory.push_back(op.address);
    }
    for (const auto& op : source_mem_ops) {
      source_memory.push_back(op.address);
    }

    classify_branch_type(header.branch_taken);
  }

  [[nodiscard]] std::size_t num_mem_ops() const { return std::size(destination_memory) + std::size(source_memory); }

  // Calculate total cache line accesses (for AMX multi-line operations)
  [[nodiscard]] std::size_t num_cache_line_accesses() const {
    std::size_t total = 0;
    if (!destination_mem_ops.empty() || !source_mem_ops.empty()) {
      for (const auto& op : destination_mem_ops) {
        total += op.num_cache_lines();
      }
      for (const auto& op : source_mem_ops) {
        total += op.num_cache_lines();
      }
    } else {
      total = num_mem_ops();
    }
    return total;
  }
};

#endif
