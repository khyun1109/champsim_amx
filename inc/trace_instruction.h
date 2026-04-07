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

#ifndef TRACE_INSTRUCTION_H
#define TRACE_INSTRUCTION_H

#include <array>
#include <cstdint>
#include <limits>

// special registers that help us identify branches
namespace champsim
{
constexpr char REG_STACK_POINTER = 6;
constexpr char REG_FLAGS = 25;
constexpr char REG_INSTRUCTION_POINTER = 26;

// Synthetic AMX tile registers (tmm0-tmm7) for dependency tracking
// These IDs are in unused register space (100+) to avoid conflicts
constexpr int16_t REG_TMM0 = 100;
constexpr int16_t REG_TMM_COUNT = 8; // tmm0-tmm7
} // namespace champsim

// instruction format
constexpr std::size_t NUM_INSTR_DESTINATIONS_SPARC = 4;
constexpr std::size_t NUM_INSTR_DESTINATIONS = 2;
constexpr std::size_t NUM_INSTR_SOURCES = 4;

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays): These classes are deliberately trivial
struct input_instr {
  // instruction pointer or PC (Program Counter)
  unsigned long long ip;

  // branch info
  unsigned char is_branch;
  unsigned char branch_taken;

  unsigned char destination_registers[NUM_INSTR_DESTINATIONS]; // output registers
  unsigned char source_registers[NUM_INSTR_SOURCES];           // input registers

  unsigned long long destination_memory[NUM_INSTR_DESTINATIONS]; // output memory
  unsigned long long source_memory[NUM_INSTR_SOURCES];           // input memory
};

struct cloudsuite_instr {
  // instruction pointer or PC (Program Counter)
  unsigned long long ip;

  // branch info
  unsigned char is_branch;
  unsigned char branch_taken;

  unsigned char destination_registers[NUM_INSTR_DESTINATIONS_SPARC]; // output registers
  unsigned char source_registers[NUM_INSTR_SOURCES];                 // input registers

  unsigned long long destination_memory[NUM_INSTR_DESTINATIONS_SPARC]; // output memory
  unsigned long long source_memory[NUM_INSTR_SOURCES];                 // input memory

  unsigned char asid[2];
};
// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

// Variable-length trace format (v2)
inline constexpr std::array<char, 16> TRACE_MAGIC_V2{{'C', 'H', 'A', 'M', 'P', 'S', 'I', 'M', 'T', 'R', 'A', 'C', 'E', 'V', '2', '_'}};
inline constexpr std::array<char, 16> TRACE_MAGIC_V2_COMPAT{{'C', 'H', 'A', 'M', 'P', 'S', 'I', 'M', 'T', 'R', 'A', 'C', 'E', '2', '_', '\0'}};
inline constexpr uint32_t TRACE_VERSION_V2 = 2;

enum class trace_instr_class : uint8_t { GENERIC = 0, AMX = 1 };

enum class trace_amx_op : uint8_t {
  NONE = 0,
  TILELOADD = 1,
  TILELOADDT1 = 2,
  TILESTORED = 3,
  TDPBF16PS = 4,
  TILEZERO = 5,
  OTHER = 255
};

enum class trace_memop_type : uint8_t { READ = 0, WRITE = 1 };

struct trace_header_v2 {
  std::array<char, 16> magic;
  uint32_t version;
  uint32_t header_size;
  uint32_t flags;
  uint32_t reserved;
};
static_assert(sizeof(trace_header_v2) == 32);

struct trace_record_v2 {
  uint64_t ip;
  uint8_t is_branch;
  uint8_t branch_taken;
  uint8_t instr_class;
  uint8_t amx_op;
  uint8_t num_dst_regs;
  uint8_t num_src_regs;
  uint8_t num_mem_ops;
  uint8_t reserved;
};
static_assert(sizeof(trace_record_v2) == 16);

struct trace_memop_v2 {
  uint64_t addr;
  uint32_t size;
  uint8_t type;
  uint8_t flags;
  uint16_t reserved;
};
static_assert(sizeof(trace_memop_v2) == 16);

#endif
