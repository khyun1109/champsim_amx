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

#include "tracereader.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "inf_stream.h"
#include "repeatable.h"

namespace champsim
{
uint64_t tracereader::instr_unique_id = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

ooo_model_instr apply_branch_target(ooo_model_instr branch, const ooo_model_instr& target)
{
  branch.branch_target = (branch.is_branch && branch.branch_taken) ? target.ip : champsim::address{};
  return branch;
}

template <template <class, class> typename R, typename T>
champsim::tracereader get_tracereader_for_type(std::string fname, uint8_t cpu)
{
  if (bool is_gzip_compressed = (fname.substr(std::size(fname) - 2) == "gz"); is_gzip_compressed) {
    return champsim::tracereader{R<T, champsim::inf_istream<champsim::decomp_tags::gzip_tag_t<>>>(cpu, fname)};
  }

  if (bool is_lzma_compressed = (fname.substr(std::size(fname) - 2) == "xz"); is_lzma_compressed) {
    return champsim::tracereader{R<T, champsim::inf_istream<champsim::decomp_tags::lzma_tag_t<>>>(cpu, fname)};
  }

  if (bool is_bzip2_compressed = (fname.substr(std::size(fname) - 3) == "bz2"); is_bzip2_compressed) {
    return champsim::tracereader{R<T, champsim::inf_istream<champsim::decomp_tags::bzip2_tag_t>>(cpu, fname)};
  }

  return champsim::tracereader{R<T, std::ifstream>(cpu, fname)};
}
void tracereader::remap_tileload_addresses(ooo_model_instr& instr)
{
  bool is_tileload = (instr.instr_class == static_cast<uint8_t>(trace_instr_class::AMX))
      && (instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD)
          || instr.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADDT1));
  if (!is_tileload)
    return;

  // DRAM address layout: [row | rank | column | bank | bankgroup | channel | offset]
  // To maximize row buffer hits, all tileload addresses should map to the
  // same bank and same row. We fix the channel/bankgroup/bank/rank bits and
  // only increment the column within a row. When columns are exhausted,
  // advance to the next row in the same bank.
  //
  // Bit layout for this config (2ch, 8bg, 4bank, 1024col, 2rank, 512K rows):
  //   offset: bits [8:0]   = 9 bits (channel_width(8) * prefetch_size(64) = 512B)
  //   channel: bit [9]     = 1 bit
  //   bankgroup: [12:10]   = 3 bits
  //   bank: [14:13]        = 2 bits
  //   column: [18:15]      = 4 bits (1024/64 = 16 columns)
  //   rank: bit [19]       = 1 bit
  //   row: [38:20]         = 19 bits
  //
  // Strategy: fix channel=0, bankgroup=0, bank=0, rank=0.
  // Only vary column and row bits. Each new cache line increments
  // by the full DRAM block stride (512B = offset size) to stay in the
  // same channel/bank but advance the column.
  constexpr uint64_t BLOCK_MASK = 63; // lower 6 bits = cache line offset
  constexpr uint64_t DRAM_BLOCK_STRIDE = 512; // offset field size: 2^9 = 512
  // To stay in same channel(0), bankgroup(0), bank(0), rank(0) but vary column:
  // column bits start at bit 15. Each column step = 2^15 = 32768.
  // But we also need to stay in same row. With 16 columns per row,
  // we can fit 16 cache lines per row in the same bank.
  // After 16 columns, we move to next row.
  constexpr uint64_t COLUMN_STEP = 1ULL << 15; // bit 15 is first column bit
  constexpr uint64_t COLUMNS_PER_ROW = 16;     // 1024/64 = 16

  auto remap = [&](uint64_t orig) -> uint64_t {
    if (orig == 0)
      return 0;
    uint64_t line = orig & ~BLOCK_MASK;
    uint64_t offset = orig & BLOCK_MASK;
    auto it = tile_addr_remap_.find(line);
    if (it == tile_addr_remap_.end()) {
      // Compute new address: base + column_index * COLUMN_STEP
      // column_index wraps around; after COLUMNS_PER_ROW, advance row
      uint64_t idx = tile_addr_remap_.size();
      uint64_t col = idx % COLUMNS_PER_ROW;
      uint64_t row = idx / COLUMNS_PER_ROW;
      // row bits start at bit 20
      uint64_t mapped = next_tile_addr_ + (row << 20) + (col * COLUMN_STEP);
      tile_addr_remap_[line] = mapped;
      return mapped | offset;
    }
    return it->second | offset;
  };

  for (auto& addr : instr.source_memory) {
    uint64_t raw = addr.to<uint64_t>();
    if (raw != 0)
      addr = champsim::address{remap(raw)};
  }

  for (auto& memop : instr.source_mem_ops) {
    uint64_t raw = memop.address.to<uint64_t>();
    if (raw != 0)
      memop.address = champsim::address{remap(raw)};
  }
}

} // namespace champsim

template <typename T, typename S>
using repeatable_reader_t = champsim::repeatable<champsim::bulk_tracereader<T, S>, uint8_t, std::string>;

template <typename T, typename S>
using repeatable_detecting_reader_t = champsim::repeatable<champsim::detecting_tracereader<T, S>, uint8_t, std::string>;

champsim::tracereader get_tracereader(const std::string& fname, uint8_t cpu, bool is_cloudsuite, bool repeat)
{
  if (is_cloudsuite && repeat) {
    return champsim::get_tracereader_for_type<repeatable_reader_t, cloudsuite_instr>(fname, cpu);
  }

  if (is_cloudsuite && !repeat) {
    return champsim::get_tracereader_for_type<champsim::bulk_tracereader, cloudsuite_instr>(fname, cpu);
  }

  if (!is_cloudsuite && repeat) {
    return champsim::get_tracereader_for_type<repeatable_detecting_reader_t, input_instr>(fname, cpu);
  }

  return champsim::get_tracereader_for_type<champsim::detecting_tracereader, input_instr>(fname, cpu);
}
