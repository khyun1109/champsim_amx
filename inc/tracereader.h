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

#ifndef TRACEREADER_H
#define TRACEREADER_H

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "instruction.h"
#include "trace_instruction.h"
#include "util/detect.h"

namespace champsim
{
class tracereader
{
  static uint64_t instr_unique_id; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
  struct reader_concept {
    virtual ~reader_concept() = default;
    virtual ooo_model_instr operator()() = 0;
    [[nodiscard]] virtual bool eof() const = 0;
  };

  template <typename T>
  struct reader_model final : public reader_concept {
    T intern_;
    reader_model(T&& val) : intern_(std::move(val)) {}

    template <typename U>
    using has_eof = decltype(std::declval<U>().eof());

    ooo_model_instr operator()() override { return intern_(); }
    [[nodiscard]] bool eof() const override
    {
      if constexpr (champsim::is_detected_v<has_eof, T>) {
        return intern_.eof();
      }
      return false; // If an eof() member function is not provided, assume the trace never ends.
    }
  };

  std::unique_ptr<reader_concept> pimpl_;

  // Tileload address remapping state
  bool remap_tileload_ = false;
  std::unordered_map<uint64_t, uint64_t> tile_addr_remap_;
  uint64_t next_tile_addr_ = 0x80000000ULL;

  void remap_tileload_addresses(ooo_model_instr& instr);

public:
  template <typename T, std::enable_if_t<!std::is_same_v<tracereader, T>, bool> = true>
  tracereader(T&& val) : pimpl_(std::make_unique<reader_model<T>>(std::forward<T>(val)))
  {
  }

  void set_tileload_remap(bool enable) { remap_tileload_ = enable; }

  auto operator()()
  {
    auto retval = (*pimpl_)();
    retval.instr_id = instr_unique_id++;
    if (remap_tileload_) remap_tileload_addresses(retval);
    return retval;
  }

  [[nodiscard]] auto eof() const { return pimpl_->eof(); }
};

template <typename T, typename F>
class bulk_tracereader
{
  static_assert(std::is_trivial_v<T>);
  static_assert(std::is_standard_layout_v<T>);

  uint8_t cpu;
  bool eof_ = false;
  F trace_file;

  constexpr static std::size_t buffer_size = 128;
  constexpr static std::size_t refresh_thresh = 1;
  std::deque<ooo_model_instr> instr_buffer;

public:
  ooo_model_instr operator()();

  bulk_tracereader(uint8_t cpu_idx, std::string tf) : cpu(cpu_idx), trace_file(tf) {}
  bulk_tracereader(uint8_t cpu_idx, F&& file) : cpu(cpu_idx), trace_file(std::move(file)) {}

  [[nodiscard]] bool eof() const { return trace_file.eof() && std::size(instr_buffer) <= refresh_thresh; }
};

template <typename StreamType>
class prefetched_istream
{
  std::vector<char> prefix_;
  std::size_t offset_ = 0;
  StreamType underlying_;
  std::streamsize gcount_ = 0;
  bool eof_ = false;

public:
  prefetched_istream(std::vector<char> prefix, StreamType&& stream) : prefix_(std::move(prefix)), underlying_(std::move(stream)) {}

  prefetched_istream& read(char* s, std::streamsize count)
  {
    gcount_ = 0;
    if (offset_ < prefix_.size()) {
      auto remaining = prefix_.size() - offset_;
      auto to_copy = std::min<std::size_t>(remaining, static_cast<std::size_t>(count));
      std::memcpy(s, prefix_.data() + offset_, to_copy);
      offset_ += to_copy;
      gcount_ += static_cast<std::streamsize>(to_copy);
    }

    if (gcount_ < count) {
      underlying_.read(s + gcount_, count - gcount_);
      gcount_ += underlying_.gcount();
    }

    eof_ = (offset_ >= prefix_.size()) && underlying_.eof();
    return *this;
  }

  [[nodiscard]] bool eof() const { return eof_; }
  [[nodiscard]] std::streamsize gcount() const { return gcount_; }
};

template <typename F>
class v2_tracereader
{
  uint8_t cpu;
  bool eof_ = false;
  F trace_file;
  trace_header_v2 header_;
  std::deque<ooo_model_instr> instr_buffer;

  constexpr static std::size_t refresh_thresh = 1;

  bool read_exact(char* dst, std::size_t count)
  {
    trace_file.read(dst, static_cast<std::streamsize>(count));
    return static_cast<std::size_t>(trace_file.gcount()) == count;
  }

  bool read_one()
  {
    trace_record_v2 header{};
    if (!read_exact(reinterpret_cast<char*>(&header), sizeof(header))) {
      eof_ = true;
      return false;
    }

    std::vector<uint8_t> dst_regs(header.num_dst_regs);
    if (!dst_regs.empty() && !read_exact(reinterpret_cast<char*>(dst_regs.data()), dst_regs.size())) {
      eof_ = true;
      return false;
    }

    std::vector<uint8_t> src_regs(header.num_src_regs);
    if (!src_regs.empty() && !read_exact(reinterpret_cast<char*>(src_regs.data()), src_regs.size())) {
      eof_ = true;
      return false;
    }

    std::vector<trace_memop_v2> mem_ops(header.num_mem_ops);
    if (!mem_ops.empty() && !read_exact(reinterpret_cast<char*>(mem_ops.data()), mem_ops.size() * sizeof(trace_memop_v2))) {
      eof_ = true;
      return false;
    }

    std::vector<memory_operation> dst_mem_ops;
    std::vector<memory_operation> src_mem_ops;
    dst_mem_ops.reserve(mem_ops.size());
    src_mem_ops.reserve(mem_ops.size());

    for (const auto& op : mem_ops) {
      auto addr = champsim::address{op.addr};
      if (op.type == static_cast<uint8_t>(trace_memop_type::WRITE)) {
        dst_mem_ops.emplace_back(addr, op.size);
      } else {
        src_mem_ops.emplace_back(addr, op.size);
      }
    }

    instr_buffer.emplace_back(cpu, header, std::move(dst_regs), std::move(src_regs), std::move(dst_mem_ops), std::move(src_mem_ops), true);
    return true;
  }

  void fill_buffer()
  {
    while (std::size(instr_buffer) <= refresh_thresh && !eof_) {
      if (!read_one()) {
        break;
      }
    }
  }

public:
  v2_tracereader(uint8_t cpu_idx, std::string tf, trace_header_v2 header) : cpu(cpu_idx), trace_file(tf), header_(header) { fill_buffer(); }
  v2_tracereader(uint8_t cpu_idx, F&& file, trace_header_v2 header) : cpu(cpu_idx), trace_file(std::move(file)), header_(header) { fill_buffer(); }

  ooo_model_instr operator()()
  {
    fill_buffer();
    if (instr_buffer.empty()) {
      return ooo_model_instr{};
    }
    auto retval = instr_buffer.front();
    instr_buffer.pop_front();
    return retval;
  }

  [[nodiscard]] bool eof() const { return eof_ && std::size(instr_buffer) <= refresh_thresh; }
};

template <typename T, typename F>
class detecting_tracereader
{
  using v1_reader = bulk_tracereader<T, prefetched_istream<F>>;
  using v2_reader = v2_tracereader<F>;

  std::variant<std::monostate, v1_reader, v2_reader> reader_;

  static trace_header_v2 parse_header(const std::array<char, sizeof(trace_header_v2)>& buf)
  {
    trace_header_v2 header{};
    std::memcpy(&header, buf.data(), sizeof(header));
    return header;
  }

public:
  detecting_tracereader(uint8_t cpu_idx, std::string tf) : detecting_tracereader(cpu_idx, F{tf}) {}
  detecting_tracereader(uint8_t cpu_idx, F&& file)
  {
    std::array<char, sizeof(trace_header_v2)> header_buf{};
    file.read(header_buf.data(), header_buf.size());
    auto bytes_read = static_cast<std::size_t>(file.gcount());

    bool is_magic = bytes_read >= TRACE_MAGIC_V2.size()
      && (std::memcmp(header_buf.data(), TRACE_MAGIC_V2.data(), TRACE_MAGIC_V2.size()) == 0
          || std::memcmp(header_buf.data(), TRACE_MAGIC_V2_COMPAT.data(), TRACE_MAGIC_V2_COMPAT.size()) == 0);

    if (is_magic) {
      if (bytes_read < sizeof(trace_header_v2)) {
        auto remaining = sizeof(trace_header_v2) - bytes_read;
        file.read(header_buf.data() + bytes_read, static_cast<std::streamsize>(remaining));
        bytes_read += static_cast<std::size_t>(file.gcount());
      }

      if (bytes_read >= sizeof(trace_header_v2)) {
        auto header = parse_header(header_buf);
        if (header.version != TRACE_VERSION_V2 || header.header_size < sizeof(trace_header_v2)) {
          is_magic = false;
        }
      }

      if (is_magic && bytes_read >= sizeof(trace_header_v2)) {
        auto header = parse_header(header_buf);
        if (header.header_size > sizeof(trace_header_v2)) {
          std::vector<char> skip(header.header_size - sizeof(trace_header_v2));
          file.read(skip.data(), static_cast<std::streamsize>(skip.size()));
        }
        reader_.template emplace<v2_reader>(cpu_idx, std::move(file), header);
        return;
      }
    }

    std::vector<char> prefix(header_buf.data(), header_buf.data() + bytes_read);
    reader_.template emplace<v1_reader>(cpu_idx, prefetched_istream<F>{std::move(prefix), std::move(file)});
  }

  ooo_model_instr operator()()
  {
    if (std::holds_alternative<std::monostate>(reader_)) {
      return ooo_model_instr{};
    }
    if (std::holds_alternative<v1_reader>(reader_)) {
      return std::get<v1_reader>(reader_)();
    }
    return std::get<v2_reader>(reader_)();
  }

  [[nodiscard]] bool eof() const
  {
    if (std::holds_alternative<std::monostate>(reader_)) {
      return true;
    }
    if (std::holds_alternative<v1_reader>(reader_)) {
      return std::get<v1_reader>(reader_).eof();
    }
    return std::get<v2_reader>(reader_).eof();
  }
};

ooo_model_instr apply_branch_target(ooo_model_instr branch, const ooo_model_instr& target);

template <typename It>
void set_branch_targets(It begin, It end)
{
  std::reverse_iterator rbegin{end};
  std::reverse_iterator rend{begin};
  std::adjacent_difference(rbegin, rend, rbegin, apply_branch_target);
}

template <typename T, typename F>
ooo_model_instr bulk_tracereader<T, F>::operator()()
{
  if (std::size(instr_buffer) <= refresh_thresh) {
    std::array<T, buffer_size - refresh_thresh> trace_read_buf;
    std::array<char, std::size(trace_read_buf) * sizeof(T)> raw_buf;
    std::size_t bytes_read;

    // Read from trace file
    trace_file.read(std::data(raw_buf), std::size(raw_buf));
    bytes_read = static_cast<std::size_t>(trace_file.gcount());
    eof_ = trace_file.eof();

    // Transform bytes into trace format instructions
    std::memcpy(std::data(trace_read_buf), std::data(raw_buf), bytes_read);

    // Inflate trace format into core model instructions
    auto begin = std::begin(trace_read_buf);
    auto end = std::next(begin, bytes_read / sizeof(T));
    std::transform(begin, end, std::back_inserter(instr_buffer), [cpu = this->cpu](T t) { return ooo_model_instr{cpu, t}; });

    // Set branch targets
    set_branch_targets(std::begin(instr_buffer), std::end(instr_buffer));
  }

  auto retval = instr_buffer.front();
  instr_buffer.pop_front();

  return retval;
}

std::string get_fptr_cmd(std::string_view fname);
} // namespace champsim

champsim::tracereader get_tracereader(const std::string& fname, uint8_t cpu, bool is_cloudsuite, bool repeat);

#endif
