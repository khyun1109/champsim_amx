#include <catch.hpp>

#include <sstream>
#include <string>

#include "tracereader.h"

namespace
{
std::string build_trace_v2()
{
  trace_header_v2 header{};
  header.magic = TRACE_MAGIC_V2;
  header.version = TRACE_VERSION_V2;
  header.header_size = sizeof(trace_header_v2);
  header.flags = 0;
  header.reserved = 0;

  trace_record_v2 record{};
  record.ip = 0x4c00133a;
  record.is_branch = 0;
  record.branch_taken = 0;
  record.instr_class = static_cast<uint8_t>(trace_instr_class::AMX);
  record.amx_op = static_cast<uint8_t>(trace_amx_op::TILELOADD);
  record.num_dst_regs = 1;
  record.num_src_regs = 2;
  record.num_mem_ops = 2;
  record.reserved = 0;

  uint8_t dst_regs[] = {0x3b};
  uint8_t src_regs[] = {0x06, 0x07};

  trace_memop_v2 mem0{};
  mem0.addr = 0x1000;
  mem0.size = 64;
  mem0.type = static_cast<uint8_t>(trace_memop_type::READ);
  mem0.flags = 0;
  mem0.reserved = 0;

  trace_memop_v2 mem1{};
  mem1.addr = 0x2000;
  mem1.size = 64;
  mem1.type = static_cast<uint8_t>(trace_memop_type::WRITE);
  mem1.flags = 0;
  mem1.reserved = 0;

  std::string out;
  out.append(reinterpret_cast<const char*>(&header), sizeof(header));
  out.append(reinterpret_cast<const char*>(&record), sizeof(record));
  out.append(reinterpret_cast<const char*>(dst_regs), sizeof(dst_regs));
  out.append(reinterpret_cast<const char*>(src_regs), sizeof(src_regs));
  out.append(reinterpret_cast<const char*>(&mem0), sizeof(mem0));
  out.append(reinterpret_cast<const char*>(&mem1), sizeof(mem1));
  return out;
}
} // namespace

TEST_CASE("A detecting_tracereader can read v2 trace format")
{
  auto trace = build_trace_v2();
  champsim::detecting_tracereader<input_instr, std::istringstream> uut{0, std::istringstream{trace}};

  auto inst = uut();
  REQUIRE(inst.ip == champsim::address{0x4c00133a});
  REQUIRE(inst.instr_class == static_cast<uint8_t>(trace_instr_class::AMX));
  REQUIRE(inst.amx_op == static_cast<uint8_t>(trace_amx_op::TILELOADD));
  REQUIRE(inst.destination_registers.size() == 1);
  REQUIRE(inst.source_registers.size() == 2);
  REQUIRE(inst.destination_registers.front() == 0x3b);
  REQUIRE(inst.source_registers.front() == 0x06);
  REQUIRE(inst.source_registers.back() == 0x07);
  REQUIRE(inst.source_memory.size() == 1);
  REQUIRE(inst.destination_memory.size() == 1);
  REQUIRE(inst.source_memory.front() == champsim::address{0x1000});
  REQUIRE(inst.destination_memory.front() == champsim::address{0x2000});
}
