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

#ifndef REPEATABLE_H
#define REPEATABLE_H

#include <memory>
#include <string>
#include <fmt/ranges.h>

#include "instruction.h"

namespace champsim
{
template <typename T, typename... Args>
struct repeatable {
  static_assert(std::is_move_constructible_v<T>);
  static_assert(std::is_move_assignable_v<T>);
  std::tuple<Args...> args_;
  T intern_{std::apply([](auto... x) { return T{x...}; }, args_)};
  bool reached_eof_ = false;
  explicit repeatable(Args... args) : args_(args...) {}

  auto operator()()
  {
    // When trace ends, mark EOF and stop — do NOT replay.
    // The core's pipeline will drain naturally and become idle.
    if (intern_.eof()) {
      if (!reached_eof_)
        fmt::print("*** Reached end of trace: {}\n", args_);
      reached_eof_ = true;
    }

    // Only read from underlying trace if not at EOF
    if (!reached_eof_)
      return intern_();

    // Return a default-constructed instruction (NOP) when past EOF
    return decltype(intern_())();
  }

  [[nodiscard]] bool eof() const { return reached_eof_; }
};
} // namespace champsim

#endif
