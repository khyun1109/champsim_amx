#include "lru.h"

#include <algorithm>
#include <cassert>
#include "tile_record.h"

lru::lru(CACHE* cache) : lru(cache, cache->NUM_SET, cache->NUM_WAY) {}

lru::lru(CACHE* cache, long sets, long ways) : replacement(cache), NUM_WAY(ways), last_used_cycles(static_cast<std::size_t>(sets * ways), 0) {}

long lru::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                      champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return std::distance(begin, victim);
}

void lru::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                 access_type type)
{
  // Tile-aware insertion: tile data is inserted at LRU position (lowest priority)
  // to prevent cache pollution from large tile loads evicting useful scalar data.
  // This mimics real HW behavior where streaming/tile data gets biased insertion.
  if (g_tile_lru_insert && intern_->current_fill_is_tile) {
    // Insert at LRU position: find the current minimum in this set and use min-1
    auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
    auto end = std::next(begin, NUM_WAY);
    auto min_it = std::min_element(begin, end);
    uint64_t min_val = *min_it;
    last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = (min_val > 0) ? min_val - 1 : 0;
  } else {
    // Normal MRU insertion
    last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;
  }
}

void lru::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                   champsim::address victim_addr, access_type type, uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (hit && access_type{type} != access_type::WRITE) // Skip this for writeback hits
    last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;
}
