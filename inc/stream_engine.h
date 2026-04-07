#ifndef STREAM_ENGINE_H
#define STREAM_ENGINE_H

#include <vector>
#include <deque>
#include <cstdint>
#include <algorithm> // for find_if, etc.
#include "champsim.h"
#include "address.h" // for champsim::address

// --- Configuration ---
// Can be moved to cmake/configure later
#define SB_SETS 256
#define SB_WAYS 16
#define SE_L2_ISSUE_BW 1 // Bandwidth: Max L2 requests per cycle from Stream Engine

// --- Stream Buffer Entry ---
struct sb_entry_t {
    bool valid = false;
    bool ready = false;   // Data is present (hit)
    bool inflight = false;// Data is requested but not yet arrived
    uint64_t tag = 0;     // Cache line address (block aligned)
    uint64_t tile_group_id = 0;
    uint64_t last_touch_cycle = 0; // for LRU
};

// --- Stream Buffer Class ---
class STREAM_BUFFER {
public:
    // Set-Associative Structure
    sb_entry_t entries[SB_SETS][SB_WAYS];

    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t merges = 0; // Inflight hits
    uint64_t evictions = 0;
    uint64_t invalidations = 0; // Store safety

    STREAM_BUFFER() {
        for(int i=0; i<SB_SETS; i++)
            for(int j=0; j<SB_WAYS; j++)
                 entries[i][j].valid = false;
    }

    // Helper: Address to Index/Tag
    uint64_t get_set_index(uint64_t addr_val) const {
        return (addr_val >> 6) % SB_SETS; // Assuming 64B lines
    }
    uint64_t get_tag(uint64_t addr_val) const {
        return (addr_val >> 6);
    }

    // Check hit
    // Returns: 0=Miss, 1=Hit(Ready), 2=Hit(Inflight/Merge)
    int check_hit(champsim::address addr) {
        uint64_t addr_val = addr.to<uint64_t>();
        uint64_t set_idx = get_set_index(addr_val);
        uint64_t tag = get_tag(addr_val);

        for (int i=0; i<SB_WAYS; ++i) {
            if (entries[set_idx][i].valid && entries[set_idx][i].tag == tag) {
                entries[set_idx][i].last_touch_cycle = 0; // TODO: use actual cycle if available
                if (entries[set_idx][i].ready) return 1; // Ready Hit
                return 2; // Inflight Merge
            }
        }
        return 0; // Miss
    }

    // Allocate Entry (on Miss) -> Returns true if success, false if full/all-locked
    bool allocate(champsim::address addr, uint64_t tile_id, uint64_t cycle) {
        uint64_t addr_val = addr.to<uint64_t>();
        uint64_t set_idx = get_set_index(addr_val);
        uint64_t tag = get_tag(addr_val);

        int lru_way = -1;
        uint64_t min_cycle = UINT64_MAX;

        // Find empty or LRU
        for (int i=0; i<SB_WAYS; ++i) {
            if (!entries[set_idx][i].valid) {
                lru_way = i;
                break;
            }
            // Prefer evicting non-inflight (safety/simplicity policy)
            // If strictly needed, allows evicting inflight (will just drop return packet)
            if (entries[set_idx][i].last_touch_cycle < min_cycle) {
                min_cycle = entries[set_idx][i].last_touch_cycle;
                lru_way = i;
            }
        }

        if (lru_way != -1) {
            if (entries[set_idx][lru_way].valid) evictions++;
            entries[set_idx][lru_way].valid = true;
            entries[set_idx][lru_way].ready = false;
            entries[set_idx][lru_way].inflight = true;
            entries[set_idx][lru_way].tag = tag;
            entries[set_idx][lru_way].tile_group_id = tile_id;
            entries[set_idx][lru_way].last_touch_cycle = cycle;
            return true;
        }
        return false;
    }

    // Fill Data (Mark Ready)
    void fill(champsim::address addr) {
        uint64_t addr_val = addr.to<uint64_t>();
        uint64_t set_idx = get_set_index(addr_val);
        uint64_t tag = get_tag(addr_val);

        for (int i=0; i<SB_WAYS; ++i) {
             if (entries[set_idx][i].valid && entries[set_idx][i].tag == tag) {
                 entries[set_idx][i].ready = true;
                 entries[set_idx][i].inflight = false;
                 return;
             }
        }
        // If not found, it was evicted while inflight. Drop.
    }

    // Invalidate (Safety for Stores)
    void invalidate(champsim::address addr) {
        uint64_t addr_val = addr.to<uint64_t>();
        uint64_t set_idx = get_set_index(addr_val);
        uint64_t tag = get_tag(addr_val);

        for (int i=0; i<SB_WAYS; ++i) {
             if (entries[set_idx][i].valid && entries[set_idx][i].tag == tag) {
                 entries[set_idx][i].valid = false;
                 invalidations++;
             }
        }
    }

    void print_stats() {
        fmt::print("STREAM_BUFFER Stats: Hits: {} Misses: {} Merges: {} Evictions: {} Invalidations: {}\n",
                   hits, misses, merges, evictions, invalidations);
    }
};

// --- Stream Descriptor ---
struct stream_desc_t {
    uint64_t tile_group_id;
    std::vector<champsim::address> line_addrs; // 16 lines
    bool issued[16] = {false}; // Track which lines sent to L2
    int issued_count = 0;
};

// --- Stream Engine Class ---
class STREAM_ENGINE {
public:
    std::deque<stream_desc_t> desc_q;
    STREAM_BUFFER sb;

    uint64_t lines_issued = 0;
    
    // Called by CPU to submit a tile prefetch
    void submit(uint64_t tile_id, const std::vector<champsim::address>& lines, uint64_t cycle) {
        // Check duplication? For now just push.
        stream_desc_t desc;
        desc.tile_group_id = tile_id;
        desc.line_addrs = lines;
        desc_q.push_back(desc);
        
        // Optimistically allocate SB entries?
        // Current policy: Allocate when issuing to L2.
    }

    // Called every cycle to issue requests to L2
    // Returns vector of packets to send to L2
    template <typename T>
    std::vector<T> issue_requests(uint64_t current_cycle) {
        std::vector<T> packets;
        int quota = SE_L2_ISSUE_BW;

        while (quota > 0 && !desc_q.empty()) {
            stream_desc_t& head = desc_q.front();
            
            bool progress = false;
            for (int i=0; i < head.line_addrs.size(); ++i) {
                if (!head.issued[i]) {
                    // Generate Packet (Virtual Address)
                    // We submit to internal_PQ to be translated.
                    // SB Allocation will happen in handle_miss (post-translation).
                    T pkt;
                    pkt.address = head.line_addrs[i];
                    pkt.v_address = head.line_addrs[i]; 
                    pkt.type = access_type::PREFETCH; // Treat as prefetch initially
                    pkt.is_stream = true;
                    pkt.fill_to_streambuf = true;
                    pkt.tile_group_id = head.tile_group_id;
                    packets.push_back(pkt);
                    
                    head.issued[i] = true;
                    head.issued_count++;
                    progress = true;
                    quota--;
                    lines_issued++;
                    
                    if (quota == 0) break;
                }
            }
            
            if (head.issued_count == head.line_addrs.size()) {
                desc_q.pop_front();
            } else if (!progress) {
                // SB Full and no progress on current desc, stop for this cycle
                break;
            }
        }
        return packets;
    }

    void print_stats() {
        fmt::print("STREAM_ENGINE Stats: Lines Issued: {}\n", lines_issued);
        sb.print_stats();
    }
};

#endif // STREAM_ENGINE_H
