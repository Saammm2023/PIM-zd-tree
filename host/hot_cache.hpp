#pragma once
// Host-side cache of hot keys for point lookups (our addition to PIM-zd-tree).
//
// The throughput-optimized PIM-zd-tree sends every lookup to the DPU that owns its key range, so a skewed query
// stream piles the queries for a few hot keys onto a few DPUs (the BSP round waits for the slowest one). The cache
// remembers keys that recent batches showed to be both hot and present, and answers those lookups on the CPU.
//
// Entries never go stale: a lookup returns the key itself exactly when the key is stored, and the workload has no
// deletes. (A delete batch would have to erase its keys from the cache.)
//
// Admission is cheap on purpose: only keys that occur at least twice in one batch (already counted by the
// in-batch deduplication) become candidates. Candidates carry an exponentially decayed count (halved per batch);
// the cache holds the `capacity` candidates with the highest count.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <parlay/parallel.h>

struct HotKeyCache {
    size_t capacity = 0;  // 0 = off
    std::unordered_set<uint64_t> keys;
    uint64_t hits = 0, lookups = 0;

    bool enabled() const { return capacity > 0; }
    bool contains(uint64_t key) const { return keys.find(key) != keys.end(); }

    // found_dups: (key, occurrences in this batch) for keys that occurred >= 2 times and were found
    void end_batch(const std::vector<std::pair<uint64_t, uint32_t>>& found_dups) {
        batch_++;
        for (auto& [key, cnt] : found_dups) {
            Candidate& c = candidates_[key];
            c.score = decayed(c) + cnt;
            c.batch = batch_;
        }
        // Keep the candidate table bounded, then take the top `capacity` as the cache
        if (candidates_.size() > capacity) {
            std::vector<std::pair<double, uint64_t>> ranked;
            ranked.reserve(candidates_.size());
            for (auto& [key, c] : candidates_) ranked.emplace_back(decayed(c), key);
            size_t keep = std::min(ranked.size(), 4 * capacity);
            std::nth_element(ranked.begin(), ranked.begin() + (keep - 1), ranked.end(),
                             [](auto& a, auto& b) { return a.first > b.first; });
            if (ranked.size() > keep) {
                for (size_t i = keep; i < ranked.size(); i++) candidates_.erase(ranked[i].second);
                ranked.resize(keep);
            }
            size_t top = std::min(capacity, ranked.size());
            std::nth_element(ranked.begin(), ranked.begin() + (top - 1), ranked.end(),
                             [](auto& a, auto& b) { return a.first > b.first; });
            keys.clear();
            for (size_t i = 0; i < top; i++) keys.insert(ranked[i].second);
        } else {
            keys.clear();
            for (auto& [key, c] : candidates_) keys.insert(key);
        }
    }

   private:
    struct Candidate {
        double score = 0;
        uint64_t batch = 0;
    };
    std::unordered_map<uint64_t, Candidate> candidates_;
    uint64_t batch_ = 0;

    double decayed(const Candidate& c) const { return c.score * std::ldexp(1.0, -(int)std::min<uint64_t>(batch_ - c.batch, 60)); }
};

// Finds the distinct keys of a lookup batch with a lock-free open-addressing table (about 10x faster than sorting
// a 50k batch). The table is kept between batches; only the slots a batch used are cleared afterwards.
// Lookup keys always have bit 0 clear (coord_to_key), so a slot stores key | 1 and 0 means empty.
struct KeyDedupTable {
    std::vector<std::atomic<uint64_t>> slot_key;
    std::vector<std::atomic<uint32_t>> slot_count;
    std::vector<uint32_t> slot_unique;  // slot -> index of its key in the batch's distinct-key list
    size_t mask = 0;

    void reserve(size_t n) {
        size_t cap = 1;
        while (cap < 4 * n) cap <<= 1;
        if (cap <= slot_key.size()) return;
        slot_key = std::vector<std::atomic<uint64_t>>(cap);
        slot_count = std::vector<std::atomic<uint32_t>>(cap);
        slot_unique.assign(cap, 0);
        parlay::parallel_for(0, cap, [&](size_t i) { slot_key[i].store(0); slot_count[i].store(0); });
        mask = cap - 1;
    }

    // Inserts key; returns its slot and whether this call created it
    std::pair<uint32_t, bool> insert(uint64_t key) {
        uint64_t k = key | 1;
        size_t h = (size_t)((k * 0x9E3779B97F4A7C15ull) >> 20) & mask;
        while (true) {
            uint64_t cur = slot_key[h].load(std::memory_order_acquire);
            if (cur == k) { slot_count[h].fetch_add(1, std::memory_order_relaxed); return {(uint32_t)h, false}; }
            if (cur == 0) {
                uint64_t expected = 0;
                if (slot_key[h].compare_exchange_strong(expected, k)) {
                    slot_count[h].fetch_add(1, std::memory_order_relaxed);
                    return {(uint32_t)h, true};
                }
                if (expected == k) { slot_count[h].fetch_add(1, std::memory_order_relaxed); return {(uint32_t)h, false}; }
            }
            h = (h + 1) & mask;
        }
    }

    void clear_slot(uint32_t h) {
        slot_key[h].store(0, std::memory_order_relaxed);
        slot_count[h].store(0, std::memory_order_relaxed);
    }
};

