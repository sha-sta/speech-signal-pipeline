#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <type_traits>

namespace pmlab {

// On-disk tick record for the binary tape ("PMTAPE01"): fixed-width little-endian, 24-byte
// stride, mmap-able in place with zero parsing. The Python exporter (scripts/export_tape.py)
// writes the same layout with a numpy structured array.
struct TapeTick {
  std::int64_t ts;
  double yes_bid;
  double yes_ask;
};
static_assert(sizeof(TapeTick) == 24 && alignof(TapeTick) == 8);
static_assert(std::is_trivially_copyable_v<TapeTick>);

// RAII mmap of a tape file: the tick span points straight into the page cache; no copies, no
// allocation proportional to file size. File layout: 8-byte magic, u64 tick count, payload.
class MappedTape {
 public:
  explicit MappedTape(const std::filesystem::path& path);  // throws std::runtime_error
  ~MappedTape();
  MappedTape(MappedTape&& other) noexcept;
  MappedTape& operator=(MappedTape&& other) noexcept;
  MappedTape(const MappedTape&) = delete;
  MappedTape& operator=(const MappedTape&) = delete;

  std::span<const TapeTick> ticks() const { return ticks_; }

 private:
  void* addr_ = nullptr;
  std::size_t len_ = 0;
  std::span<const TapeTick> ticks_;
};

void write_tape(std::span<const TapeTick> ticks, const std::filesystem::path& path);

}  // namespace pmlab
