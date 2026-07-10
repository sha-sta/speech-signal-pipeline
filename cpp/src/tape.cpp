#include "pmlab_engine/tape.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace pmlab {
namespace {

constexpr char kTapeMagic[8] = {'P', 'M', 'T', 'A', 'P', 'E', '0', '1'};
constexpr std::size_t kHeaderBytes = sizeof(kTapeMagic) + sizeof(std::uint64_t);

[[noreturn]] void fail(const std::filesystem::path& path, const char* what) {
  throw std::runtime_error(path.string() + ": " + what);
}

}  // namespace

MappedTape::MappedTape(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail(path, "cannot open");
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail(path, "fstat failed");
  }
  len_ = static_cast<std::size_t>(st.st_size);
  if (len_ < kHeaderBytes) {
    ::close(fd);
    fail(path, "truncated tape header");
  }
  addr_ = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps its own reference
  if (addr_ == MAP_FAILED) {
    addr_ = nullptr;
    fail(path, "mmap failed");
  }
  // A throwing constructor never runs the destructor: unmap manually on validation failure.
  const char* base = static_cast<const char*>(addr_);
  std::uint64_t n = 0;
  if (std::memcmp(base, kTapeMagic, sizeof(kTapeMagic)) == 0) {
    std::memcpy(&n, base + sizeof(kTapeMagic), sizeof(n));
  } else {
    ::munmap(addr_, len_);
    addr_ = nullptr;
    fail(path, "bad tape magic");
  }
  if (len_ < kHeaderBytes + n * sizeof(TapeTick)) {
    ::munmap(addr_, len_);
    addr_ = nullptr;
    fail(path, "truncated tape payload");
  }
  ticks_ = {reinterpret_cast<const TapeTick*>(base + kHeaderBytes), n};
}

MappedTape::~MappedTape() {
  if (addr_ != nullptr) {
    ::munmap(addr_, len_);
  }
}

MappedTape::MappedTape(MappedTape&& other) noexcept
    : addr_(std::exchange(other.addr_, nullptr)),
      len_(std::exchange(other.len_, 0)),
      ticks_(std::exchange(other.ticks_, {})) {}

MappedTape& MappedTape::operator=(MappedTape&& other) noexcept {
  if (this != &other) {
    if (addr_ != nullptr) {
      ::munmap(addr_, len_);
    }
    addr_ = std::exchange(other.addr_, nullptr);
    len_ = std::exchange(other.len_, 0);
    ticks_ = std::exchange(other.ticks_, {});
  }
  return *this;
}

void write_tape(std::span<const TapeTick> ticks, const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) fail(path, "cannot open for write");
  out.write(kTapeMagic, sizeof(kTapeMagic));
  const std::uint64_t n = ticks.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(ticks.data()),
            static_cast<std::streamsize>(ticks.size_bytes()));
  if (!out) fail(path, "write failed");
}

}  // namespace pmlab
