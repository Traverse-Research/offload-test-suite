//===- Buffer.h - Offload API Buffer --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOADTEST_API_BUFFER_H
#define OFFLOADTEST_API_BUFFER_H

#include "API/Resources.h"

#include "llvm/ADT/BitmaskEnum.h"

#include <cstdint>

namespace offloadtest {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

enum class BufferUsage : uint32_t {
  None = 0,
  Sampled = 1 << 0,      // Read-only shader access (SRV).
  Storage = 1 << 1,      // Read-write shader access (UAV).
  Constant = 1 << 2,     // Constant/uniform buffer (CBV).
  VertexBuffer = 1 << 3, // Vertex buffer.
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ VertexBuffer)
};

struct BufferCreateDesc {
  MemoryLocation Location;
  MemoryBacking Backing;
  BufferUsage Usage;
};

class Buffer {
public:
  virtual ~Buffer() = default;
  virtual size_t getSizeInBytes() const = 0;

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

protected:
  Buffer() = default;
};

} // namespace offloadtest

#endif // OFFLOADTEST_API_BUFFER_H
