//===- AccelerationStructure.h - RT Acceleration Structure Types
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOADTEST_API_ACCELERATIONSTRUCTURE_H
#define OFFLOADTEST_API_ACCELERATIONSTRUCTURE_H

#include "API/API.h"
#include "API/Buffer.h"
#include "API/Resources.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace offloadtest {

enum class IndexFormat { Uint16, Uint32 };

enum AccelerationStructureBuildFlags : uint32_t {
  BuildFlagNone = 0,
  AllowUpdate = 1 << 0,
  PreferFastTrace = 1 << 1,
  PreferFastBuild = 1 << 2,
};

inline AccelerationStructureBuildFlags
operator|(AccelerationStructureBuildFlags A,
          AccelerationStructureBuildFlags B) {
  return static_cast<AccelerationStructureBuildFlags>(static_cast<uint32_t>(A) |
                                                      static_cast<uint32_t>(B));
}

inline AccelerationStructureBuildFlags
operator&(AccelerationStructureBuildFlags A,
          AccelerationStructureBuildFlags B) {
  return static_cast<AccelerationStructureBuildFlags>(static_cast<uint32_t>(A) &
                                                      static_cast<uint32_t>(B));
}

struct AccelerationStructureSizes {
  uint64_t ResultDataMaxSizeInBytes = 0;
  uint64_t ScratchDataSizeInBytes = 0;
  uint64_t UpdateScratchDataSizeInBytes = 0;
};

struct TriangleGeometryDesc {
  Buffer *VertexBuffer = nullptr;
  uint64_t VertexBufferOffset = 0;
  uint32_t VertexCount = 0;
  uint32_t VertexStride = 0;
  Format VertexFormat = Format::RGB32Float;
  Buffer *IndexBuffer = nullptr;
  uint64_t IndexBufferOffset = 0;
  uint32_t IndexCount = 0;
  IndexFormat IdxFormat = IndexFormat::Uint32;
  bool Opaque = true;
};

struct AABBGeometryDesc {
  Buffer *AABBBuffer = nullptr;
  uint64_t AABBBufferOffset = 0;
  uint32_t AABBCount = 0;
  uint32_t AABBStride = 24;
  bool Opaque = true;
};

class AccelerationStructure;

struct AccelerationStructureInstance {
  float Transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  uint32_t InstanceID = 0;
  uint8_t InstanceMask = 0xFF;
  AccelerationStructure *BLAS = nullptr;
};

struct BLASBuildRequest {
  llvm::SmallVector<TriangleGeometryDesc> Triangles;
  llvm::SmallVector<AABBGeometryDesc> AABBs;
  AccelerationStructureBuildFlags Flags = BuildFlagNone;
  AccelerationStructureSizes Sizes;
};

struct TLASBuildRequest {
  llvm::SmallVector<AccelerationStructureInstance> Instances;
  AccelerationStructureBuildFlags Flags = BuildFlagNone;
  AccelerationStructureSizes Sizes;
};

inline llvm::Error validateTriangleGeometryDesc(const TriangleGeometryDesc &D) {
  if (!D.VertexBuffer)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TriangleGeometryDesc: VertexBuffer is null.");
  if (!isPositionCompatible(D.VertexFormat))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TriangleGeometryDesc: VertexFormat '%s' is not position-compatible.",
        getFormatName(D.VertexFormat).data());
  if (D.VertexStride < getFormatSizeInBytes(D.VertexFormat))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TriangleGeometryDesc: VertexStride (%u) must be >= format size (%u).",
        D.VertexStride, getFormatSizeInBytes(D.VertexFormat));
  if (D.VertexCount == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "TriangleGeometryDesc: VertexCount is 0.");
  if (D.IndexBuffer && D.IndexCount == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TriangleGeometryDesc: IndexBuffer is set but IndexCount is 0.");
  if (!D.IndexBuffer && D.IndexCount != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TriangleGeometryDesc: IndexCount is set but IndexBuffer is null.");
  return llvm::Error::success();
}

inline llvm::Error validateAABBGeometryDesc(const AABBGeometryDesc &D) {
  if (!D.AABBBuffer)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "AABBGeometryDesc: AABBBuffer is null.");
  if (D.AABBCount == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "AABBGeometryDesc: AABBCount is 0.");
  if (D.AABBStride < 24)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "AABBGeometryDesc: AABBStride (%u) must be >= 24.", D.AABBStride);
  return llvm::Error::success();
}

inline llvm::Error validateBLASBuildRequest(const BLASBuildRequest &Req) {
  if (Req.Triangles.empty() && Req.AABBs.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "BLASBuildRequest: Must have at least one geometry descriptor.");
  for (const auto &T : Req.Triangles)
    if (auto Err = validateTriangleGeometryDesc(T))
      return Err;
  for (const auto &A : Req.AABBs)
    if (auto Err = validateAABBGeometryDesc(A))
      return Err;
  return llvm::Error::success();
}

inline llvm::Error validateTLASBuildRequest(const TLASBuildRequest &Req) {
  if (Req.Instances.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "TLASBuildRequest: Must have at least one instance.");
  for (size_t I = 0; I < Req.Instances.size(); ++I)
    if (!Req.Instances[I].BLAS)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "TLASBuildRequest: Instance %zu has a null BLAS pointer.", I);
  return llvm::Error::success();
}

class AccelerationStructure {
  GPUAPI API;

public:
  virtual ~AccelerationStructure();
  AccelerationStructure(const AccelerationStructure &) = delete;
  AccelerationStructure &operator=(const AccelerationStructure &) = delete;

  GPUAPI getAPI() const { return API; }

protected:
  explicit AccelerationStructure(GPUAPI API) : API(API) {}
};

} // namespace offloadtest

#endif // OFFLOADTEST_API_ACCELERATIONSTRUCTURE_H
