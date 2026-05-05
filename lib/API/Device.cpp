//===- DX/Device.cpp - DirectX Device API ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "API/Device.h"
#include "API/Encoder.h"
#include "API/FormatConversion.h"

#include "Config.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace offloadtest;

CommandEncoder::~CommandEncoder() {}

Buffer::~Buffer() {}

CommandBuffer::~CommandBuffer() {}

Fence::~Fence() {}

Queue::~Queue() {}

Texture::~Texture() {}

AccelerationStructure::~AccelerationStructure() {}

Device::~Device() {}

llvm::Expected<llvm::SmallVector<std::unique_ptr<Device>>>
offloadtest::initializeDevices(const DeviceConfig Config) {
  llvm::SmallVector<std::unique_ptr<Device>> Devices;
  llvm::Error Err = llvm::Error::success();

#ifdef OFFLOADTEST_ENABLE_D3D12
  if (auto E = initializeDX12Devices(Config, Devices))
    Err = llvm::joinErrors(std::move(Err), std::move(E));
#endif

#ifdef OFFLOADTEST_ENABLE_VULKAN
  if (auto E = initializeVulkanDevices(Config, Devices))
    Err = llvm::joinErrors(std::move(Err), std::move(E));
#endif

#ifdef OFFLOADTEST_ENABLE_METAL
  if (auto E = initializeMetalDevices(Config, Devices))
    Err = llvm::joinErrors(std::move(Err), std::move(E));
#endif

  if (Devices.empty()) {
    if (Err)
      return std::move(Err);
    return llvm::createStringError(std::errc::no_such_device,
                                   "No GPU devices found.");
  }
  // Log errors from backends that failed while others succeeded.
  if (Err)
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs());
  return Devices;
}

llvm::Expected<std::unique_ptr<Texture>>
offloadtest::createRenderTargetFromCPUBuffer(Device &Dev,
                                             const CPUBuffer &Buf) {
  auto TexFmtOrErr = toFormat(Buf.Format, Buf.Channels);
  if (!TexFmtOrErr)
    return TexFmtOrErr.takeError();

  TextureCreateDesc Desc = {};
  Desc.Location = MemoryLocation::GpuOnly;
  Desc.Usage = TextureUsage::RenderTarget;
  Desc.Fmt = *TexFmtOrErr;
  Desc.Width = Buf.OutputProps.Width;
  Desc.Height = Buf.OutputProps.Height;
  Desc.MipLevels = 1;
  Desc.OptimizedClearValue = ClearColor{};

  if (auto Err = validateTextureDescMatchesCPUBuffer(Desc, Buf))
    return Err;

  return Dev.createTexture("RenderTarget", Desc);
}

llvm::Expected<std::unique_ptr<Buffer>>
offloadtest::createBufferWithData(Device &Dev, llvm::StringRef Name,
                                  const BufferCreateDesc &Desc,
                                  llvm::ArrayRef<uint8_t> Data) {
  BufferCreateDesc Mut = Desc;
  auto BufOrErr = Dev.createBuffer(Name.str(), Mut, Data.size());
  if (!BufOrErr)
    return BufOrErr.takeError();
  auto Buf = std::move(*BufOrErr);

  auto PtrOrErr = Buf->map();
  if (!PtrOrErr)
    return PtrOrErr.takeError();
  memcpy(*PtrOrErr, Data.data(), Data.size());
  if (auto Err = Buf->unmap())
    return std::move(Err);

  return Buf;
}

llvm::Expected<std::unique_ptr<Buffer>>
offloadtest::createVertexBufferFromCPUBuffer(Device &Dev,
                                             const CPUBuffer &Buf) {
  BufferCreateDesc BufDesc = {};
  BufDesc.Location = MemoryLocation::CpuToGpu;
  BufDesc.Usage = BufferUsage::VertexBuffer;
  // TODO: Currently uses a single CpuToGpu mapped buffer.
  // On discrete GPUs consider using a staging buffer + copy to a GpuOnly vertex
  // buffer for optimal GPU read performance.
  return createBufferWithData(
      Dev, "VertexBuffer", BufDesc,
      llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(Buf.Data[0].get()), Buf.size()));
}

llvm::Error offloadtest::buildPipelineAccelerationStructures(
    Device &Dev, ComputeEncoder &Enc, Pipeline &P,
    llvm::SmallVectorImpl<std::unique_ptr<AccelerationStructure>> &OutAS,
    llvm::SmallVectorImpl<std::unique_ptr<Buffer>> &OutInputBuffers) {
  if (P.AccelStructs.BLAS.empty() && P.AccelStructs.TLAS.empty())
    return llvm::Error::success();

  // Storage upload-heap buffer flags. Backends that need extra native flags
  // for AS-build inputs (e.g. Vulkan SHADER_DEVICE_ADDRESS +
  // ACCEL_BUILD_INPUT_READ_ONLY) add them implicitly when ray tracing is
  // supported.
  const BufferCreateDesc UploadDesc{MemoryLocation::CpuToGpu,
                                    BufferUsage::Storage};

  // Stash the request structs while we build them up — the encoder reads
  // them through pointers stored in ASBuildItem.
  llvm::SmallVector<BLASBuildRequest> BLASRequests;
  BLASRequests.reserve(P.AccelStructs.BLAS.size());
  llvm::SmallVector<size_t> BLASOutIdx;
  BLASOutIdx.reserve(P.AccelStructs.BLAS.size());
  llvm::StringMap<size_t> BLASIndex;

  for (const auto &BD : P.AccelStructs.BLAS) {
    llvm::SmallVector<TriangleGeometryDesc> Triangles;
    Triangles.reserve(BD.Triangles.size());
    for (const auto &T : BD.Triangles) {
      assert(T.VertexBufferPtr && "VertexBufferPtr not resolved");
      auto VBOrErr = createBufferWithData(
          Dev, "AS-Vertices", UploadDesc,
          llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                                      T.VertexBufferPtr->Data[0].get()),
                                  T.VertexBufferPtr->size()));
      if (!VBOrErr)
        return VBOrErr.takeError();

      TriangleGeometryDesc TGD;
      TGD.VertexBuffer = VBOrErr->get();
      TGD.VertexCount = T.VertexCount;
      TGD.VertexStride = T.VertexStride;
      TGD.VertexFormat = T.VertexFormat;
      TGD.Opaque = T.Opaque;

      OutInputBuffers.push_back(std::move(*VBOrErr));

      if (T.IndexBufferPtr) {
        auto IBOrErr = createBufferWithData(
            Dev, "AS-Indices", UploadDesc,
            llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                                        T.IndexBufferPtr->Data[0].get()),
                                    T.IndexBufferPtr->size()));
        if (!IBOrErr)
          return IBOrErr.takeError();
        TGD.IndexBuffer = IBOrErr->get();
        TGD.IndexCount = T.IndexCount;
        TGD.IdxFormat = T.IdxFormat;
        OutInputBuffers.push_back(std::move(*IBOrErr));
      }
      Triangles.push_back(TGD);
    }
    // TODO: AABB geometry support (would mirror the triangle path).

    auto ReqOrErr = Dev.createBLASBuildRequest(Triangles, {}, BuildFlagNone);
    if (!ReqOrErr)
      return ReqOrErr.takeError();
    auto ASOrErr = Dev.createAccelerationStructure(*ReqOrErr);
    if (!ASOrErr)
      return ASOrErr.takeError();

    BLASIndex[BD.Name] = OutAS.size();
    BLASOutIdx.push_back(OutAS.size());
    OutAS.push_back(std::move(*ASOrErr));
    BLASRequests.push_back(std::move(*ReqOrErr));
  }

  llvm::SmallVector<ASBuildItem> BLASBatch;
  BLASBatch.reserve(BLASRequests.size());
  for (size_t I = 0; I < BLASRequests.size(); ++I) {
    BLASBatch.push_back(
        {OutAS[BLASOutIdx[I]].get(),
         {static_cast<const BLASBuildRequest *>(&BLASRequests[I])}});
  }
  if (!BLASBatch.empty())
    if (auto Err = Enc.batchBuildAS(BLASBatch))
      return Err;

  // TLAS pass — references BLASes built in the previous batch.
  llvm::SmallVector<TLASBuildRequest> TLASRequests;
  TLASRequests.reserve(P.AccelStructs.TLAS.size());
  llvm::SmallVector<size_t> TLASOutIdx;
  TLASOutIdx.reserve(P.AccelStructs.TLAS.size());

  for (const auto &TD : P.AccelStructs.TLAS) {
    llvm::SmallVector<AccelerationStructureInstance> Instances;
    Instances.reserve(TD.Instances.size());
    for (const auto &I : TD.Instances) {
      auto It = BLASIndex.find(I.BLAS);
      if (It == BLASIndex.end())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "TLAS '%s' references unknown BLAS '%s'",
                                       TD.Name.c_str(), I.BLAS.c_str());

      AccelerationStructureInstance Inst;
      static_assert(sizeof(Inst.Transform) == sizeof(I.Transform),
                    "Transform layout mismatch");
      memcpy(Inst.Transform, I.Transform, sizeof(I.Transform));
      Inst.InstanceID = I.InstanceID;
      Inst.InstanceMask = I.InstanceMask;
      Inst.BLAS = OutAS[It->second].get();
      Instances.push_back(Inst);
    }

    auto ReqOrErr = Dev.createTLASBuildRequest(Instances, BuildFlagNone);
    if (!ReqOrErr)
      return ReqOrErr.takeError();
    auto ASOrErr = Dev.createAccelerationStructure(*ReqOrErr);
    if (!ASOrErr)
      return ASOrErr.takeError();

    TLASOutIdx.push_back(OutAS.size());
    OutAS.push_back(std::move(*ASOrErr));
    TLASRequests.push_back(std::move(*ReqOrErr));
  }

  llvm::SmallVector<ASBuildItem> TLASBatch;
  TLASBatch.reserve(TLASRequests.size());
  for (size_t I = 0; I < TLASRequests.size(); ++I) {
    TLASBatch.push_back(
        {OutAS[TLASOutIdx[I]].get(),
         {static_cast<const TLASBuildRequest *>(&TLASRequests[I])}});
  }
  if (!TLASBatch.empty())
    if (auto Err = Enc.batchBuildAS(TLASBatch))
      return Err;

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Texture>>
offloadtest::createDefaultDepthStencilTarget(Device &Dev, uint32_t Width,
                                             uint32_t Height) {
  TextureCreateDesc Desc = {};
  Desc.Location = MemoryLocation::GpuOnly;
  Desc.Usage = TextureUsage::DepthStencil;
  Desc.Fmt = Format::D32FloatS8Uint;
  Desc.Width = Width;
  Desc.Height = Height;
  Desc.MipLevels = 1;
  Desc.OptimizedClearValue = ClearDepthStencil{1.0f, 0};

  return Dev.createTexture("DepthStencil", Desc);
}
