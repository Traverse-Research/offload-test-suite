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

#include "Config.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <memory>

using namespace offloadtest;

Queue::~Queue() {}

Device::~Device() {}

llvm::Expected<llvm::SmallVector<std::unique_ptr<Device>>>
offloadtest::initializeDevices(const DeviceConfig Config) {
  llvm::SmallVector<std::unique_ptr<Device>> Devices;

#ifdef OFFLOADTEST_ENABLE_D3D12
  if (auto Err = initializeDX12Devices(Config, Devices))
    return Err;
#endif

#ifdef OFFLOADTEST_ENABLE_VULKAN
  if (auto Err = initializeVulkanDevices(Config, Devices))
    return Err;
#endif

#ifdef OFFLOADTEST_ENABLE_METAL
  if (auto Err = initializeMetalDevices(Config, Devices))
    return Err;
#endif

  return Devices;
}

llvm::Expected<std::shared_ptr<Texture>>
offloadtest::createRenderTarget(Device &Dev, const CPUBuffer &Buf) {
  auto TexFmtOrErr = toTextureFormat(Buf.Format, Buf.Channels);
  if (!TexFmtOrErr)
    return TexFmtOrErr.takeError();

  TextureCreateDesc Desc = {};
  Desc.Location = MemoryLocation::GpuOnly;
  Desc.Usage = TextureUsage::RenderTarget;
  Desc.Format = *TexFmtOrErr;
  Desc.Width = Buf.OutputProps.Width;
  Desc.Height = Buf.OutputProps.Height;
  Desc.MipLevels = 1;
  Desc.OptimizedClearValue = ClearColor{};

  if (auto Err = validateTextureDescMatchesCPUBuffer(Desc, Buf))
    return Err;

  return Dev.createTexture("RenderTarget", Desc);
}

llvm::Expected<std::shared_ptr<Texture>>
offloadtest::createDepthStencil(Device &Dev, uint32_t Width, uint32_t Height) {
  TextureCreateDesc Desc = {};
  Desc.Location = MemoryLocation::GpuOnly;
  Desc.Usage = TextureUsage::DepthStencil;
  Desc.Format = TextureFormat::D32FloatS8Uint;
  Desc.Width = Width;
  Desc.Height = Height;
  Desc.MipLevels = 1;
  Desc.OptimizedClearValue = ClearDepthStencil{1.0f, 0};

  return Dev.createTexture("DepthStencil", Desc);
}
