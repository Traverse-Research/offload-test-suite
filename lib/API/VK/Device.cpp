//===- VX/Device.cpp - Vulkan Device API ----------------------------------===//
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
#include "API/FormatConversion.h"
#include "API/VertexBuffer.h"
#include "Support/Pipeline.h"
#include "VKResources.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <system_error>
#include <vulkan/vulkan.h>

using namespace offloadtest;

#define VKFormats(FMT, BITS)                                                   \
  if (Channels == 1)                                                           \
    return VK_FORMAT_R##BITS##_##FMT;                                          \
  if (Channels == 2)                                                           \
    return VK_FORMAT_R##BITS##G##BITS##_##FMT;                                 \
  if (Channels == 3)                                                           \
    return VK_FORMAT_R##BITS##G##BITS##B##BITS##_##FMT;                        \
  if (Channels == 4)                                                           \
    return VK_FORMAT_R##BITS##G##BITS##B##BITS##A##BITS##_##FMT;

static VkFormat getVKFormat(DataFormat Format, int Channels) {
  switch (Format) {
  case DataFormat::Int16:
    VKFormats(SINT, 16) break;
  case DataFormat::UInt16:
    VKFormats(UINT, 16) break;
  case DataFormat::Int32:
    VKFormats(SINT, 32) break;
  case DataFormat::UInt32:
    VKFormats(UINT, 32) break;
  case DataFormat::Float32:
    VKFormats(SFLOAT, 32) break;
  case DataFormat::Int64:
    VKFormats(SINT, 64) break;
  case DataFormat::UInt64:
    VKFormats(UINT, 64) break;
  case DataFormat::Float64:
    VKFormats(SFLOAT, 64) break;
  case DataFormat::Depth32:
    if (Channels != 1)
      llvm_unreachable("Depth32 format only supports a single channel.");
    return VK_FORMAT_D32_SFLOAT;
  default:
    llvm_unreachable("Unsupported Resource format specified");
  }
  return VK_FORMAT_UNDEFINED;
}

static VkDescriptorType getDescriptorType(const ResourceKind RK) {
  switch (RK) {
  case ResourceKind::Buffer:
  case ResourceKind::RWBuffer:
    return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
  case ResourceKind::Texture2D:
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  case ResourceKind::RWTexture2D:
    return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  case ResourceKind::ByteAddressBuffer:
  case ResourceKind::RWByteAddressBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::RWStructuredBuffer:
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  case ResourceKind::ConstantBuffer:
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  case ResourceKind::Sampler:
    return VK_DESCRIPTOR_TYPE_SAMPLER;
  case ResourceKind::SampledTexture2D:
    return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  }
  llvm_unreachable("All cases handled");
}

static VkFilter getVKFilter(FilterMode Mode) {
  switch (Mode) {
  case FilterMode::Nearest:
    return VK_FILTER_NEAREST;
  case FilterMode::Linear:
    return VK_FILTER_LINEAR;
  }
  llvm_unreachable("All filter cases handled");
}

static VkSamplerAddressMode getVKAddressMode(AddressMode Mode) {
  switch (Mode) {
  case AddressMode::Clamp:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case AddressMode::Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case AddressMode::Mirror:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case AddressMode::Border:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  case AddressMode::MirrorOnce:
    return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  }
  llvm_unreachable("All address mode cases handled");
}

static VkCompareOp getVKCompareOp(CompareFunction Func) {
  switch (Func) {
  case CompareFunction::Never:
    return VK_COMPARE_OP_NEVER;
  case CompareFunction::Less:
    return VK_COMPARE_OP_LESS;
  case CompareFunction::Equal:
    return VK_COMPARE_OP_EQUAL;
  case CompareFunction::LessEqual:
    return VK_COMPARE_OP_LESS_OR_EQUAL;
  case CompareFunction::Greater:
    return VK_COMPARE_OP_GREATER;
  case CompareFunction::NotEqual:
    return VK_COMPARE_OP_NOT_EQUAL;
  case CompareFunction::GreaterEqual:
    return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case CompareFunction::Always:
    return VK_COMPARE_OP_ALWAYS;
  }
  llvm_unreachable("All compare op cases handled");
}

static VkImageViewType getImageViewType(const ResourceKind RK) {
  switch (RK) {
  case ResourceKind::Texture2D:
  case ResourceKind::RWTexture2D:
  case ResourceKind::SampledTexture2D:
    return VK_IMAGE_VIEW_TYPE_2D;
  case ResourceKind::Buffer:
  case ResourceKind::RWBuffer:
  case ResourceKind::ByteAddressBuffer:
  case ResourceKind::RWByteAddressBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::RWStructuredBuffer:
  case ResourceKind::ConstantBuffer:
  case ResourceKind::Sampler:
    llvm_unreachable("Not an image view!");
  }
  llvm_unreachable("All cases handled");
}

static VkImageType getVKImageType(const ResourceKind RK) {
  switch (RK) {
  case ResourceKind::Texture2D:
  case ResourceKind::RWTexture2D:
  case ResourceKind::SampledTexture2D:
    return VK_IMAGE_TYPE_2D;
  default:
    llvm_unreachable("Unsupported image kind");
  }
  llvm_unreachable("All cases handled");
}

static VkShaderStageFlagBits getShaderStageFlag(Stages Stage) {
  switch (Stage) {
  case Stages::Compute:
    return VK_SHADER_STAGE_COMPUTE_BIT;
  case Stages::Vertex:
    return VK_SHADER_STAGE_VERTEX_BIT;
  case Stages::Pixel:
    return VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  llvm_unreachable("All cases handled");
}

static std::string getMessageSeverityString(
    VkDebugUtilsMessageSeverityFlagBitsEXT MessageSeverity) {
  if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    return "Error";
  if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    return "Warning";
  if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    return "Info";
  if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
    return "Verbose";
  return "Unknown";
}

static VkBool32
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT MessageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT MessageType,
              const VkDebugUtilsMessengerCallbackDataEXT *Data, void *) {
  // Only interested in messages from the validation layers.
  if (!(MessageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
    return VK_FALSE;

  llvm::dbgs() << "Validation " << getMessageSeverityString(MessageSeverity);
  llvm::dbgs() << ": [ " << Data->pMessageIdName << " ]\n";
  llvm::dbgs() << Data->pMessage;

  for (uint32_t I = 0; I < Data->objectCount; I++) {
    llvm::dbgs() << '\n';
    if (Data->pObjects[I].pObjectName) {
      llvm::dbgs() << "[" << Data->pObjects[I].pObjectName << "]";
    }
  }
  llvm::dbgs() << '\n';

  // Return true to turn the validation error or warning into an error in the
  // vulkan API. This should causes tests to fail.
  const bool IsErrorOrWarning =
      MessageSeverity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
  if (IsErrorOrWarning)
    return VK_TRUE;

  // Continue to run even with VERBOSE and INFO messages.
  return VK_FALSE;
}

static VkDebugUtilsMessengerEXT registerDebugUtilCallback(VkInstance Instance) {
  VkDebugUtilsMessengerCreateInfoEXT CreateInfo = {};
  CreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  CreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  CreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  CreateInfo.pfnUserCallback = debugCallback;
  CreateInfo.pUserData = nullptr; // Optional
  auto Func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      Instance, "vkCreateDebugUtilsMessengerEXT");
  if (Func == nullptr)
    return VK_NULL_HANDLE;

  VkDebugUtilsMessengerEXT DebugMessenger;
  Func(Instance, &CreateInfo, nullptr, &DebugMessenger);
  return DebugMessenger;
}

static llvm::Expected<uint32_t>
getMemoryIndex(VkPhysicalDevice Device, uint32_t MemoryTypeBits,
               VkMemoryPropertyFlags MemoryFlags) {
  VkPhysicalDeviceMemoryProperties MemProperties;
  vkGetPhysicalDeviceMemoryProperties(Device, &MemProperties);
  for (uint32_t I = 0; I < MemProperties.memoryTypeCount; ++I) {
    const uint32_t Bit = (1u << I);
    if ((MemoryTypeBits & Bit) == 0)
      continue;
    if ((MemProperties.memoryTypes[I].propertyFlags & MemoryFlags) ==
        MemoryFlags)
      return I;
  }
  return llvm::createStringError(std::errc::not_enough_memory,
                                 "Could not identify appropriate memory.");
}

static llvm::SmallVector<VkLayerProperties, 0> queryInstanceLayers() {
  uint32_t LayerCount;
  vkEnumerateInstanceLayerProperties(&LayerCount, nullptr);

  llvm::SmallVector<VkLayerProperties, 0> Layers;
  if (LayerCount == 0)
    return Layers;

  Layers.resize(LayerCount);
  vkEnumerateInstanceLayerProperties(&LayerCount, Layers.data());

  return Layers;
}

static bool
isLayerSupported(const llvm::SmallVector<VkLayerProperties, 0> &Layers,
                 llvm::StringRef QueryName) {
  for (auto &Layer : Layers) {
    if (Layer.layerName == QueryName)
      return true;
  }
  return false;
}

static llvm::SmallVector<VkExtensionProperties, 0>
queryInstanceExtensions(const char *InstanceLayer) {
  uint32_t ExtCount;
  vkEnumerateInstanceExtensionProperties(InstanceLayer, &ExtCount, nullptr);

  llvm::SmallVector<VkExtensionProperties, 0> Extensions;
  if (ExtCount == 0)
    return Extensions;

  Extensions.resize(ExtCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &ExtCount, Extensions.data());

  return Extensions;
}

static llvm::SmallVector<VkExtensionProperties, 0>
queryDeviceExtensions(VkPhysicalDevice PhysicalDevice) {
  uint32_t ExtCount;
  vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &ExtCount,
                                       nullptr);

  llvm::SmallVector<VkExtensionProperties, 0> Extensions;
  if (ExtCount == 0)
    return Extensions;

  Extensions.resize(ExtCount);
  vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &ExtCount,
                                       Extensions.data());

  return Extensions;
}

static bool isExtensionSupported(
    const llvm::SmallVector<VkExtensionProperties, 0> &Extensions,
    llvm::StringRef QueryName) {
  for (const auto &Ext : Extensions) {
    if (Ext.extensionName == QueryName)
      return true;
  }
  return false;
}

struct VulkanInstance {
  VkInstance Instance;
  VkDebugUtilsMessengerEXT DebugMessenger;

  VulkanInstance(VkInstance Instance, VkDebugUtilsMessengerEXT DebugMessenger)
      : Instance(Instance), DebugMessenger(DebugMessenger) {}
  VulkanInstance(const VulkanInstance &) = delete;
  VulkanInstance(VulkanInstance &&) = delete;
  VulkanInstance &operator=(const VulkanInstance &) = delete;
  VulkanInstance &operator=(VulkanInstance &&) = delete;
  ~VulkanInstance() {
    if (DebugMessenger) {
      auto Func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          Instance, "vkDestroyDebugUtilsMessengerEXT");
      assert(Func != nullptr);
      Func(Instance, DebugMessenger, nullptr);
    }
    vkDestroyInstance(Instance, nullptr);
  }
};

namespace {

class VulkanBuffer : public offloadtest::Buffer {
public:
  VkDevice Dev; // Needed for clean-up
  VkBuffer Buffer;
  VkDeviceMemory Memory;
  std::string Name;
  BufferCreateDesc Desc;
  size_t SizeInBytes;

  VulkanBuffer(VkDevice Dev, VkBuffer Buffer, VkDeviceMemory Memory,
               llvm::StringRef Name, BufferCreateDesc Desc, size_t SizeInBytes)
      : Dev(Dev), Buffer(Buffer), Memory(Memory), Name(Name), Desc(Desc),
        SizeInBytes(SizeInBytes) {}

  size_t getSizeInBytes() const override { return SizeInBytes; }

  ~VulkanBuffer() override {
    vkDestroyBuffer(Dev, Buffer, nullptr);
    vkFreeMemory(Dev, Memory, nullptr);
  }
};

class VulkanTexture : public offloadtest::Texture {
public:
  VkDevice Dev;
  VkImage Image;
  VkDeviceMemory Memory;
  // TODO:
  // RenderTarget and DepthStencil views are created at texture creation time.
  // Ideally Sampled/Storage image views would also live here, but they are
  // currently created during descriptor set setup, which determines their
  // binding layout.
  VkImageView View = VK_NULL_HANDLE;
  std::string Name;
  TextureCreateDesc Desc;

  VulkanTexture(VkDevice Dev, VkImage Image, VkDeviceMemory Memory,
                llvm::StringRef Name, TextureCreateDesc Desc)
      : Dev(Dev), Image(Image), Memory(Memory), Name(Name), Desc(Desc) {}

  const TextureCreateDesc &getDesc() const override { return Desc; }

  ~VulkanTexture() override {
    if (View)
      vkDestroyImageView(Dev, View, nullptr);
    vkDestroyImage(Dev, Image, nullptr);
    vkFreeMemory(Dev, Memory, nullptr);
  }
};

class VulkanQueue : public offloadtest::Queue {
public:
  VkQueue Queue = VK_NULL_HANDLE;
  uint32_t QueueFamilyIdx = 0;
  VulkanQueue(VkQueue Q, uint32_t QueueFamilyIdx)
      : Queue(Q), QueueFamilyIdx(QueueFamilyIdx) {}
};

class VulkanDevice : public offloadtest::Device {
private:
  std::shared_ptr<VulkanInstance> Instance;
  VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties Props;
  VkPhysicalDeviceProperties2 Props2;
  VkPhysicalDeviceFloatControlsProperties FloatControlProp;
  VkPhysicalDeviceDriverProperties DriverProps;
  VkDevice Device = VK_NULL_HANDLE;
  VulkanQueue GraphicsQueue;
  Capabilities Caps;
  using LayerVector = llvm::SmallVector<VkLayerProperties, 0>;
  LayerVector InstanceLayers;
  using ExtensionVector = llvm::SmallVector<VkExtensionProperties, 0>;
  ExtensionVector DeviceExtensions;

  // A GPU resource created for a descriptor set binding. Holds either a
  // texture or a buffer, plus VK-specific layout and sampler state.
  struct BoundResource {
    std::variant<std::shared_ptr<VulkanTexture>, std::shared_ptr<VulkanBuffer>>
        Resource;

    // Current image layout for barrier transitions (VK-specific).
    VkImageLayout CurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Sampler for combined image sampler descriptors (VK-specific).
    VkSampler Sampler = VK_NULL_HANDLE;

    // Device-local counter buffer (VK-specific, for UAV append/consume).
    std::shared_ptr<VulkanBuffer> CounterBuffer;

    VkImage getImage() const {
      return std::get<std::shared_ptr<VulkanTexture>>(Resource)->Image;
    }

    VkBuffer getBuffer() const {
      return std::get<std::shared_ptr<VulkanBuffer>>(Resource)->Buffer;
    }

    VkDeviceMemory getMemory() const {
      return std::visit(
          [](const auto &R) -> VkDeviceMemory { return R->Memory; }, Resource);
    }

    bool isTexture() const {
      return std::holds_alternative<std::shared_ptr<VulkanTexture>>(Resource);
    }

    size_t getSizeInBytes() const {
      return std::visit([](const auto &R) { return R->getSizeInBytes(); },
                        Resource);
    }
  };

  // Pairs a source GPU resource with a CPU-readable readback buffer for
  // copying results back after execution.
  struct CounterBinding {
    std::shared_ptr<VulkanBuffer> Source;      // Device-local counter buffer.
    std::shared_ptr<VulkanBuffer> Destination; // Host-visible readback buffer.
  };

  struct ReadbackBinding {
    std::shared_ptr<BoundResource> Source;
    std::shared_ptr<VulkanBuffer> Destination;
    offloadtest::Resource *PipelineResource; // For readback data routing.
    uint32_t ArrayIndex;                     // Index within resource array.
    std::optional<CounterBinding> Counter;
  };

  using ResourceBundle = llvm::SmallVector<std::shared_ptr<BoundResource>>;
  using ResourcePair = std::pair<offloadtest::Resource *, ResourceBundle>;

  struct CompiledShader {
    Stages Stage;
    std::string Entry;
    VkShaderModule Shader;
  };

  struct InvocationState {
    VkCommandPool CmdPool = VK_NULL_HANDLE;
    VkCommandBuffer CmdBuffer = VK_NULL_HANDLE;
    VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool Pool = VK_NULL_HANDLE;
    VkPipelineCache PipelineCache = VK_NULL_HANDLE;
    VkPipeline Pipeline = VK_NULL_HANDLE;

    // FrameBuffer associated data for offscreen rendering.
    VkFramebuffer FrameBuffer = VK_NULL_HANDLE;
    std::shared_ptr<VulkanTexture> RenderTarget;
    std::shared_ptr<VulkanBuffer> RTReadback;
    std::shared_ptr<VulkanTexture> DepthStencil;
    std::optional<offloadtest::VertexBuffer> VB = std::nullopt;

    VkRenderPass RenderPass = VK_NULL_HANDLE;
    uint32_t ShaderStageMask = 0;

    llvm::SmallVector<CompiledShader> Shaders;
    llvm::SmallVector<VkDescriptorSetLayout> DescriptorSetLayouts;
    llvm::SmallVector<ResourcePair> Resources;
    llvm::SmallVector<VkDescriptorSet> DescriptorSets;
    llvm::SmallVector<VkBufferView> BufferViews;
    llvm::SmallVector<VkImageView> ImageViews;

    // Staging/upload buffers kept alive until GPU finishes.
    llvm::SmallVector<std::shared_ptr<VulkanBuffer>> ResourcesKeepAlive;
    // Readback bindings for copying results back to CPU after execution.
    llvm::SmallVector<ReadbackBinding> ReadbackBindings;

    uint32_t getFullShaderStageMask() {
      if (0 != ShaderStageMask)
        return ShaderStageMask;
      for (const auto &S : Shaders)
        ShaderStageMask |= getShaderStageFlag(S.Stage);
      return ShaderStageMask;
    }
  };

public:
  static llvm::Expected<std::unique_ptr<VulkanDevice>>
  create(std::shared_ptr<VulkanInstance> Instance,
         VkPhysicalDevice PhysicalDevice,
         llvm::SmallVector<VkLayerProperties, 0> InstanceLayers) {
    VkPhysicalDeviceProperties Props;
    vkGetPhysicalDeviceProperties(PhysicalDevice, &Props);

    // Find a queue family that supports both graphics and compute.
    uint32_t QueueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &QueueCount,
                                             nullptr);
    if (QueueCount == 0)
      return llvm::createStringError(std::errc::no_such_device,
                                     "No queue families reported.");

    const std::unique_ptr<VkQueueFamilyProperties[]> QueueFamilyProps(
        new VkQueueFamilyProperties[QueueCount]);
    vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &QueueCount,
                                             QueueFamilyProps.get());

    std::optional<uint32_t> SelectedIdx;
    for (uint32_t I = 0; I < QueueCount; ++I) {
      const VkQueueFlags Flags = QueueFamilyProps[I].queueFlags;
      // Prefer family supporting both GRAPHICS and COMPUTE
      if ((Flags & VK_QUEUE_GRAPHICS_BIT) && (Flags & VK_QUEUE_COMPUTE_BIT)) {
        SelectedIdx = static_cast<int>(I);
        break;
      }
    }

    if (!SelectedIdx)
      return llvm::createStringError(std::errc::no_such_device,
                                     "No suitable queue family found.");

    const uint32_t QueueFamilyIdx = *SelectedIdx;

    VkDeviceQueueCreateInfo QueueInfo = {};
    const float QueuePriority = 1.0f;
    QueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    QueueInfo.queueFamilyIndex = QueueFamilyIdx;
    QueueInfo.queueCount = 1;
    QueueInfo.pQueuePriorities = &QueuePriority;

    VkDeviceCreateInfo DeviceInfo = {};
    DeviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    DeviceInfo.queueCreateInfoCount = 1;
    DeviceInfo.pQueueCreateInfos = &QueueInfo;

    VkPhysicalDeviceFeatures2 Features{};
    Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceVulkan11Features Features11{};
    Features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features Features12{};
    Features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features Features13{};
    Features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
#ifdef VK_VERSION_1_4
    VkPhysicalDeviceVulkan14Features Features14{};
    Features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
#endif

    Features.pNext = &Features11;
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0))
      Features11.pNext = &Features12;
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0))
      Features12.pNext = &Features13;
#ifdef VK_VERSION_1_4
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 4, 0))
      Features13.pNext = &Features14;
#endif
    vkGetPhysicalDeviceFeatures2(PhysicalDevice, &Features);

    DeviceInfo.pEnabledFeatures = &Features.features;
    DeviceInfo.pNext = Features.pNext;

    VkDevice Device = VK_NULL_HANDLE;
    if (vkCreateDevice(PhysicalDevice, &DeviceInfo, nullptr, &Device))
      return llvm::createStringError(std::errc::no_such_device,
                                     "Could not create Vulkan logical device.");
    VkQueue DeviceQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(Device, QueueFamilyIdx, 0, &DeviceQueue);

    const VulkanQueue GraphicsQueue = VulkanQueue(DeviceQueue, QueueFamilyIdx);

    return std::make_unique<VulkanDevice>(Instance, PhysicalDevice, Props,
                                          Device, std::move(GraphicsQueue),
                                          std::move(InstanceLayers));
  }

  VulkanDevice(std::shared_ptr<VulkanInstance> I, VkPhysicalDevice P,
               VkPhysicalDeviceProperties Props, VkDevice D, VulkanQueue Q,
               llvm::SmallVector<VkLayerProperties, 0> InstanceLayers)
      : Instance(I), PhysicalDevice(P), Props(Props), Device(D),
        GraphicsQueue(std::move(Q)), InstanceLayers(std::move(InstanceLayers)) {
    const uint64_t DeviceNameSz =
        strnlen(Props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
    Description = std::string(Props.deviceName, DeviceNameSz);

    FloatControlProp.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    FloatControlProp.pNext = nullptr;

    DriverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    DriverProps.pNext = &FloatControlProp;

    Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    Props2.pNext = &DriverProps;
    vkGetPhysicalDeviceProperties2(PhysicalDevice, &Props2);

    const uint64_t DriverNameSz =
        strnlen(DriverProps.driverName, VK_MAX_DRIVER_NAME_SIZE);
    DriverName = std::string(DriverProps.driverName, DriverNameSz);
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple silicon Macs may have multiple Vulkan drivers sharing one device
    // name. Include the driver name in the description to enable
    // adapter-regex matching.
    Description += " (" + DriverName + ")";
#endif

    DeviceExtensions = queryDeviceExtensions(PhysicalDevice);
  }
  VulkanDevice(const VulkanDevice &) = delete;

  ~VulkanDevice() override {
    if (Device != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(Device);
      vkDestroyDevice(Device, nullptr);
    }
  }

  llvm::StringRef getAPIName() const override { return "Vulkan"; }
  GPUAPI getAPI() const override { return GPUAPI::Vulkan; }

  Queue &getGraphicsQueue() override { return GraphicsQueue; }

  llvm::Expected<std::shared_ptr<offloadtest::Buffer>>
  createBuffer(std::string Name, BufferCreateDesc &Desc,
               size_t SizeInBytes) override {
    VkBufferCreateInfo BufInfo = {};
    BufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BufInfo.size = SizeInBytes;
    BufInfo.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((Desc.Usage & BufferUsage::Sampled) != BufferUsage::None)
      BufInfo.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                       VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((Desc.Usage & BufferUsage::Storage) != BufferUsage::None)
      BufInfo.usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((Desc.Usage & BufferUsage::Constant) != BufferUsage::None)
      BufInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if ((Desc.Usage & BufferUsage::VertexBuffer) != BufferUsage::None)
      BufInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    BufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer DeviceBuffer;
    if (vkCreateBuffer(Device, &BufInfo, nullptr, &DeviceBuffer))
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to create device buffer.");

    VkMemoryRequirements MemReqs;
    vkGetBufferMemoryRequirements(Device, DeviceBuffer, &MemReqs);

    VkMemoryAllocateInfo AllocInfo = {};
    AllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    AllocInfo.allocationSize = MemReqs.size;
    auto MemIdx = getMemoryIndex(PhysicalDevice, MemReqs.memoryTypeBits,
                                 getVulkanMemoryFlags(Desc.Location));
    if (!MemIdx)
      return MemIdx.takeError();
    AllocInfo.memoryTypeIndex = *MemIdx;

    VkDeviceMemory DeviceMemory;
    if (vkAllocateMemory(Device, &AllocInfo, nullptr, &DeviceMemory))
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to allocate device memory.");
    if (vkBindBufferMemory(Device, DeviceBuffer, DeviceMemory, 0))
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to bind device buffer memory.");

    return std::make_shared<VulkanBuffer>(Device, DeviceBuffer, DeviceMemory,
                                          Name, Desc, SizeInBytes);
  }

  llvm::Expected<std::shared_ptr<offloadtest::Texture>>
  createTexture(std::string Name, TextureCreateDesc &Desc) override {
    if (auto Err = validateTextureCreateDesc(Desc))
      return Err;

    VkImageCreateInfo ImageInfo = {};
    ImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ImageInfo.imageType = VK_IMAGE_TYPE_2D;
    ImageInfo.format = getVulkanFormat(Desc.Format);
    ImageInfo.extent = {Desc.Width, Desc.Height, 1};
    ImageInfo.mipLevels = Desc.MipLevels;
    ImageInfo.arrayLayers = 1;
    ImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    ImageInfo.usage = getVulkanImageUsage(Desc.Usage);
    ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage Image;
    if (vkCreateImage(Device, &ImageInfo, nullptr, &Image))
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to create image.");

    VkMemoryRequirements MemReqs;
    vkGetImageMemoryRequirements(Device, Image, &MemReqs);

    VkMemoryAllocateInfo AllocInfo = {};
    AllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    AllocInfo.allocationSize = MemReqs.size;
    auto MemIdx = getMemoryIndex(PhysicalDevice, MemReqs.memoryTypeBits,
                                 getVulkanMemoryFlags(Desc.Location));
    if (!MemIdx) {
      vkDestroyImage(Device, Image, nullptr);
      return MemIdx.takeError();
    }
    AllocInfo.memoryTypeIndex = *MemIdx;

    VkDeviceMemory DeviceMemory;
    if (vkAllocateMemory(Device, &AllocInfo, nullptr, &DeviceMemory)) {
      vkDestroyImage(Device, Image, nullptr);
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to allocate image memory.");
    }
    if (vkBindImageMemory(Device, Image, DeviceMemory, 0)) {
      vkDestroyImage(Device, Image, nullptr);
      vkFreeMemory(Device, DeviceMemory, nullptr);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to bind image memory.");
    }

    auto Tex = std::make_shared<VulkanTexture>(Device, Image, DeviceMemory,
                                               Name, Desc);

    const bool IsRT =
        (Desc.Usage & TextureUsage::RenderTarget) != TextureUsage::None;
    const bool IsDS =
        (Desc.Usage & TextureUsage::DepthStencil) != TextureUsage::None;
    if (IsRT || IsDS) {
      VkImageViewCreateInfo ViewCi = {};
      ViewCi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ViewCi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ViewCi.format = getVulkanFormat(Desc.Format);
      ViewCi.subresourceRange.baseMipLevel = 0;
      ViewCi.subresourceRange.levelCount = 1;
      ViewCi.subresourceRange.baseArrayLayer = 0;
      ViewCi.subresourceRange.layerCount = 1;
      ViewCi.image = Image;
      if (IsRT) {
        ViewCi.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                             VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        ViewCi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      } else {
        ViewCi.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
      }
      if (vkCreateImageView(Device, &ViewCi, nullptr, &Tex->View)) {
        // Tex destructor will clean up Image + Memory.
        return llvm::createStringError(std::errc::device_or_resource_busy,
                                       "Failed to create image view.");
      }
    }

    return Tex;
  }

  const Capabilities &getCapabilities() override {
    if (Caps.empty())
      queryCapabilities();
    return Caps;
  }

  void printExtra(llvm::raw_ostream &OS) override {
    OS << "  Layers:\n";
    for (auto &Layer : InstanceLayers) {
      uint64_t Sz = strnlen(Layer.layerName, VK_MAX_EXTENSION_NAME_SIZE);
      OS << "  - LayerName: " << llvm::StringRef(Layer.layerName, Sz) << "\n";
      OS << "    SpecVersion: " << Layer.specVersion << "\n";
      OS << "    ImplVersion: " << Layer.implementationVersion << "\n";
      Sz = strnlen(Layer.description, VK_MAX_DESCRIPTION_SIZE);
      OS << "    LayerDesc: " << llvm::StringRef(Layer.description, Sz) << "\n";
    }

    OS << "  Extensions:\n";
    for (const auto &Ext : DeviceExtensions) {
      OS << "  - ExtensionName: " << llvm::StringRef(Ext.extensionName) << "\n";
      OS << "    SpecVersion: " << Ext.specVersion << "\n";
    }
  }

  const VkPhysicalDeviceProperties &getProps() const { return Props; }

private:
  void queryCapabilities() {

    VkPhysicalDeviceFeatures2 Features{};
    Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceVulkan11Features Features11{};
    Features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features Features12{};
    Features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features Features13{};
    Features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
#ifdef VK_VERSION_1_4
    VkPhysicalDeviceVulkan14Features Features14{};
    Features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
#endif

    Features.pNext = &Features11;
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0))
      Features11.pNext = &Features12;
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0))
      Features12.pNext = &Features13;
#ifdef VK_VERSION_1_4
    if (Props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 4, 0))
      Features13.pNext = &Features14;
#endif
    vkGetPhysicalDeviceFeatures2(PhysicalDevice, &Features);

    Caps.insert(std::make_pair(
        "APIMajorVersion",
        makeCapability<uint32_t>("APIMajorVersion",
                                 VK_API_VERSION_MAJOR(Props.apiVersion))));

    Caps.insert(std::make_pair(
        "APIMinorVersion",
        makeCapability<uint32_t>("APIMinorVersion",
                                 VK_API_VERSION_MINOR(Props.apiVersion))));

#define VULKAN_FLOAT_CONTROLS_FEATURE_BOOL(Name)                               \
  Caps.insert(std::make_pair(                                                  \
      #Name, makeCapability<bool>(#Name, FloatControlProp.Name)));
#define VULKAN_FEATURE_BOOL(Name)                                              \
  Caps.insert(std::make_pair(                                                  \
      #Name, makeCapability<bool>(#Name, Features.features.Name)));
#define VULKAN11_FEATURE_BOOL(Name)                                            \
  Caps.insert(                                                                 \
      std::make_pair(#Name, makeCapability<bool>(#Name, Features11.Name)));
#define VULKAN12_FEATURE_BOOL(Name)                                            \
  Caps.insert(                                                                 \
      std::make_pair(#Name, makeCapability<bool>(#Name, Features12.Name)));
#define VULKAN13_FEATURE_BOOL(Name)                                            \
  Caps.insert(                                                                 \
      std::make_pair(#Name, makeCapability<bool>(#Name, Features13.Name)));
#ifdef VK_VERSION_1_4
#define VULKAN14_FEATURE_BOOL(Name)                                            \
  Caps.insert(                                                                 \
      std::make_pair(#Name, makeCapability<bool>(#Name, Features14.Name)));
#endif
#include "VKFeatures.def"
  }

public:
  llvm::Error createDevice(InvocationState &IS) {
    VkCommandPoolCreateInfo CmdPoolInfo = {};
    CmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    CmdPoolInfo.queueFamilyIndex = GraphicsQueue.QueueFamilyIdx;
    CmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(Device, &CmdPoolInfo, nullptr, &IS.CmdPool))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Could not create command pool.");
    return llvm::Error::success();
  }

  llvm::Error createCommandBuffer(InvocationState &IS) {
    VkCommandBufferAllocateInfo CBufAllocInfo = {};
    CBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    CBufAllocInfo.commandPool = IS.CmdPool;
    CBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    CBufAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(Device, &CBufAllocInfo, &IS.CmdBuffer))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Could not create command buffer.");
    VkCommandBufferBeginInfo BufferInfo = {};
    BufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(IS.CmdBuffer, &BufferInfo))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Could not begin command buffer.");
    return llvm::Error::success();
  }

  llvm::Expected<VkSampler> createVkSampler(const Sampler &S) {
    VkSamplerCreateInfo SamplerInfo = {};
    SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    SamplerInfo.magFilter = getVKFilter(S.MagFilter);
    SamplerInfo.minFilter = getVKFilter(S.MinFilter);
    SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    SamplerInfo.addressModeU = getVKAddressMode(S.Address);
    SamplerInfo.addressModeV = getVKAddressMode(S.Address);
    SamplerInfo.addressModeW = getVKAddressMode(S.Address);
    SamplerInfo.mipLodBias = S.MipLODBias;
    SamplerInfo.anisotropyEnable = VK_FALSE;
    SamplerInfo.maxAnisotropy = 1.0f;
    SamplerInfo.compareEnable =
        S.Kind == SamplerKind::SamplerComparison ? VK_TRUE : VK_FALSE;
    SamplerInfo.compareOp = getVKCompareOp(S.ComparisonOp);
    SamplerInfo.minLod = S.MinLOD;
    SamplerInfo.maxLod = S.MaxLOD;
    SamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    SamplerInfo.unnormalizedCoordinates = VK_FALSE;

    VkSampler Sampler;
    if (vkCreateSampler(Device, &SamplerInfo, nullptr, &Sampler))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create sampler.");
    return Sampler;
  }

  llvm::Expected<std::shared_ptr<BoundResource>>
  createTextureGPUResource(Resource &R, TextureUsage Usage,
                           MemoryBacking Backing, InvocationState &IS) {
    const CPUBuffer &B = *R.BufferPtr;
    auto FmtOrErr = toFormat(B.Format, B.Channels);
    if (!FmtOrErr)
      return FmtOrErr.takeError();

    const char *Name =
        (Usage & TextureUsage::Storage) != TextureUsage::None ? "UAV" : "SRV";

    TextureCreateDesc TexDesc = {};
    TexDesc.Location = MemoryLocation::GpuOnly;
    TexDesc.Backing = Backing;
    TexDesc.Usage = Usage;
    TexDesc.Format = *FmtOrErr;
    TexDesc.Width = B.OutputProps.Width;
    TexDesc.Height = B.OutputProps.Height;
    TexDesc.MipLevels = B.OutputProps.MipLevels;

    auto TexOrErr = createTexture(Name, TexDesc);
    if (!TexOrErr)
      return TexOrErr.takeError();
    auto Tex = std::static_pointer_cast<VulkanTexture>(*TexOrErr);

    auto BR = std::make_shared<BoundResource>();
    BR->Resource = Tex;
    BR->CurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Sampled textures use combined-image-sampler descriptors and need a
    // sampler handle.
    if (R.isSampledTexture()) {
      auto SamplerOrErr = createVkSampler(*R.SamplerPtr);
      if (!SamplerOrErr)
        return SamplerOrErr.takeError();
      BR->Sampler = *SamplerOrErr;
    }

    return BR;
  }

  llvm::Error uploadResourceData(BoundResource &BR, const void *Data,
                                 size_t DataSize, InvocationState &IS) {
    const size_t UploadSize = BR.getSizeInBytes();

    BufferCreateDesc StagingDesc = {};
    StagingDesc.Location = MemoryLocation::CpuToGpu;
    StagingDesc.Backing = MemoryBacking::Automatic;
    StagingDesc.Usage = BufferUsage::None;
    auto StagingOrErr = createBuffer("Upload", StagingDesc, UploadSize);
    if (!StagingOrErr)
      return StagingOrErr.takeError();
    auto Staging = std::static_pointer_cast<VulkanBuffer>(*StagingOrErr);

    // Map, fill, flush.
    void *Mapped = nullptr;
    if (vkMapMemory(Device, Staging->Memory, 0, VK_WHOLE_SIZE, 0, &Mapped))
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to map staging buffer.");
    if (UploadSize > DataSize)
      memset(Mapped, 0, UploadSize);
    memcpy(Mapped, Data, DataSize);

    VkMappedMemoryRange Range = {};
    Range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    Range.memory = Staging->Memory;
    Range.offset = 0;
    Range.size = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(Device, 1, &Range);
    vkUnmapMemory(Device, Staging->Memory);

    if (BR.isTexture()) {
      auto &Tex = *std::get<std::shared_ptr<VulkanTexture>>(BR.Resource);
      copyBufferToTexture(IS.CmdBuffer, Tex, *Staging, BR.CurrentLayout);
      BR.CurrentLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
      auto &Buf = *std::get<std::shared_ptr<VulkanBuffer>>(BR.Resource);
      copyBufferToBuffer(IS.CmdBuffer, Buf, *Staging);
    }

    IS.ResourcesKeepAlive.push_back(Staging);
    return llvm::Error::success();
  }

  llvm::Expected<std::shared_ptr<BoundResource>>
  createBufferGPUResource(Resource &R, BufferUsage Usage, MemoryBacking Backing,
                          InvocationState &IS) {
    const char *Name = "SRV";
    if ((Usage & BufferUsage::Storage) != BufferUsage::None)
      Name = "UAV";
    else if ((Usage & BufferUsage::Constant) != BufferUsage::None)
      Name = "CBV";

    BufferCreateDesc BufDesc = {};
    BufDesc.Location = MemoryLocation::GpuOnly;
    BufDesc.Backing = Backing;
    BufDesc.Usage = Usage;
    auto BufOrErr = createBuffer(Name, BufDesc, R.size());
    if (!BufOrErr)
      return BufOrErr.takeError();
    auto Buf = std::static_pointer_cast<VulkanBuffer>(*BufOrErr);

    auto BR = std::make_shared<BoundResource>();
    BR->Resource = Buf;
    return BR;
  }

  llvm::Expected<ResourceBundle> createDescriptorResource(Resource &R,
                                                          InvocationState &IS) {
    ResourceBundle Bundle;

    // Handle standalone samplers — no backing data, just a VkSampler.
    if (R.isSampler()) {
      auto BR = std::make_shared<BoundResource>();
      // Standalone sampler: no texture/buffer resource, just sampler handle.
      // Use a null VulkanTexture placeholder.
      BR->Resource = std::shared_ptr<VulkanTexture>(nullptr);

      auto SamplerOrErr = createVkSampler(*R.SamplerPtr);
      if (!SamplerOrErr)
        return SamplerOrErr.takeError();
      BR->Sampler = *SamplerOrErr;
      Bundle.push_back(BR);
      return Bundle;
    }

    // Map resource kind to usage.
    const bool IsReadWrite = R.isReadWrite();
    TextureUsage TexUsage =
        IsReadWrite ? TextureUsage::Storage : TextureUsage::Sampled;
    BufferUsage BufUsage =
        IsReadWrite ? BufferUsage::Storage : BufferUsage::Sampled;
    if (R.Kind == ResourceKind::ConstantBuffer)
      BufUsage = BufferUsage::Constant;

    uint32_t ArrayIndex = 0;
    for (const auto &ResData : R.BufferPtr->Data) {
      // Create the GPU resource.
      auto BROrErr =
          R.isTexture() ? createTextureGPUResource(R, TexUsage,
                                                   MemoryBacking::Automatic, IS)
                        : createBufferGPUResource(R, BufUsage,
                                                  MemoryBacking::Automatic, IS);
      if (!BROrErr)
        return BROrErr.takeError();
      auto BR = *BROrErr;

      // Upload initial data.
      if (auto Err = uploadResourceData(*BR, ResData.get(), R.size(), IS))
        return Err;

      // For read-write resources, create a readback binding.
      if (IsReadWrite) {
        BufferCreateDesc ReadbackDesc = {};
        ReadbackDesc.Location = MemoryLocation::GpuToCpu;
        ReadbackDesc.Backing = MemoryBacking::Automatic;
        ReadbackDesc.Usage = BufferUsage::None;
        auto ReadbackOrErr =
            createBuffer("Readback", ReadbackDesc, BR->getSizeInBytes());
        if (!ReadbackOrErr)
          return ReadbackOrErr.takeError();

        ReadbackBinding RB;
        RB.Source = BR;
        RB.Destination = std::static_pointer_cast<VulkanBuffer>(*ReadbackOrErr);
        RB.PipelineResource = &R;
        RB.ArrayIndex = ArrayIndex;

        // Counter resources (VK-specific: separate buffer per counter).
        if (R.HasCounter) {
          BufferCreateDesc CounterDevDesc = {};
          CounterDevDesc.Location = MemoryLocation::GpuOnly;
          CounterDevDesc.Backing = MemoryBacking::Automatic;
          CounterDevDesc.Usage = BufferUsage::Storage;
          auto CounterDevOrErr =
              createBuffer("Counter", CounterDevDesc, sizeof(uint32_t));
          if (!CounterDevOrErr)
            return CounterDevOrErr.takeError();
          auto CounterDev =
              std::static_pointer_cast<VulkanBuffer>(*CounterDevOrErr);

          // Upload initial counter value (0) via staging buffer.
          BufferCreateDesc CounterStagingDesc = {};
          CounterStagingDesc.Location = MemoryLocation::CpuToGpu;
          CounterStagingDesc.Backing = MemoryBacking::Automatic;
          CounterStagingDesc.Usage = BufferUsage::None;
          auto CounterStagingOrErr = createBuffer(
              "CounterUpload", CounterStagingDesc, sizeof(uint32_t));
          if (!CounterStagingOrErr)
            return CounterStagingOrErr.takeError();
          auto CounterStaging =
              std::static_pointer_cast<VulkanBuffer>(*CounterStagingOrErr);

          // Map and zero-fill the staging buffer.
          void *CounterMapped = nullptr;
          vkMapMemory(Device, CounterStaging->Memory, 0, sizeof(uint32_t), 0,
                      &CounterMapped);
          memset(CounterMapped, 0, sizeof(uint32_t));
          VkMappedMemoryRange CounterRange = {};
          CounterRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
          CounterRange.memory = CounterStaging->Memory;
          CounterRange.size = VK_WHOLE_SIZE;
          vkFlushMappedMemoryRanges(Device, 1, &CounterRange);
          vkUnmapMemory(Device, CounterStaging->Memory);

          VkBufferCopy Copy = {};
          Copy.size = sizeof(uint32_t);
          vkCmdCopyBuffer(IS.CmdBuffer, CounterStaging->Buffer,
                          CounterDev->Buffer, 1, &Copy);
          IS.ResourcesKeepAlive.push_back(CounterStaging);

          BR->CounterBuffer = CounterDev;

          BufferCreateDesc CounterReadbackDesc = {};
          CounterReadbackDesc.Location = MemoryLocation::GpuToCpu;
          CounterReadbackDesc.Backing = MemoryBacking::Automatic;
          CounterReadbackDesc.Usage = BufferUsage::None;
          auto CounterReadbackOrErr = createBuffer(
              "CounterReadback", CounterReadbackDesc, sizeof(uint32_t));
          if (!CounterReadbackOrErr)
            return CounterReadbackOrErr.takeError();
          auto CounterReadback =
              std::static_pointer_cast<VulkanBuffer>(*CounterReadbackOrErr);

          RB.Counter = CounterBinding{CounterDev, CounterReadback};
        }

        IS.ReadbackBindings.push_back(RB);
      }

      Bundle.push_back(BR);
      ArrayIndex++;
    }
    return Bundle;
  }

  static void copyReadbackBindings(InvocationState &IS) {
    for (const auto &RB : IS.ReadbackBindings) {
      if (RB.Source->isTexture()) {
        auto &Tex =
            *std::get<std::shared_ptr<VulkanTexture>>(RB.Source->Resource);

        // Barrier: compute shader → transfer read.
        VkImageMemoryBarrier ImageBarrier = {};
        ImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ImageBarrier.subresourceRange = getImageSubresourceRange(Tex.Desc);
        ImageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ImageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ImageBarrier.oldLayout = RB.Source->CurrentLayout;
        ImageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ImageBarrier.image = Tex.Image;
        vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &ImageBarrier);

        copyTextureToBuffer(IS.CmdBuffer, *RB.Destination, Tex);
      } else {
        auto &Buf =
            *std::get<std::shared_ptr<VulkanBuffer>>(RB.Source->Resource);

        // Barrier: compute shader → transfer read.
        VkBufferMemoryBarrier Barrier = {};
        Barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        Barrier.size = VK_WHOLE_SIZE;
        Barrier.srcAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        Barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        Barrier.buffer = Buf.Buffer;
        vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                             &Barrier, 0, nullptr);

        VkBufferCopy Copy = {};
        Copy.size = RB.Source->getSizeInBytes();
        vkCmdCopyBuffer(IS.CmdBuffer, Buf.Buffer, RB.Destination->Buffer, 1,
                        &Copy);
      }

      // Counter readback.
      if (RB.Counter) {
        VkBufferMemoryBarrier Barrier = {};
        Barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        Barrier.size = VK_WHOLE_SIZE;
        Barrier.srcAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        Barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        Barrier.buffer = RB.Counter->Source->Buffer;
        vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                             &Barrier, 0, nullptr);

        VkBufferCopy CounterCopy = {};
        CounterCopy.size = sizeof(uint32_t);
        vkCmdCopyBuffer(IS.CmdBuffer, RB.Counter->Source->Buffer,
                        RB.Counter->Destination->Buffer, 1, &CounterCopy);
      }

      // Barrier: transfer write → host read on destination buffers.
      VkBufferMemoryBarrier HostBarrier = {};
      HostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      HostBarrier.size = VK_WHOLE_SIZE;
      HostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      HostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      HostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      HostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      HostBarrier.buffer = RB.Destination->Buffer;
      vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                           &HostBarrier, 0, nullptr);

      if (RB.Counter) {
        HostBarrier.buffer = RB.Counter->Destination->Buffer;
        vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                             &HostBarrier, 0, nullptr);
      }
    }
  }

  llvm::Error createRenderTarget(Pipeline &P, InvocationState &IS) {
    if (!P.Bindings.RTargetBufferPtr)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "No render target bound for graphics pipeline.");
    const CPUBuffer &RTBuf = *P.Bindings.RTargetBufferPtr;

    auto TexOrErr = offloadtest::createRenderTargetFromCPUBuffer(*this, RTBuf);
    if (!TexOrErr)
      return TexOrErr.takeError();

    IS.RenderTarget = std::static_pointer_cast<VulkanTexture>(*TexOrErr);

    // Create a host-visible staging buffer for readback.
    BufferCreateDesc BufDesc = {};
    BufDesc.Location = MemoryLocation::GpuToCpu;
    BufDesc.Backing = MemoryBacking::Automatic;
    BufDesc.Usage = BufferUsage::None;
    auto BufOrErr = createBuffer("RTReadback", BufDesc, RTBuf.size());
    if (!BufOrErr)
      return BufOrErr.takeError();
    IS.RTReadback = std::static_pointer_cast<VulkanBuffer>(*BufOrErr);

    return llvm::Error::success();
  }

  llvm::Error createDepthStencil(Pipeline &P, InvocationState &IS) {
    auto TexOrErr = offloadtest::createDefaultDepthStencilTarget(
        *this, IS.RenderTarget->Desc.Width, IS.RenderTarget->Desc.Height);
    if (!TexOrErr)
      return TexOrErr.takeError();
    IS.DepthStencil = std::static_pointer_cast<VulkanTexture>(*TexOrErr);
    return llvm::Error::success();
  }

  llvm::Error createResources(Pipeline &P, InvocationState &IS) {
    for (auto &D : P.Sets) {
      for (auto &R : D.Resources) {
        auto ExRes = createDescriptorResource(R, IS);
        if (!ExRes)
          return ExRes.takeError();
        IS.Resources.push_back(std::make_pair(&R, *ExRes));
      }
    }

    if (P.isGraphics()) {
      if (auto Err = createRenderTarget(P, IS))
        return Err;
      // TODO: Always created for graphics pipelines. Consider making this
      // conditional on the pipeline definition.
      if (auto Err = createDepthStencil(P, IS))
        return Err;

      if (!P.Bindings.VertexBufferPtr)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "No vertex buffer specified for graphics pipeline.");

      const ParsedVertexBuffer &PVB = *P.Bindings.VertexBufferPtr;

      BufferCreateDesc VBBufDesc = {};
      VBBufDesc.Location = MemoryLocation::CpuToGpu;
      VBBufDesc.Backing = MemoryBacking::Automatic;
      VBBufDesc.Usage = BufferUsage::VertexBuffer;
      auto BufOrErr =
          createBuffer("VertexBuffer", VBBufDesc, PVB.InterleavedSize);
      if (!BufOrErr)
        return BufOrErr.takeError();

      VertexBufferDesc VBDesc;
      for (const auto &S : PVB.Streams)
        VBDesc.Streams.push_back({S.Name, S.Fmt});

      IS.VB = offloadtest::VertexBuffer{VBDesc, *BufOrErr};

      // TODO: Currently uses a single CpuToGpu mapped buffer. For optimal GPU
      // performance on discrete GPUs, use a staging buffer + copy to a GpuOnly
      // vertex buffer instead.
      auto *VKBuf = static_cast<VulkanBuffer *>(IS.VB->Data.get());
      const size_t BufSize = IS.VB->Data->getSizeInBytes();
      void *Mapped = nullptr;
      vkMapMemory(Device, VKBuf->Memory, 0, BufSize, 0, &Mapped);
      memcpy(Mapped, PVB.InterleavedData.get(), BufSize);
      vkUnmapMemory(Device, VKBuf->Memory);
    }

    return llvm::Error::success();
  }

  llvm::Error executeCommandBuffer(InvocationState &IS) {
    if (vkEndCommandBuffer(IS.CmdBuffer))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Could not end command buffer.");

    VkSubmitInfo SubmitInfo = {};
    SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SubmitInfo.commandBufferCount = 1;
    SubmitInfo.pCommandBuffers = &IS.CmdBuffer;
    VkFenceCreateInfo FenceInfo = {};
    FenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence Fence;
    if (vkCreateFence(Device, &FenceInfo, nullptr, &Fence))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Could not create fence.");

    // Submit to the queue
    if (vkQueueSubmit(GraphicsQueue.Queue, 1, &SubmitInfo, Fence))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to submit to queue.");
    if (vkWaitForFences(Device, 1, &Fence, VK_TRUE, UINT64_MAX))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed waiting for fence.");

    vkDestroyFence(Device, Fence, nullptr);
    vkFreeCommandBuffers(Device, IS.CmdPool, 1, &IS.CmdBuffer);
    return llvm::Error::success();
  }

  llvm::Error createDescriptorPool(Pipeline &P, InvocationState &IS) {

    constexpr VkDescriptorType DescriptorTypes[] = {
        VK_DESCRIPTOR_TYPE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    constexpr size_t DescriptorTypesSize =
        sizeof(DescriptorTypes) / sizeof(VkDescriptorType);
    uint32_t DescriptorCounts[DescriptorTypesSize] = {0};
    for (const auto &S : P.Sets) {
      for (const auto &R : S.Resources) {
        DescriptorCounts[getDescriptorType(R.Kind)] += R.getArraySize();
        if (R.HasCounter)
          DescriptorCounts[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] +=
              R.getArraySize();
      }
    }
    llvm::SmallVector<VkDescriptorPoolSize> PoolSizes;
    for (const VkDescriptorType Type : DescriptorTypes) {
      if (DescriptorCounts[Type] > 0) {
        llvm::outs() << "Descriptors: { type = " << Type
                     << ", count = " << DescriptorCounts[Type] << " }\n";
        VkDescriptorPoolSize PoolSize = {};
        PoolSize.type = Type;
        PoolSize.descriptorCount = DescriptorCounts[Type];
        PoolSizes.push_back(PoolSize);
      }
    }

    if (P.Sets.size() > 0) {
      VkDescriptorPoolCreateInfo PoolCreateInfo = {};
      PoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      PoolCreateInfo.poolSizeCount = PoolSizes.size();
      PoolCreateInfo.pPoolSizes = PoolSizes.data();
      PoolCreateInfo.maxSets = P.Sets.size();
      if (vkCreateDescriptorPool(Device, &PoolCreateInfo, nullptr, &IS.Pool))
        return llvm::createStringError(std::errc::device_or_resource_busy,
                                       "Failed to create descriptor pool.");
    }
    return llvm::Error::success();
  }

  llvm::Error createDescriptorSets(Pipeline &P, InvocationState &IS) {
    for (const auto &S : P.Sets) {
      std::vector<VkDescriptorSetLayoutBinding> Bindings;
      for (const auto &R : S.Resources) {
        VkDescriptorSetLayoutBinding Binding = {};
        if (!R.VKBinding.has_value())
          return llvm::createStringError(std::errc::invalid_argument,
                                         "No VulkanBinding provided for '%s'",
                                         R.Name.c_str());
        if (R.HasCounter && !R.VKBinding->CounterBinding)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "No CounterBinding provided for resource '%s' with a counter",
              R.Name.c_str());
        Binding.binding = R.VKBinding->Binding;
        Binding.descriptorType = getDescriptorType(R.Kind);
        Binding.descriptorCount = R.getArraySize();
        Binding.stageFlags = IS.getFullShaderStageMask();
        Bindings.push_back(Binding);
        if (R.HasCounter) {
          VkDescriptorSetLayoutBinding CounterBinding = {};
          CounterBinding.binding = *R.VKBinding->CounterBinding;
          CounterBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          CounterBinding.descriptorCount = R.getArraySize();
          CounterBinding.stageFlags = IS.getFullShaderStageMask();
          Bindings.push_back(CounterBinding);
        }
      }
      VkDescriptorSetLayoutCreateInfo LayoutCreateInfo = {};
      LayoutCreateInfo.sType =
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      LayoutCreateInfo.bindingCount = Bindings.size();
      LayoutCreateInfo.pBindings = Bindings.data();
      llvm::outs() << "Binding " << Bindings.size() << " descriptors.\n";
      VkDescriptorSetLayout Layout;
      if (vkCreateDescriptorSetLayout(Device, &LayoutCreateInfo, nullptr,
                                      &Layout))
        return llvm::createStringError(
            std::errc::device_or_resource_busy,
            "Failed to create descriptor set layout.");
      IS.DescriptorSetLayouts.push_back(Layout);
    }

    VkPipelineLayoutCreateInfo PipelineCreateInfo = {};
    PipelineCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    PipelineCreateInfo.setLayoutCount = IS.DescriptorSetLayouts.size();
    PipelineCreateInfo.pSetLayouts = IS.DescriptorSetLayouts.data();

    llvm::SmallVector<VkPushConstantRange, 1> Ranges;
    for (const auto &PCB : P.PushConstants) {
      const VkPushConstantRange R = {
          static_cast<VkShaderStageFlags>(getShaderStageFlag(PCB.Stage)),
          /* offset= */ 0, static_cast<uint32_t>(PCB.size())};
      Ranges.emplace_back(std::move(R));
    }
    PipelineCreateInfo.pushConstantRangeCount = Ranges.size();
    PipelineCreateInfo.pPushConstantRanges = Ranges.data();

    if (vkCreatePipelineLayout(Device, &PipelineCreateInfo, nullptr,
                               &IS.PipelineLayout))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create pipeline layout.");

    if (P.Sets.size() == 0)
      return llvm::Error::success();

    VkDescriptorSetAllocateInfo DSAllocInfo = {};
    DSAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    DSAllocInfo.descriptorPool = IS.Pool;
    DSAllocInfo.descriptorSetCount = IS.DescriptorSetLayouts.size();
    DSAllocInfo.pSetLayouts = IS.DescriptorSetLayouts.data();
    assert(IS.DescriptorSets.empty());
    IS.DescriptorSets.insert(IS.DescriptorSets.begin(),
                             IS.DescriptorSetLayouts.size(), VkDescriptorSet());
    llvm::outs() << "Num Descriptor sets: " << IS.DescriptorSetLayouts.size()
                 << "\n";
    if (vkAllocateDescriptorSets(Device, &DSAllocInfo,
                                 IS.DescriptorSets.data()))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to allocate descriptor sets.");

    // Calculate the number of infos/views we are going to need for each type
    uint32_t ImageInfoCount = 0;
    uint32_t BufferInfoCount = 0;
    uint32_t BufferViewCount = 0;
    for (auto &D : P.Sets) {
      for (auto &R : D.Resources) {
        if (R.isSampler()) {
          ImageInfoCount += 1;
          continue;
        }
        const uint32_t Count = R.getArraySize();
        if (R.isTexture())
          ImageInfoCount += Count;
        else if (R.isRaw())
          BufferInfoCount += Count;
        else
          BufferViewCount += Count;
        if (R.HasCounter)
          BufferInfoCount += Count;
      }
    }

    // reserve enough space for the descriptor infos so it never needs to be
    // resized (we need the memory fixed in place)
    llvm::SmallVector<VkDescriptorImageInfo> ImageInfos;
    llvm::SmallVector<VkDescriptorBufferInfo> BufferInfos;
    llvm::SmallVector<VkBufferView> BufferViews;
    ImageInfos.reserve(ImageInfoCount);
    BufferInfos.reserve(BufferInfoCount);
    BufferViews.reserve(BufferViewCount);

    llvm::SmallVector<VkWriteDescriptorSet> WriteDescriptors;
    WriteDescriptors.reserve(ImageInfoCount + BufferInfoCount +
                             BufferViewCount);
    assert(IS.BufferViews.empty());

    uint32_t OverallResIdx = 0;
    for (uint32_t SetIdx = 0; SetIdx < P.Sets.size(); ++SetIdx) {
      for (uint32_t RIdx = 0; RIdx < P.Sets[SetIdx].Resources.size();
           ++RIdx, ++OverallResIdx) {
        const Resource &R = P.Sets[SetIdx].Resources[RIdx];
        const ResourceBundle &Bundle = IS.Resources[OverallResIdx].second;
        uint32_t IndexOfFirstInArray;

        if (R.isSampler()) {
          IndexOfFirstInArray = ImageInfos.size();
          for (const auto &BR : Bundle) {
            const VkDescriptorImageInfo ImageInfo = {
                BR->Sampler, VK_NULL_HANDLE,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            ImageInfos.push_back(ImageInfo);
          }
        } else if (R.isTexture()) {
          IndexOfFirstInArray = ImageInfos.size();
          for (const auto &BR : Bundle) {
            auto &Tex = *std::get<std::shared_ptr<VulkanTexture>>(BR->Resource);
            const TextureCreateDesc &Desc = Tex.Desc;

            VkImageViewCreateInfo ViewCreateInfo = {};
            ViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ViewCreateInfo.viewType = getImageViewType(R.Kind);
            ViewCreateInfo.format = getVulkanFormat(Desc.Format);
            ViewCreateInfo.components = {
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
            ViewCreateInfo.subresourceRange.aspectMask =
                isDepthFormat(Desc.Format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
            ViewCreateInfo.subresourceRange.baseMipLevel = 0;
            ViewCreateInfo.subresourceRange.baseArrayLayer = 0;
            ViewCreateInfo.subresourceRange.layerCount = 1;
            ViewCreateInfo.subresourceRange.levelCount = Desc.MipLevels;
            ViewCreateInfo.image = Tex.Image;

            VkImageView View = VK_NULL_HANDLE;
            if (vkCreateImageView(Device, &ViewCreateInfo, nullptr, &View))
              return llvm::createStringError(std::errc::device_or_resource_busy,
                                             "Failed to create image view.");
            IS.ImageViews.push_back(View);
            const VkDescriptorImageInfo ImageInfo = {BR->Sampler, View,
                                                     VK_IMAGE_LAYOUT_GENERAL};
            ImageInfos.push_back(ImageInfo);
          }
        } else if (R.isRaw()) {
          IndexOfFirstInArray = BufferInfos.size();
          for (const auto &BR : Bundle) {
            const VkDescriptorBufferInfo BI = {BR->getBuffer(), 0,
                                               VK_WHOLE_SIZE};
            BufferInfos.push_back(BI);
          }
        } else {
          // Typed buffer (texel buffer view).
          VkBufferViewCreateInfo ViewCreateInfo = {};
          ViewCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
          ViewCreateInfo.format =
              getVKFormat(R.BufferPtr->Format, R.BufferPtr->Channels);
          ViewCreateInfo.range = VK_WHOLE_SIZE;
          IndexOfFirstInArray = BufferViews.size();
          for (const auto &BR : Bundle) {
            ViewCreateInfo.buffer = BR->getBuffer();
            VkBufferView View = VK_NULL_HANDLE;
            if (vkCreateBufferView(Device, &ViewCreateInfo, nullptr, &View))
              return llvm::createStringError(std::errc::device_or_resource_busy,
                                             "Failed to create buffer view.");
            IS.BufferViews.push_back(View);
            BufferViews.push_back(View);
          }
        }

        VkWriteDescriptorSet WDS = {};
        WDS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WDS.dstSet = IS.DescriptorSets[SetIdx];
        WDS.dstBinding = R.VKBinding->Binding;
        WDS.descriptorCount = R.getArraySize();
        WDS.descriptorType = getDescriptorType(R.Kind);
        if (R.isTexture() || R.isSampler())
          WDS.pImageInfo = &ImageInfos[IndexOfFirstInArray];
        else if (R.isRaw())
          WDS.pBufferInfo = &BufferInfos[IndexOfFirstInArray];
        else
          WDS.pTexelBufferView = &BufferViews[IndexOfFirstInArray];
        llvm::outs() << "Updating Descriptor [" << OverallResIdx << "] { "
                     << SetIdx << ", " << RIdx << " }\n";
        WriteDescriptors.push_back(WDS);

        if (R.HasCounter) {
          IndexOfFirstInArray = BufferInfos.size();
          for (const auto &BR : Bundle) {
            const VkDescriptorBufferInfo BI = {BR->CounterBuffer->Buffer, 0,
                                               VK_WHOLE_SIZE};
            BufferInfos.push_back(BI);
          }

          VkWriteDescriptorSet CounterWDS = {};
          CounterWDS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          CounterWDS.dstSet = IS.DescriptorSets[SetIdx];
          CounterWDS.dstBinding = *R.VKBinding->CounterBinding;
          CounterWDS.descriptorCount = R.getArraySize();
          CounterWDS.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          CounterWDS.pBufferInfo = &BufferInfos[IndexOfFirstInArray];
          llvm::outs() << "Updating Counter Descriptor [" << OverallResIdx
                       << "] { " << SetIdx << ", " << RIdx << " }\n";
          llvm::outs() << "Binding = " << CounterWDS.dstBinding << "\n";
          WriteDescriptors.push_back(CounterWDS);
        }
      }
    }
    assert(ImageInfos.size() == ImageInfoCount &&
           BufferInfos.size() == BufferInfoCount &&
           BufferViews.size() == BufferViewCount &&
           "size of buffer infos does not match expected count");

    llvm::outs() << "WriteDescriptors: " << WriteDescriptors.size() << "\n";
    vkUpdateDescriptorSets(Device, WriteDescriptors.size(),
                           WriteDescriptors.data(), 0, nullptr);
    return llvm::Error::success();
  }

  llvm::Error createShaderModules(Pipeline &P, InvocationState &IS) {
    for (const auto &Shader : P.Shaders) {
      const llvm::StringRef Program = Shader.Shader->getBuffer();
      VkShaderModuleCreateInfo ShaderCreateInfo = {};
      ShaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      ShaderCreateInfo.codeSize = Program.size();
      ShaderCreateInfo.pCode =
          reinterpret_cast<const uint32_t *>(Program.data());
      CompiledShader CS = {Shader.Stage, Shader.Entry, 0};
      if (vkCreateShaderModule(Device, &ShaderCreateInfo, nullptr, &CS.Shader))
        return llvm::createStringError(std::errc::not_supported,
                                       "Failed to create shader module.");
      IS.Shaders.emplace_back(CS);
    }
    return llvm::Error::success();
  }

  llvm::Error createRenderPass(InvocationState &IS) {
    std::array<VkAttachmentDescription, 2> Attachments = {};

    Attachments[0].format = getVulkanFormat(IS.RenderTarget->Desc.Format);
    Attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    Attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    Attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    Attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    Attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    Attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    Attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    Attachments[1].format = getVulkanFormat(IS.DepthStencil->Desc.Format);
    Attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    Attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    Attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    Attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    Attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    Attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    Attachments[1].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference ColorReference = {};
    ColorReference.attachment = 0;
    ColorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference DepthReference = {};
    DepthReference.attachment = 1;
    DepthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription SubpassDescription = {};
    SubpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    SubpassDescription.colorAttachmentCount = 1;
    SubpassDescription.pColorAttachments = &ColorReference;
    SubpassDescription.pDepthStencilAttachment = &DepthReference;
    SubpassDescription.inputAttachmentCount = 0;
    SubpassDescription.pInputAttachments = nullptr;
    SubpassDescription.preserveAttachmentCount = 0;
    SubpassDescription.pPreserveAttachments = nullptr;
    SubpassDescription.pResolveAttachments = nullptr;

    std::array<VkSubpassDependency, 2> Dependencies = {};

    Dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    Dependencies[0].dstSubpass = 0;
    Dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    Dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    Dependencies[0].srcAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    Dependencies[0].dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    Dependencies[0].dependencyFlags = 0;

    Dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    Dependencies[1].dstSubpass = 0;
    Dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    Dependencies[1].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    Dependencies[1].srcAccessMask = 0;
    Dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    Dependencies[1].dependencyFlags = 0;

    VkRenderPassCreateInfo RPCI = {};
    RPCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    RPCI.attachmentCount = static_cast<uint32_t>(Attachments.size());
    RPCI.pAttachments = Attachments.data();
    RPCI.subpassCount = 1;
    RPCI.pSubpasses = &SubpassDescription;
    RPCI.dependencyCount = static_cast<uint32_t>(Dependencies.size());
    RPCI.pDependencies = Dependencies.data();

    if (vkCreateRenderPass(Device, &RPCI, nullptr, &IS.RenderPass))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create render pass.");
    return llvm::Error::success();
  }

  llvm::Error createFrameBuffer(InvocationState &IS) {
    std::array<VkImageView, 2> Views = {IS.RenderTarget->View,
                                        IS.DepthStencil->View};

    VkFramebufferCreateInfo FbufCreateInfo = {};
    FbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    FbufCreateInfo.renderPass = IS.RenderPass;
    FbufCreateInfo.attachmentCount = Views.size();
    FbufCreateInfo.pAttachments = Views.data();
    FbufCreateInfo.width = IS.RenderTarget->Desc.Width;
    FbufCreateInfo.height = IS.RenderTarget->Desc.Height;
    FbufCreateInfo.layers = 1;

    if (vkCreateFramebuffer(Device, &FbufCreateInfo, nullptr, &IS.FrameBuffer))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create frame buffer.");
    return llvm::Error::success();
  }

  static llvm::Error
  parseSpecializationConstant(const SpecializationConstant &SpecConst,
                              VkSpecializationMapEntry &Entry,
                              llvm::SmallVector<char> &SpecData) {
    Entry.constantID = SpecConst.ConstantID;
    Entry.offset = SpecData.size();
    switch (SpecConst.Type) {
    case DataFormat::Float32: {
      float Value = 0.0f;
      double Tmp = 0.0;
      if (llvm::StringRef(SpecConst.Value).getAsDouble(Tmp))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid float value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Value = static_cast<float>(Tmp);
      Entry.size = sizeof(float);
      SpecData.resize(SpecData.size() + sizeof(float));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(float));
      break;
    }
    case DataFormat::Float64: {
      double Value = 0.0;
      if (llvm::StringRef(SpecConst.Value).getAsDouble(Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid double value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(double);
      SpecData.resize(SpecData.size() + sizeof(double));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(double));
      break;
    }
    case DataFormat::Int16: {
      int16_t Value = 0;
      if (llvm::StringRef(SpecConst.Value).getAsInteger(0, Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid int16 value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(int16_t);
      SpecData.resize(SpecData.size() + sizeof(int16_t));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(int16_t));
      break;
    }
    case DataFormat::UInt16: {
      uint16_t Value = 0;
      if (llvm::StringRef(SpecConst.Value).getAsInteger(0, Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid uint16 value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(uint16_t);
      SpecData.resize(SpecData.size() + sizeof(uint16_t));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(uint16_t));
      break;
    }
    case DataFormat::Int32: {
      int32_t Value = 0;
      if (llvm::StringRef(SpecConst.Value).getAsInteger(0, Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid int32 value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(int32_t);
      SpecData.resize(SpecData.size() + sizeof(int32_t));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(int32_t));
      break;
    }
    case DataFormat::UInt32: {
      uint32_t Value = 0;
      if (llvm::StringRef(SpecConst.Value).getAsInteger(0, Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid uint32 value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(uint32_t);
      SpecData.resize(SpecData.size() + sizeof(uint32_t));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(uint32_t));
      break;
    }
    case DataFormat::Bool: {
      bool Value = false;
      if (llvm::StringRef(SpecConst.Value).getAsInteger(0, Value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Invalid bool value for specialization constant '%s'",
            SpecConst.Value.c_str());
      Entry.size = sizeof(bool);
      SpecData.resize(SpecData.size() + sizeof(bool));
      memcpy(SpecData.data() + Entry.offset, &Value, sizeof(bool));
      break;
    }
    default:
      llvm_unreachable("Unsupported specialization constant type");
    }
    return llvm::Error::success();
  }

  llvm::Error createPipeline(Pipeline &P, InvocationState &IS) {
    VkPipelineCacheCreateInfo CacheCreateInfo = {};
    CacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (vkCreatePipelineCache(Device, &CacheCreateInfo, nullptr,
                              &IS.PipelineCache))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create pipeline cache.");

    if (P.isCompute()) {
      const offloadtest::Shader &Shader = P.Shaders[0];
      assert(IS.Shaders.size() == 1 &&
             "Currently only support one compute shader");
      const CompiledShader &S = IS.Shaders[0];
      VkPipelineShaderStageCreateInfo StageInfo = {};
      StageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      StageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      StageInfo.module = S.Shader;
      StageInfo.pName = S.Entry.c_str();

      llvm::SmallVector<VkSpecializationMapEntry> SpecEntries;
      llvm::SmallVector<char> SpecData;
      VkSpecializationInfo SpecInfo = {};
      if (!Shader.SpecializationConstants.empty()) {
        llvm::DenseSet<uint32_t> SeenConstantIDs;
        for (const auto &SpecConst : Shader.SpecializationConstants) {
          if (!SeenConstantIDs.insert(SpecConst.ConstantID).second)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "Test configuration contains multiple entries for "
                "specialization constant ID %u.",
                SpecConst.ConstantID);

          VkSpecializationMapEntry Entry;
          if (auto Err =
                  parseSpecializationConstant(SpecConst, Entry, SpecData))
            return Err;
          SpecEntries.push_back(Entry);
        }

        SpecInfo.mapEntryCount = SpecEntries.size();
        SpecInfo.pMapEntries = SpecEntries.data();
        SpecInfo.dataSize = SpecData.size();
        SpecInfo.pData = SpecData.data();
        StageInfo.pSpecializationInfo = &SpecInfo;
      }

      VkComputePipelineCreateInfo PipelineCreateInfo = {};
      PipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      PipelineCreateInfo.stage = StageInfo;
      PipelineCreateInfo.layout = IS.PipelineLayout;
      if (vkCreateComputePipelines(Device, IS.PipelineCache, 1,
                                   &PipelineCreateInfo, nullptr, &IS.Pipeline))
        return llvm::createStringError(std::errc::device_or_resource_busy,
                                       "Failed to create pipeline.");
      return llvm::Error::success();
    }

    llvm::SmallVector<VkPipelineShaderStageCreateInfo> Stages;
    for (const auto &S : IS.Shaders) {
      VkPipelineShaderStageCreateInfo StageInfo = {};
      StageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      StageInfo.stage = getShaderStageFlag(S.Stage);
      StageInfo.module = S.Shader;
      StageInfo.pName = S.Entry.c_str();
      Stages.emplace_back(StageInfo);
    }

    VkPipelineInputAssemblyStateCreateInfo InputAssemblyCI = {};
    InputAssemblyCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    InputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo RastStateCI = {};
    RastStateCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    RastStateCI.polygonMode = VK_POLYGON_MODE_FILL;
    RastStateCI.cullMode = VK_CULL_MODE_NONE;
    RastStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    RastStateCI.depthClampEnable = VK_FALSE;
    RastStateCI.rasterizerDiscardEnable = VK_FALSE;
    RastStateCI.depthBiasEnable = VK_FALSE;
    RastStateCI.lineWidth = 1.0f;

    VkPipelineColorBlendAttachmentState BlendState = {};
    BlendState.colorWriteMask = 0xf;
    BlendState.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo BlendStateCI = {};
    BlendStateCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    BlendStateCI.attachmentCount = 1;
    BlendStateCI.pAttachments = &BlendState;

    VkPipelineViewportStateCreateInfo ViewStateCI = {};
    ViewStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    ViewStateCI.viewportCount = 1;
    ViewStateCI.scissorCount = 1;

    const VkDynamicState DynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo DynamicStateCI = {};
    DynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    DynamicStateCI.pDynamicStates = &DynamicStates[0];
    DynamicStateCI.dynamicStateCount = 2;

    VkPipelineDepthStencilStateCreateInfo DepthStencilStateCI = {};
    DepthStencilStateCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    DepthStencilStateCI.depthTestEnable = VK_TRUE;
    DepthStencilStateCI.depthWriteEnable = VK_TRUE;
    DepthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    DepthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
    DepthStencilStateCI.back.failOp = VK_STENCIL_OP_KEEP;
    DepthStencilStateCI.back.passOp = VK_STENCIL_OP_KEEP;
    DepthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;
    DepthStencilStateCI.stencilTestEnable = VK_FALSE;
    DepthStencilStateCI.front = DepthStencilStateCI.back;

    VkPipelineMultisampleStateCreateInfo MultisampleStateCI = {};
    MultisampleStateCI.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    MultisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Build vertex input state from the vertex buffer description.
    if (!IS.VB)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Vertex buffer not initialized.");
    const VertexBufferDesc &VBDesc = IS.VB->Desc;

    VkVertexInputBindingDescription VertexInputBinding{};
    VertexInputBinding.binding = 0;
    VertexInputBinding.stride = VBDesc.getStride();
    VertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    llvm::SmallVector<VkVertexInputAttributeDescription> Attributes;
    for (uint32_t I = 0; I < VBDesc.Streams.size(); ++I) {
      const VertexStream &S = VBDesc.Streams[I];
      VkVertexInputAttributeDescription VkVA = {};
      VkVA.location = I;
      VkVA.binding = 0;
      VkVA.format = getVulkanFormat(S.Fmt);
      VkVA.offset = VBDesc.getOffset(I);
      Attributes.push_back(VkVA);
    }

    VkPipelineVertexInputStateCreateInfo VertexInputStateCi = {};
    VertexInputStateCi.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VertexInputStateCi.vertexBindingDescriptionCount = 1;
    VertexInputStateCi.pVertexBindingDescriptions = &VertexInputBinding;
    VertexInputStateCi.vertexAttributeDescriptionCount = Attributes.size();
    VertexInputStateCi.pVertexAttributeDescriptions = Attributes.data();

    VkGraphicsPipelineCreateInfo PipelineCreateInfo = {};
    PipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    PipelineCreateInfo.stageCount = Stages.size();
    PipelineCreateInfo.pStages = Stages.data();
    PipelineCreateInfo.pVertexInputState = &VertexInputStateCi;
    PipelineCreateInfo.pInputAssemblyState = &InputAssemblyCI;
    PipelineCreateInfo.pRasterizationState = &RastStateCI;
    PipelineCreateInfo.pColorBlendState = &BlendStateCI;
    PipelineCreateInfo.pMultisampleState = &MultisampleStateCI;
    PipelineCreateInfo.pViewportState = &ViewStateCI;
    PipelineCreateInfo.pDepthStencilState = &DepthStencilStateCI;
    PipelineCreateInfo.pDynamicState = &DynamicStateCI;
    PipelineCreateInfo.renderPass = IS.RenderPass;
    PipelineCreateInfo.layout = IS.PipelineLayout;

    if (vkCreateGraphicsPipelines(Device, IS.PipelineCache, 1,
                                  &PipelineCreateInfo, nullptr, &IS.Pipeline))
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create graphics pipeline.");

    return llvm::Error::success();
  }

  static llvm::SmallVector<VkBufferImageCopy>
  getImageCopyRegions(const TextureCreateDesc &Desc) {
    const VkImageAspectFlags AspectMask = isDepthFormat(Desc.Format)
                                              ? VK_IMAGE_ASPECT_DEPTH_BIT
                                              : VK_IMAGE_ASPECT_COLOR_BIT;
    llvm::SmallVector<VkBufferImageCopy> Regions;
    uint64_t CurrentOffset = 0;
    for (uint32_t I = 0; I < Desc.MipLevels; ++I) {
      VkBufferImageCopy Region = {};
      Region.imageSubresource.aspectMask = AspectMask;
      Region.imageSubresource.mipLevel = I;
      Region.imageSubresource.baseArrayLayer = 0;
      Region.imageSubresource.layerCount = 1;
      Region.imageExtent.width = std::max(1u, Desc.Width >> I);
      Region.imageExtent.height = std::max(1u, Desc.Height >> I);
      Region.imageExtent.depth = 1;
      Region.bufferOffset = CurrentOffset;
      Regions.push_back(Region);
      CurrentOffset += static_cast<uint64_t>(Region.imageExtent.width) *
                       Region.imageExtent.height * getFormatSize(Desc.Format);
    }
    return Regions;
  }

  static VkImageSubresourceRange
  getImageSubresourceRange(const TextureCreateDesc &Desc) {
    VkImageSubresourceRange SubRange = {};
    SubRange.aspectMask = isDepthFormat(Desc.Format)
                              ? VK_IMAGE_ASPECT_DEPTH_BIT
                              : VK_IMAGE_ASPECT_COLOR_BIT;
    SubRange.baseMipLevel = 0;
    SubRange.levelCount = Desc.MipLevels;
    SubRange.layerCount = 1;
    return SubRange;
  }

  static void copyBufferToTexture(VkCommandBuffer CmdBuffer,
                                  VulkanTexture &Dst, VulkanBuffer &Src,
                                  VkImageLayout OldLayout) {
    const auto Regions = getImageCopyRegions(Dst.Desc);
    const auto SubRange = getImageSubresourceRange(Dst.Desc);

    VkImageMemoryBarrier Barrier = {};
    Barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    Barrier.subresourceRange = SubRange;
    Barrier.srcAccessMask = 0;
    Barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    Barrier.oldLayout = OldLayout;
    Barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    Barrier.image = Dst.Image;
    vkCmdPipelineBarrier(CmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &Barrier);

    vkCmdCopyBufferToImage(CmdBuffer, Src.Buffer, Dst.Image,
                           VK_IMAGE_LAYOUT_GENERAL, Regions.size(),
                           Regions.data());
  }

  static void copyTextureToBuffer(VkCommandBuffer CmdBuffer,
                                  VulkanBuffer &Dst, VulkanTexture &Src) {
    const auto Regions = getImageCopyRegions(Src.Desc);
    vkCmdCopyImageToBuffer(CmdBuffer, Src.Image, VK_IMAGE_LAYOUT_GENERAL,
                           Dst.Buffer, Regions.size(), Regions.data());
  }

  static void copyBufferToBuffer(VkCommandBuffer CmdBuffer, VulkanBuffer &Dst,
                                 VulkanBuffer &Src) {
    VkBufferCopy Copy = {};
    Copy.size = Src.SizeInBytes;
    vkCmdCopyBuffer(CmdBuffer, Src.Buffer, Dst.Buffer, 1, &Copy);
  }

  // Record commands to copy a texture into a readback buffer.
  void copyTextureToReadback(VkCommandBuffer CmdBuffer,
                             const VulkanTexture &Tex,
                             const VulkanBuffer &Readback,
                             VkImageLayout OldLayout,
                             VkAccessFlags SrcAccessMask,
                             VkPipelineStageFlags SrcStageMask) {
    const VkImageAspectFlags AspectMask = isDepthFormat(Tex.Desc.Format)
                                              ? VK_IMAGE_ASPECT_DEPTH_BIT
                                              : VK_IMAGE_ASPECT_COLOR_BIT;

    // Transition texture to transfer source.
    VkImageSubresourceRange SubRange = {};
    SubRange.aspectMask = AspectMask;
    SubRange.baseMipLevel = 0;
    SubRange.levelCount = 1;
    SubRange.layerCount = 1;

    VkImageMemoryBarrier ImageBarrier = {};
    ImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ImageBarrier.subresourceRange = SubRange;
    ImageBarrier.srcAccessMask = SrcAccessMask;
    ImageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ImageBarrier.oldLayout = OldLayout;
    ImageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ImageBarrier.image = Tex.Image;
    vkCmdPipelineBarrier(CmdBuffer, SrcStageMask,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &ImageBarrier);

    // Copy image to readback buffer.
    VkBufferImageCopy Region = {};
    Region.imageSubresource.aspectMask = AspectMask;
    Region.imageSubresource.mipLevel = 0;
    Region.imageSubresource.baseArrayLayer = 0;
    Region.imageSubresource.layerCount = 1;
    Region.imageExtent.width = Tex.Desc.Width;
    Region.imageExtent.height = Tex.Desc.Height;
    Region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(CmdBuffer, Tex.Image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           Readback.Buffer, 1, &Region);

    // Barrier to make the readback buffer visible to the host.
    VkBufferMemoryBarrier BufBarrier = {};
    BufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    BufBarrier.size = VK_WHOLE_SIZE;
    BufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    BufBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    BufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    BufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    BufBarrier.buffer = Readback.Buffer;
    vkCmdPipelineBarrier(CmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &BufBarrier, 0, nullptr);
  }

  llvm::Error createCommands(Pipeline &P, InvocationState &IS) {
    // Image resources were uploaded in CB1 (which has completed). Vulkan
    // requires an explicit image memory barrier in CB2 to declare the layout
    // transition to the shader stage — the queue wait between CBs only
    // provides memory availability, not layout transitions.
    for (const auto &RP : IS.Resources) {
      for (const auto &BR : RP.second) {
        if (!BR->isTexture() || BR->CurrentLayout == VK_IMAGE_LAYOUT_UNDEFINED)
          continue;
        auto &Tex = *std::get<std::shared_ptr<VulkanTexture>>(BR->Resource);
        const TextureCreateDesc &Desc = Tex.Desc;

        VkImageSubresourceRange SubRange = {};
        SubRange.aspectMask = isDepthFormat(Desc.Format)
                                  ? VK_IMAGE_ASPECT_DEPTH_BIT
                                  : VK_IMAGE_ASPECT_COLOR_BIT;
        SubRange.baseMipLevel = 0;
        SubRange.levelCount = Desc.MipLevels;
        SubRange.layerCount = 1;

        VkImageMemoryBarrier ImageBarrier = {};
        ImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ImageBarrier.subresourceRange = SubRange;
        ImageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ImageBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            (RP.first->isReadWrite() ? VK_ACCESS_SHADER_WRITE_BIT : 0);
        ImageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ImageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ImageBarrier.image = Tex.Image;
        vkCmdPipelineBarrier(IS.CmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &ImageBarrier);
      }
    }

    if (P.isGraphics()) {
      const auto *ColorCV =
          std::get_if<ClearColor>(&*IS.RenderTarget->Desc.OptimizedClearValue);
      if (!ColorCV)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Render target clear value must be a ClearColor.");
      const auto *DepthCV = std::get_if<ClearDepthStencil>(
          &*IS.DepthStencil->Desc.OptimizedClearValue);
      if (!DepthCV)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Depth/stencil clear value must be a ClearDepthStencil.");
      VkClearValue ClearValues[2] = {};
      ClearValues[0].color = {{ColorCV->R, ColorCV->G, ColorCV->B, ColorCV->A}};
      ClearValues[1].depthStencil = {DepthCV->Depth, DepthCV->Stencil};

      VkRenderPassBeginInfo RenderPassBeginInfo = {};
      RenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      RenderPassBeginInfo.renderPass = IS.RenderPass;
      RenderPassBeginInfo.framebuffer = IS.FrameBuffer;
      RenderPassBeginInfo.renderArea.extent.width = IS.RenderTarget->Desc.Width;
      RenderPassBeginInfo.renderArea.extent.height =
          IS.RenderTarget->Desc.Height;
      RenderPassBeginInfo.clearValueCount = 2;
      RenderPassBeginInfo.pClearValues = ClearValues;

      vkCmdBeginRenderPass(IS.CmdBuffer, &RenderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport Viewport = {};
      Viewport.x = 0.0f;
      Viewport.y = 0.0f;
      Viewport.width = static_cast<float>(IS.RenderTarget->Desc.Width);
      Viewport.height = static_cast<float>(IS.RenderTarget->Desc.Height);
      Viewport.minDepth = 0.0f;
      Viewport.maxDepth = 1.0f;
      vkCmdSetViewport(IS.CmdBuffer, 0, 1, &Viewport);

      VkRect2D Scissor = {};
      Scissor.offset = {0, 0};
      Scissor.extent.width = IS.RenderTarget->Desc.Width;
      Scissor.extent.height = IS.RenderTarget->Desc.Height;
      vkCmdSetScissor(IS.CmdBuffer, 0, 1, &Scissor);
    }

    const VkPipelineBindPoint BindPoint = P.isGraphics()
                                              ? VK_PIPELINE_BIND_POINT_GRAPHICS
                                              : VK_PIPELINE_BIND_POINT_COMPUTE;
    vkCmdBindPipeline(IS.CmdBuffer, BindPoint, IS.Pipeline);
    if (IS.DescriptorSets.size() > 0)
      vkCmdBindDescriptorSets(IS.CmdBuffer, BindPoint, IS.PipelineLayout, 0,
                              IS.DescriptorSets.size(),
                              IS.DescriptorSets.data(), 0, 0);

    for (const auto &PCB : P.PushConstants) {
      llvm::SmallVector<uint8_t, 4> Data;
      PCB.getContent(Data);
      vkCmdPushConstants(IS.CmdBuffer, IS.PipelineLayout,
                         getShaderStageFlag(PCB.Stage), 0, Data.size(),
                         Data.data());
    }

    if (P.isCompute()) {
      const llvm::ArrayRef<int> DispatchSize =
          llvm::ArrayRef<int>(P.Shaders[0].DispatchSize);
      vkCmdDispatch(IS.CmdBuffer, DispatchSize[0], DispatchSize[1],
                    DispatchSize[2]);
      llvm::outs() << "Dispatched compute shader: { " << DispatchSize[0] << ", "
                   << DispatchSize[1] << ", " << DispatchSize[2] << " }\n";
    } else {
      if (!IS.VB)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Vertex buffer not initialized.");
      VkDeviceSize Offsets[1]{0};
      VkBuffer VBHandle =
          static_cast<VulkanBuffer *>(IS.VB->Data.get())->Buffer;
      vkCmdBindVertexBuffers(IS.CmdBuffer, 0, 1, &VBHandle, Offsets);
      // instanceCount must be >=1 to draw; previously was 0 which draws nothing
      vkCmdDraw(IS.CmdBuffer, IS.VB->getVertexCount(), 1, 0, 0);
      llvm::outs() << "Drew " << IS.VB->getVertexCount() << " vertices.\n";
      vkCmdEndRenderPass(IS.CmdBuffer);
      copyTextureToReadback(IS.CmdBuffer, *IS.RenderTarget, *IS.RTReadback,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    copyReadbackBindings(IS);
    return llvm::Error::success();
  }

  llvm::Error readBackData(Pipeline &P, InvocationState &IS) {
    VkMappedMemoryRange Range = {};
    Range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    Range.offset = 0;
    Range.size = VK_WHOLE_SIZE;

    for (const auto &RB : IS.ReadbackBindings) {
      Resource *R = RB.PipelineResource;

      void *Mapped = nullptr;
      vkMapMemory(Device, RB.Destination->Memory, 0, VK_WHOLE_SIZE, 0, &Mapped);
      Range.memory = RB.Destination->Memory;
      vkInvalidateMappedMemoryRanges(Device, 1, &Range);
      memcpy(R->BufferPtr->Data[RB.ArrayIndex].get(), Mapped, R->size());
      vkUnmapMemory(Device, RB.Destination->Memory);

      if (RB.Counter) {
        void *CounterMapped = nullptr;
        vkMapMemory(Device, RB.Counter->Destination->Memory, 0, VK_WHOLE_SIZE,
                    0, &CounterMapped);
        Range.memory = RB.Counter->Destination->Memory;
        vkInvalidateMappedMemoryRanges(Device, 1, &Range);
        R->BufferPtr->Counters.push_back(
            *static_cast<uint32_t *>(CounterMapped));
        vkUnmapMemory(Device, RB.Counter->Destination->Memory);
      }
    }

    // Copy back the render target data if this was a graphics pipeline.
    if (P.isGraphics()) {
      void *Mapped = nullptr;
      vkMapMemory(Device, IS.RTReadback->Memory, 0, VK_WHOLE_SIZE, 0, &Mapped);
      Range.memory = IS.RTReadback->Memory;
      vkInvalidateMappedMemoryRanges(Device, 1, &Range);

      const CPUBuffer &B = *P.Bindings.RTargetBufferPtr;
      memcpy(B.Data[0].get(), Mapped, B.size());
      vkUnmapMemory(Device, IS.RTReadback->Memory);
    }
    return llvm::Error::success();
  }

  void cleanup(InvocationState &IS) {
    vkQueueWaitIdle(GraphicsQueue.Queue);
    for (auto &V : IS.BufferViews)
      vkDestroyBufferView(Device, V, nullptr);

    for (auto &V : IS.ImageViews)
      vkDestroyImageView(Device, V, nullptr);

    // Destroy VkSampler handles — these are not owned by VulkanTexture.
    for (const auto &RP : IS.Resources)
      for (const auto &BR : RP.second)
        if (BR->Sampler != VK_NULL_HANDLE)
          vkDestroySampler(Device, BR->Sampler, nullptr);

    // shared_ptr destructors handle VulkanTexture/VulkanBuffer cleanup.
    IS.Resources.clear();
    IS.ReadbackBindings.clear();
    IS.ResourcesKeepAlive.clear();

    if (IS.getFullShaderStageMask() != VK_SHADER_STAGE_COMPUTE_BIT) {
      vkDestroyFramebuffer(Device, IS.FrameBuffer, nullptr);
      vkDestroyRenderPass(Device, IS.RenderPass, nullptr);
    }

    if (IS.Pipeline)
      vkDestroyPipeline(Device, IS.Pipeline, nullptr);

    for (auto &S : IS.Shaders)
      vkDestroyShaderModule(Device, S.Shader, nullptr);

    if (IS.PipelineCache)
      vkDestroyPipelineCache(Device, IS.PipelineCache, nullptr);

    if (IS.PipelineLayout)
      vkDestroyPipelineLayout(Device, IS.PipelineLayout, nullptr);

    for (auto &L : IS.DescriptorSetLayouts)
      vkDestroyDescriptorSetLayout(Device, L, nullptr);

    if (IS.Pool)
      vkDestroyDescriptorPool(Device, IS.Pool, nullptr);

    if (IS.CmdPool)
      vkDestroyCommandPool(Device, IS.CmdPool, nullptr);
  }

  llvm::Error executeProgram(Pipeline &P) override {
    InvocationState State;
    auto CleanupState = llvm::scope_exit([&]() {
      cleanup(State);
      llvm::outs() << "Cleanup complete.\n";
    });

    if (auto Err = createDevice(State))
      return Err;
    llvm::outs() << "Physical device created.\n";
    if (auto Err = createShaderModules(P, State))
      return Err;
    llvm::outs() << "Shader module created.\n";
    if (auto Err = createCommandBuffer(State))
      return Err;
    llvm::outs() << "Copy command buffer created.\n";
    if (auto Err = createResources(P, State))
      return Err;
    if (P.isGraphics()) {
      if (auto Err = createRenderPass(State))
        return Err;
      llvm::outs() << "Render pass created.\n";
      if (auto Err = createFrameBuffer(State))
        return Err;
      llvm::outs() << "Frame buffer created.\n";
    }
    llvm::outs() << "Memory buffers created.\n";
    if (auto Err = executeCommandBuffer(State))
      return Err;
    llvm::outs() << "Executed copy command buffer.\n";
    if (auto Err = createCommandBuffer(State))
      return Err;
    llvm::outs() << "Execute command buffer created.\n";
    if (auto Err = createDescriptorPool(P, State))
      return Err;
    llvm::outs() << "Descriptor pool created.\n";
    if (auto Err = createDescriptorSets(P, State))
      return Err;
    llvm::outs() << "Descriptor sets created.\n";
    if (auto Err = createPipeline(P, State))
      return Err;
    llvm::outs() << "Compute pipeline created.\n";
    if (auto Err = createCommands(P, State))
      return Err;
    llvm::outs() << "Commands created.\n";
    if (auto Err = executeCommandBuffer(State))
      return Err;
    llvm::outs() << "Executed compute command buffer.\n";
    if (auto Err = readBackData(P, State))
      return Err;
    llvm::outs() << "Compute pipeline created.\n";

    return llvm::Error::success();
  }
};
} // namespace

llvm::Error offloadtest::initializeVulkanDevices(
    const DeviceConfig Config,
    llvm::SmallVectorImpl<std::unique_ptr<Device>> &Devices) {
  // Request the highest supported API version
  uint32_t ApiVersion = 0;
  vkEnumerateInstanceVersion(&ApiVersion);

  VkApplicationInfo AppInfo = {};
  AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  AppInfo.pApplicationName = "OffloadTest";
  AppInfo.apiVersion = ApiVersion;

  VkInstanceCreateInfo CreateInfo = {};
  CreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  CreateInfo.pApplicationInfo = &AppInfo;

  llvm::SmallVector<const char *> EnabledInstanceExtensions;
  llvm::SmallVector<const char *> EnabledLayers;
#if __APPLE__
  // If we build Vulkan support for Apple platforms the VK_KHR_PORTABILITY
  // extension is required, so we can just force this one on. If it fails, the
  // whole device would fail anyways.
  EnabledInstanceExtensions.push_back(
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  CreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  const llvm::SmallVector<VkLayerProperties, 0> AvailableInstanceLayers =
      queryInstanceLayers();
  if (Config.EnableValidationLayer) {
    const llvm::StringRef ValidationLayer = "VK_LAYER_KHRONOS_validation";
    if (isLayerSupported(AvailableInstanceLayers, ValidationLayer))
      EnabledLayers.push_back(ValidationLayer.data());
  }
  const llvm::SmallVector<VkExtensionProperties, 0> AvailableExtensions =
      queryInstanceExtensions(nullptr);
  if (Config.EnableDebugLayer) {
    const llvm::StringRef DebugUtilsExtensionName = "VK_EXT_debug_utils";
    if (isExtensionSupported(AvailableExtensions, DebugUtilsExtensionName))
      EnabledInstanceExtensions.push_back(DebugUtilsExtensionName.data());
  }

  CreateInfo.ppEnabledLayerNames = EnabledLayers.data();
  CreateInfo.enabledLayerCount = EnabledLayers.size();
  CreateInfo.ppEnabledExtensionNames = EnabledInstanceExtensions.data();
  CreateInfo.enabledExtensionCount = EnabledInstanceExtensions.size();

  VkInstance Instance = VK_NULL_HANDLE;
  const VkResult Res = vkCreateInstance(&CreateInfo, NULL, &Instance);
  if (Res == VK_ERROR_INCOMPATIBLE_DRIVER)
    return llvm::createStringError(std::errc::no_such_device,
                                   "Cannot find a base Vulkan device");
  if (Res)
    return llvm::createStringError(std::errc::no_such_device,
                                   "Unknown Vulkan initialization error: %d",
                                   Res);

#ifndef NDEBUG
  VkDebugUtilsMessengerEXT DebugMessenger = registerDebugUtilCallback(Instance);
#else
  VkDebugUtilsMessengerEXT DebugMessenger = VK_NULL_HANDLE;
#endif

  const std::shared_ptr<VulkanInstance> VulkanInstanceShPtr =
      std::make_shared<VulkanInstance>(Instance, DebugMessenger);

  uint32_t DeviceCount = 0;
  if (vkEnumeratePhysicalDevices(Instance, &DeviceCount, nullptr))
    return llvm::createStringError(std::errc::no_such_device,
                                   "Failed to get device count");
  std::vector<VkPhysicalDevice> PhysicalDevices(DeviceCount);
  if (vkEnumeratePhysicalDevices(Instance, &DeviceCount,
                                 PhysicalDevices.data()))
    return llvm::createStringError(std::errc::no_such_device,
                                   "Failed to enumerate devices");

  for (const auto &PDev : PhysicalDevices) {
    auto DeviceOrErr = VulkanDevice::create(VulkanInstanceShPtr, PDev,
                                            AvailableInstanceLayers);
    if (!DeviceOrErr) {
      return DeviceOrErr.takeError();
    }
    Devices.push_back(std::move(*DeviceOrErr));
  }

  return llvm::Error::success();
}
