#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#define IR_RUNTIME_METALCPP
#define IR_PRIVATE_IMPLEMENTATION
#include "metal_irconverter_runtime.h"

#include "API/Device.h"
#include "API/Encoder.h"
#include "API/FormatConversion.h"
#include "MTLResources.h"
#include "Support/Pipeline.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace offloadtest;

static llvm::Error toError(NS::Error *Err) {
  if (!Err)
    return llvm::Error::success();
  const std::error_code EC =
      std::error_code(static_cast<int>(Err->code()), std::system_category());
  llvm::SmallString<256> ErrMsg;
  llvm::raw_svector_ostream OS(ErrMsg);
  OS << Err->localizedDescription()->utf8String() << ": ";
  OS << Err->localizedFailureReason()->utf8String();
  return llvm::createStringError(EC, ErrMsg);
}

#define MTLFormats(FMT)                                                        \
  if (Channels == 1)                                                           \
    return MTL::PixelFormatR##FMT;                                             \
  if (Channels == 2)                                                           \
    return MTL::PixelFormatRG##FMT;                                            \
  if (Channels == 4)                                                           \
    return MTL::PixelFormatRGBA##FMT;

static MTL::PixelFormat getMTLFormat(DataFormat Format, int Channels) {
  switch (Format) {
  case DataFormat::Int32:
    MTLFormats(32Sint) break;
  case DataFormat::Float32:
    MTLFormats(32Float) break;
  default:
    llvm_unreachable("Unsupported Resource format specified");
  }
  return MTL::PixelFormatInvalid;
}

namespace {

class MTLFence : public offloadtest::Fence {
public:
  MTLFence(MTL::SharedEvent *Event, llvm::StringRef Name)
      : Name(Name), Event(Event) {}
  std::string Name;
  MTL::SharedEvent *Event;

  static llvm::Expected<std::unique_ptr<MTLFence>>
  create(MTL::Device *Device, llvm::StringRef Name) {
    MTL::SharedEvent *Event = Device->newSharedEvent();
    if (!Event)
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create shared event.");
    return std::make_unique<MTLFence>(Event, Name);
  }

  ~MTLFence() {
    if (Event)
      Event->release();
  }

  uint64_t getFenceValue() override { return Event->signaledValue(); }

  llvm::Error waitForCompletion(uint64_t SignalValue) override {
    if (!Event->waitUntilSignaledValue(SignalValue, UINT64_MAX))
      return llvm::createStringError(std::errc::timed_out,
                                     "Timed out waiting on shared event.");
    return llvm::Error::success();
  }
};

class MTLQueue : public offloadtest::Queue {
public:
  using Queue::submit;

  MTL::CommandQueue *Queue;
  std::unique_ptr<MTLFence> SubmitFence;
  uint64_t FenceCounter = 0;

  // Batches of command buffers submitted to the GPU that may still be
  // in-flight.  Each batch records the fence value it signals so we can
  // non-blockingly query progress and release completed batches.
  struct InFlightBatch {
    uint64_t FenceValue;
    llvm::SmallVector<std::unique_ptr<offloadtest::CommandBuffer>> CBs;
  };
  llvm::SmallVector<InFlightBatch> InFlightBatches;

  MTLQueue(MTL::CommandQueue *Queue, std::unique_ptr<MTLFence> SubmitFence)
      : Queue(Queue), SubmitFence(std::move(SubmitFence)) {}
  ~MTLQueue() override {
    if (Queue)
      Queue->release();
  }

  llvm::Expected<offloadtest::SubmitResult>
  submit(llvm::SmallVector<std::unique_ptr<offloadtest::CommandBuffer>> CBs)
      override;
};

class MTLPipelineState : public offloadtest::PipelineState {
public:
  std::string Name;
  MTL::ComputePipelineState *ComputePipeline = nullptr;
  MTL::RenderPipelineState *RenderPipeline = nullptr;

  MTLPipelineState(llvm::StringRef Name,
                   MTL::ComputePipelineState *ComputePipeline)
      : offloadtest::PipelineState(GPUAPI::Metal), Name(Name),
        ComputePipeline(ComputePipeline) {}

  MTLPipelineState(llvm::StringRef Name,
                   MTL::RenderPipelineState *RenderPipeline)
      : offloadtest::PipelineState(GPUAPI::Metal), Name(Name),
        RenderPipeline(RenderPipeline) {}

  ~MTLPipelineState() override {
    if (ComputePipeline)
      ComputePipeline->release();
    if (RenderPipeline)
      RenderPipeline->release();
  }

  static bool classof(const offloadtest::PipelineState *B) {
    return B->getAPI() == GPUAPI::Metal;
  }
};

class MTLBuffer : public offloadtest::Buffer {
public:
  MTL::Buffer *Buf;
  std::string Name;
  BufferCreateDesc Desc;
  size_t SizeInBytes;

  MTLBuffer(MTL::Buffer *Buf, llvm::StringRef Name, BufferCreateDesc Desc,
            size_t SizeInBytes)
      : offloadtest::Buffer(GPUAPI::Metal), Buf(Buf), Name(Name), Desc(Desc),
        SizeInBytes(SizeInBytes) {}

  size_t getSizeInBytes() const override { return SizeInBytes; }

  llvm::Expected<void *> map() override {
    if (Desc.Location == MemoryLocation::GpuOnly)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Cannot map a GpuOnly buffer.");
    return Buf->contents();
  }

  llvm::Error unmap() override {
    // Managed storage (CpuToGpu) requires an explicit didModifyRange to
    // propagate CPU-side writes to the GPU. Shared storage (GpuToCpu) is
    // coherent and needs no action.
    if (Desc.Location == MemoryLocation::CpuToGpu)
      Buf->didModifyRange(NS::Range::Make(0, SizeInBytes));
    return llvm::Error::success();
  }

  ~MTLBuffer() override {
    if (Buf)
      Buf->release();
  }

  static bool classof(const offloadtest::Buffer *B) {
    return B->getAPI() == GPUAPI::Metal;
  }
};

class MTLTexture : public offloadtest::Texture {
public:
  MTL::Texture *Tex;
  std::string Name;
  TextureCreateDesc Desc;

  MTLTexture(MTL::Texture *Tex, llvm::StringRef Name, TextureCreateDesc Desc)
      : offloadtest::Texture(GPUAPI::Metal), Tex(Tex), Name(Name), Desc(Desc) {}

  ~MTLTexture() override {
    if (Tex)
      Tex->release();
  }

  static bool classof(const offloadtest::Texture *T) {
    return T->getAPI() == GPUAPI::Metal;
  }
};

class MTLDevice; // forward decl — defined below in this same anon ns

class MTLCommandBuffer : public offloadtest::CommandBuffer {
public:
  MTL::CommandBuffer *CmdBuffer = nullptr;
  /// Back-pointer to the owning device; used by encoders that need to
  /// allocate scratch / instance buffers for AS builds.
  MTLDevice *Dev = nullptr;
  /// MTL::Buffer objects (e.g. AS scratch and TLAS instance buffers) that
  /// must outlive command-buffer submission. Released when the CB is
  /// destroyed.
  llvm::SmallVector<MTL::Buffer *> KeepAliveMTLBuffers;

  static llvm::Expected<std::unique_ptr<MTLCommandBuffer>>
  create(MTL::CommandQueue *Queue) {
    auto CB = std::unique_ptr<MTLCommandBuffer>(new MTLCommandBuffer());
    CB->CmdBuffer = Queue->commandBuffer();
    if (!CB->CmdBuffer)
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create Metal command buffer.");
    return CB;
  }

  ~MTLCommandBuffer() override {
    for (auto *B : KeepAliveMTLBuffers)
      if (B)
        B->release();
  }

  static bool classof(const CommandBuffer *CB) {
    return CB->getKind() == GPUAPI::Metal;
  }

  llvm::Expected<std::unique_ptr<offloadtest::ComputeEncoder>>
  createComputeEncoder() override;

private:
  MTLCommandBuffer() : CommandBuffer(GPUAPI::Metal) {}
};

class MTLAccelStruct : public offloadtest::AccelerationStructure {
public:
  MTL::AccelerationStructure *AccelStruct;

  MTLAccelStruct(MTL::AccelerationStructure *AccelStruct)
      : offloadtest::AccelerationStructure(GPUAPI::Metal),
        AccelStruct(AccelStruct) {}

  ~MTLAccelStruct() override {
    if (AccelStruct)
      AccelStruct->release();
  }

  static bool classof(const offloadtest::AccelerationStructure *AS) {
    return AS->getAPI() == GPUAPI::Metal;
  }
};

llvm::Expected<offloadtest::SubmitResult> MTLQueue::submit(
    llvm::SmallVector<std::unique_ptr<offloadtest::CommandBuffer>> CBs) {
  // Non-blocking: query how far the GPU has progressed and release
  // command buffers from completed submissions.
  {
    const uint64_t Completed = SubmitFence->getFenceValue();
    llvm::erase_if(InFlightBatches, [Completed](const InFlightBatch &B) {
      return B.FenceValue <= Completed;
    });
  }

  // Metal serial queues guarantee that command buffers execute in commit order,
  // so no explicit wait on prior work is needed here.
  const uint64_t SignalValue = ++FenceCounter;

  for (size_t I = 0; I < CBs.size(); ++I) {
    auto &MCB = llvm::cast<MTLCommandBuffer>(*CBs[I].get());
    // Signal the submit fence when the last command buffer completes.
    if (I == CBs.size() - 1)
      MCB.CmdBuffer->encodeSignalEvent(SubmitFence->Event, SignalValue);
    MCB.CmdBuffer->commit();
  }

  // Keep submitted command buffers alive until the GPU is done with them.
  InFlightBatches.push_back({SignalValue, std::move(CBs)});

  return offloadtest::SubmitResult{SubmitFence.get(), SignalValue};
}

class MTLComputeEncoder : public offloadtest::ComputeEncoder {
  MTLCommandBuffer *CB = nullptr;
  MTL::CommandBuffer *CmdBuffer;
  MTL::ComputeCommandEncoder *ComputeEnc = nullptr;
  MTL::BlitCommandEncoder *BlitEnc = nullptr;
  /// Lazy AS encoder, created when batchBuildAS is called and torn down at
  /// the next encoder transition (via endEncodingImpl).
  MTL::AccelerationStructureCommandEncoder *ASEnc = nullptr;

  /// Threadgroup size from shader reflection (the numthreads() attribute
  /// persisted in the transpiled Metallib). Must be set via
  /// setThreadGroupSize() before dispatching.
  MTL::Size ThreadsPerGroup = {1, 1, 1};

  /// Accumulated barrier scope from commands recorded since the last barrier.
  MTL::BarrierScope PendingScope = MTL::BarrierScope(0);

  /// Record that a command touched the given resource types.  The accumulated
  /// scope is flushed as a memoryBarrier before the next command.
  void addBarrierScope(MTL::BarrierScope Scope) { PendingScope |= Scope; }

  void flushBarrier() {
    if (ComputeEnc && PendingScope != MTL::BarrierScope(0)) {
      ComputeEnc->memoryBarrier(PendingScope);
      PendingScope = MTL::BarrierScope(0);
    }
  }

  /// End the blit encoder if active, lazily (re-)create the compute encoder.
  /// Metal requires a dedicated BlitCommandEncoder for copy operations. Metal 4
  /// moves blit operations onto the compute encoder, removing this separation.
  llvm::Error ensureComputeEncoder() {
    if (ComputeEnc)
      return llvm::Error::success();
    endEncodingImpl();
    ComputeEnc = CmdBuffer->computeCommandEncoder();
    if (!ComputeEnc)
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create Metal compute encoder.");
    ComputeEnc->pushDebugGroup(
        NS::String::string("ComputeEncoder", NS::UTF8StringEncoding));
    return llvm::Error::success();
  }

  /// End the compute encoder if active, lazily create the blit encoder.
  llvm::Error ensureBlitEncoder() {
    if (BlitEnc)
      return llvm::Error::success();
    endEncodingImpl();
    BlitEnc = CmdBuffer->blitCommandEncoder();
    if (!BlitEnc)
      return llvm::createStringError(std::errc::device_or_resource_busy,
                                     "Failed to create Metal blit encoder.");
    return llvm::Error::success();
  }

public:
  MTLComputeEncoder(MTLCommandBuffer *CB, MTL::CommandBuffer *CmdBuffer,
                    MTL::ComputeCommandEncoder *Encoder)
      : ComputeEncoder(GPUAPI::Metal), CB(CB), CmdBuffer(CmdBuffer),
        ComputeEnc(Encoder) {}

  ~MTLComputeEncoder() override { endEncoding(); }

  static bool classof(const CommandEncoder *E) {
    return E->getAPI() == GPUAPI::Metal;
  }

  MTL::ComputeCommandEncoder *getNative() const { return ComputeEnc; }

  /// Set the threadgroup size for subsequent dispatch calls. The values must
  /// come from shader reflection (the numthreads() attribute in the HLSL
  /// source, persisted in the transpiled Metallib).
  void setThreadGroupSize(NS::UInteger X, NS::UInteger Y, NS::UInteger Z) {
    ThreadsPerGroup = MTL::Size(X, Y, Z);
  }

  MTL::CommandEncoder *getActiveEncoder() const {
    if (ComputeEnc)
      return ComputeEnc;
    return BlitEnc;
  }

  void pushDebugGroup(llvm::StringRef Label) override {
    if (auto *Enc = getActiveEncoder())
      Enc->pushDebugGroup(
          NS::String::string(Label.data(), NS::UTF8StringEncoding));
  }

  void popDebugGroup() override {
    if (auto *Enc = getActiveEncoder())
      Enc->popDebugGroup();
  }

  void insertDebugSignpost(llvm::StringRef Label) override {
    if (auto *Enc = getActiveEncoder())
      Enc->insertDebugSignpost(
          NS::String::string(Label.data(), NS::UTF8StringEncoding));
  }

  llvm::Error dispatch(uint32_t GroupCountX, uint32_t GroupCountY,
                       uint32_t GroupCountZ) override {
    if (auto Err = ensureComputeEncoder())
      return Err;
    flushBarrier();
    insertDebugSignpost(llvm::formatv("Dispatch [{0},{1},{2}]", GroupCountX,
                                      GroupCountY, GroupCountZ)
                            .str());
    const MTL::Size GridSize(ThreadsPerGroup.width * GroupCountX,
                             ThreadsPerGroup.height * GroupCountY,
                             ThreadsPerGroup.depth * GroupCountZ);
    ComputeEnc->dispatchThreads(GridSize, ThreadsPerGroup);
    addBarrierScope(MTL::BarrierScopeBuffers | MTL::BarrierScopeTextures);
    return llvm::Error::success();
  }

  llvm::Error copyBufferToBuffer(offloadtest::Buffer &Src, size_t SrcOffset,
                                 offloadtest::Buffer &Dst, size_t DstOffset,
                                 size_t Size) override {
    if (auto Err = ensureBlitEncoder())
      return Err;
    auto &MTLSrc = static_cast<MTLBuffer &>(Src);
    auto &MTLDst = static_cast<MTLBuffer &>(Dst);
    insertDebugSignpost(llvm::formatv("CopyBuffer {0}B", Size).str());
    BlitEnc->copyFromBuffer(MTLSrc.Buf, SrcOffset, MTLDst.Buf, DstOffset, Size);
    addBarrierScope(MTL::BarrierScopeBuffers);
    return llvm::Error::success();
  }

  // Defined out-of-line below — needs MTLDevice's full type for access to the
  // MTL::Device handle (used to allocate scratch and instance buffers).
  llvm::Error batchBuildAS(llvm::ArrayRef<ASBuildItem> Items) override;

  /// Lazily transition into an AccelerationStructureCommandEncoder; mirrors
  /// the existing compute↔blit lazy switch.
  llvm::Error ensureASEncoder() {
    if (ASEnc)
      return llvm::Error::success();
    endEncodingImpl();
    ASEnc = CmdBuffer->accelerationStructureCommandEncoder();
    if (!ASEnc)
      return llvm::createStringError(
          std::errc::device_or_resource_busy,
          "Failed to create Metal acceleration-structure encoder.");
    return llvm::Error::success();
  }

  void endEncodingImpl() override {
    if (ComputeEnc) {
      flushBarrier();
      ComputeEnc->popDebugGroup();
      ComputeEnc->endEncoding();
      ComputeEnc = nullptr;
    }
    if (BlitEnc) {
      BlitEnc->endEncoding();
      BlitEnc = nullptr;
    }
    if (ASEnc) {
      ASEnc->endEncoding();
      ASEnc = nullptr;
    }
  }
};

llvm::Expected<std::unique_ptr<offloadtest::ComputeEncoder>>
MTLCommandBuffer::createComputeEncoder() {
  MTL::ComputeCommandEncoder *NativeEncoder =
      CmdBuffer->computeCommandEncoder();
  if (!NativeEncoder)
    return llvm::createStringError(
        std::errc::device_or_resource_busy,
        "Failed to create Metal compute command encoder.");
  NativeEncoder->pushDebugGroup(
      NS::String::string("ComputeEncoder", NS::UTF8StringEncoding));
  return std::make_unique<MTLComputeEncoder>(this, CmdBuffer, NativeEncoder);
}
class MTLDevice : public offloadtest::Device {
  // MTLComputeEncoder needs access to the MTL::Device handle for AS scratch
  // and instance buffer allocation.
  friend class MTLComputeEncoder;

  Capabilities Caps;
  MTL::Device *Device;
  MTLQueue GraphicsQueue;

  struct InvocationState {
    InvocationState() { Pool = NS::AutoreleasePool::alloc()->init(); }
    ~InvocationState() {
      for (MTL::Texture *T : Textures)
        T->release();
      for (MTL::Buffer *B : Buffers)
        B->release();

      Pool->release();
    }

    NS::AutoreleasePool *Pool = nullptr;
    MTL::Buffer *ArgBuffer;
    std::unique_ptr<offloadtest::Buffer> VB;
    llvm::SmallVector<MTL::Texture *> Textures;
    llvm::SmallVector<MTL::Buffer *> Buffers;
    std::unique_ptr<offloadtest::Texture> FrameBufferTexture;
    std::unique_ptr<offloadtest::Buffer> FrameBufferReadback;
    std::unique_ptr<offloadtest::Texture> DepthStencil;
    std::unique_ptr<MTLCommandBuffer> CB;
    std::unique_ptr<PipelineState> Pipeline;
  };

  llvm::Error createDescriptor(Resource &R, InvocationState &IS,
                               const uint32_t HeapIdx) {
    auto *TablePtr = (IRDescriptorTableEntry *)IS.ArgBuffer->contents();

    assert(R.BufferPtr->ArraySize == 1 &&
           "Resource arrays are not yet supported on Metal.");

    if (R.isRaw()) {
      MTL::Buffer *Buf =
          Device->newBuffer(R.BufferPtr->Data.back().get(), R.size(),
                            MTL::ResourceStorageModeManaged);
      IRBufferView View = {};
      View.buffer = Buf;
      View.bufferSize = R.size();

      IRDescriptorTableSetBufferView(&TablePtr[HeapIdx], &View);
      IS.Buffers.push_back(Buf);
    } else {
      const uint64_t Width = R.isTexture() ? R.BufferPtr->OutputProps.Width
                                           : R.size() / R.getElementSize();
      const uint64_t Height =
          R.isTexture() ? R.BufferPtr->OutputProps.Height : 1;
      MTL::TextureUsage UsageFlags = MTL::ResourceUsageRead;
      if (R.isReadWrite())
        UsageFlags |= MTL::ResourceUsageWrite;
      MTL::TextureDescriptor *Desc = nullptr;
      const MTL::PixelFormat Format =
          getMTLFormat(R.BufferPtr->Format, R.BufferPtr->Channels);
      switch (R.Kind) {
      case ResourceKind::Buffer:
      case ResourceKind::RWBuffer:
        Desc = MTL::TextureDescriptor::textureBufferDescriptor(
            Format, Width, MTL::ResourceStorageModeManaged, UsageFlags);
        break;
      case ResourceKind::Texture2D:
      case ResourceKind::RWTexture2D:
        Desc = MTL::TextureDescriptor::texture2DDescriptor(Format, Width,
                                                           Height, false);
        break;
      case ResourceKind::Sampler:
        llvm_unreachable("Not implemented yet.");
      case ResourceKind::SampledTexture2D:
        llvm_unreachable("SampledTextures aren't supported in Metal.");
      case ResourceKind::StructuredBuffer:
      case ResourceKind::RWStructuredBuffer:
      case ResourceKind::ByteAddressBuffer:
      case ResourceKind::RWByteAddressBuffer:
      case ResourceKind::ConstantBuffer:
        llvm_unreachable("Raw is checked above");
      case ResourceKind::AccelerationStructure:
        llvm_unreachable("Acceleration structures use a separate path!");
      }

      MTL::Texture *NewTex = Device->newTexture(Desc);
      NewTex->replaceRegion(MTL::Region(0, 0, Width, Height), 0,
                            R.BufferPtr->Data.back().get(),
                            Width * R.getElementSize());

      IS.Textures.push_back(NewTex);

      IRDescriptorTableSetTexture(&TablePtr[HeapIdx], NewTex, 0, 0);
    }

    return llvm::Error::success();
  }

  llvm::Error createBuffers(Pipeline &P, InvocationState &IS) {
    const size_t ResourceCount = P.getDescriptorCount();
    const size_t TableSize = sizeof(IRDescriptorTableEntry) * ResourceCount;

    if (TableSize > 0) {
      IS.ArgBuffer =
          Device->newBuffer(TableSize, MTL::ResourceStorageModeManaged);
      uint32_t HeapIndex = 0;
      for (auto &D : P.Sets) {
        for (auto &R : D.Resources) {
          if (auto Err = createDescriptor(R, IS, HeapIndex++))
            return Err;
        }
      }
      IS.ArgBuffer->didModifyRange(NS::Range::Make(0, IS.ArgBuffer->length()));
    }
    if (P.isGraphics()) {
      if (!P.Bindings.VertexBufferPtr)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "No vertex buffer specified for graphics pipeline.");

      auto VBOrErr = offloadtest::createVertexBufferFromCPUBuffer(
          *this, *P.Bindings.VertexBufferPtr);
      if (!VBOrErr)
        return VBOrErr.takeError();
      IS.VB = std::move(*VBOrErr);
    }
    return llvm::Error::success();
  }

  llvm::Error createComputeCommands(Pipeline &P, InvocationState &IS) {
    auto EncoderOrErr = IS.CB->createComputeEncoder();
    if (!EncoderOrErr)
      return EncoderOrErr.takeError();
    auto &Encoder = llvm::cast<MTLComputeEncoder>(*EncoderOrErr.get());
    MTL::ComputeCommandEncoder *NativeEncoder = Encoder.getNative();

    const auto &PS = llvm::cast<MTLPipelineState>(IS.Pipeline.get());
    NativeEncoder->setComputePipelineState(PS->ComputePipeline);
    NativeEncoder->setBuffer(IS.ArgBuffer, 0, 2);
    for (uint64_t I = 0; I < IS.Textures.size(); ++I)
      NativeEncoder->useResource(IS.Textures[I], MTL::ResourceUsageRead |
                                                     MTL::ResourceUsageWrite);
    for (uint64_t I = 0; I < IS.Buffers.size(); ++I)
      NativeEncoder->useResource(IS.Buffers[I], MTL::ResourceUsageRead |
                                                    MTL::ResourceUsageWrite);

    NS::UInteger TGS[3] = {PS->ComputePipeline->maxTotalThreadsPerThreadgroup(),
                           1, 1};
    if (P.Shaders[0].Reflection) {
      llvm::Expected<llvm::json::Value> E = llvm::json::parse(
          llvm::StringRef(P.Shaders[0].Reflection->getBuffer()));
      if (!E)
        return E.takeError();
      llvm::json::Value Reflection = *E;

      const llvm::json::Object *ReflectionObj = Reflection.getAsObject();
      if (!ReflectionObj)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Shader reflection must be a JSON object.");
      auto StateIt = ReflectionObj->find("state");
      if (StateIt == ReflectionObj->end())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Key 'state' not found in shader reflection.");
      const llvm::json::Object *State = StateIt->second.getAsObject();
      auto TGSize = State->find("tg_size");
      if (TGSize == State->end())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Key 'tg_size' not found in shader reflection.");
      const llvm::json::Array *TGSizeArr = TGSize->second.getAsArray();
      if (TGSizeArr->size() != 3)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Threadgroup size in reflection must have three components.");
      for (size_t I = 0; I < 3; ++I) {
        auto OpVal = (*TGSizeArr)[I].getAsUINT64();
        if (!OpVal)
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Threadgroup size components in "
                                         "reflection must be integers.");
        TGS[I] = *OpVal;
      }
    }
    Encoder.setThreadGroupSize(TGS[0], TGS[1], TGS[2]);

    const llvm::ArrayRef<int> DispatchSize =
        llvm::ArrayRef<int>(P.Shaders[0].DispatchSize);
    if (auto Err =
            Encoder.dispatch(DispatchSize[0], DispatchSize[1], DispatchSize[2]))
      return Err;
    Encoder.endEncoding();
    return llvm::Error::success();
  }

  llvm::Error createRenderTarget(Pipeline &P, InvocationState &IS) {
    if (!P.Bindings.RTargetBufferPtr)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "No render target bound for graphics pipeline.");
    const CPUBuffer &OutBuf = *P.Bindings.RTargetBufferPtr;

    auto TexOrErr = offloadtest::createRenderTargetFromCPUBuffer(*this, OutBuf);
    if (!TexOrErr)
      return TexOrErr.takeError();

    IS.FrameBufferTexture = std::move(*TexOrErr);

    // Create a readback buffer for copying render target data to the CPU.
    BufferCreateDesc BufDesc = {};
    BufDesc.Location = MemoryLocation::GpuToCpu;
    BufDesc.Usage = BufferUsage::Storage;
    auto BufOrErr = createBuffer("RTReadback", BufDesc, OutBuf.size());
    if (!BufOrErr)
      return BufOrErr.takeError();
    IS.FrameBufferReadback = std::move(*BufOrErr);

    return llvm::Error::success();
  }

  llvm::Error createDepthStencil(Pipeline &P, InvocationState &IS) {
    auto TexOrErr = offloadtest::createDefaultDepthStencilTarget(
        *this, P.Bindings.RTargetBufferPtr->OutputProps.Width,
        P.Bindings.RTargetBufferPtr->OutputProps.Height);
    if (!TexOrErr)
      return TexOrErr.takeError();
    IS.DepthStencil = std::move(*TexOrErr);
    return llvm::Error::success();
  }

  llvm::Error createGraphicsCommands(Pipeline &P, InvocationState &IS) {
    if (auto Err = createRenderTarget(P, IS))
      return Err;
    // TODO: Always created for graphics pipelines. Consider making this
    // conditional on the pipeline definition.
    if (auto Err = createDepthStencil(P, IS))
      return Err;

    auto &FBTex = llvm::cast<MTLTexture>(*IS.FrameBufferTexture);
    auto &DS = llvm::cast<MTLTexture>(*IS.DepthStencil);
    auto &FBReadback = llvm::cast<MTLBuffer>(*IS.FrameBufferReadback);

    MTL::RenderPassDescriptor *Desc =
        MTL::RenderPassDescriptor::alloc()->init();

    const uint64_t Width = FBTex.Desc.Width;
    const uint64_t Height = FBTex.Desc.Height;

    // Color attachment.
    auto *CADesc = MTL::RenderPassColorAttachmentDescriptor::alloc()->init();
    CADesc->setTexture(FBTex.Tex);
    CADesc->setLoadAction(MTL::LoadActionClear);
    const auto *ColorCV =
        std::get_if<ClearColor>(&*FBTex.Desc.OptimizedClearValue);
    if (!ColorCV)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Render target clear value must be a ClearColor.");

    CADesc->setClearColor(
        MTL::ClearColor(ColorCV->R, ColorCV->G, ColorCV->B, ColorCV->A));
    CADesc->setStoreAction(MTL::StoreActionStore);
    Desc->colorAttachments()->setObject(CADesc, 0);

    // Depth/stencil attachment.
    const auto *DepthCV =
        std::get_if<ClearDepthStencil>(&*DS.Desc.OptimizedClearValue);
    if (!DepthCV)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Depth/stencil clear value must be a ClearDepthStencil.");

    auto *DADesc = Desc->depthAttachment();
    DADesc->setTexture(DS.Tex);
    DADesc->setLoadAction(MTL::LoadActionClear);
    DADesc->setClearDepth(DepthCV->Depth);
    DADesc->setStoreAction(MTL::StoreActionDontCare);

    auto *SADesc = Desc->stencilAttachment();
    SADesc->setTexture(DS.Tex);
    SADesc->setLoadAction(MTL::LoadActionClear);
    SADesc->setClearStencil(DepthCV->Stencil);
    SADesc->setStoreAction(MTL::StoreActionDontCare);

    MTL::RenderCommandEncoder *CmdEncoder =
        IS.CB->CmdBuffer->renderCommandEncoder(Desc);

    const auto &PS = llvm::cast<MTLPipelineState>(IS.Pipeline.get());
    CmdEncoder->setRenderPipelineState(PS->RenderPipeline);

    // Configure depth stencil state: depth test enabled, write all, less.
    MTL::DepthStencilDescriptor *DSDesc =
        MTL::DepthStencilDescriptor::alloc()->init();
    DSDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    DSDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState *DSState = Device->newDepthStencilState(DSDesc);
    CmdEncoder->setDepthStencilState(DSState);
    DSDesc->release();
    DSState->release();

    // Explicitly set viewport to texture dimensions.
    CmdEncoder->setViewport(
        MTL::Viewport{0.0, 0.0, (double)Width, (double)Height, 0.0, 1.0});
    CmdEncoder->setCullMode(MTL::CullModeNone);

    // Bind vertex buffer at slot 0 to match the vertex descriptor which
    // references buffer index 0.
    CmdEncoder->setVertexBuffer(llvm::cast<MTLBuffer>(*IS.VB).Buf, 0, 0);

    CmdEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0),
                               P.Bindings.getVertexCount());

    CmdEncoder->endEncoding();

    // Blit the render target into the readback buffer for CPU access.
    MTL::BlitCommandEncoder *Blit = IS.CB->CmdBuffer->blitCommandEncoder();
    const size_t ElemSize = getFormatSizeInBytes(FBTex.Desc.Fmt);
    const size_t RowBytes = Width * ElemSize;
    Blit->copyFromTexture(FBTex.Tex, 0, 0, MTL::Origin(0, 0, 0),
                          MTL::Size(Width, Height, 1), FBReadback.Buf, 0,
                          RowBytes, 0);
    Blit->endEncoding();

    return llvm::Error::success();
  }

  llvm::Expected<offloadtest::SubmitResult>
  executeCommands(InvocationState &IS) {
    return GraphicsQueue.submit(std::move(IS.CB));
  }

  llvm::Error copyBack(Pipeline &P, InvocationState &IS) {
    uint32_t TextureIndex = 0;
    uint32_t BufferIndex = 0;
    for (auto &D : P.Sets) {
      for (auto &R : D.Resources) {
        assert(R.BufferPtr->ArraySize == 1 &&
               "Resource arrays are not yet supported on Metal.");
        if (R.isReadOnly()) {
          if (R.isRaw())
            ++BufferIndex;
          else
            ++TextureIndex;
          continue;
        }
        if (R.isRaw()) {
          memcpy(R.BufferPtr->Data.back().get(),
                 IS.Buffers[BufferIndex++]->contents(), R.size());
          continue;
        }
        const uint64_t Width = R.isTexture() ? R.BufferPtr->OutputProps.Width
                                             : R.size() / R.getElementSize();
        const uint64_t Height =
            R.isTexture() ? R.BufferPtr->OutputProps.Height : 1;
        IS.Textures[TextureIndex++]->getBytes(
            R.BufferPtr->Data.back().get(), Width * R.getElementSize(),
            MTL::Region(0, 0, Width, Height), 0);
      }
    }
    if (P.isGraphics()) {
      CPUBuffer *RTarget = P.Bindings.RTargetBufferPtr;
      const uint64_t Width = RTarget->OutputProps.Width;
      const uint64_t Height = RTarget->OutputProps.Height;
      const size_t ElemSize = RTarget->getElementSize();
      const size_t RowBytes = Width * ElemSize;

      // Read from the readback buffer. The blit copied the texture data in
      // GPU layout order, so we flip rows here to produce an upright image.
      auto &FBReadback = llvm::cast<MTLBuffer>(*IS.FrameBufferReadback);
      const unsigned char *Src =
          reinterpret_cast<const unsigned char *>(FBReadback.Buf->contents());
      unsigned char *Buf =
          reinterpret_cast<unsigned char *>(RTarget->Data[0].get());
      for (uint64_t R = 0; R < Height; ++R) {
        const uint64_t SrcRow = (Height - 1) - R;
        memcpy(Buf + R * RowBytes, Src + SrcRow * RowBytes, RowBytes);
      }
    }
    return llvm::Error::success();
  }

public:
  MTLDevice(MTL::Device *D, MTL::CommandQueue *Q,
            std::unique_ptr<MTLFence> SubmitFence)
      : Device(D), GraphicsQueue(Q, std::move(SubmitFence)) {
    Description = Device->name()->utf8String();
  }
  const Capabilities &getCapabilities() override {
    if (Caps.empty())
      queryCapabilities();
    return Caps;
  }

  llvm::StringRef getAPIName() const override { return "Metal"; };
  GPUAPI getAPI() const override { return GPUAPI::Metal; };

  Queue &getGraphicsQueue() override { return GraphicsQueue; }

  llvm::Expected<std::unique_ptr<offloadtest::Fence>>
  createFence(llvm::StringRef Name) override {
    return MTLFence::create(Device, Name);
  }

  llvm::Expected<std::unique_ptr<offloadtest::Buffer>>
  createBuffer(std::string Name, BufferCreateDesc &Desc,
               size_t SizeInBytes) override {
    MTL::Buffer *Buf = Device->newBuffer(
        SizeInBytes, getMetalBufferResourceOptions(Desc.Location));
    if (!Buf)
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to create Metal buffer.");
    return std::make_unique<MTLBuffer>(Buf, Name, Desc, SizeInBytes);
  }

  llvm::Expected<std::unique_ptr<offloadtest::Texture>>
  createTexture(std::string Name, TextureCreateDesc &Desc) override {
    if (auto Err = validateTextureCreateDesc(Desc))
      return Err;

    MTL::TextureDescriptor *TDesc = MTL::TextureDescriptor::texture2DDescriptor(
        getMetalPixelFormat(Desc.Fmt), Desc.Width, Desc.Height,
        Desc.MipLevels > 1);
    TDesc->setMipmapLevelCount(Desc.MipLevels);
    TDesc->setStorageMode(getMetalTextureStorageMode(Desc.Location));
    TDesc->setUsage(getMetalTextureUsage(Desc.Usage));

    MTL::Texture *Tex = Device->newTexture(TDesc);
    if (!Tex)
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to create Metal texture.");
    return std::make_unique<MTLTexture>(Tex, Name, Desc);
  }

  llvm::Expected<std::unique_ptr<offloadtest::CommandBuffer>>
  createCommandBuffer() override {
    auto CBOrErr = MTLCommandBuffer::create(GraphicsQueue.Queue);
    if (!CBOrErr)
      return CBOrErr.takeError();
    (*CBOrErr)->Dev = this;
    return std::unique_ptr<offloadtest::CommandBuffer>(std::move(*CBOrErr));
  }

  llvm::Expected<std::unique_ptr<PipelineState>>
  createPipelineCs(llvm::StringRef Name,
                   const BindingsDesc & /*unused on metal*/,
                   ShaderContainer CS) override {
    NS::Error *Error = nullptr;
    const llvm::StringRef Program = CS.Shader->getBuffer();
    dispatch_data_t Data = dispatch_data_create(Program.data(), Program.size(),
                                                dispatch_get_main_queue(),
                                                ^{
                                                });
    MTL::Library *Lib = Device->newLibrary(Data, &Error);
    if (Error)
      return toError(Error);
    auto LibScope = llvm::scope_exit([&] { Lib->release(); });

    MTL::Function *Fn = Lib->newFunction(
        NS::String::string(CS.EntryPoint.c_str(), NS::UTF8StringEncoding));
    if (!Fn)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Failed to find entry point '%s' in Metal library.",
          CS.EntryPoint.c_str());
    auto FnScope = llvm::scope_exit([&] { Fn->release(); });

    MTL::ComputePipelineState *PSO =
        Device->newComputePipelineState(Fn, &Error);
    if (Error)
      return toError(Error);

    return std::make_unique<MTLPipelineState>(Name, PSO);
  }

  llvm::Expected<std::unique_ptr<PipelineState>> createPipelineVsPs(
      llvm::StringRef Name, const BindingsDesc & /*unused on metal*/,
      llvm::ArrayRef<InputLayoutDesc> InputLayout,
      llvm::ArrayRef<Format> RTFormats, std::optional<Format> DSFormat,
      ShaderContainer VS, ShaderContainer PS) override {
    NS::Error *Error = nullptr;

    // Load vertex shader.
    const llvm::StringRef VSProgram = VS.Shader->getBuffer();
    dispatch_data_t VSData = dispatch_data_create(
        VSProgram.data(), VSProgram.size(), dispatch_get_main_queue(),
        ^{
        });
    MTL::Library *VSLib = Device->newLibrary(VSData, &Error);
    if (Error)
      return toError(Error);
    auto VSLibScope = llvm::scope_exit([&] { VSLib->release(); });

    MTL::Function *VSFn = VSLib->newFunction(
        NS::String::string(VS.EntryPoint.c_str(), NS::UTF8StringEncoding));
    if (!VSFn)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Failed to find vertex entry point '%s' in Metal library.",
          VS.EntryPoint.c_str());
    auto VSFnScope = llvm::scope_exit([&] { VSFn->release(); });

    // Load pixel/fragment shader.
    const llvm::StringRef PSProgram = PS.Shader->getBuffer();
    dispatch_data_t PSData = dispatch_data_create(
        PSProgram.data(), PSProgram.size(), dispatch_get_main_queue(),
        ^{
        });
    MTL::Library *PSLib = Device->newLibrary(PSData, &Error);
    if (Error)
      return toError(Error);
    auto PSLibScope = llvm::scope_exit([&] { PSLib->release(); });

    MTL::Function *PSFn = PSLib->newFunction(
        NS::String::string(PS.EntryPoint.c_str(), NS::UTF8StringEncoding));
    if (!PSFn)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Failed to find fragment entry point '%s' in Metal library.",
          PS.EntryPoint.c_str());
    auto PSFnScope = llvm::scope_exit([&] { PSFn->release(); });

    MTL::RenderPipelineDescriptor *Desc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    auto DescScope = llvm::scope_exit([&] { Desc->release(); });
    Desc->setVertexFunction(VSFn);
    Desc->setFragmentFunction(PSFn);

    // Build vertex descriptor from InputLayout.
    if (!InputLayout.empty()) {
      NS::Array *FnAttrs = VSFn->vertexAttributes();
      // Currently we error on vertex shaders without any vertex attributes.
      // However, this is a valid use case that should be supported in the
      // future.
      if (!FnAttrs)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Vertex shader has no vertex attributes.");

      if (FnAttrs->count() != InputLayout.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Mismatch between vertex shader attribute count and pipeline "
            "vertex input count.");

      // Collect the attribute indices the shader expects so that we can map the
      // specified attributes onto the correct indices.
      llvm::StringMap<uint32_t> ShaderAttrIndices;
      for (uint32_t I = 0; I < FnAttrs->count(); ++I) {
        auto *A = static_cast<MTL::VertexAttribute *>(FnAttrs->object(I));
        if (A && A->isActive()) {
          ShaderAttrIndices.insert(std::make_pair(
              llvm::StringRef(A->name()->utf8String()), A->attributeIndex()));
          llvm::errs() << "Shader attr: " << A->name()->utf8String()
                       << " at index " << A->attributeIndex() << "\n";
        }
      }

      MTL::VertexDescriptor *VtxDesc = MTL::VertexDescriptor::alloc()->init();
      auto VtxDescScope = llvm::scope_exit([&] { VtxDesc->release(); });
      uint32_t Stride = 0;
      for (uint32_t I = 0; I < static_cast<uint32_t>(InputLayout.size()); ++I) {
        const InputLayoutDesc &Elem = InputLayout[I];
        assert(!Elem.InstanceStepRate &&
               "Instance step rate is currently not supported.");

        llvm::SmallString<32> AttrName(Elem.Name);
        llvm::transform(AttrName, AttrName.begin(), tolower);
        // Append a zero since we're only supporting one attribute per name.
        // We'll need to revisit this if we ever support indexed attributes.
        AttrName += "0";

        const uint32_t ElemSize = getFormatSizeInBytes(Elem.Fmt);
        MTL::VertexAttributeDescriptor *AttrDesc =
            MTL::VertexAttributeDescriptor::alloc()->init();
        AttrDesc->setBufferIndex(0);
        AttrDesc->setOffset(Elem.OffsetInBytes);
        AttrDesc->setFormat(getMetalVertexFormat(Elem.Fmt));
        VtxDesc->attributes()->setObject(AttrDesc, ShaderAttrIndices[AttrName]);
        AttrDesc->release();
        Stride = std::max(Stride, Elem.OffsetInBytes + ElemSize);
      }

      MTL::VertexBufferLayoutDescriptor *LDesc =
          MTL::VertexBufferLayoutDescriptor::alloc()->init();
      LDesc->setStride(Stride);
      LDesc->setStepRate(1);
      LDesc->setStepFunction(MTL::VertexStepFunctionPerVertex);
      VtxDesc->layouts()->setObject(LDesc, 0);
      LDesc->release();

      Desc->setVertexDescriptor(VtxDesc);
    }

    // Configure render target color attachments.
    for (size_t I = 0; I < RTFormats.size(); ++I) {
      MTL::RenderPipelineColorAttachmentDescriptor *RPCA =
          MTL::RenderPipelineColorAttachmentDescriptor::alloc()->init();
      RPCA->setPixelFormat(getMetalPixelFormat(RTFormats[I]));
      Desc->colorAttachments()->setObject(RPCA, I);
      RPCA->release();
    }

    // Configure depth/stencil attachment.
    if (DSFormat) {
      const MTL::PixelFormat DSPixelFormat = getMetalPixelFormat(*DSFormat);
      Desc->setDepthAttachmentPixelFormat(DSPixelFormat);
      if (isStencilFormat(*DSFormat))
        Desc->setStencilAttachmentPixelFormat(DSPixelFormat);
    }

    MTL::RenderPipelineState *PSO =
        Device->newRenderPipelineState(Desc, &Error);
    if (Error)
      return toError(Error);

    return std::make_unique<MTLPipelineState>(Name, PSO);
  }

  llvm::Expected<BLASBuildRequest>
  createBLASBuildRequest(llvm::ArrayRef<TriangleGeometryDesc> Triangles,
                         llvm::ArrayRef<AABBGeometryDesc> AABBs,
                         AccelerationStructureBuildFlags Flags) override {
    if (!Device->supportsRaytracing())
      return llvm::createStringError(
          std::errc::not_supported,
          "Ray tracing is not supported on this device.");

    BLASBuildRequest Req;
    Req.Triangles.assign(Triangles.begin(), Triangles.end());
    Req.AABBs.assign(AABBs.begin(), AABBs.end());
    Req.Flags = Flags;

    if (auto Err = validateBLASBuildRequest(Req))
      return Err;

    NS::Array *GeomDescs = nullptr;
    // Build an array of geometry descriptors.
    llvm::SmallVector<MTL::AccelerationStructureGeometryDescriptor *> Descs;

    for (const auto &T : Triangles) {
      auto *TD =
          MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
      auto *VB = llvm::cast<MTLBuffer>(T.VertexBuffer);
      TD->setVertexBuffer(VB->Buf);
      TD->setVertexBufferOffset(T.VertexBufferOffset);
      TD->setVertexStride(T.VertexStride);
      TD->setVertexFormat(getMetalPositionFormat(T.VertexFormat));
      TD->setTriangleCount(T.IndexBuffer ? T.IndexCount / 3
                                         : T.VertexCount / 3);
      if (T.IndexBuffer) {
        auto *IB = llvm::cast<MTLBuffer>(T.IndexBuffer);
        TD->setIndexBuffer(IB->Buf);
        TD->setIndexBufferOffset(T.IndexBufferOffset);
        TD->setIndexType(getMetalIndexType(T.IdxFormat));
      }
      TD->setOpaque(T.Opaque);
      Descs.push_back(TD);
    }

    for (const auto &A : AABBs) {
      auto *AD =
          MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
              ->init();
      auto *BB = llvm::cast<MTLBuffer>(A.AABBBuffer);
      AD->setBoundingBoxBuffer(BB->Buf);
      AD->setBoundingBoxBufferOffset(A.AABBBufferOffset);
      AD->setBoundingBoxStride(A.AABBStride);
      AD->setBoundingBoxCount(A.AABBCount);
      AD->setOpaque(A.Opaque);
      Descs.push_back(AD);
    }

    GeomDescs = NS::Array::array(
        reinterpret_cast<NS::Object *const *>(Descs.data()), Descs.size());

    auto *Descriptor =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    Descriptor->setGeometryDescriptors(GeomDescs);

    MTL::AccelerationStructureSizes Sizes =
        Device->accelerationStructureSizes(Descriptor);

    Req.Sizes.ResultDataMaxSizeInBytes = Sizes.accelerationStructureSize;
    Req.Sizes.ScratchDataSizeInBytes = Sizes.buildScratchBufferSize;
    Req.Sizes.UpdateScratchDataSizeInBytes = Sizes.refitScratchBufferSize;

    Descriptor->release();
    for (auto *D : Descs)
      D->release();

    return Req;
  }

  llvm::Expected<TLASBuildRequest> createTLASBuildRequest(
      llvm::ArrayRef<AccelerationStructureInstance> Instances,
      AccelerationStructureBuildFlags Flags) override {
    if (!Device->supportsRaytracing())
      return llvm::createStringError(
          std::errc::not_supported,
          "Ray tracing is not supported on this device.");

    TLASBuildRequest Req;
    Req.Instances.assign(Instances.begin(), Instances.end());
    Req.Flags = Flags;

    if (auto Err = validateTLASBuildRequest(Req))
      return Err;

    auto *Descriptor =
        MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    Descriptor->setInstanceCount(Instances.size());

    MTL::AccelerationStructureSizes Sizes =
        Device->accelerationStructureSizes(Descriptor);

    Req.Sizes.ResultDataMaxSizeInBytes = Sizes.accelerationStructureSize;
    Req.Sizes.ScratchDataSizeInBytes = Sizes.buildScratchBufferSize;
    Req.Sizes.UpdateScratchDataSizeInBytes = Sizes.refitScratchBufferSize;

    Descriptor->release();

    return Req;
  }

  llvm::Expected<std::unique_ptr<offloadtest::AccelerationStructure>>
  createAccelerationStructure(const BLASBuildRequest &Request) override {
    if (!Device->supportsRaytracing())
      return llvm::createStringError(
          std::errc::not_supported,
          "Ray tracing is not supported on this device.");

    MTL::AccelerationStructure *AS = Device->newAccelerationStructure(
        Request.Sizes.ResultDataMaxSizeInBytes);
    if (!AS)
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to create Metal BLAS.");
    return std::make_unique<MTLAccelStruct>(AS);
  }

  llvm::Expected<std::unique_ptr<offloadtest::AccelerationStructure>>
  createAccelerationStructure(const TLASBuildRequest &Request) override {
    if (!Device->supportsRaytracing())
      return llvm::createStringError(
          std::errc::not_supported,
          "Ray tracing is not supported on this device.");

    MTL::AccelerationStructure *AS = Device->newAccelerationStructure(
        Request.Sizes.ResultDataMaxSizeInBytes);
    if (!AS)
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to create Metal TLAS.");
    return std::make_unique<MTLAccelStruct>(AS);
  }

  llvm::Error executeProgram(Pipeline &P) override {
    InvocationState IS;

    auto CBOrErr = MTLCommandBuffer::create(GraphicsQueue.Queue);
    if (!CBOrErr)
      return CBOrErr.takeError();
    IS.CB = std::move(*CBOrErr);
    IS.CB->Dev = this;

    if (auto Err = createBuffers(P, IS))
      return Err;

    BindingsDesc Bindings = {};
    for (auto &S : P.Sets) {
      DescriptorSetLayoutDesc Layout;
      for (auto &R : S.Resources) {
        ResourceBindingDesc ResourceBinding = {};
        ResourceBinding.Kind = R.Kind;
        ResourceBinding.DXBinding.Register = R.DXBinding.Register;
        ResourceBinding.DXBinding.Space = R.DXBinding.Space;
        ResourceBinding.VKBinding = R.VKBinding;
        ResourceBinding.DescriptorCount = R.getArraySize();
        Layout.ResourceBindings.push_back(ResourceBinding);
      }
      Bindings.DescriptorSetDescs.push_back(Layout);
    }

    if (P.isCompute()) {
      if (P.Shaders.size() != 1 || P.Shaders[0].Stage != Stages::Compute)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Compute pipeline must have exactly one compute shader.");

      ShaderContainer CS = {};
      CS.EntryPoint = P.Shaders[0].Entry;
      CS.Shader = P.Shaders[0].Shader.get();

      auto PipelineStateOrErr =
          createPipelineCs("Compute Pipeline State", Bindings, CS);
      if (!PipelineStateOrErr)
        return PipelineStateOrErr.takeError();
      IS.Pipeline = std::move(*PipelineStateOrErr);
      llvm::outs() << "Compute Pipeline created.\n";

      if (auto Err = createComputeCommands(P, IS))
        return Err;
    } else {
      ShaderContainer VS = {};
      ShaderContainer PS = {};
      for (auto &Shader : P.Shaders) {
        if (Shader.Stage == Stages::Vertex) {
          VS.EntryPoint = Shader.Entry;
          VS.Shader = Shader.Shader.get();
        } else if (Shader.Stage == Stages::Pixel) {
          PS.EntryPoint = Shader.Entry;
          PS.Shader = Shader.Shader.get();
        }
      }

      llvm::SmallVector<InputLayoutDesc> InputLayout;
      for (auto &Attr : P.Bindings.VertexAttributes) {
        auto FormatOrErr = toFormat(Attr.Format, Attr.Channels);
        if (!FormatOrErr)
          return FormatOrErr.takeError();

        InputLayoutDesc Desc = {};
        Desc.Name = Attr.Name;
        Desc.Fmt = *FormatOrErr;
        Desc.OffsetInBytes = Attr.Offset;
        InputLayout.push_back(Desc);
      }

      auto FormatOrErr = toFormat(P.Bindings.RTargetBufferPtr->Format,
                                  P.Bindings.RTargetBufferPtr->Channels);
      if (!FormatOrErr)
        return FormatOrErr.takeError();

      llvm::SmallVector<Format> RTFormats;
      RTFormats.push_back(*FormatOrErr);

      auto PipelineStateOrErr =
          createPipelineVsPs("Graphics Pipeline State", Bindings, InputLayout,
                             RTFormats, Format::D32FloatS8Uint, VS, PS);
      if (!PipelineStateOrErr)
        return PipelineStateOrErr.takeError();
      IS.Pipeline = std::move(*PipelineStateOrErr);

      if (auto Err = createGraphicsCommands(P, IS))
        return Err;
    }

    auto SubmitResult = executeCommands(IS);
    if (!SubmitResult)
      return SubmitResult.takeError();

    if (auto Err = SubmitResult->waitForCompletion())
      return Err;

    if (auto Err = copyBack(P, IS))
      return Err;
    return llvm::Error::success();
  }

  virtual ~MTLDevice() {};

private:
  void queryCapabilities() {}
};

llvm::Error MTLComputeEncoder::batchBuildAS(llvm::ArrayRef<ASBuildItem> Items) {
  if (Items.empty())
    return llvm::Error::success();
  if (!CB || !CB->Dev)
    return llvm::createStringError(
        std::errc::not_supported,
        "Metal command buffer has no associated MTLDevice.");
  MTL::Device *MTLDev = CB->Dev->Device;
  if (!MTLDev->supportsRaytracing())
    return llvm::createStringError(
        std::errc::not_supported,
        "Ray tracing is not supported on this Metal device.");

  if (auto Err = ensureASEncoder())
    return Err;

  for (const auto &Item : Items) {
    auto *AS = llvm::cast<MTLAccelStruct>(Item.AS);
    MTL::AccelerationStructureDescriptor *Desc = nullptr;
    uint64_t ScratchSize = 0;

    if (const auto *BLAS = llvm::dyn_cast<const BLASBuildRequest *>(Item.Req)) {
      llvm::SmallVector<MTL::AccelerationStructureGeometryDescriptor *> Geoms;
      for (const auto &T : BLAS->Triangles) {
        auto *TD = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
                       ->init();
        auto *VB = llvm::cast<MTLBuffer>(T.VertexBuffer);
        TD->setVertexBuffer(VB->Buf);
        TD->setVertexBufferOffset(T.VertexBufferOffset);
        TD->setVertexStride(T.VertexStride);
        TD->setVertexFormat(getMetalPositionFormat(T.VertexFormat));
        TD->setTriangleCount(T.IndexBuffer ? T.IndexCount / 3
                                           : T.VertexCount / 3);
        if (T.IndexBuffer) {
          auto *IB = llvm::cast<MTLBuffer>(T.IndexBuffer);
          TD->setIndexBuffer(IB->Buf);
          TD->setIndexBufferOffset(T.IndexBufferOffset);
          TD->setIndexType(getMetalIndexType(T.IdxFormat));
        }
        TD->setOpaque(T.Opaque);
        Geoms.push_back(TD);
      }
      for (const auto &A : BLAS->AABBs) {
        auto *AD =
            MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
                ->init();
        auto *BB = llvm::cast<MTLBuffer>(A.AABBBuffer);
        AD->setBoundingBoxBuffer(BB->Buf);
        AD->setBoundingBoxBufferOffset(A.AABBBufferOffset);
        AD->setBoundingBoxStride(A.AABBStride);
        AD->setBoundingBoxCount(A.AABBCount);
        AD->setOpaque(A.Opaque);
        Geoms.push_back(AD);
      }
      auto *PD = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
      NS::Array *GeomArr = NS::Array::array(
          reinterpret_cast<NS::Object *const *>(Geoms.data()), Geoms.size());
      PD->setGeometryDescriptors(GeomArr);
      Desc = PD;
      ScratchSize = BLAS->Sizes.ScratchDataSizeInBytes;
      for (auto *G : Geoms)
        G->release();
    } else {
      const auto *TLAS = llvm::cast<const TLASBuildRequest *>(Item.Req);

      // Metal's MTLAccelerationStructureInstanceDescriptor references BLASes
      // by index into a separate `instancedAccelerationStructures` array,
      // not by GPU address. Deduplicate the BLAS pointers and remember
      // their indices.
      llvm::SmallVector<MTL::AccelerationStructure *> UniqueBLASes;
      llvm::SmallVector<uint32_t> InstanceASIdx;
      InstanceASIdx.reserve(TLAS->Instances.size());
      for (const auto &Inst : TLAS->Instances) {
        auto *MTLBLAS = llvm::cast<MTLAccelStruct>(Inst.BLAS);
        auto It = std::find(UniqueBLASes.begin(), UniqueBLASes.end(),
                            MTLBLAS->AccelStruct);
        uint32_t Idx;
        if (It == UniqueBLASes.end()) {
          Idx = static_cast<uint32_t>(UniqueBLASes.size());
          UniqueBLASes.push_back(MTLBLAS->AccelStruct);
        } else {
          Idx = static_cast<uint32_t>(It - UniqueBLASes.begin());
        }
        InstanceASIdx.push_back(Idx);
      }

      // Pack instance descriptors. Layout differs from VK/DX12: 32-byte
      // entries with an index instead of a GPU address.
      const size_t InstByteSize =
          TLAS->Instances.size() *
          sizeof(MTL::AccelerationStructureInstanceDescriptor);
      MTL::Buffer *InstBuf =
          MTLDev->newBuffer(InstByteSize, MTL::ResourceStorageModeShared);
      if (!InstBuf)
        return llvm::createStringError(
            std::errc::not_enough_memory,
            "Failed to allocate TLAS instance buffer.");
      auto *InstPtr =
          static_cast<MTL::AccelerationStructureInstanceDescriptor *>(
              InstBuf->contents());
      for (size_t I = 0; I < TLAS->Instances.size(); ++I) {
        const auto &Src = TLAS->Instances[I];
        auto &D = InstPtr[I];
        // Metal stores transform as packed 4x3 column-major; our high-level
        // Transform[3][4] is row-major. Transpose into Metal's layout.
        for (int Row = 0; Row < 3; ++Row)
          for (int Col = 0; Col < 4; ++Col)
            D.transformationMatrix.columns[Col][Row] = Src.Transform[Row][Col];
        D.options = MTL::AccelerationStructureInstanceOptionNone;
        D.mask = Src.InstanceMask;
        D.intersectionFunctionTableOffset = 0;
        D.accelerationStructureIndex = InstanceASIdx[I];
      }
      CB->KeepAliveMTLBuffers.push_back(InstBuf);

      auto *ID = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
      ID->setInstanceDescriptorBuffer(InstBuf);
      ID->setInstanceCount(TLAS->Instances.size());
      NS::Array *BLASArr = NS::Array::array(
          reinterpret_cast<NS::Object *const *>(UniqueBLASes.data()),
          UniqueBLASes.size());
      ID->setInstancedAccelerationStructures(BLASArr);
      Desc = ID;
      ScratchSize = TLAS->Sizes.ScratchDataSizeInBytes;
    }

    MTL::Buffer *Scratch =
        MTLDev->newBuffer(ScratchSize, MTL::ResourceStorageModePrivate);
    if (!Scratch) {
      Desc->release();
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "Failed to allocate AS scratch buffer.");
    }
    CB->KeepAliveMTLBuffers.push_back(Scratch);

    insertDebugSignpost("BuildAccelerationStructure");
    ASEnc->buildAccelerationStructure(AS->AccelStruct, Desc, Scratch, 0);
    Desc->release();
  }

  return llvm::Error::success();
}
} // namespace

llvm::Error offloadtest::initializeMetalDevices(
    const DeviceConfig /*Config*/,
    llvm::SmallVectorImpl<std::unique_ptr<Device>> &Devices) {
  MTL::Device *MetalDevice = MTL::CreateSystemDefaultDevice();
  MTL::CommandQueue *MetalQueue = MetalDevice->newCommandQueue();

  auto SubmitFenceOrErr = MTLFence::create(MetalDevice, "QueueSubmitFence");
  if (!SubmitFenceOrErr)
    return SubmitFenceOrErr.takeError();

  auto DefaultDev = std::make_unique<MTLDevice>(MetalDevice, MetalQueue,
                                                std::move(*SubmitFenceOrErr));
  Devices.push_back(std::move(DefaultDev));

  return llvm::Error::success();
}
