#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/BinaryOps.h>

namespace at::native {
namespace mps {

static const char* METAL_BINARY = R"BINARY_METAL(

#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void fmax(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmax(*input, *other);
}

template<typename T>
kernel void fmin(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmin(*input, *other);
}

#define REGISTER_FMAX_OP(DTYPE)                       \
template                                               \
[[host_name("fmax_" #DTYPE)]]                         \
kernel void fmax<DTYPE>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

#define REGISTER_FMIN_OP(DTYPE)                       \
template                                               \
[[host_name("fmin_" #DTYPE)]]                         \
kernel void fmin<DTYPE>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

REGISTER_FMAX_OP(float);
REGISTER_FMAX_OP(half);
REGISTER_FMIN_OP(float);
REGISTER_FMIN_OP(half);

)BINARY_METAL";

using namespace mps;

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> binaryLibrary = nil;
  if (binaryLibrary) {
    return binaryLibrary;
  }

  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion: MTLLanguageVersion2_3];
  binaryLibrary  = [device newLibraryWithSource:[NSString stringWithCString: METAL_BINARY encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(binaryLibrary, "Failed to create metal binary library, error: ", [[error description] UTF8String]);
  return binaryLibrary;
}

static id<MTLComputePipelineState> binaryPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> binaryLib = compileBinaryOpsLibrary(device);
  id<MTLFunction> binaryFunc = [binaryLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(binaryFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:binaryFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

void fmax_fmin_mps_impl(TensorIteratorBase& iter, const std::string max_min) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output(0);
  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(input);
  id<MTLBuffer> otherBuffer  = getMTLBufferStorage(other);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(out);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      NSError* error = nil;
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      for (const auto i: c10::irange(iterShape.size())) {
        TORCH_CHECK(i <= UINT32_MAX);
        iterShapeData[i] = (uint32_t)(iterShape[i]);
      }

      for (const auto i: c10::irange(nDim)) {
        for (const auto offset: c10::irange(nOffsets)) {
            strides[i][offset] = iter.strides(offset)[i];
        }
      }

      id<MTLFunction> kernelDataOffsetsFunction = MPSDevice::getInstance()->metalIndexingFunction("kernel_index_offsets", nil);
      id<MTLComputePipelineState> kernelDataOffsetsPSO = [[device newComputePipelineStateWithFunction: kernelDataOffsetsFunction
                                                                                                error: &error] autorelease];
      id<MTLBuffer> kernelDataOffsets = [[device newBufferWithLength: numThreads * sizeof(simd_uint3)
                                                             options: 0] autorelease];
      TORCH_CHECK(kernelDataOffsetsPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);
      [computeEncoder setComputePipelineState:kernelDataOffsetsPSO];
      [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
      [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
      [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];
      [computeEncoder setBytes:&nOffsets length:sizeof(uint32_t) atIndex:4];

      NSUInteger kernelOffsetsTGSize = kernelDataOffsetsPSO.maxTotalThreadsPerThreadgroup;
      if (kernelOffsetsTGSize > numThreads)
          kernelOffsetsTGSize = numThreads;

      MTLSize kernelOffsetsThreadGroupSize = MTLSizeMake(kernelOffsetsTGSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: kernelOffsetsThreadGroupSize];

      const std::string kernel = "f" + max_min + "_" + scalarToMetalTypeString(out.scalar_type());
      id<MTLComputePipelineState> fmaxfminPSO = binaryPipelineState(device, kernel);
      [computeEncoder setComputePipelineState:fmaxfminPSO];
      [computeEncoder setBuffer:inputBuffer  offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:otherBuffer  offset:other.storage_offset() * other.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:out.storage_offset() * out.element_size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];

      NSUInteger tgSize = fmaxfminPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];

      [computeEncoder endEncoding];
      mpsStream->commit(true);
    }
  });
}
} // namespace mps

void fmax_mps_kernel(TensorIteratorBase& iter) {
    if (isFloatingType(iter.common_dtype())) {
        mps::fmax_fmin_mps_impl(iter, "max");
    } else {
        at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
    }
}
void fmin_mps_kernel(TensorIteratorBase& iter) {
    if (isFloatingType(iter.common_dtype())) {
        mps::fmax_fmin_mps_impl(iter, "min");
    } else {
        at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
    }
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel);
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel);

} // namespace at::native
