// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../ffx_fsr2.h"
#include "ffx_fsr2_gl.h"
#include "glad/gl.h"
#include "shaders/ffx_fsr2_shaders_gl.h"  // include all the precompiled SPIR-V (GL) shaders for the FSR2 passes
#include "../ffx_fsr2_private.h"
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <codecvt>
#include <cstdio>
#include <string>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h> // GetModuleHandleA

// prototypes for functions in the interface
FfxErrorCode GetDeviceCapabilitiesGL(FfxFsr2Interface* backendInterface, FfxDeviceCapabilities* deviceCapabilities, FfxDevice);
FfxErrorCode CreateBackendContextGL(FfxFsr2Interface* backendInterface, FfxDevice);
FfxErrorCode DestroyBackendContextGL(FfxFsr2Interface* backendInterface);
FfxErrorCode CreateResourceGL(FfxFsr2Interface* backendInterface, const FfxCreateResourceDescription* desc, FfxResourceInternal* outResource);
FfxErrorCode RegisterResourceGL(FfxFsr2Interface* backendInterface, const FfxResource* inResource, FfxResourceInternal* outResourceInternal);
FfxErrorCode UnregisterResourcesGL(FfxFsr2Interface* backendInterface);
FfxResourceDescription GetResourceDescriptorGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource);
FfxErrorCode DestroyResourceGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource);
FfxErrorCode CreatePipelineGL(FfxFsr2Interface* backendInterface, FfxFsr2Pass passId, const FfxPipelineDescription* desc, FfxPipelineState* outPass);
FfxErrorCode DestroyPipelineGL(FfxFsr2Interface* backendInterface, FfxPipelineState* pipeline);
FfxErrorCode ScheduleGpuJobGL(FfxFsr2Interface* backendInterface, const FfxGpuJobDescription* job);
FfxErrorCode ExecuteGpuJobsGL(FfxFsr2Interface* backendInterface, FfxCommandList);

namespace
{
  constexpr uint32_t FSR2_MAX_QUEUED_FRAMES          = 4;
  constexpr uint32_t FSR2_MAX_RESOURCE_COUNT         = 64;
  constexpr uint32_t FSR2_MAX_STAGING_RESOURCE_COUNT = 8;
  constexpr uint32_t FSR2_MAX_GPU_JOBS               = 32;
  constexpr uint32_t FSR2_MAX_UNIFORM_BUFFERS        = 4;
  constexpr uint32_t FSR2_MAX_IMAGE_VIEWS            = 32;
  constexpr uint32_t FSR2_MAX_BUFFERED_DESCRIPTORS   = FFX_FSR2_PASS_COUNT * FSR2_MAX_QUEUED_FRAMES;
  constexpr uint32_t FSR2_UBO_RING_BUFFER_SIZE       = FSR2_MAX_BUFFERED_DESCRIPTORS * FSR2_MAX_UNIFORM_BUFFERS;
  constexpr uint32_t FSR2_UBO_SIZE                   = 256;
  constexpr uint32_t FSR2_DEFAULT_SUBGROUP_SIZE      = 32;

  namespace GL
  {
    struct Texture { GLuint id = {}; };
    struct Buffer { GLuint id = {}; };
    struct Sampler { GLuint id = {}; };
  }

  // Per-job GPU timing, enabled with FSR2_GL_PROFILE=1 in the environment. Timestamps
  // bracket each job; the previous frame's results are read (and occasionally printed)
  // at the start of the next ExecuteGpuJobsGL, so no stalls in steady state.
  struct JobProfiler
  {
    static constexpr int MAXQ = 32;
    int enabled = -1;  // -1 = not yet checked
    bool inited = false;
    PFNGLGENQUERIESPROC pglGenQueries = nullptr;
    PFNGLQUERYCOUNTERPROC pglQueryCounter = nullptr;
    PFNGLGETQUERYOBJECTUI64VPROC pglGetQueryObjectui64v = nullptr;
    GLuint queries[MAXQ * 2] = {};
    char labels[MAXQ][64] = {};
    int prevCount = 0;
    double sumMs[MAXQ] = {};
    int frames = 0;
  };
  JobProfiler g_jobprof;
}

struct BackendContext_GL {

  enum class Aspect { UNDEFINED, COLOR, DEPTH };

  // store for resources and resourceViews
  struct Resource
  {
#ifdef _DEBUG
    char                    resourceName[64] = {};
#endif
    FfxResourceDescription  resourceDescription;

    GL::Buffer buffer = {};

    GL::Texture textureAllMipsView = {};
    GL::Texture textureSingleMipViews[FSR2_MAX_IMAGE_VIEWS] = {};
    Aspect textureAspect = {};
  };

  struct UniformBuffer
  {
    GL::Buffer bufferResource = {};
    uint8_t* pData = {};
  };

  struct GLFunctionTable
  {
    ffx_glGetProcAddress glGetProcAddress = nullptr;
    PFNGLGETINTEGERVPROC glGetIntegerv = nullptr;
    PFNGLGETSTRINGIPROC glGetStringi = nullptr;
    PFNGLGETSTRINGPROC glGetString = nullptr;
    PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
    PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
    PFNGLOBJECTLABELPROC glObjectLabel = nullptr;
    PFNGLCREATESAMPLERSPROC glCreateSamplers = nullptr;
    PFNGLSAMPLERPARAMETERIPROC glSamplerParameteri = nullptr;
    PFNGLSAMPLERPARAMETERFPROC glSamplerParameterf = nullptr;
    PFNGLCREATEBUFFERSPROC glCreateBuffers = nullptr;
    PFNGLNAMEDBUFFERSTORAGEPROC glNamedBufferStorage = nullptr;
    PFNGLNAMEDBUFFERSUBDATAPROC glNamedBufferSubData = nullptr;
    PFNGLCREATETEXTURESPROC glCreateTextures = nullptr;
    PFNGLGENTEXTURESPROC glGenTextures = nullptr;
    PFNGLTEXTUREVIEWPROC glTextureView = nullptr;
    PFNGLTEXTURESTORAGE1DPROC glTextureStorage1D = nullptr;
    PFNGLTEXTURESTORAGE2DPROC glTextureStorage2D = nullptr;
    PFNGLTEXTURESTORAGE3DPROC glTextureStorage3D = nullptr;
    PFNGLCREATESHADERPROC glCreateShader = nullptr;
    PFNGLSHADERBINARYPROC glShaderBinary = nullptr;
    PFNGLSPECIALIZESHADERPROC glSpecializeShader = nullptr;
    PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
    PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
    PFNGLATTACHSHADERPROC glAttachShader = nullptr;
    PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
    PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
    PFNGLDELETETEXTURESPROC glDeleteTextures = nullptr;
    PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
    PFNGLDELETESAMPLERSPROC glDeleteSamplers = nullptr;
    PFNGLDELETESHADERPROC glDeleteShader = nullptr;
    PFNGLMAPNAMEDBUFFERRANGEPROC glMapNamedBufferRange = nullptr;
    PFNGLUNMAPNAMEDBUFFERPROC glUnmapNamedBuffer = nullptr;
    PFNGLMEMORYBARRIERPROC glMemoryBarrier = nullptr;
    PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
    PFNGLPROGRAMUNIFORM1IPROC glProgramUniform1i = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
    PFNGLBINDTEXTUREUNITPROC glBindTextureUnit = nullptr;
    PFNGLBINDSAMPLERPROC glBindSampler = nullptr;
    PFNGLBINDBUFFERRANGEPROC glBindBufferRange = nullptr;
    PFNGLBINDIMAGETEXTUREPROC glBindImageTexture = nullptr;
    PFNGLDISPATCHCOMPUTEPROC glDispatchCompute = nullptr;
    PFNGLCOPYNAMEDBUFFERSUBDATAPROC glCopyNamedBufferSubData = nullptr;
    PFNGLCOPYIMAGESUBDATAPROC glCopyImageSubData = nullptr;
    PFNGLTEXTURESUBIMAGE1DPROC glTextureSubImage1D = nullptr;
    PFNGLTEXTURESUBIMAGE2DPROC glTextureSubImage2D = nullptr;
    PFNGLTEXTURESUBIMAGE3DPROC glTextureSubImage3D = nullptr;
    PFNGLCLEARTEXIMAGEPROC glClearTexImage = nullptr;
  };

  GLFunctionTable         glFunctionTable = {};
  FfxDeviceCapabilities   capabilities = {};

  uint32_t                gpuJobCount = 0;
  FfxGpuJobDescription    gpuJobs[FSR2_MAX_GPU_JOBS] = {};

  uint32_t                nextStaticResource = 0;
  uint32_t                nextDynamicResource = 0;
  uint32_t                stagingResourceCount = 0;
  Resource                resources[FSR2_MAX_RESOURCE_COUNT] = {};
  FfxResourceInternal     stagingResources[FSR2_MAX_STAGING_RESOURCE_COUNT] = {};

  GL::Sampler             pointSampler = {};
  GL::Sampler             linearSampler = {};

  UniformBuffer           uboRingBuffer[FSR2_UBO_RING_BUFFER_SIZE] = {};
  uint32_t                uboRingBufferIndex = 0;
};

FFX_API size_t ffxFsr2GetScratchMemorySizeGL()
{
  return sizeof(BackendContext_GL);
}

FfxErrorCode ffxFsr2GetInterfaceGL(
  FfxFsr2Interface* outInterface,
  void* scratchBuffer,
  size_t scratchBufferSize,
  ffx_glGetProcAddress getProcAddress)
{
  FFX_RETURN_ON_ERROR(
    outInterface,
    FFX_ERROR_INVALID_POINTER);
  FFX_RETURN_ON_ERROR(
    scratchBuffer,
    FFX_ERROR_INVALID_POINTER);
  FFX_RETURN_ON_ERROR(
    scratchBufferSize >= ffxFsr2GetScratchMemorySizeGL(),
    FFX_ERROR_INSUFFICIENT_MEMORY);

  outInterface->fpGetDeviceCapabilities = GetDeviceCapabilitiesGL;
  outInterface->fpCreateBackendContext = CreateBackendContextGL;
  outInterface->fpDestroyBackendContext = DestroyBackendContextGL;
  outInterface->fpCreateResource = CreateResourceGL;
  outInterface->fpRegisterResource = RegisterResourceGL;
  outInterface->fpUnregisterResources = UnregisterResourcesGL;
  outInterface->fpGetResourceDescription = GetResourceDescriptorGL;
  outInterface->fpDestroyResource = DestroyResourceGL;
  outInterface->fpCreatePipeline = CreatePipelineGL;
  outInterface->fpDestroyPipeline = DestroyPipelineGL;
  outInterface->fpScheduleGpuJob = ScheduleGpuJobGL;
  outInterface->fpExecuteGpuJobs = ExecuteGpuJobsGL;
  outInterface->scratchBuffer = scratchBuffer;
  outInterface->scratchBufferSize = scratchBufferSize;

  BackendContext_GL* context = (BackendContext_GL*)scratchBuffer;

  context->glFunctionTable.glGetProcAddress = getProcAddress;

  return FFX_OK;
}

static void loadGLFunctions(BackendContext_GL* backendContext, ffx_glGetProcAddress getProcAddress)
{
  FFX_ASSERT(NULL != backendContext);

  backendContext->glFunctionTable.glObjectLabel = (PFNGLOBJECTLABELPROC)getProcAddress("glObjectLabel");
  backendContext->glFunctionTable.glGetIntegerv = (PFNGLGETINTEGERVPROC)getProcAddress("glGetIntegerv");
  backendContext->glFunctionTable.glGetString = (PFNGLGETSTRINGPROC)getProcAddress("glGetString");
  backendContext->glFunctionTable.glGetStringi = (PFNGLGETSTRINGIPROC)getProcAddress("glGetStringi");
  backendContext->glFunctionTable.glGetShaderiv = (PFNGLGETSHADERIVPROC)getProcAddress("glGetShaderiv");
  backendContext->glFunctionTable.glGetProgramiv = (PFNGLGETPROGRAMIVPROC)getProcAddress("glGetProgramiv");
  backendContext->glFunctionTable.glCreateSamplers = (PFNGLCREATESAMPLERSPROC)getProcAddress("glCreateSamplers");
  backendContext->glFunctionTable.glSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC)getProcAddress("glSamplerParameteri");
  backendContext->glFunctionTable.glSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC)getProcAddress("glSamplerParameterf");
  backendContext->glFunctionTable.glCreateBuffers = (PFNGLCREATEBUFFERSPROC)getProcAddress("glCreateBuffers");
  backendContext->glFunctionTable.glNamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC)getProcAddress("glNamedBufferStorage");
  backendContext->glFunctionTable.glNamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC)getProcAddress("glNamedBufferSubData");
  backendContext->glFunctionTable.glCreateTextures = (PFNGLCREATETEXTURESPROC)getProcAddress("glCreateTextures");
  backendContext->glFunctionTable.glGenTextures = (PFNGLGENTEXTURESPROC)getProcAddress("glGenTextures");
  backendContext->glFunctionTable.glTextureView = (PFNGLTEXTUREVIEWPROC)getProcAddress("glTextureView");
  backendContext->glFunctionTable.glTextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC)getProcAddress("glTextureStorage1D");
  backendContext->glFunctionTable.glTextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC)getProcAddress("glTextureStorage2D");
  backendContext->glFunctionTable.glTextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC)getProcAddress("glTextureStorage3D");
  backendContext->glFunctionTable.glCreateShader = (PFNGLCREATESHADERPROC)getProcAddress("glCreateShader");
  backendContext->glFunctionTable.glShaderBinary = (PFNGLSHADERBINARYPROC)getProcAddress("glShaderBinary");
  backendContext->glFunctionTable.glSpecializeShader = (PFNGLSPECIALIZESHADERPROC)getProcAddress("glSpecializeShader");
  backendContext->glFunctionTable.glCompileShader = (PFNGLCOMPILESHADERPROC)getProcAddress("glCompileShader");
  backendContext->glFunctionTable.glCreateProgram = (PFNGLCREATEPROGRAMPROC)getProcAddress("glCreateProgram");
  backendContext->glFunctionTable.glAttachShader = (PFNGLATTACHSHADERPROC)getProcAddress("glAttachShader");
  backendContext->glFunctionTable.glLinkProgram = (PFNGLLINKPROGRAMPROC)getProcAddress("glLinkProgram");
  backendContext->glFunctionTable.glDeleteProgram = (PFNGLDELETEPROGRAMPROC)getProcAddress("glDeleteProgram");
  backendContext->glFunctionTable.glDeleteTextures = (PFNGLDELETETEXTURESPROC)getProcAddress("glDeleteTextures");
  backendContext->glFunctionTable.glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)getProcAddress("glDeleteBuffers");
  backendContext->glFunctionTable.glDeleteSamplers = (PFNGLDELETESAMPLERSPROC)getProcAddress("glDeleteSamplers");
  backendContext->glFunctionTable.glDeleteShader = (PFNGLDELETESHADERPROC)getProcAddress("glDeleteShader");
  backendContext->glFunctionTable.glMapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC)getProcAddress("glMapNamedBufferRange");
  backendContext->glFunctionTable.glUnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC)getProcAddress("glUnmapNamedBuffer");
  backendContext->glFunctionTable.glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)getProcAddress("glMemoryBarrier");
  backendContext->glFunctionTable.glUseProgram = (PFNGLUSEPROGRAMPROC)getProcAddress("glUseProgram");
  backendContext->glFunctionTable.glProgramUniform1i = (PFNGLPROGRAMUNIFORM1IPROC)getProcAddress("glProgramUniform1i");
  backendContext->glFunctionTable.glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)getProcAddress("glGetUniformLocation");
  backendContext->glFunctionTable.glBindTextureUnit = (PFNGLBINDTEXTUREUNITPROC)getProcAddress("glBindTextureUnit");
  backendContext->glFunctionTable.glBindSampler = (PFNGLBINDSAMPLERPROC)getProcAddress("glBindSampler");
  backendContext->glFunctionTable.glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)getProcAddress("glBindBufferRange");
  backendContext->glFunctionTable.glBindImageTexture = (PFNGLBINDIMAGETEXTUREPROC)getProcAddress("glBindImageTexture");
  backendContext->glFunctionTable.glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)getProcAddress("glDispatchCompute");
  backendContext->glFunctionTable.glCopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC)getProcAddress("glCopyNamedBufferSubData");
  backendContext->glFunctionTable.glCopyImageSubData = (PFNGLCOPYIMAGESUBDATAPROC)getProcAddress("glCopyImageSubData");
  backendContext->glFunctionTable.glTextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC)getProcAddress("glTextureSubImage1D");
  backendContext->glFunctionTable.glTextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC)getProcAddress("glTextureSubImage2D");
  backendContext->glFunctionTable.glTextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC)getProcAddress("glTextureSubImage3D");
  backendContext->glFunctionTable.glClearTexImage = (PFNGLCLEARTEXIMAGEPROC)getProcAddress("glClearTexImage");
}

static GLenum getGLFormatFromSurfaceFormat(FfxSurfaceFormat fmt)
{
  switch (fmt)
  {
  case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
    return GL_RGBA32F;
  case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
    return GL_RGBA32F;
  case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
    return GL_RGBA16F;
  case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:
    return GL_RGBA16;
  case FFX_SURFACE_FORMAT_R32G32_FLOAT:
    return GL_RG32F;
  case FFX_SURFACE_FORMAT_R32_UINT:
    return GL_R32UI;
  case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
    return GL_RGBA8;
  case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
    return GL_RGBA8;
  case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
    return GL_R11F_G11F_B10F;
  case FFX_SURFACE_FORMAT_R16G16_FLOAT:
    return GL_RG16F;
  case FFX_SURFACE_FORMAT_R16G16_UINT:
    return GL_RG16UI;
  case FFX_SURFACE_FORMAT_R16_FLOAT:
    return GL_R16F;
  case FFX_SURFACE_FORMAT_R16_UINT:
    return GL_R16UI;
  case FFX_SURFACE_FORMAT_R16_UNORM:
    return GL_R16;
  case FFX_SURFACE_FORMAT_R16_SNORM:
    return GL_R16_SNORM;
  case FFX_SURFACE_FORMAT_R8_UNORM:
    return GL_R8;
  case FFX_SURFACE_FORMAT_R8G8_UNORM:
    return GL_RG8;
  case FFX_SURFACE_FORMAT_R32_FLOAT:
    return GL_R32F;
  case FFX_SURFACE_FORMAT_R8_UINT:
    return GL_R8UI;
  default:
    FFX_ASSERT_FAIL("");
    return 0;
  }
}

static GLenum getGLUploadFormatFromSurfaceFormat(FfxSurfaceFormat fmt)
{
  switch (fmt)
  {
  case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
  case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:
  case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
  case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
    return GL_RGBA;
  case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
    return GL_RGB;
  case FFX_SURFACE_FORMAT_R32G32_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16_UINT:
  case FFX_SURFACE_FORMAT_R8G8_UNORM:
    return GL_RG;
  case FFX_SURFACE_FORMAT_R16_FLOAT:
  case FFX_SURFACE_FORMAT_R16_UNORM:
  case FFX_SURFACE_FORMAT_R16_SNORM:
  case FFX_SURFACE_FORMAT_R8_UNORM:
  case FFX_SURFACE_FORMAT_R32_FLOAT:
    return GL_RED;
  case FFX_SURFACE_FORMAT_R8_UINT:
  case FFX_SURFACE_FORMAT_R16_UINT:
  case FFX_SURFACE_FORMAT_R32_UINT:
    return GL_RED_INTEGER;
  default: FFX_ASSERT_FAIL(""); return 0;
  }
}

static GLenum getGLUploadTypeFromSurfaceFormat(FfxSurfaceFormat fmt)
{
  switch (fmt)
  {
  case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
  case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
  case FFX_SURFACE_FORMAT_R32G32_FLOAT:
  case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
  case FFX_SURFACE_FORMAT_R16G16_FLOAT:
  case FFX_SURFACE_FORMAT_R16_FLOAT:
  case FFX_SURFACE_FORMAT_R32_FLOAT:
    return GL_FLOAT;
  case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
  case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
  case FFX_SURFACE_FORMAT_R8G8_UNORM:
  case FFX_SURFACE_FORMAT_R8_UNORM:
    return GL_UNSIGNED_BYTE;
  case FFX_SURFACE_FORMAT_R32_UINT:
    return GL_UNSIGNED_INT;
  case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:
  case FFX_SURFACE_FORMAT_R16_UNORM:
  case FFX_SURFACE_FORMAT_R16G16_UINT:
  case FFX_SURFACE_FORMAT_R16_UINT:
  case FFX_SURFACE_FORMAT_R8_UINT:
    return GL_UNSIGNED_SHORT;
  case FFX_SURFACE_FORMAT_R16_SNORM:
    return GL_SHORT;
  default: FFX_ASSERT_FAIL(""); return 0;
  }
}

FfxSurfaceFormat ffxGetSurfaceFormatGL(GLenum fmt)
{
  switch (fmt)
  {
  case GL_RGBA32F:
    return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
  case GL_RGBA16F:
    return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
  case GL_RGBA16:
    return FFX_SURFACE_FORMAT_R16G16B16A16_UNORM;
  case GL_RG32F:
    return FFX_SURFACE_FORMAT_R32G32_FLOAT;
  case GL_R32UI:
    return FFX_SURFACE_FORMAT_R32_UINT;
  case GL_RGBA8:
    return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
  case GL_R11F_G11F_B10F:
    return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;
  case GL_RG16F:
    return FFX_SURFACE_FORMAT_R16G16_FLOAT;
  case GL_RG16UI:
    return FFX_SURFACE_FORMAT_R16G16_UINT;
  case GL_R16F:
    return FFX_SURFACE_FORMAT_R16_FLOAT;
  case GL_R16UI:
    return FFX_SURFACE_FORMAT_R16_UINT;
  case GL_R16:
    return FFX_SURFACE_FORMAT_R16_UNORM;
  case GL_R16_SNORM:
    return FFX_SURFACE_FORMAT_R16_SNORM;
  case GL_R8:
    return FFX_SURFACE_FORMAT_R8_UNORM;
  case GL_R32F:
    return FFX_SURFACE_FORMAT_R32_FLOAT;
  case GL_R8UI:
    return FFX_SURFACE_FORMAT_R8_UINT;
  default:
    return FFX_SURFACE_FORMAT_UNKNOWN;
  }
}

static BackendContext_GL::UniformBuffer accquireDynamicUBO(BackendContext_GL* backendContext, uint32_t size, const void* pData)
{
  // the ubo ring buffer is pre-populated with buffer objects of 256-bytes to prevent creating buffers at runtime
  FFX_ASSERT(size <= 256);

  BackendContext_GL::UniformBuffer& ubo = backendContext->uboRingBuffer[backendContext->uboRingBufferIndex];

  if (pData)
  {
    backendContext->glFunctionTable.glNamedBufferSubData(ubo.bufferResource.id, 0, size, pData);
  }

  backendContext->uboRingBufferIndex++;

  if (backendContext->uboRingBufferIndex >= FSR2_UBO_RING_BUFFER_SIZE)
  {
    backendContext->uboRingBufferIndex = 0;
  }

  return ubo;
}

FfxResource ffxGetTextureResourceGL(GLuint textureGL, uint32_t width, uint32_t height, GLenum imgFormat, const wchar_t* name)
{
  FfxResource resource = {};
  resource.resource = reinterpret_cast<void*>(static_cast<uintptr_t>(textureGL));
  resource.descriptorData = 0;
  resource.description.flags = FFX_RESOURCE_FLAGS_NONE;
  resource.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
  resource.description.width = width;
  resource.description.height = height;
  resource.description.depth = 1;
  resource.description.mipCount = 1;
  resource.description.format = ffxGetSurfaceFormatGL(imgFormat);

  switch (imgFormat)
  {
  case GL_DEPTH_COMPONENT16:
  case GL_DEPTH_COMPONENT24:
  case GL_DEPTH_COMPONENT32F:
  case GL_DEPTH24_STENCIL8:
  case GL_DEPTH32F_STENCIL8:
  {
    resource.isDepth = true;
    break;
  }
  default:
  {
    resource.isDepth = false;
    break;
  }
  }

#ifdef _DEBUG
  if (name)
  {
    wcscpy_s(resource.name, name);
  }
#endif

  return resource;
}

FfxResource ffxGetBufferResourceGL(GLuint bufferGL, uint32_t size, const wchar_t* name)
{
  FfxResource resource = {};
  resource.resource = reinterpret_cast<void*>(static_cast<uintptr_t>(bufferGL));
  resource.descriptorData = 0;
  resource.description.flags = FFX_RESOURCE_FLAGS_NONE;
  resource.description.type = FFX_RESOURCE_TYPE_BUFFER;
  resource.description.width = size;
  resource.description.height = 1;
  resource.description.depth = 1;
  resource.description.mipCount = 1;
  resource.description.format = FFX_SURFACE_FORMAT_UNKNOWN;
  resource.isDepth = false;

#ifdef _DEBUG
  if (name)
  {
    wcscpy_s(resource.name, name);
  }
#endif

  return resource;
}

GLuint ffxGetGLImage(FfxFsr2Context* context, uint32_t resId)
{
  FFX_ASSERT(context);

  FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)(context);
  BackendContext_GL* backendContext = (BackendContext_GL*)(contextPrivate->contextDescription.callbacks.scratchBuffer);

  int32_t internalIndex = contextPrivate->uavResources[resId].internalIndex;

  return (internalIndex == -1) ? 0 : backendContext->resources[internalIndex].textureAllMipsView.id;
}

FfxErrorCode RegisterResourceGL(
  FfxFsr2Interface* backendInterface,
  const FfxResource* inFfxResource,
  FfxResourceInternal* outFfxResourceInternal
)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)(backendInterface->scratchBuffer);

  if (inFfxResource->resource == nullptr) {

    outFfxResourceInternal->internalIndex = FFX_FSR2_RESOURCE_IDENTIFIER_NULL;
    return FFX_OK;
  }

  FFX_ASSERT(backendContext->nextDynamicResource > backendContext->nextStaticResource);
  outFfxResourceInternal->internalIndex = backendContext->nextDynamicResource--;

  BackendContext_GL::Resource* backendResource = &backendContext->resources[outFfxResourceInternal->internalIndex];

  backendResource->resourceDescription = inFfxResource->description;

#ifdef _DEBUG
  size_t retval = 0;
  wcstombs_s(&retval, backendResource->resourceName, sizeof(backendResource->resourceName), inFfxResource->name, sizeof(backendResource->resourceName));
  if (retval >= 64) backendResource->resourceName[63] = '\0';
#endif

  if (inFfxResource->description.type == FFX_RESOURCE_TYPE_BUFFER)
  {
    const auto buffer = static_cast<GLuint>(reinterpret_cast<uintptr_t>(inFfxResource->resource));

    backendResource->buffer = { buffer };
  }
  else
  {
    const auto texture = static_cast<GLuint>(reinterpret_cast<uintptr_t>(inFfxResource->resource));

    backendResource->textureAllMipsView = { texture };
    backendResource->textureSingleMipViews[0] = { texture };

    if (texture) {
      if (inFfxResource->isDepth)
      {
        backendResource->textureAspect = BackendContext_GL::Aspect::DEPTH;
      }
      else
      {
        backendResource->textureAspect = BackendContext_GL::Aspect::COLOR;
      }
    }
  }

  return FFX_OK;
}

// dispose dynamic resources: This should be called at the end of the frame
FfxErrorCode UnregisterResourcesGL(FfxFsr2Interface* backendInterface)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)(backendInterface->scratchBuffer);

  backendContext->nextDynamicResource = FSR2_MAX_RESOURCE_COUNT - 1;

  return FFX_OK;
}

FfxErrorCode GetDeviceCapabilitiesGL(FfxFsr2Interface* backendInterface, FfxDeviceCapabilities* deviceCapabilities, FfxDevice)
{
  FFX_ASSERT(backendInterface);
  FFX_ASSERT(deviceCapabilities);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  // no shader model in vulkan so assume the minimum
  deviceCapabilities->minimumSupportedShaderModel = FFX_SHADER_MODEL_5_1;
  deviceCapabilities->waveLaneCountMin = 0;
  deviceCapabilities->waveLaneCountMax = 0;
  deviceCapabilities->fp16Supported = false;
  deviceCapabilities->raytracingSupported = false;

  // check if extensions are supported

  // Workaround: latest AMD driver does not report GL_KHR_shader_subgroup, despite the extension being supported
  bool vendorIsAmd = false;
  const auto* vendor = reinterpret_cast<const char*>(backendContext->glFunctionTable.glGetString(GL_VENDOR));
  if (strstr(vendor, "ATI"))
  {
    vendorIsAmd = true;
  }

  bool subgroupSupported = false;
  GLint numExtensions{};
  backendContext->glFunctionTable.glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
  for (GLint i = 0; i < numExtensions; i++)
  {
    const auto* extensionString = reinterpret_cast<const char*>(backendContext->glFunctionTable.glGetStringi(GL_EXTENSIONS, i));
    if (vendorIsAmd || (strcmp(extensionString, "GL_KHR_shader_subgroup") == 0))
    {
      GLint supportedStages{};
      backendContext->glFunctionTable.glGetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &supportedStages);
      if (supportedStages & GL_COMPUTE_SHADER_BIT)
      {
        subgroupSupported = true;
      }
    }

    if (strcmp(extensionString, "GL_NV_gpu_shader5") == 0 || strcmp(extensionString, "GL_AMD_gpu_shader_half_float") == 0)
    {
      deviceCapabilities->fp16Supported = true;
    }
  }

  // Workaround: RenderDoc prevents many extensions from being reported
  // In this case, shaders that use GL_KHR_shader_subgroup will still work, but API calls using constants from it will not
  bool renderDocIsAttached = GetModuleHandleA("renderdoc.dll");
  if (renderDocIsAttached)
  {
    subgroupSupported = true;
  }

  if (!subgroupSupported)
  {
    return FFX_ERROR_BACKEND_API_ERROR; // GL_KHR_shader_subgroup is required
  }

  GLint subgroupSize = FSR2_DEFAULT_SUBGROUP_SIZE;
  if (!renderDocIsAttached)
  {
    backendContext->glFunctionTable.glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
  }
  deviceCapabilities->waveLaneCountMin = static_cast<uint32_t>(subgroupSize);
  deviceCapabilities->waveLaneCountMax = static_cast<uint32_t>(subgroupSize);
  
  return FFX_OK;
}

FfxErrorCode CreateBackendContextGL(FfxFsr2Interface* backendInterface, FfxDevice)
{
  FFX_ASSERT(backendInterface);

  // set up some internal resources we need (space for resource views and constant buffers)
  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  backendContext->nextStaticResource = 0;
  backendContext->nextDynamicResource = FSR2_MAX_RESOURCE_COUNT - 1;

  // load OpenGL functions
  loadGLFunctions(backendContext, backendContext->glFunctionTable.glGetProcAddress);

  FFX_VALIDATE(GetDeviceCapabilitiesGL(backendInterface, &backendContext->capabilities, nullptr));

  // create samplers
  backendContext->glFunctionTable.glCreateSamplers(1, &backendContext->pointSampler.id);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->pointSampler.id, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->pointSampler.id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->pointSampler.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->pointSampler.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->pointSampler.id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->pointSampler.id, GL_TEXTURE_MIN_LOD, -1000);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->pointSampler.id, GL_TEXTURE_MAX_LOD, 1000);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->pointSampler.id, GL_TEXTURE_MAX_ANISOTROPY, 1);

  backendContext->glFunctionTable.glCreateSamplers(1, &backendContext->linearSampler.id);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->linearSampler.id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->linearSampler.id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->linearSampler.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->linearSampler.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameteri(backendContext->linearSampler.id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->linearSampler.id, GL_TEXTURE_MIN_LOD, -1000);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->linearSampler.id, GL_TEXTURE_MAX_LOD, 1000);
  backendContext->glFunctionTable.glSamplerParameterf(backendContext->linearSampler.id, GL_TEXTURE_MAX_ANISOTROPY, 1);

  // allocate ring buffer of uniform buffers
  for (uint32_t i = 0; i < FSR2_UBO_RING_BUFFER_SIZE; i++)
  {
    BackendContext_GL::UniformBuffer& ubo = backendContext->uboRingBuffer[i];
    backendContext->glFunctionTable.glCreateBuffers(1, &ubo.bufferResource.id);
    // Device-local storage updated with glNamedBufferSubData: a persistently+coherently
    // mapped buffer lives in snooped system memory on NVIDIA, and every shader read of
    // the constants then goes over the bus instead of through the cache hierarchy.
    backendContext->glFunctionTable.glNamedBufferStorage(ubo.bufferResource.id, FSR2_UBO_SIZE, nullptr, GL_DYNAMIC_STORAGE_BIT);
    ubo.pData = nullptr;
  }
  }

  backendContext->gpuJobCount = 0;
  backendContext->stagingResourceCount = 0;
  backendContext->uboRingBufferIndex = 0;

  return FFX_OK;
}

FfxErrorCode DestroyBackendContextGL(FfxFsr2Interface* backendInterface)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  for (uint32_t i = 0; i < backendContext->stagingResourceCount; i++)
  {
    DestroyResourceGL(backendInterface, backendContext->stagingResources[i]);
  }

  for (uint32_t i = 0; i < FSR2_UBO_RING_BUFFER_SIZE; i++)
  {
    BackendContext_GL::UniformBuffer& ubo = backendContext->uboRingBuffer[i];

    // buffer is implicitly unmapped by deleting it
    backendContext->glFunctionTable.glDeleteBuffers(1, &ubo.bufferResource.id);
  }

  backendContext->glFunctionTable.glDeleteSamplers(1, &backendContext->pointSampler.id);
  backendContext->glFunctionTable.glDeleteSamplers(1, &backendContext->linearSampler.id);

  // clear all the fields of the context
  *backendContext = {};

  return FFX_OK;
}

// create a internal resource that will stay alive until effect gets shut down
FfxErrorCode CreateResourceGL(
  FfxFsr2Interface* backendInterface,
  const FfxCreateResourceDescription* createResourceDescription,
  FfxResourceInternal* outResource)
{
  FFX_ASSERT(backendInterface);
  FFX_ASSERT(createResourceDescription);
  FFX_ASSERT(outResource);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  FFX_ASSERT(backendContext->nextStaticResource + 1 < backendContext->nextDynamicResource);
  outResource->internalIndex = backendContext->nextStaticResource++;
  BackendContext_GL::Resource* res = &backendContext->resources[outResource->internalIndex];
  res->resourceDescription = createResourceDescription->resourceDescription;
  res->resourceDescription.mipCount = createResourceDescription->resourceDescription.mipCount;

  if (res->resourceDescription.mipCount == 0)
  {
    res->resourceDescription.mipCount = (uint32_t)(1 + floor(log2(FFX_MAXIMUM(FFX_MAXIMUM(createResourceDescription->resourceDescription.width, createResourceDescription->resourceDescription.height), createResourceDescription->resourceDescription.depth))));
  }

#ifdef _DEBUG
  size_t retval = 0;
  wcstombs_s(&retval, res->resourceName, sizeof(res->resourceName), createResourceDescription->name, sizeof(res->resourceName));
  if (retval >= 64) res->resourceName[63] = '\0';
#endif

  switch (createResourceDescription->resourceDescription.type)
  {
  case FFX_RESOURCE_TYPE_BUFFER:
  {
    if (createResourceDescription->initData)
    {
      FFX_ASSERT(createResourceDescription->resourceDescription.width == createResourceDescription->initDataSize);
    }

    backendContext->glFunctionTable.glCreateBuffers(1, &res->buffer.id);
    backendContext->glFunctionTable.glNamedBufferStorage(
      res->buffer.id,
      createResourceDescription->resourceDescription.width,
      createResourceDescription->initData,
      0);

#ifdef _DEBUG
    backendContext->glFunctionTable.glObjectLabel(GL_BUFFER, res->buffer.id, -1, res->resourceName);
#endif
    break;
  }
  case FFX_RESOURCE_TYPE_TEXTURE1D:
  {
    backendContext->glFunctionTable.glCreateTextures(GL_TEXTURE_1D, 1, &res->textureAllMipsView.id);
    backendContext->glFunctionTable.glTextureStorage1D(
      res->textureAllMipsView.id,
      res->resourceDescription.mipCount,
      getGLFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
      createResourceDescription->resourceDescription.width);

    if (createResourceDescription->initData)
    {
      backendContext->glFunctionTable.glad_glTextureSubImage1D(
        res->textureAllMipsView.id,
        0,
        0,
        createResourceDescription->resourceDescription.width,
        getGLUploadFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        getGLUploadTypeFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        createResourceDescription->initData);
    }

    break;
  }
  case FFX_RESOURCE_TYPE_TEXTURE2D:
  {
    backendContext->glFunctionTable.glCreateTextures(GL_TEXTURE_2D, 1, &res->textureAllMipsView.id);
    backendContext->glFunctionTable.glTextureStorage2D(
      res->textureAllMipsView.id,
      res->resourceDescription.mipCount,
      getGLFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
      createResourceDescription->resourceDescription.width,
      createResourceDescription->resourceDescription.height);

    if (createResourceDescription->initData)
    {
      backendContext->glFunctionTable.glad_glTextureSubImage2D(
        res->textureAllMipsView.id,
        0,
        0,
        0,
        createResourceDescription->resourceDescription.width,
        createResourceDescription->resourceDescription.height,
        getGLUploadFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        getGLUploadTypeFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        createResourceDescription->initData);
    }

    break;
  }
  case FFX_RESOURCE_TYPE_TEXTURE3D:
  {
    backendContext->glFunctionTable.glCreateTextures(GL_TEXTURE_3D, 1, &res->textureAllMipsView.id);
    backendContext->glFunctionTable.glTextureStorage3D(
      res->textureAllMipsView.id,
      res->resourceDescription.mipCount,
      getGLFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
      createResourceDescription->resourceDescription.width,
      createResourceDescription->resourceDescription.height,
      createResourceDescription->resourceDescription.depth);

    if (createResourceDescription->initData)
    {
      backendContext->glFunctionTable.glad_glTextureSubImage3D(
        res->textureAllMipsView.id,
        0,
        0,
        0,
        0,
        createResourceDescription->resourceDescription.width,
        createResourceDescription->resourceDescription.height,
        createResourceDescription->resourceDescription.depth,
        getGLUploadFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        getGLUploadTypeFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        createResourceDescription->initData);
    }
    break;
  }
  default:;
  }

  if (createResourceDescription->resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER)
  {
    GLenum type = 0;
    switch (createResourceDescription->resourceDescription.type)
    {
    case FFX_RESOURCE_TYPE_TEXTURE1D: type = GL_TEXTURE_1D; break;
    case FFX_RESOURCE_TYPE_TEXTURE2D: type = GL_TEXTURE_2D; break;
    case FFX_RESOURCE_TYPE_TEXTURE3D: type = GL_TEXTURE_3D; break;
    }

    res->textureAspect = BackendContext_GL::Aspect::COLOR;

    for (uint32_t i = 0; i < res->resourceDescription.mipCount; i++)
    {
      backendContext->glFunctionTable.glGenTextures(1, &res->textureSingleMipViews[i].id);
      backendContext->glFunctionTable.glTextureView(
        res->textureSingleMipViews[i].id,
        type,
        res->textureAllMipsView.id,
        getGLFormatFromSurfaceFormat(createResourceDescription->resourceDescription.format),
        i,
        1,
        0,
        1);

      // texture view name
#ifdef _DEBUG
      backendContext->glFunctionTable.glObjectLabel(GL_TEXTURE, res->textureSingleMipViews[i].id, -1, res->resourceName);
#endif
    }

    // texture name
#ifdef _DEBUG
    backendContext->glFunctionTable.glObjectLabel(GL_TEXTURE, res->textureAllMipsView.id, -1, res->resourceName);
#endif
  }

  return FFX_OK;
}

FfxResourceDescription GetResourceDescriptorGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  if (resource.internalIndex == -1)
  {
    return {};
  }

  return backendContext->resources[resource.internalIndex].resourceDescription;
}

// --- Native-GLSL pipeline path --------------------------------------------------
// The shipped pipelines are SPIR-V blobs fed to glShaderBinary, but NVIDIA's GL
// SPIR-V front end generates dramatically worse code than its GLSL front end for
// these shaders (accumulate 3.0ms vs 1.0ms, depth clip 0.8ms vs 0.13ms at
// 1920x1035 -> 4K on a 3080; the whole effect drops from ~4.2ms to ~1.5ms). So on
// NVIDIA the same sources are compiled through the GLSL front end at runtime, from
// ffx_fsr2_glsl_embedded.h (generated by generate_glsl_embedded.py, which also
// pre-applies the few native-GL compatibility transforms). FSR2_GL_SPIRV=1 in the
// environment forces the SPIR-V path, FSR2_GL_GLSL=1 forces GLSL on any vendor;
// any failure falls back to SPIR-V.

#include "ffx_fsr2_glsl_embedded.h"

static const char* fsr2GlslFindEmbedded(const char* name)
{
  for (size_t i = 0; i < sizeof(g_glslEmbeddedFiles) / sizeof(g_glslEmbeddedFiles[0]); i++)
  {
    if (strcmp(g_glslEmbeddedFiles[i].name, name) == 0) return g_glslEmbeddedFiles[i].data;
  }
  return nullptr;
}

static std::string fsr2GlslExpandIncludes(const char* text, int depth)
{
  if (depth > 8 || !text) return {};
  std::string src = text;
  std::string out;
  size_t pos = 0;
  while (pos < src.size())
  {
    size_t eol = src.find('\n', pos);
    if (eol == std::string::npos) eol = src.size();
    std::string line = src.substr(pos, eol - pos);
    size_t inc = line.find("#include");
    if (inc != std::string::npos && line.find_first_not_of(" \t") == inc)
    {
      size_t q0 = line.find('"', inc);
      size_t q1 = (q0 == std::string::npos) ? std::string::npos : line.find('"', q0 + 1);
      if (q1 != std::string::npos)
      {
        std::string file = line.substr(q0 + 1, q1 - q0 - 1);
        // Includes inside preprocessor-dead fp16 blocks are not embedded; skip them.
        const char* body = fsr2GlslFindEmbedded(file.c_str());
        if (body)
        {
          out += fsr2GlslExpandIncludes(body, depth + 1);
          out += '\n';
        }
        pos = eol + 1;
        continue;
      }
    }
    out += line;
    out += '\n';
    pos = eol + 1;
  }
  return out;
}

static const char* fsr2GlslPassFile(FfxFsr2Pass pass)
{
  switch (pass)
  {
  case FFX_FSR2_PASS_DEPTH_CLIP:                  return "ffx_fsr2_depth_clip_pass.glsl2";
  case FFX_FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH:  return "ffx_fsr2_reconstruct_previous_depth_pass.glsl2";
  case FFX_FSR2_PASS_LOCK:                        return "ffx_fsr2_lock_pass.glsl2";
  case FFX_FSR2_PASS_ACCUMULATE:
  case FFX_FSR2_PASS_ACCUMULATE_SHARPEN:          return "ffx_fsr2_accumulate_pass.glsl2";
  case FFX_FSR2_PASS_RCAS:                        return "ffx_fsr2_rcas_pass.glsl2";
  case FFX_FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID:   return "ffx_fsr2_compute_luminance_pyramid_pass.glsl2";
  case FFX_FSR2_PASS_GENERATE_REACTIVE:           return "ffx_fsr2_autogen_reactive_pass.glsl2";
  case FFX_FSR2_PASS_TCR_AUTOGENERATE:            return "ffx_fsr2_tcr_autogen_pass.glsl2";
  default: return nullptr;
  }
}

// Returns a linked program or 0. On success the pipeline's slot indices are rewritten:
// the glsl2 sources declare opaque uniforms without binding qualifiers (the offline
// glslang build auto-assigned units into the SPIR-V), so units are assigned here in
// pipeline order. UBO blocks carry explicit bindings and are queried.
static GLuint fsr2TryCreateGlslPipeline(BackendContext_GL* backendContext, FfxFsr2Pass pass, uint32_t flags, FfxPipelineState* outPipeline)
{
  const char* passFile = fsr2GlslPassFile(pass);
  if (!passFile) return 0;
  std::string src = fsr2GlslExpandIncludes(fsr2GlslFindEmbedded(passFile), 0);
  if (src.empty()) return 0;

  // Prelude: subgroup extensions (some sources declare them mid-file, which NVIDIA
  // rejects, so the generator strips them all) and the permutation defines matching
  // what the offline compile passed for this blob. fp16 stays off: the sources use
  // Vulkan-only fp16 extensions, and the accumulate pass disables it on NVIDIA anyway.
  char prelude[1024];
  snprintf(prelude, sizeof(prelude),
    "#extension GL_KHR_shader_subgroup_basic : require\n"
    "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
    "#extension GL_KHR_shader_subgroup_ballot : require\n"
    "#extension GL_KHR_shader_subgroup_quad : require\n"
    "#extension GL_KHR_shader_subgroup_shuffle : require\n"
    "#define FFX_GPU 1\n"
    "#define FFX_GLSL 1\n"
    "#define FFX_HALF 0\n"
    "#define FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE %d\n"
    "#define FFX_FSR2_OPTION_HDR_COLOR_INPUT %d\n"
    "#define FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS %d\n"
    "#define FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS %d\n"
    "#define FFX_FSR2_OPTION_INVERTED_DEPTH %d\n"
    "#define FFX_FSR2_OPTION_APPLY_SHARPENING %d\n",
    (flags & FSR2_SHADER_PERMUTATION_REPROJECT_USE_LANCZOS_TYPE) ? 1 : 0,
    (flags & FSR2_SHADER_PERMUTATION_HDR_COLOR_INPUT) ? 1 : 0,
    (flags & FSR2_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS) ? 1 : 0,
    (flags & FSR2_SHADER_PERMUTATION_JITTER_MOTION_VECTORS) ? 1 : 0,
    (flags & FSR2_SHADER_PERMUTATION_DEPTH_INVERTED) ? 1 : 0,
    (flags & FSR2_SHADER_PERMUTATION_ENABLE_SHARPENING) ? 1 : 0);

  size_t verPos = src.find("#version");
  if (verPos == std::string::npos) return 0;
  size_t verEol = src.find('\n', verPos);
  src.insert(verEol + 1, prelude);

  const auto getProc = backendContext->glFunctionTable.glGetProcAddress;
  auto pglShaderSource = (PFNGLSHADERSOURCEPROC)getProc("glShaderSource");
  auto pglGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)getProc("glGetShaderInfoLog");
  auto pglGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)getProc("glGetUniformBlockIndex");
  auto pglGetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIVPROC)getProc("glGetActiveUniformBlockiv");
  if (!pglShaderSource || !pglGetShaderInfoLog || !pglGetUniformBlockIndex || !pglGetActiveUniformBlockiv) return 0;

  GLuint shader = backendContext->glFunctionTable.glCreateShader(GL_COMPUTE_SHADER);
  const char* csrc = src.c_str();
  pglShaderSource(shader, 1, &csrc, nullptr);
  backendContext->glFunctionTable.glCompileShader(shader);
  GLint status = 0;
  backendContext->glFunctionTable.glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status)
  {
    static char log[16384] = {};
    pglGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    printf("FSR2 GLSL: pass %d compile failed:\n%s\n", (int)pass, log);
    backendContext->glFunctionTable.glDeleteShader(shader);
    return 0;
  }
  GLuint program = backendContext->glFunctionTable.glCreateProgram();
  backendContext->glFunctionTable.glAttachShader(program, shader);
  backendContext->glFunctionTable.glLinkProgram(program);
  backendContext->glFunctionTable.glDeleteShader(shader);
  GLint linked = 0;
  backendContext->glFunctionTable.glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked)
  {
    printf("FSR2 GLSL: pass %d link failed\n", (int)pass);
    backendContext->glFunctionTable.glDeleteProgram(program);
    return 0;
  }

  // Staged: a failure must leave the tables untouched for the SPIR-V fallback.
  uint32_t newCbSlots[FFX_MAX_NUM_CONST_BUFFERS] = {};
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  for (uint32_t i = 0; i < outPipeline->constCount; i++)
  {
    std::string name = conv.to_bytes(outPipeline->cbResourceBindings[i].name);
    GLuint bi = pglGetUniformBlockIndex(program, name.c_str());
    if (bi == GL_INVALID_INDEX)
    {
      // The blob reflects the instance name; GLSL indexes blocks by type name.
      bi = pglGetUniformBlockIndex(program, (name + "_t").c_str());
    }
    if (bi == GL_INVALID_INDEX)
    {
      printf("FSR2 GLSL: pass %d cb %s not found\n", (int)pass, name.c_str());
      backendContext->glFunctionTable.glDeleteProgram(program);
      return 0;
    }
    GLint binding = 0;
    pglGetActiveUniformBlockiv(program, bi, GL_UNIFORM_BLOCK_BINDING, &binding);
    newCbSlots[i] = (uint32_t)binding;
  }
  for (uint32_t i = 0; i < outPipeline->srvCount; i++)
  {
    std::string name = conv.to_bytes(outPipeline->srvResourceBindings[i].name);
    GLint loc = backendContext->glFunctionTable.glGetUniformLocation(program, name.c_str());
    // A location of -1 means the compiler proved the uniform unused; the unit is
    // still reserved so the executor's bind lands harmlessly.
    if (loc >= 0) backendContext->glFunctionTable.glProgramUniform1i(program, loc, (GLint)i);
    outPipeline->srvResourceBindings[i].slotIndex = i;
  }
  for (uint32_t i = 0; i < outPipeline->uavCount; i++)
  {
    std::string name = conv.to_bytes(outPipeline->uavResourceBindings[i].name);
    GLint loc = backendContext->glFunctionTable.glGetUniformLocation(program, name.c_str());
    if (loc >= 0) backendContext->glFunctionTable.glProgramUniform1i(program, loc, (GLint)i);
    outPipeline->uavResourceBindings[i].slotIndex = i;
  }
  for (uint32_t i = 0; i < outPipeline->constCount; i++) outPipeline->cbResourceBindings[i].slotIndex = newCbSlots[i];
  return program;
}

FfxErrorCode CreatePipelineGL(FfxFsr2Interface* backendInterface, FfxFsr2Pass pass, const FfxPipelineDescription* pipelineDescription, FfxPipelineState* outPipeline)
{
  FFX_ASSERT(backendInterface);
  FFX_ASSERT(pipelineDescription);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  // query device capabilities 
  FfxDeviceCapabilities deviceCapabilities;

  GetDeviceCapabilitiesGL(backendInterface, &deviceCapabilities, nullptr);
  
  bool useLut = false;

  if (deviceCapabilities.waveLaneCountMax == 64)
  {
    useLut = true;
  }

  // check if we have 16bit floating point.
  bool supportedFP16 = deviceCapabilities.fp16Supported;
  
  if (pass == FFX_FSR2_PASS_ACCUMULATE || pass == FFX_FSR2_PASS_ACCUMULATE_SHARPEN)
  {
    // Workaround: Disable FP16 path for the accumulate pass on NVIDIA due to reduced occupancy and high VRAM throughput.
    const auto* vendor = reinterpret_cast<const char*>(backendContext->glFunctionTable.glGetString(GL_VENDOR));
    if (strstr(vendor, "NVIDIA"))
    {
      supportedFP16 = false;
    }
  }

  // work out what permutation to load.
  uint32_t flags = 0;
  flags |= (pipelineDescription->contextFlags & FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE) ? FSR2_SHADER_PERMUTATION_HDR_COLOR_INPUT : 0;
  flags |= (pipelineDescription->contextFlags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? 0 : FSR2_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS;
  flags |= (pipelineDescription->contextFlags & FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? FSR2_SHADER_PERMUTATION_JITTER_MOTION_VECTORS : 0;
  flags |= (pipelineDescription->contextFlags & FFX_FSR2_ENABLE_DEPTH_INVERTED) ? FSR2_SHADER_PERMUTATION_DEPTH_INVERTED : 0;
  flags |= (pass == FFX_FSR2_PASS_ACCUMULATE_SHARPEN) ? FSR2_SHADER_PERMUTATION_ENABLE_SHARPENING : 0;
  flags |= (useLut) ? FSR2_SHADER_PERMUTATION_REPROJECT_USE_LANCZOS_TYPE : 0;
  //flags |= (canForceWave64) ? FSR2_SHADER_PERMUTATION_FORCE_WAVE64 : 0; // cannot force wave64 in OpenGL
  flags |= (supportedFP16 && (pass != FFX_FSR2_PASS_RCAS)) ? FSR2_SHADER_PERMUTATION_ALLOW_FP16 : 0;

  const Fsr2ShaderBlobGL shaderBlob = fsr2GetPermutationBlobByIndexGL(pass, flags);
  FFX_ASSERT(shaderBlob.data && shaderBlob.size);

  // populate the pass.
  outPipeline->srvCount = shaderBlob.combinedSamplerCount;
  outPipeline->uavCount = shaderBlob.storageImageCount;
  outPipeline->constCount = shaderBlob.uniformBufferCount;

  FFX_ASSERT(shaderBlob.storageImageCount < FFX_MAX_NUM_UAVS);
  FFX_ASSERT(shaderBlob.combinedSamplerCount < FFX_MAX_NUM_SRVS);
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

  for (uint32_t srvIndex = 0; srvIndex < outPipeline->srvCount; ++srvIndex)
  {
    outPipeline->srvResourceBindings[srvIndex].slotIndex = shaderBlob.boundCombinedSamplerBindings[srvIndex];
    wcscpy_s(outPipeline->srvResourceBindings[srvIndex].name, converter.from_bytes(shaderBlob.boundCombinedSamplerNames[srvIndex]).c_str());
  }
  for (uint32_t uavIndex = 0; uavIndex < outPipeline->uavCount; ++uavIndex)
  {
    outPipeline->uavResourceBindings[uavIndex].slotIndex = shaderBlob.boundStorageImageBindings[uavIndex];
    wcscpy_s(outPipeline->uavResourceBindings[uavIndex].name, converter.from_bytes(shaderBlob.boundStorageImageNames[uavIndex]).c_str());
  }
  for (uint32_t cbIndex = 0; cbIndex < outPipeline->constCount; ++cbIndex)
  {
    outPipeline->cbResourceBindings[cbIndex].slotIndex = shaderBlob.boundUniformBufferBindings[cbIndex];
    wcscpy_s(outPipeline->cbResourceBindings[cbIndex].name, converter.from_bytes(shaderBlob.boundUniformBufferNames[cbIndex]).c_str());
  }

  // Native GLSL by default on NVIDIA (see fsr2TryCreateGlslPipeline); SPIR-V for
  // other vendors (AMD's GL SPIR-V path performs as expected) and as the fallback.
  bool wantGlsl = getenv("FSR2_GL_GLSL") != nullptr;
  if (!wantGlsl && !getenv("FSR2_GL_SPIRV"))
  {
    const auto* vendorStr = reinterpret_cast<const char*>(backendContext->glFunctionTable.glGetString(GL_VENDOR));
    wantGlsl = vendorStr && strstr(vendorStr, "NVIDIA");
  }
  if (wantGlsl)
  {
    GLuint glslProgram = fsr2TryCreateGlslPipeline(backendContext, pass, flags, outPipeline);
    if (glslProgram)
    {
      outPipeline->pipeline = reinterpret_cast<FfxPipeline>(static_cast<uintptr_t>(glslProgram));
      outPipeline->rootSignature = nullptr;
      return FFX_OK;
    }
    printf("FSR2 GLSL: pass %d falling back to SPIR-V\n", (int)pass);
  }

  // create the shader module
  GLuint shader = backendContext->glFunctionTable.glCreateShader(GL_COMPUTE_SHADER);
  backendContext->glFunctionTable.glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V, shaderBlob.data, shaderBlob.size);
  backendContext->glFunctionTable.glSpecializeShader(shader, "main", 0, nullptr, nullptr);

  GLint compileStatus{};
  backendContext->glFunctionTable.glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
  if (compileStatus == GL_FALSE)
  {
    return FFX_ERROR_BACKEND_API_ERROR;
  }

  // create the compute pipeline
  GLuint program = backendContext->glFunctionTable.glCreateProgram();
  backendContext->glFunctionTable.glAttachShader(program, shader);
  backendContext->glFunctionTable.glLinkProgram(program);

  GLint linkStatus{};
  backendContext->glFunctionTable.glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
  if (linkStatus == GL_FALSE)
  {
    backendContext->glFunctionTable.glDeleteShader(shader);
    return FFX_ERROR_BACKEND_API_ERROR;
  }

  backendContext->glFunctionTable.glDeleteShader(shader);

  outPipeline->pipeline = reinterpret_cast<FfxPipeline>(static_cast<uintptr_t>(program));
  outPipeline->rootSignature = nullptr;

  return FFX_OK;
}

FfxErrorCode ScheduleGpuJobGL(FfxFsr2Interface* backendInterface, const FfxGpuJobDescription* job)
{
  FFX_ASSERT(backendInterface);
  FFX_ASSERT(job);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  FFX_ASSERT(backendContext->gpuJobCount < FSR2_MAX_GPU_JOBS);

  backendContext->gpuJobs[backendContext->gpuJobCount] = *job;

  if (job->jobType == FFX_GPU_JOB_COMPUTE)
  {
    // needs to copy SRVs and UAVs in case they are on the stack only
    FfxComputeJobDescription* computeJob = &backendContext->gpuJobs[backendContext->gpuJobCount].computeJobDescriptor;
    const uint32_t numConstBuffers = job->computeJobDescriptor.pipeline.constCount;
    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < numConstBuffers; ++currentRootConstantIndex)
    {
      computeJob->cbs[currentRootConstantIndex].uint32Size = job->computeJobDescriptor.cbs[currentRootConstantIndex].uint32Size;
      memcpy(computeJob->cbs[currentRootConstantIndex].data, job->computeJobDescriptor.cbs[currentRootConstantIndex].data, computeJob->cbs[currentRootConstantIndex].uint32Size * sizeof(uint32_t));
    }
  }

  backendContext->gpuJobCount++;
  
  return FFX_OK;
}

static void addBarrier(const BackendContext_GL* backendContext, bool isBufferBarrier, FfxResourceStates newState)
{
  FFX_ASSERT(backendContext);

  if (isBufferBarrier)
  {
    GLbitfield barriers = 0;
    barriers |= (newState & FFX_RESOURCE_STATE_UNORDERED_ACCESS) ? GL_SHADER_STORAGE_BARRIER_BIT : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COMPUTE_READ) ? GL_UNIFORM_BARRIER_BIT : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COPY_SRC) ? (GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT) : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COPY_DEST) ? (GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT) : 0;
    backendContext->glFunctionTable.glMemoryBarrier(barriers);
  }
  else
  {
    GLbitfield barriers = 0;
    barriers |= (newState & FFX_RESOURCE_STATE_UNORDERED_ACCESS) ? GL_SHADER_IMAGE_ACCESS_BARRIER_BIT : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COMPUTE_READ) ? (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT) : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COPY_SRC) ? GL_TEXTURE_UPDATE_BARRIER_BIT : 0;
    barriers |= (newState & FFX_RESOURCE_STATE_COPY_DEST) ? GL_TEXTURE_UPDATE_BARRIER_BIT : 0;
    backendContext->glFunctionTable.glMemoryBarrier(barriers);
  }
}

static FfxErrorCode executeGpuJobCompute(BackendContext_GL* backendContext, FfxGpuJobDescription* job)
{
  FFX_ASSERT(backendContext);

  const auto program = static_cast<GLuint>(reinterpret_cast<uintptr_t>(job->computeJobDescriptor.pipeline.pipeline));

  // One combined barrier covers both the image stores the UAVs of this dispatch will
  // read-modify-write and the texture fetches of its SRVs.
  backendContext->glFunctionTable.glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

  for (uint32_t uav = 0; uav < job->computeJobDescriptor.pipeline.uavCount; ++uav)
  {
    BackendContext_GL::Resource ffxResource = backendContext->resources[job->computeJobDescriptor.uavs[uav].internalIndex];

    backendContext->glFunctionTable.glBindImageTexture(
      job->computeJobDescriptor.pipeline.uavResourceBindings[uav].slotIndex,
      ffxResource.textureSingleMipViews[job->computeJobDescriptor.uavMip[uav]].id,
      0,
      true,
      0,
      GL_READ_WRITE,
      getGLFormatFromSurfaceFormat(ffxResource.resourceDescription.format)
    );
  }

  // bind srvs (sampled textures)

  for (uint32_t srv = 0; srv < job->computeJobDescriptor.pipeline.srvCount; ++srv)
  {
    BackendContext_GL::Resource ffxResource = backendContext->resources[job->computeJobDescriptor.srvs[srv].internalIndex];

    // Integer textures are incomplete under a linear-filtering sampler; give them the
    // point sampler (matching what the shaders do with them anyway).
    GLuint sampler = backendContext->linearSampler.id;
    switch (ffxResource.resourceDescription.format)
    {
    case FFX_SURFACE_FORMAT_R32_UINT:
    case FFX_SURFACE_FORMAT_R16_UINT:
    case FFX_SURFACE_FORMAT_R16G16_UINT:
    case FFX_SURFACE_FORMAT_R8_UINT:
      sampler = backendContext->pointSampler.id;
      break;
    default:;
    }

    backendContext->glFunctionTable.glBindTextureUnit(job->computeJobDescriptor.pipeline.srvResourceBindings[srv].slotIndex, ffxResource.textureAllMipsView.id);
    backendContext->glFunctionTable.glBindSampler(job->computeJobDescriptor.pipeline.srvResourceBindings[srv].slotIndex, sampler);
  }

  // update ubos (uniform buffers)
  for (uint32_t i = 0; i < job->computeJobDescriptor.pipeline.constCount; ++i)
  {
    auto ubo = accquireDynamicUBO(backendContext, job->computeJobDescriptor.cbs[i].uint32Size * sizeof(uint32_t), job->computeJobDescriptor.cbs[i].data);
    backendContext->glFunctionTable.glBindBufferRange(
      GL_UNIFORM_BUFFER,
      job->computeJobDescriptor.pipeline.cbResourceBindings[i].slotIndex,
      ubo.bufferResource.id,
      0,
      FSR2_UBO_SIZE
    );
  }

  backendContext->glFunctionTable.glUseProgram(program);
  backendContext->glFunctionTable.glDispatchCompute(job->computeJobDescriptor.dimensions[0], job->computeJobDescriptor.dimensions[1], job->computeJobDescriptor.dimensions[2]);
  if (getenv("FSR2_GL_DEBUG"))
  {
    auto pglGetError = (PFNGLGETERRORPROC)backendContext->glFunctionTable.glGetProcAddress("glGetError");
    GLenum err = pglGetError();
    if (err) printf("FSR2 GL_DEBUG: glGetError 0x%x after dispatch (program %u)\n", err, program);
  }

  return FFX_OK;
}

static FfxErrorCode executeGpuJobClearFloat(BackendContext_GL* backendContext, FfxGpuJobDescription* job)
{
  FFX_ASSERT(backendContext);

  uint32_t idx = job->clearJobDescriptor.target.internalIndex;
  BackendContext_GL::Resource ffxResource = backendContext->resources[idx];

  if (ffxResource.resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER)
  {
    addBarrier(backendContext, false, FFX_RESOURCE_STATE_COPY_DEST);

    auto texture = ffxResource.textureAllMipsView;

    float clearColorValue[4] = {};
    clearColorValue[0] = job->clearJobDescriptor.color[0];
    clearColorValue[1] = job->clearJobDescriptor.color[1];
    clearColorValue[2] = job->clearJobDescriptor.color[2];
    clearColorValue[3] = job->clearJobDescriptor.color[3];

    for (uint32_t i = 0; i < ffxResource.resourceDescription.mipCount; i++)
    {
      backendContext->glFunctionTable.glClearTexImage(texture.id, i, GL_RGBA, GL_FLOAT, clearColorValue);
    }
  }

  return FFX_OK;
}

FfxErrorCode ExecuteGpuJobsGL(FfxFsr2Interface* backendInterface, FfxCommandList)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  FfxErrorCode errorCode = FFX_OK;

  if (g_jobprof.enabled == -1)
  {
    g_jobprof.enabled = getenv("FSR2_GL_PROFILE") ? 1 : 0;
    if (g_jobprof.enabled)
    {
      g_jobprof.pglGenQueries = (PFNGLGENQUERIESPROC)backendContext->glFunctionTable.glGetProcAddress("glGenQueries");
      g_jobprof.pglQueryCounter = (PFNGLQUERYCOUNTERPROC)backendContext->glFunctionTable.glGetProcAddress("glQueryCounter");
      g_jobprof.pglGetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC)backendContext->glFunctionTable.glGetProcAddress("glGetQueryObjectui64v");
      if (g_jobprof.pglGenQueries && g_jobprof.pglQueryCounter && g_jobprof.pglGetQueryObjectui64v)
      {
        g_jobprof.pglGenQueries(JobProfiler::MAXQ * 2, g_jobprof.queries);
        g_jobprof.inited = true;
      }
    }
  }
  const bool prof = g_jobprof.enabled == 1 && g_jobprof.inited;

  // Collect the previous frame's timings first; by now they are ready.
  if (prof && g_jobprof.prevCount)
  {
    for (int i = 0; i < g_jobprof.prevCount; i++)
    {
      GLuint64 t0 = 0, t1 = 0;
      g_jobprof.pglGetQueryObjectui64v(g_jobprof.queries[i * 2], GL_QUERY_RESULT, &t0);
      g_jobprof.pglGetQueryObjectui64v(g_jobprof.queries[i * 2 + 1], GL_QUERY_RESULT, &t1);
      g_jobprof.sumMs[i] += double(t1 - t0) / 1.0e6;
    }
    g_jobprof.frames++;
    if (g_jobprof.frames == 300)
    {
      double total = 0;
      printf("== FSR2 GL per-job GPU ms (avg over %d frames) ==\n", g_jobprof.frames);
      for (int i = 0; i < g_jobprof.prevCount; i++)
      {
        printf("  job %2d  %-40s %7.3f\n", i, g_jobprof.labels[i], g_jobprof.sumMs[i] / g_jobprof.frames);
        total += g_jobprof.sumMs[i] / g_jobprof.frames;
        g_jobprof.sumMs[i] = 0;
      }
      printf("  TOTAL %7.3f\n", total);
      fflush(stdout);
      g_jobprof.frames = 0;
    }
  }

  // execute all renderjobs
  for (uint32_t i = 0; i < backendContext->gpuJobCount; ++i)
  {
    FfxGpuJobDescription* gpuJob = &backendContext->gpuJobs[i];

    if (prof && i < JobProfiler::MAXQ)
    {
      if (gpuJob->jobType == FFX_GPU_JOB_COMPUTE)
        snprintf(g_jobprof.labels[i], sizeof(g_jobprof.labels[i]), "compute %ux%u",
                 gpuJob->computeJobDescriptor.dimensions[0], gpuJob->computeJobDescriptor.dimensions[1]);
      else
        snprintf(g_jobprof.labels[i], sizeof(g_jobprof.labels[i]), "jobtype %d", (int)gpuJob->jobType);
      g_jobprof.pglQueryCounter(g_jobprof.queries[i * 2], GL_TIMESTAMP);
    }

    switch (gpuJob->jobType)
    {
    case FFX_GPU_JOB_CLEAR_FLOAT:
    {
      errorCode = executeGpuJobClearFloat(backendContext, gpuJob);
      break;
    }
    case FFX_GPU_JOB_COPY:
    {
      FFX_ASSERT_FAIL("Copy job is not implemented in OpenGL backend");
      break;
    }
    case FFX_GPU_JOB_COMPUTE:
    {
      errorCode = executeGpuJobCompute(backendContext, gpuJob);
      break;
    }
    default:;
    }

    if (prof && i < JobProfiler::MAXQ)
      g_jobprof.pglQueryCounter(g_jobprof.queries[i * 2 + 1], GL_TIMESTAMP);
  }

  if (prof)
    g_jobprof.prevCount = (int)((backendContext->gpuJobCount < JobProfiler::MAXQ) ? backendContext->gpuJobCount : JobProfiler::MAXQ);

  // Sampler objects bound above stay bound for whatever the application renders next
  // and silently override its textures' own parameters (e.g. forcing mip filtering on
  // a mipless texture makes it incomplete, sampling solid black). Unbind them from
  // every unit a pass may have used.
  for (uint32_t i = 0; i < FFX_MAX_NUM_SRVS + 8; i++)
  {
    backendContext->glFunctionTable.glBindSampler(i, 0);
  }

  // check the execute function returned cleanly.
  FFX_RETURN_ON_ERROR(
    errorCode == FFX_OK,
    FFX_ERROR_BACKEND_API_ERROR);

  backendContext->gpuJobCount = 0;

  return FFX_OK;
}

FfxErrorCode DestroyResourceGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource)
{
  FFX_ASSERT(backendInterface);

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  if (resource.internalIndex != -1)
  {
    BackendContext_GL::Resource& res = backendContext->resources[resource.internalIndex];

    if (res.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
    {
      if (res.buffer.id)
      {
        backendContext->glFunctionTable.glDeleteBuffers(1, &res.buffer.id);
        res.buffer = {};
      }
    }
    else
    {
      if (res.textureAllMipsView.id)
      {
        backendContext->glFunctionTable.glDeleteTextures(1, &res.textureAllMipsView.id);
        res.textureAllMipsView = {};
      }

      for (uint32_t i = 0; i < res.resourceDescription.mipCount; i++)
      {
        if (res.textureSingleMipViews[i].id)
        {
          backendContext->glFunctionTable.glDeleteTextures(1, &res.textureSingleMipViews[i].id);
          res.textureSingleMipViews[i] = {};
        }
      }
    }
  }

  return FFX_OK;
}

FfxErrorCode DestroyPipelineGL(FfxFsr2Interface* backendInterface, FfxPipelineState* pipeline)
{
  FFX_ASSERT(backendInterface);

  if (!pipeline)
  {
    return FFX_OK;
  }

  BackendContext_GL* backendContext = (BackendContext_GL*)backendInterface->scratchBuffer;

  // destroy pipeline 
  const auto program = static_cast<GLuint>(reinterpret_cast<uintptr_t>(pipeline->pipeline));
  if (program)
  {
    backendContext->glFunctionTable.glDeleteProgram(program);
    pipeline->pipeline = nullptr;
  }

  return FFX_OK;
}
