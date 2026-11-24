#pragma once
// SphSurface.h - screen space fluid rendering for SphSolver (particles drawn as a smooth, lit water surface).
//
// Passes (all recorded by RecordOffscreen(), outside the main render pass):
//   1. depth      : particles as sphere impostors -> linear view depth (R32F) + z-test (D32)
//   2. thickness  : same impostors, additive -> thickness along the view ray (R16F)
//   3. smoothing  : separable bilateral filter of the depth (nBlurIterations x (horizontal + vertical))
// Then RecordComposite(), INSIDE the main render pass (subpass 0, after the scene):
//   4. composite  : normals from the smoothed depth, Fresnel reflection, refraction + Beer-Lambert absorption,
//                   specular; writes gl_FragDepth so that the scene geometry still occludes the fluid.
//
// The refracted background is procedural (sky gradient): the scene is rendered in an MSAA subpass and cannot be
// sampled as a texture. The sphere radius, filter and shading are tuned with SphSurface::Settings.
//
// Naming: same convention as SphSolver (m_ prefix, b = bool, n = int / uint32_t, v = std::vector, a = C array,
// p = pointer). The members of the structs are not renamed.

// 1. PROJECT
#include "Utility.h"
#include "vulkan_device.hpp"
#include "vulkan_ubo.hpp"

// 2. LIB
#include <vulkan/vulkan.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
using namespace glm;

// 3. WIN
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
using namespace std;

extern uint32_t g_FramesInFlight;

class SphSurface {
public:
    struct Settings {
        float radiusScale     = 1.0f;                       // impostor radius = radiusScale * spacing
        int   nBlurIterations = 2;                          // horizontal + vertical passes, at least 1
        float filterRadius    = 2.0f;                       // world radius of the smoothing, in impostor radii
        float depthSigma      = 1.0f;                       // depth range of the bilateral filter, in impostor radii
        int   nMaxFilterPixels = 24;                        // cap of the filter radius on the screen (cost)
        vec3  waterColor      = vec3(0.19f, 0.28f, 0.33f);  // tint of the deep water
        float absorption      = 1.0f;                       // 1/m, scales (1 - waterColor)
        float thicknessScale  = 1.0f;                       // 1 = physical thickness
        float fresnelF0       = 0.02f;                      // reflectance at normal incidence (water)
        float specularPower   = 200.0f;
        float lightAzimuth    = 35.0f;                      // degrees
        float lightElevation  = 50.0f;                      // degrees
        float exposure        = 1.0f;
    };

    explicit SphSurface(shared_ptr<VulkanDevice> vulkanDevice);
    ~SphSurface();
    SphSurface(const SphSurface&)            = delete;
    SphSurface& operator=(const SphSurface&) = delete;

    // (Re)creates the targets and the pipelines: call after InitVulkan and after every swapchain / render pass recreation.
    void Create(VkRenderPass mainRenderPass, VkExtent2D extent);
    void Destroy();

    // Outside any render pass, after the simulation (the particle buffer is read as a vertex buffer, stride 32).
    void RecordOffscreen(VkCommandBuffer cmd, uint32_t nFrame, VkBuffer particles, uint32_t nCount, float spacing,
                         const mat4& view, const mat4& proj, const Settings& settings);

    // Inside the main render pass, subpass 0, after the scene geometry.
    void RecordComposite(VkCommandBuffer cmd, uint32_t nFrame);

private:
    struct Target {
        VkImage         image  = VK_NULL_HANDLE;
        VkDeviceMemory  memory = VK_NULL_HANDLE;
        VkImageView     view   = VK_NULL_HANDLE;
    };

    struct PipelineDesc {
        VkShaderModule          vert        = VK_NULL_HANDLE;
        VkShaderModule          frag        = VK_NULL_HANDLE;
        bool                    bParticles  = false;                // vertex input = particle buffer (one point per instance)
        VkPrimitiveTopology     topology    = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkSampleCountFlagBits   samples     = VK_SAMPLE_COUNT_1_BIT;
        bool                    bDepthTest  = false;
        bool                    bDepthWrite = false;
        VkCompareOp             depthOp     = VK_COMPARE_OP_LESS;
        bool                    bBlendAdd   = false;
        VkPipelineLayout        layout      = VK_NULL_HANDLE;
        VkRenderPass            renderPass  = VK_NULL_HANDLE;
    };

    uint32_t        FindMemoryType(uint32_t nTypeBits, VkMemoryPropertyFlags props) const;
    void            CreateTarget(Target& target, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
    void            DestroyTarget(Target& target);
    VkRenderPass    MakeRenderPass(VkFormat colorFormat, VkAttachmentLoadOp colorLoad, bool bDepth);
    VkFramebuffer   MakeFramebuffer(VkRenderPass renderPass, const VkImageView* pViews, uint32_t nViews);
    VkPipeline      BuildPipeline(const PipelineDesc& desc);

    void            CreateTargets();
    void            CreateRenderPasses();
    void            CreateLayouts();
    void            CreatePipelines(VkRenderPass mainRenderPass);
    void            CreateDescriptors();

    void            UpdateUbo(uint32_t nFrame, float spacing, const mat4& view, const mat4& proj, const Settings& settings);
    void            BeginPass(VkCommandBuffer cmd, VkRenderPass renderPass, VkFramebuffer framebuffer, const VkClearValue* pClears, uint32_t nClears);
    void            RecordBlurPass(VkCommandBuffer cmd, VkFramebuffer framebuffer, uint32_t nInputSet, float dirX, float dirY,
                                   float filterRadius, float projScale, float depthSigma, float maxRadius);

    shared_ptr<VulkanDevice>    m_VulkanDevice;
    VkExtent2D                  m_Extent{};
    bool                        m_bCreated = false;

    // Targets: raw linear depth, z-test depth, ping-pong smoothed depth (final result in B), thickness
    Target          m_RawDepth, m_ZDepth, m_SmoothA, m_SmoothB, m_Thickness;
    VkSampler       m_Sampler = VK_NULL_HANDLE;

    VkRenderPass    m_RpDepth = VK_NULL_HANDLE, m_RpThick = VK_NULL_HANDLE, m_RpBlur = VK_NULL_HANDLE;
    VkFramebuffer   m_FbDepth = VK_NULL_HANDLE, m_FbThick = VK_NULL_HANDLE, m_FbBlurA = VK_NULL_HANDLE, m_FbBlurB = VK_NULL_HANDLE;

    VkDescriptorSetLayout   m_UboSetLayout  = VK_NULL_HANDLE;   // set of the particle passes: UBO
    VkDescriptorSetLayout   m_TexSetLayout  = VK_NULL_HANDLE;   // set of the blur: 1 sampler
    VkDescriptorSetLayout   m_CompSetLayout = VK_NULL_HANDLE;   // set of the composite: UBO + depth + thickness
    VkPipelineLayout        m_ParticleLayout  = VK_NULL_HANDLE;
    VkPipelineLayout        m_BlurLayout      = VK_NULL_HANDLE;
    VkPipelineLayout        m_CompositeLayout = VK_NULL_HANDLE;
    VkPipeline              m_PipeDepth = VK_NULL_HANDLE, m_PipeThick = VK_NULL_HANDLE, m_PipeBlur = VK_NULL_HANDLE, m_PipeComposite = VK_NULL_HANDLE;

    VkDescriptorPool            m_Pool = VK_NULL_HANDLE;
    vector<VkDescriptorSet>     m_vUboSets;                     // 1 per frame in flight
    vector<VkDescriptorSet>     m_vCompSets;                    // 1 per frame in flight
    VkDescriptorSet             m_aBlurSets[3] = {};            // 0 = raw depth, 1 = smooth A, 2 = smooth B
    vector<unique_ptr<VulkanUBO>> m_vUbo;                       // 1 per frame in flight
};
