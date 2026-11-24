// SphSurface.cpp - screen space fluid rendering (see SphSurface.h for the overview).

#include "SphSurface.h"

namespace {

constexpr VkFormat kZDepthFormat    = VK_FORMAT_D32_SFLOAT;     // z-test of the impostors
constexpr VkFormat kLinearFormat    = VK_FORMAT_R32_SFLOAT;     // linear view-space depth
constexpr VkFormat kThicknessFormat = VK_FORMAT_R16_SFLOAT;     // additive thickness
constexpr uint32_t kParticleStride  = 32;                       // sizeof(Particle) in SphSolver.h
constexpr uint32_t kMaxPointPixels  = 128;                      // keep under the pointSizeRange of the GPU

// Uniform block "SurfaceUBO" (std140) of the sph_surf_*.vert/.frag shaders. Keep it identical.
struct SurfaceUbo {
    mat4 view;
    mat4 proj;
    mat4 invView;
    vec4 light;         // xyz = direction towards the light (world space)
    vec4 screen;        // x = width, y = height, z = 1 / width, w = 1 / height
    vec4 params0;       // x = impostor radius (m)
    vec4 absorption;    // rgb = absorption coefficients (1/m)
    vec4 waterColor;    // rgb
    vec4 params1;       // x = F0, y = specular power, z = exposure, w = thickness scale
};
static_assert(sizeof(SurfaceUbo) == 288, "SurfaceUbo must match the std140 block of the shaders");

// Push constants of sph_surf_blur.frag
struct BlurPush {
    float dirX, dirY;
    float filterRadius;     // m
    float projScale;        // pixels per meter at a distance of 1 m
    float depthSigma;       // m
    float maxRadius;        // pixels
    float pad0, pad1;
};
static_assert(sizeof(BlurPush) == 32, "BlurPush must match the push constant block");

} // namespace


// ============================================================================
// Construction
// ============================================================================

SphSurface::SphSurface(shared_ptr<VulkanDevice> vulkanDevice)
    : m_VulkanDevice(vulkanDevice)
{
}

SphSurface::~SphSurface()
{
    Destroy();
}

void SphSurface::Create(VkRenderPass mainRenderPass, VkExtent2D extent)
{
    Destroy();
    m_Extent = extent;

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };    // nearest: the shaders use texelFetch
    si.magFilter    = VK_FILTER_NEAREST;
    si.minFilter    = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(m_VulkanDevice->device, &si, nullptr, &m_Sampler);

    CreateTargets();
    CreateRenderPasses();
    CreateLayouts();
    CreatePipelines(mainRenderPass);
    CreateDescriptors();
    m_bCreated = true;
}

void SphSurface::Destroy()
{
    if (!m_VulkanDevice || !m_bCreated)
        return;

    VkDevice dev = m_VulkanDevice->device;
    vkDeviceWaitIdle(dev);

    if (m_PipeDepth     != VK_NULL_HANDLE) vkDestroyPipeline(dev, m_PipeDepth, nullptr);
    if (m_PipeThick     != VK_NULL_HANDLE) vkDestroyPipeline(dev, m_PipeThick, nullptr);
    if (m_PipeBlur      != VK_NULL_HANDLE) vkDestroyPipeline(dev, m_PipeBlur, nullptr);
    if (m_PipeComposite != VK_NULL_HANDLE) vkDestroyPipeline(dev, m_PipeComposite, nullptr);
    m_PipeDepth = m_PipeThick = m_PipeBlur = m_PipeComposite = VK_NULL_HANDLE;

    if (m_ParticleLayout  != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, m_ParticleLayout, nullptr);
    if (m_BlurLayout      != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, m_BlurLayout, nullptr);
    if (m_CompositeLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, m_CompositeLayout, nullptr);
    m_ParticleLayout = m_BlurLayout = m_CompositeLayout = VK_NULL_HANDLE;

    if (m_UboSetLayout  != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, m_UboSetLayout, nullptr);
    if (m_TexSetLayout  != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, m_TexSetLayout, nullptr);
    if (m_CompSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, m_CompSetLayout, nullptr);
    m_UboSetLayout = m_TexSetLayout = m_CompSetLayout = VK_NULL_HANDLE;

    if (m_Pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, m_Pool, nullptr);    // frees the sets
    m_Pool = VK_NULL_HANDLE;
    m_vUboSets.clear();
    m_vCompSets.clear();
    for (VkDescriptorSet& set : m_aBlurSets)
        set = VK_NULL_HANDLE;
    m_vUbo.clear();

    VkFramebuffer* apFb[4] = { &m_FbDepth, &m_FbThick, &m_FbBlurA, &m_FbBlurB };
    for (VkFramebuffer* pFb : apFb)
    {
        if (*pFb != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, *pFb, nullptr);
        *pFb = VK_NULL_HANDLE;
    }
    VkRenderPass* apRp[3] = { &m_RpDepth, &m_RpThick, &m_RpBlur };
    for (VkRenderPass* pRp : apRp)
    {
        if (*pRp != VK_NULL_HANDLE) vkDestroyRenderPass(dev, *pRp, nullptr);
        *pRp = VK_NULL_HANDLE;
    }

    DestroyTarget(m_RawDepth);
    DestroyTarget(m_ZDepth);
    DestroyTarget(m_SmoothA);
    DestroyTarget(m_SmoothB);
    DestroyTarget(m_Thickness);

    if (m_Sampler != VK_NULL_HANDLE) vkDestroySampler(dev, m_Sampler, nullptr);
    m_Sampler = VK_NULL_HANDLE;

    m_bCreated = false;
}


// ============================================================================
// Low level helpers
// ============================================================================

uint32_t SphSurface::FindMemoryType(uint32_t nTypeBits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_VulkanDevice->physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((nTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

void SphSurface::CreateTarget(Target& target, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    VkDevice dev = m_VulkanDevice->device;

    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = format;
    ci.extent        = { m_Extent.width, m_Extent.height, 1 };
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(dev, &ci, nullptr, &target.image);

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, target.image, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(dev, &ai, nullptr, &target.memory);
    vkBindImageMemory(dev, target.image, target.memory, 0);

    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image                       = target.image;
    vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                      = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    vkCreateImageView(dev, &vi, nullptr, &target.view);
}

void SphSurface::DestroyTarget(Target& target)
{
    VkDevice dev = m_VulkanDevice->device;
    if (target.view   != VK_NULL_HANDLE) vkDestroyImageView(dev, target.view, nullptr);
    if (target.image  != VK_NULL_HANDLE) vkDestroyImage(dev, target.image, nullptr);
    if (target.memory != VK_NULL_HANDLE) vkFreeMemory(dev, target.memory, nullptr);
    target = Target();
}

VkRenderPass SphSurface::MakeRenderPass(VkFormat colorFormat, VkAttachmentLoadOp colorLoad, bool bDepth)
{
    // One color attachment left in SHADER_READ_ONLY layout (sampled by the next pass), optionally a z-test depth.
    VkAttachmentDescription aAtt[2] = {};
    aAtt[0].format         = colorFormat;
    aAtt[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    aAtt[0].loadOp         = colorLoad;
    aAtt[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    aAtt[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    aAtt[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    aAtt[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    aAtt[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (bDepth)
    {
        aAtt[1].format         = kZDepthFormat;
        aAtt[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        aAtt[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        aAtt[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        aAtt[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        aAtt[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        aAtt[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        aAtt[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colorRef;
    sub.pDepthStencilAttachment = bDepth ? &depthRef : nullptr;

    // External -> subpass: the previous readers (the sampling of the previous passes / frames) must be finished
    VkPipelineStageFlags srcStage  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkAccessFlags        srcAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkPipelineStageFlags dstStage  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkAccessFlags        dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    if (bDepth)
    {
        srcStage  |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        srcAccess |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dstStage  |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dstAccess |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkSubpassDependency aDeps[2] = {};
    aDeps[0] = { VK_SUBPASS_EXTERNAL, 0, srcStage, dstStage, srcAccess, dstAccess, 0 };
    // Subpass -> external: the color writes must be visible to the sampling of the next passes
    aDeps[1] = { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, 0 };

    VkRenderPassCreateInfo ci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    ci.attachmentCount = bDepth ? 2u : 1u;
    ci.pAttachments    = aAtt;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;
    ci.dependencyCount = 2;
    ci.pDependencies   = aDeps;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    vkCreateRenderPass(m_VulkanDevice->device, &ci, nullptr, &renderPass);
    return renderPass;
}

VkFramebuffer SphSurface::MakeFramebuffer(VkRenderPass renderPass, const VkImageView* pViews, uint32_t nViews)
{
    VkFramebufferCreateInfo ci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    ci.renderPass      = renderPass;
    ci.attachmentCount = nViews;
    ci.pAttachments    = pViews;
    ci.width           = m_Extent.width;
    ci.height          = m_Extent.height;
    ci.layers          = 1;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    vkCreateFramebuffer(m_VulkanDevice->device, &ci, nullptr, &framebuffer);
    return framebuffer;
}

VkPipeline SphSurface::BuildPipeline(const PipelineDesc& desc)
{
    VkPipelineShaderStageCreateInfo aStages[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   desc.vert, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, desc.frag, "main" }
    };

    // Vertex input: the particle buffer (position at offset 0, one instance = one point), or nothing (full-screen triangle)
    VkVertexInputBindingDescription   binding{ 0, kParticleStride, VK_VERTEX_INPUT_RATE_INSTANCE };
    VkVertexInputAttributeDescription attrib{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    if (desc.bParticles)
    {
        vertexInput.vertexBindingDescriptionCount   = 1;
        vertexInput.pVertexBindingDescriptions      = &binding;
        vertexInput.vertexAttributeDescriptionCount = 1;
        vertexInput.pVertexAttributeDescriptions    = &attrib;
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = desc.topology;

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = desc.samples;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable  = desc.bDepthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = desc.bDepthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp   = desc.depthOp;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (desc.bBlendAdd)
    {
        blendAttachment.blendEnable         = VK_TRUE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    // Viewport and scissor are dynamic: the targets have the size of the swapchain, set at record time
    VkDynamicState aDynamic[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = aDynamic;

    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.stageCount          = 2;
    pi.pStages             = aStages;
    pi.pVertexInputState   = &vertexInput;
    pi.pInputAssemblyState = &inputAssembly;
    pi.pViewportState      = &viewportState;
    pi.pRasterizationState = &rasterizer;
    pi.pMultisampleState   = &multisample;
    pi.pDepthStencilState  = &depthStencil;
    pi.pColorBlendState    = &colorBlend;
    pi.pDynamicState       = &dynamicState;
    pi.layout              = desc.layout;
    pi.renderPass          = desc.renderPass;
    pi.subpass             = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(m_VulkanDevice->device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
    return pipeline;
}


// ============================================================================
// Creation of the targets, passes, pipelines and descriptors
// ============================================================================

void SphSurface::CreateTargets()
{
    const VkImageUsageFlags colorSampled = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    CreateTarget(m_RawDepth,  kLinearFormat,    colorSampled, VK_IMAGE_ASPECT_COLOR_BIT);
    CreateTarget(m_SmoothA,   kLinearFormat,    colorSampled, VK_IMAGE_ASPECT_COLOR_BIT);
    CreateTarget(m_SmoothB,   kLinearFormat,    colorSampled, VK_IMAGE_ASPECT_COLOR_BIT);
    CreateTarget(m_Thickness, kThicknessFormat, colorSampled, VK_IMAGE_ASPECT_COLOR_BIT);
    CreateTarget(m_ZDepth,    kZDepthFormat,    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void SphSurface::CreateRenderPasses()
{
    m_RpDepth = MakeRenderPass(kLinearFormat,    VK_ATTACHMENT_LOAD_OP_CLEAR,     true);
    m_RpThick = MakeRenderPass(kThicknessFormat, VK_ATTACHMENT_LOAD_OP_CLEAR,     false);
    m_RpBlur  = MakeRenderPass(kLinearFormat,    VK_ATTACHMENT_LOAD_OP_DONT_CARE, false);   // fully overwritten

    const VkImageView aDepthViews[2] = { m_RawDepth.view, m_ZDepth.view };
    m_FbDepth = MakeFramebuffer(m_RpDepth, aDepthViews, 2);
    m_FbThick = MakeFramebuffer(m_RpThick, &m_Thickness.view, 1);
    m_FbBlurA = MakeFramebuffer(m_RpBlur,  &m_SmoothA.view, 1);
    m_FbBlurB = MakeFramebuffer(m_RpBlur,  &m_SmoothB.view, 1);
}

void SphSurface::CreateLayouts()
{
    VkDevice dev = m_VulkanDevice->device;

    auto makeSetLayout = [&](const VkDescriptorSetLayoutBinding* pBindings, uint32_t nBindings) {
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = nBindings;
        ci.pBindings    = pBindings;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(dev, &ci, nullptr, &layout);
        return layout;
    };

    const VkDescriptorSetLayoutBinding uboBinding{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    m_UboSetLayout = makeSetLayout(&uboBinding, 1);

    const VkDescriptorSetLayoutBinding texBinding{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    m_TexSetLayout = makeSetLayout(&texBinding, 1);

    const VkDescriptorSetLayoutBinding aCompBindings[3] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
    };
    m_CompSetLayout = makeSetLayout(aCompBindings, 3);

    auto makePipelineLayout = [&](VkDescriptorSetLayout setLayout, const VkPushConstantRange* pRange) {
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount         = 1;
        ci.pSetLayouts            = &setLayout;
        ci.pushConstantRangeCount = pRange ? 1u : 0u;
        ci.pPushConstantRanges    = pRange;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(dev, &ci, nullptr, &layout);
        return layout;
    };

    const VkPushConstantRange blurRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BlurPush) };
    m_ParticleLayout  = makePipelineLayout(m_UboSetLayout,  nullptr);
    m_BlurLayout      = makePipelineLayout(m_TexSetLayout,  &blurRange);
    m_CompositeLayout = makePipelineLayout(m_CompSetLayout, nullptr);
}

void SphSurface::CreatePipelines(VkRenderPass mainRenderPass)
{
    VkDevice dev = m_VulkanDevice->device;

    auto loadShader = [&](const char* path) {
        auto vCode = CompileShaderRuntime(path);
        return CreateShaderModule(dev, vCode);
    };
    VkShaderModule particleVert = loadShader("Resources/Shaders/Sph/sph_surf_particle.vert");
    VkShaderModule depthFrag    = loadShader("Resources/Shaders/Sph/sph_surf_depth.frag");
    VkShaderModule thickFrag    = loadShader("Resources/Shaders/Sph/sph_surf_thick.frag");
    VkShaderModule fullVert     = loadShader("Resources/Shaders/Sph/sph_surf_fullscreen.vert");
    VkShaderModule blurFrag     = loadShader("Resources/Shaders/Sph/sph_surf_blur.frag");
    VkShaderModule compFrag     = loadShader("Resources/Shaders/Sph/sph_surf_composite.frag");

    // 1. Depth of the sphere impostors (z-test, linear depth in the color target)
    PipelineDesc d;
    d.vert = particleVert; d.frag = depthFrag; d.bParticles = true; d.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    d.bDepthTest = true; d.bDepthWrite = true; d.depthOp = VK_COMPARE_OP_LESS;
    d.layout = m_ParticleLayout; d.renderPass = m_RpDepth;
    m_PipeDepth = BuildPipeline(d);

    // 2. Thickness: additive, no depth test
    d = PipelineDesc();
    d.vert = particleVert; d.frag = thickFrag; d.bParticles = true; d.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    d.bBlendAdd = true;
    d.layout = m_ParticleLayout; d.renderPass = m_RpThick;
    m_PipeThick = BuildPipeline(d);

    // 3. Bilateral smoothing (full-screen triangle)
    d = PipelineDesc();
    d.vert = fullVert; d.frag = blurFrag;
    d.layout = m_BlurLayout; d.renderPass = m_RpBlur;
    m_PipeBlur = BuildPipeline(d);

    // 4. Composite in the main render pass (MSAA, subpass 0), depth test against the scene and depth write
    d = PipelineDesc();
    d.vert = fullVert; d.frag = compFrag;
    d.samples = m_VulkanDevice->msaaSamples;
    d.bDepthTest = true; d.bDepthWrite = true; d.depthOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    d.layout = m_CompositeLayout; d.renderPass = mainRenderPass;
    m_PipeComposite = BuildPipeline(d);

    vkDestroyShaderModule(dev, particleVert, nullptr);
    vkDestroyShaderModule(dev, depthFrag, nullptr);
    vkDestroyShaderModule(dev, thickFrag, nullptr);
    vkDestroyShaderModule(dev, fullVert, nullptr);
    vkDestroyShaderModule(dev, blurFrag, nullptr);
    vkDestroyShaderModule(dev, compFrag, nullptr);
}

void SphSurface::CreateDescriptors()
{
    VkDevice dev = m_VulkanDevice->device;
    const uint32_t nFrames = g_FramesInFlight;

    m_vUbo.resize(nFrames);
    for (uint32_t i = 0; i < nFrames; ++i)
        m_vUbo[i] = make_unique<VulkanUBO>(m_VulkanDevice, sizeof(SurfaceUbo));

    // Pool: UBO + depth + thickness per frame (UBO also for the particle passes), 3 samplers for the blur
    VkDescriptorPoolSize aSizes[2] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2 * nFrames },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 + 2 * nFrames },
    };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 2;
    pi.pPoolSizes    = aSizes;
    pi.maxSets       = 2 * nFrames + 3;
    vkCreateDescriptorPool(dev, &pi, nullptr, &m_Pool);

    auto allocate = [&](VkDescriptorSetLayout layout, uint32_t nCount, VkDescriptorSet* pSets) {
        vector<VkDescriptorSetLayout> vLayouts(nCount, layout);
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool     = m_Pool;
        ai.descriptorSetCount = nCount;
        ai.pSetLayouts        = vLayouts.data();
        vkAllocateDescriptorSets(dev, &ai, pSets);
    };
    m_vUboSets.resize(nFrames);
    m_vCompSets.resize(nFrames);
    allocate(m_UboSetLayout,  nFrames, m_vUboSets.data());
    allocate(m_CompSetLayout, nFrames, m_vCompSets.data());
    allocate(m_TexSetLayout,  3,       m_aBlurSets);

    auto writeSampler = [&](VkDescriptorSet set, uint32_t nBinding, VkImageView view) {
        VkDescriptorImageInfo info{ m_Sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = set;
        w.dstBinding      = nBinding;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &info;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    };
    auto writeUbo = [&](VkDescriptorSet set, VkBuffer buffer) {
        VkDescriptorBufferInfo info{ buffer, 0, sizeof(SurfaceUbo) };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo     = &info;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    };

    for (uint32_t i = 0; i < nFrames; ++i)
    {
        writeUbo(m_vUboSets[i],  m_vUbo[i]->buffer);
        writeUbo(m_vCompSets[i], m_vUbo[i]->buffer);
        writeSampler(m_vCompSets[i], 1, m_SmoothB.view);        // final smoothed depth
        writeSampler(m_vCompSets[i], 2, m_Thickness.view);
    }
    writeSampler(m_aBlurSets[0], 0, m_RawDepth.view);
    writeSampler(m_aBlurSets[1], 0, m_SmoothA.view);
    writeSampler(m_aBlurSets[2], 0, m_SmoothB.view);
}


// ============================================================================
// Recording
// ============================================================================

void SphSurface::UpdateUbo(uint32_t nFrame, float spacing, const mat4& view, const mat4& proj, const Settings& settings)
{
    const float radius = settings.radiusScale * spacing;
    const float az     = glm::radians(settings.lightAzimuth);
    const float el     = glm::radians(settings.lightElevation);

    // The impostors overlap: the sum of their chords is (4/3 pi r^3 / s^3) times the real thickness
    const float ratio        = settings.radiusScale;
    const float thickPhysics = 1.0f / (4.18879f * ratio * ratio * ratio);

    SurfaceUbo u{};
    u.view       = view;
    u.proj       = proj;
    u.invView    = glm::inverse(view);
    u.light      = vec4(std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az), 0.0f);
    u.screen     = vec4((float)m_Extent.width, (float)m_Extent.height, 1.0f / (float)m_Extent.width, 1.0f / (float)m_Extent.height);
    u.params0    = vec4(radius, 0.0f, 0.0f, 0.0f);
    u.absorption = vec4((vec3(1.0f) - settings.waterColor) * settings.absorption, 0.0f);
    u.waterColor = vec4(settings.waterColor, 0.0f);
    u.params1    = vec4(settings.fresnelF0, settings.specularPower, settings.exposure, settings.thicknessScale * thickPhysics);
    memcpy(m_vUbo[nFrame]->data, &u, sizeof(SurfaceUbo));
}

void SphSurface::BeginPass(VkCommandBuffer cmd, VkRenderPass renderPass, VkFramebuffer framebuffer, const VkClearValue* pClears, uint32_t nClears)
{
    VkRenderPassBeginInfo bi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    bi.renderPass          = renderPass;
    bi.framebuffer         = framebuffer;
    bi.renderArea.extent   = m_Extent;
    bi.clearValueCount     = nClears;
    bi.pClearValues        = pClears;
    vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{ 0.0f, 0.0f, (float)m_Extent.width, (float)m_Extent.height, 0.0f, 1.0f };
    VkRect2D   scissor{ { 0, 0 }, m_Extent };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void SphSurface::RecordBlurPass(VkCommandBuffer cmd, VkFramebuffer framebuffer, uint32_t nInputSet, float dirX, float dirY,
                                float filterRadius, float projScale, float depthSigma, float maxRadius)
{
    BeginPass(cmd, m_RpBlur, framebuffer, nullptr, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipeBlur);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_BlurLayout, 0, 1, &m_aBlurSets[nInputSet], 0, nullptr);

    BlurPush pc{};
    pc.dirX = dirX;  pc.dirY = dirY;
    pc.filterRadius = filterRadius;  pc.projScale = projScale;  pc.depthSigma = depthSigma;  pc.maxRadius = maxRadius;
    vkCmdPushConstants(cmd, m_BlurLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BlurPush), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void SphSurface::RecordOffscreen(VkCommandBuffer cmd, uint32_t nFrame, VkBuffer particles, uint32_t nCount, float spacing,
                                 const mat4& view, const mat4& proj, const Settings& settings)
{
    if (!m_bCreated || nCount == 0 || nFrame >= m_vUbo.size())
        return;

    UpdateUbo(nFrame, spacing, view, proj, settings);

    const VkDeviceSize offset = 0;

    // 1. Depth: linear view depth in the color target, z-test on the D32 target
    {
        VkClearValue aClears[2] = {};
        aClears[0].color        = { { 0.0f, 0.0f, 0.0f, 0.0f } };   // 0 = no fluid
        aClears[1].depthStencil = { 1.0f, 0 };
        BeginPass(cmd, m_RpDepth, m_FbDepth, aClears, 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipeDepth);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticleLayout, 0, 1, &m_vUboSets[nFrame], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &particles, &offset);
        vkCmdDraw(cmd, 1, nCount, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    // 2. Thickness: additive blending
    {
        VkClearValue clear{};
        clear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        BeginPass(cmd, m_RpThick, m_FbThick, &clear, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipeThick);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticleLayout, 0, 1, &m_vUboSets[nFrame], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &particles, &offset);
        vkCmdDraw(cmd, 1, nCount, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    // 3. Smoothing: raw -> A (horizontal) -> B (vertical), then B -> A -> B for the next iterations
    const float radius       = settings.radiusScale * spacing;
    const float projScale    = std::abs(proj[1][1]) * (float)m_Extent.height * 0.5f;
    const float filterRadius = settings.filterRadius * radius;
    const float depthSigma   = std::max(settings.depthSigma * radius, 1e-5f);
    const float maxRadius    = (float)std::clamp(settings.nMaxFilterPixels, 1, 64);
    const int   nIterations  = std::max(1, settings.nBlurIterations);
    for (int it = 0; it < nIterations; ++it)
    {
        RecordBlurPass(cmd, m_FbBlurA, (it == 0) ? 0u : 2u, 1.0f, 0.0f, filterRadius, projScale, depthSigma, maxRadius);
        RecordBlurPass(cmd, m_FbBlurB, 1u,                  0.0f, 1.0f, filterRadius, projScale, depthSigma, maxRadius);
    }
}

void SphSurface::RecordComposite(VkCommandBuffer cmd, uint32_t nFrame)
{
    if (!m_bCreated || nFrame >= m_vCompSets.size())
        return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipeComposite);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_CompositeLayout, 0, 1, &m_vCompSets[nFrame], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}
