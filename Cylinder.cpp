#include "Cylinder.h"

extern uint32_t g_FramesInFlight;


Cylinder::Cylinder(shared_ptr<VulkanDevice>& vulkanDevice)
{
    mVulkanDevice = vulkanDevice;
}
Cylinder::~Cylinder()
{
    mPipeline.destroy(mVulkanDevice->device);
}

void Cylinder::SetCylinder(const vec2& centerXZ, float radius, float yMin, float yMax)
{
    mCenter = centerXZ;
    mRadius = radius;
    mYMin   = yMin;
    mYMax   = yMax;
}

void Cylinder::CreateBuffers()
{
    vkDeviceWaitIdle(mVulkanDevice->device);
    mVertexBuffer.reset();
    mIndexBuffer.reset();

    // Horizontal rings: bottom and top, plus intermediate ones every mRingSpacing
    const float height = std::max(mYMax - mYMin, 1e-3f);
    const int   rings  = std::max(2, (int)std::lround(height / mRingSpacing) + 1);
    const int   n      = std::max(3, mSegments);
    const float twoPi  = 6.28318530717958f;

    vector<sCylinderVertex> vertices;
    vertices.reserve((size_t)rings * n);
    for (int r = 0; r < rings; ++r)
    {
        const float y = mYMin + height * (float)r / (float)(rings - 1);
        for (int i = 0; i < n; ++i)
        {
            const float a = twoPi * (float)i / (float)n;
            vertices.push_back({ vec3(mCenter.x + mRadius * std::cos(a), y, mCenter.y + mRadius * std::sin(a)) });
        }
    }

    vector<uint32_t> indices;
    for (int r = 0; r < rings; ++r)
    {
        const uint32_t base = (uint32_t)(r * n);
        // Circle
        for (int i = 0; i < n; ++i)
        {
            indices.push_back(base + (uint32_t)i);
            indices.push_back(base + (uint32_t)((i + 1) % n));
        }
        // Vertical lines to the next ring
        if (r + 1 < rings)
        {
            for (int i = 0; i < n; i += std::max(1, mVerticalEvery))
            {
                indices.push_back(base + (uint32_t)i);
                indices.push_back(base + (uint32_t)(n + i));
            }
        }
    }
    mIndexCount = (uint32_t)indices.size();

    // Vertex buffer
    VkDeviceSize vertexSize = sizeof(sCylinderVertex) * vertices.size();
    VulkanBuffer stagingVertex(mVulkanDevice, vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* data;
    vkMapMemory(mVulkanDevice->device, stagingVertex.bufferMemory, 0, vertexSize, 0, &data);
    memcpy(data, vertices.data(), vertexSize);
    vkUnmapMemory(mVulkanDevice->device, stagingVertex.bufferMemory);

    mVertexBuffer = std::make_unique<VulkanBuffer>(mVulkanDevice, vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mVertexBuffer->CopyIntoBuffer(stagingVertex, vertexSize);

    // Index buffer
    VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
    VulkanBuffer stagingIndex(mVulkanDevice, indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(mVulkanDevice->device, stagingIndex.bufferMemory, 0, indexSize, 0, &data);
    memcpy(data, indices.data(), indexSize);
    vkUnmapMemory(mVulkanDevice->device, stagingIndex.bufferMemory);

    mIndexBuffer = std::make_unique<VulkanBuffer>(mVulkanDevice, indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mIndexBuffer->CopyIntoBuffer(stagingIndex, indexSize);
}

void Cylinder::CreateCylinderPipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent)
{
    CreateBuffers();

    // Previous cleanup
    mPipeline.destroy(mVulkanDevice->device);

    // 1. Shaders (same as the bounding box: position + color UBO)
    auto vertCode = CompileShaderRuntime("Resources/Shaders/Model/model_bbox.vert");
    auto fragCode = CompileShaderRuntime("Resources/Shaders/Model/model_bbox.frag");
    VkShaderModule vertModule = CreateShaderModule(mVulkanDevice->device, vertCode);
    VkShaderModule fragModule = CreateShaderModule(mVulkanDevice->device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main" }
    };

    // 2. Vertex input (position only)
    auto bindingDescription    = sCylinderVertex::getBindingDescription();
    auto attributeDescriptions = sCylinderVertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 1;
    vertexInputInfo.pVertexBindingDescriptions      = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions    = attributeDescriptions.data();

    // 3. DescriptorSetLayout (sMatrixColorUBO)
    array<VkDescriptorSetLayoutBinding, 1> bindings = { {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr }
    } };
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings    = bindings.data();
    vkCreateDescriptorSetLayout(mVulkanDevice->device, &layoutInfo, nullptr, &mPipeline.descSetLayout);

    // 4. PipelineLayout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts    = &mPipeline.descSetLayout;
    vkCreatePipelineLayout(mVulkanDevice->device, &pipelineLayoutInfo, nullptr, &mPipeline.pipelineLayout);

    // 5. Input assembly (lines)
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 6. Viewport & scissor
    VkViewport viewport{ 0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, swapChainExtent };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    // 7. Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // 8. Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = mVulkanDevice->msaaSamples;

    // 9. Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    // 10. Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    // 11. Pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.layout              = mPipeline.pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;
    vkCreateGraphicsPipelines(mVulkanDevice->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipeline.pipeline);

    vkDestroyShaderModule(mVulkanDevice->device, vertModule, nullptr);
    vkDestroyShaderModule(mVulkanDevice->device, fragModule, nullptr);

    CreateDescriptors();
}

void Cylinder::CreateDescriptors()
{
    mPipeline.descSet.resize(g_FramesInFlight);
    mPipeline.ubo.resize(g_FramesInFlight);

    for (size_t i = 0; i < g_FramesInFlight; i++)
        mPipeline.ubo[i] = make_unique<VulkanUBO>(mVulkanDevice, sizeof(sMatrixColorUBO));

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)g_FramesInFlight }
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = poolSizes;
    poolInfo.maxSets       = (uint32_t)g_FramesInFlight;
    vkCreateDescriptorPool(mVulkanDevice->device, &poolInfo, nullptr, &mPipeline.descPool);

    // 1 descriptor set per frame in flight
    vector<VkDescriptorSetLayout> layouts(g_FramesInFlight, mPipeline.descSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = mPipeline.descPool;
    allocInfo.descriptorSetCount = (uint32_t)g_FramesInFlight;
    allocInfo.pSetLayouts        = layouts.data();
    vkAllocateDescriptorSets(mVulkanDevice->device, &allocInfo, mPipeline.descSet.data());

    UpdateDescriptors();
}

void Cylinder::UpdateDescriptors()
{
    for (size_t i = 0; i < g_FramesInFlight; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mPipeline.ubo[i]->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range  = mPipeline.ubo[i]->GetSize();

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mPipeline.descSet[i];
        write.dstBinding      = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &bufferInfo;
        vkUpdateDescriptorSets(mVulkanDevice->device, 1, &write, 0, nullptr);
    }
    mHasPipeline = true;
}

void Cylinder::Render(VkCommandBuffer cmd, int iCurrentFrame, Camera& camera, mat4 model, vec4 color)
{
    if (!bVisible || !mHasPipeline || mIndexCount == 0)
        return;

    // UBO of the current frame
    sMatrixColorUBO* ubo = static_cast<sMatrixColorUBO*>(mPipeline.ubo[iCurrentFrame]->data);
    *ubo = { model, camera.GetView(), camera.GetProjection(), color };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipelineLayout, 0, 1, &mPipeline.descSet[iCurrentFrame], 0, nullptr);

    VkBuffer     vertexBuffers[] = { mVertexBuffer->buffer };
    VkDeviceSize offsets[]       = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, mIndexBuffer->buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, mIndexCount, 1, 0, 0, 0);
}

// Recreate the pipeline if the SwapChain has changed
void Cylinder::RecreatePipelines(VkRenderPass renderPass, VkExtent2D swapChainExtent)
{
    vkDeviceWaitIdle(mVulkanDevice->device);
    if (mHasPipeline)
        CreateCylinderPipeline(renderPass, swapChainExtent);
}
