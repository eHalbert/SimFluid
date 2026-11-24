/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#include "HullMesh.h"

extern uint32_t g_FramesInFlight;


// LINE MESH (simple lines) ////////////////////////////////////////////////////////

LineMesh::LineMesh(shared_ptr<VulkanDevice> vulkanDevice, const vector<vec3>& vertices)
{
    mVulkanDevice = vulkanDevice;

    CreateVertexBuffer(vertices);
}
LineMesh::~LineMesh()
{
    mPipeline.destroy(mVulkanDevice->device);
    mVertexBuffer.reset();
    mStagingVertexBuffer.reset();
}

void LineMesh::CreateVertexBuffer(const vector<vec3>& vertices)
{
    mVertexCount = static_cast<uint32_t>(vertices.size());
    size_t vertexBufferSize = sizeof(sVertexLine) * vertices.size();

    // GPU Buffer
    mVertexBuffer = make_unique<VulkanBuffer>(mVulkanDevice, vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Reusable staging buffer
    mStagingVertexBuffer = make_unique<VulkanBuffer>(mVulkanDevice, vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Upload initial
    UpdateVertices(vertices);
}
void LineMesh::CreatePipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent, VkPrimitiveTopology topology)
{
    mPipeline.destroy(mVulkanDevice->device);

    // 1. Shaders
    auto vertCode = CompileShaderRuntime("Resources/Shaders/Model/line_unicolor.vert");
    auto fragCode = CompileShaderRuntime("Resources/Shaders/Model/line_unicolor.frag");
    VkShaderModule vertModule = CreateShaderModule(mVulkanDevice->device, vertCode);
    VkShaderModule fragModule = CreateShaderModule(mVulkanDevice->device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main" }
    };

    // 2. Vertex input (only position)
    auto bindingDesc = sVertexLine::getBindingDescription();
    auto attrDescs = sVertexLine::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs.data();

    // 3. Descriptor Set Layout (UBO)
    array<VkDescriptorSetLayoutBinding, 1> bindings = { {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },
    } };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(mVulkanDevice->device, &layoutInfo, nullptr, &mPipeline.descSetLayout);

    // 4. Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mPipeline.descSetLayout;
    vkCreatePipelineLayout(mVulkanDevice->device, &pipelineLayoutInfo, nullptr, &mPipeline.pipelineLayout);

    // 5. Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 6. Viewport & scissors
    VkViewport viewport{ 0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, swapChainExtent };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // 7. Rasterizer → Mode ligne
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // 8. Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = mVulkanDevice->msaaSamples;

    // 9. Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // 10. Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

	// 11. Dynamic state
    
    // 12. Pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = mPipeline.pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(mVulkanDevice->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipeline.pipeline);

    vkDestroyShaderModule(mVulkanDevice->device, fragModule, nullptr);
    vkDestroyShaderModule(mVulkanDevice->device, vertModule, nullptr);

    CreateDescriptor();
}
void LineMesh::CreateDescriptor()
{
    mPipeline.descSet.resize(g_FramesInFlight);
    mPipeline.ubo.resize(g_FramesInFlight); 
    
    for (uint32_t i = 0; i < g_FramesInFlight; ++i)
        mPipeline.ubo[i] = make_unique<VulkanUBO>(mVulkanDevice, sizeof(sMatrixColorUBO));

    VkDescriptorPoolSize poolSizes[1] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * g_FramesInFlight}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = g_FramesInFlight;
    vkCreateDescriptorPool(mVulkanDevice->device, &poolInfo, nullptr, &mPipeline.descPool);

    // Allocate 1 descriptor set per frame in flight
    vector<VkDescriptorSetLayout> layouts(g_FramesInFlight, mPipeline.descSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mPipeline.descPool;
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(mVulkanDevice->device, &allocInfo, mPipeline.descSet.data());

    UpdateDescriptor();
}
void LineMesh::UpdateDescriptor()
{
    for (uint32_t i = 0; i < g_FramesInFlight; ++i)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mPipeline.ubo[i]->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = mPipeline.ubo[i]->GetSize();

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = mPipeline.descSet[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(mVulkanDevice->device, 1, &descriptorWrite, 0, nullptr);
    }
}

void LineMesh::UpdateUBO(uint32_t currentFrame, Camera& camera, mat4& model, vec4 color)
{
    sMatrixColorUBO* ubo = static_cast<sMatrixColorUBO*>(mPipeline.ubo[currentFrame]->data);
    *ubo = { model, camera.GetView(), camera.GetProjection(), color };
}
void LineMesh::UpdateVertices(const vector<vec3>& vertices)
{
    mVertexCount = static_cast<uint32_t>(vertices.size());
    size_t vertexBufferSize = sizeof(sVertexLine) * vertices.size();

    // Conversion vec3 → VertexLine
    vector<sVertexLine> vertexLines(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        vertexLines[i].pos[0] = vertices[i].x;
        vertexLines[i].pos[1] = vertices[i].y;
        vertexLines[i].pos[2] = vertices[i].z;
    }

    // Upload
    void* data;
    vkMapMemory(mVulkanDevice->device, mStagingVertexBuffer->bufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertexLines.data(), vertexBufferSize);
    vkUnmapMemory(mVulkanDevice->device, mStagingVertexBuffer->bufferMemory);

    mVertexBuffer->CopyIntoBuffer(*mStagingVertexBuffer, vertexBufferSize);
}
void LineMesh::Render(VkCommandBuffer cmd, uint32_t currentFrame)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipelineLayout, 0, 1, &mPipeline.descSet[currentFrame], 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mVertexBuffer->buffer, &offset);

    vkCmdDraw(cmd, mVertexCount, 1, 0, 0);
}

void LineMesh::RecreatePipelines(VkRenderPass renderPass, VkExtent2D newExtent, VkPrimitiveTopology topology)
{
    CreatePipeline(renderPass, newExtent, topology);
}


// GRID MESH //////////////////////////////////////////////////////////////////

GridMesh::GridMesh(shared_ptr<VulkanDevice> vulkanDevice, VkRenderPass renderPass, VkExtent2D swapChainExtent, uint32_t gridSize, float cellSize)
{
    mVulkanDevice = vulkanDevice;
    mGridSize = gridSize;
    mCellSize = cellSize;

    CreateVertices();
    CreateVertexBuffer();
    CreatePipeline(renderPass, swapChainExtent);
}
GridMesh::~GridMesh()
{
    mPipeline.destroy(mVulkanDevice->device);
    mVertexBuffer.reset();
    mStagingVertexBuffer.reset();
}

void GridMesh::CreateVertices() 
{
    float halfSize = (mGridSize * mCellSize) / 2.0f;

    // Lines parallel to the X axis (Z fixed, Y=0)
    for (uint32_t i = 0; i <= mGridSize; ++i) 
    {
        float z = -halfSize + i * mCellSize;

        mVertices.push_back({ -halfSize, 0.0f, z }); // start
        mVertices.push_back({ +halfSize, 0.0f, z }); // end
    }

    // Lines parallel to the Z axis (X fixed, Y=0)
    for (uint32_t i = 0; i <= mGridSize; ++i) 
    {
        float x = -halfSize + i * mCellSize;

        mVertices.push_back({ x, 0.0f, -halfSize }); // start
        mVertices.push_back({ x, 0.0f, +halfSize }); // end
    }
}
void GridMesh::CreateVertexBuffer() 
{
    size_t vertexBufferSize = sizeof(sVertexLine) * mVertices.size();

    // GPU buffer (DEVICE_LOCAL)
    mVertexBuffer = make_unique<VulkanBuffer>(mVulkanDevice, vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Staging (HOST_VISIBLE + HOST_COHERENT)
    mStagingVertexBuffer = make_unique<VulkanBuffer>(mVulkanDevice, vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(mVulkanDevice->device, mStagingVertexBuffer->bufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, mVertices.data(), vertexBufferSize);
    vkUnmapMemory(mVulkanDevice->device, mStagingVertexBuffer->bufferMemory);

    mVertexBuffer->CopyIntoBuffer(*mStagingVertexBuffer, vertexBufferSize);
}
void GridMesh::CreatePipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent)
{
	mPipeline.destroy(mVulkanDevice->device);

    // 1. Shaders (mono‑color)
    auto vertCode = CompileShaderRuntime("Resources/Shaders/Model/line_unicolor.vert");
    auto fragCode = CompileShaderRuntime("Resources/Shaders/Model/line_unicolor.frag");
    VkShaderModule vertModule = CreateShaderModule(mVulkanDevice->device, vertCode);
    VkShaderModule fragModule = CreateShaderModule(mVulkanDevice->device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main" }
    };

    // 2. Vertex input
    auto bindingDesc = sVertexLine::getBindingDescription();
    auto attrDescs = sVertexLine::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs.data();

    // 3. Descriptor set layout (UBO)
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboBinding;
    vkCreateDescriptorSetLayout(mVulkanDevice->device, &layoutInfo, nullptr, &mPipeline.descSetLayout);

    // 4. Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mPipeline.descSetLayout;
    vkCreatePipelineLayout(mVulkanDevice->device, &pipelineLayoutInfo, nullptr, &mPipeline.pipelineLayout);

    // 5. Input assembly : LINES
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; // each pair of points = a segment
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 6. Viewport & scissors
    VkViewport viewport{ 0.0f, 0.0f, float(swapChainExtent.width), float(swapChainExtent.height), 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, swapChainExtent };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // 7. Rasterizer (mode ligne)
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // 8. Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = mVulkanDevice->msaaSamples;

    // 9. Depth & stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // 10. Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // 11. Dynamic state (no dynamic state)
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 0;
    dynamicState.pDynamicStates = nullptr;

    // 12. Pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = mPipeline.pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(mVulkanDevice->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipeline.pipeline);

    vkDestroyShaderModule(mVulkanDevice->device, fragModule, nullptr);
    vkDestroyShaderModule(mVulkanDevice->device, vertModule, nullptr);

    CreateDescriptor();
}
void GridMesh::CreateDescriptor() 
{
    mPipeline.descSet.resize(g_FramesInFlight);
    mPipeline.ubo.resize(g_FramesInFlight);
    
    // 1. Descriptor pool
    VkDescriptorPoolSize poolSizes[1] = { {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, g_FramesInFlight  // 1 UBO per frame → 2 UBOs for 2 frames
    } };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = g_FramesInFlight;  // descriptor sets

    vkCreateDescriptorPool(mVulkanDevice->device, &poolInfo, nullptr, &mPipeline.descPool);

    // 2. UBO for each frame
    for (uint32_t i = 0; i < g_FramesInFlight; ++i) 
        mPipeline.ubo[i] = make_unique<VulkanUBO>(mVulkanDevice, sizeof(sMatrixColorUBO));

    // 3. Allocate 1 descriptor set per frame in flight
    vector<VkDescriptorSetLayout> layouts(g_FramesInFlight, mPipeline.descSetLayout);

    // 4. Allocation of 2 descriptor sets
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mPipeline.descPool;
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    vkAllocateDescriptorSets(mVulkanDevice->device, &allocInfo, mPipeline.descSet.data());

    // 5. Write the UBOs into each descriptor set
    for (uint32_t i = 0; i < g_FramesInFlight; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mPipeline.ubo[i]->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = mPipeline.ubo[i]->GetSize();

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mPipeline.descSet[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(mVulkanDevice->device, 1, &write, 0, nullptr);
    }
}

void GridMesh::UpdateUBO(uint32_t currentFrame, Camera& camera, const mat4& model, const vec4& color)
{
    sMatrixColorUBO* ubo = static_cast<sMatrixColorUBO*>(mPipeline.ubo[currentFrame]->data);
    *ubo = { model, camera.GetView(), camera.GetProjection(), color};
}
void GridMesh::Render(VkCommandBuffer cmd, uint32_t currentFrame)
{
    if (!bVisible) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline.pipelineLayout, 0, 1, &mPipeline.descSet[currentFrame], 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mVertexBuffer->buffer, &offset);

    uint32_t vertexCount = static_cast<uint32_t>(mVertices.size());
    vkCmdDraw(cmd, vertexCount, 1, 0, 0); // LINE_LIST : each pair of vertices = a segment
}

void GridMesh::RecreatePipeline(VkRenderPass renderPassScene, VkExtent2D newExtent)
{
    CreatePipeline(renderPassScene, newExtent);
}
