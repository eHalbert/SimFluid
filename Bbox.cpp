/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#include "Bbox.h"


extern uint32_t                 g_FramesInFlight;


Bbox::Bbox(shared_ptr<VulkanDevice>& vulkanDevice)
{
	mVulkanDevice       = vulkanDevice;
}
Bbox::~Bbox()
{
	mBboxPipeline.destroy(mVulkanDevice->device);
}

// Pipeline of the bounding box
void Bbox::CreateBboxBuffers()
{
    vkDeviceWaitIdle(mVulkanDevice->device); 
    mBboxVertexBuffer.reset();
    mBboxIndexBuffer.reset();

    // 8 vertices of the BBox
    vector<sBboxVertex> vertices(8);
    vertices[0] = { mBbox.min };                                    // 0: min
    vertices[1] = { vec3(mBbox.max.x, mBbox.min.y, mBbox.min.z) };  // 1: x+
    vertices[2] = { vec3(mBbox.max.x, mBbox.max.y, mBbox.min.z) };  // 2: x+y+
    vertices[3] = { vec3(mBbox.min.x, mBbox.max.y, mBbox.min.z) };  // 3: y+
    vertices[4] = { vec3(mBbox.min.x, mBbox.min.y, mBbox.max.z) };  // 4: z+
    vertices[5] = { vec3(mBbox.max.x, mBbox.min.y, mBbox.max.z) };  // 5: x+z+
    vertices[6] = { mBbox.max };                                    // 6: x+y+z+
    vertices[7] = { vec3(mBbox.min.x, mBbox.max.y, mBbox.max.z) };  // 7: y+z+

    // 24 indices for 12 edges (each edge = 2 indices)
    vector<uint32_t> indices = {
        0,1, 1,2, 2,3, 3,0,  // Front face (Z min)
        4,5, 5,6, 6,7, 7,4,  // Back face (Z max)
        0,4, 1,5, 2,6, 3,7   // Side connections
    };

    // Vertex buffer
    VkDeviceSize vertexSize = sizeof(sBboxVertex) * vertices.size();
    VulkanBuffer stagingBuffer(mVulkanDevice, vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(mVulkanDevice->device, stagingBuffer.bufferMemory, 0, vertexSize, 0, &data);
    memcpy(data, vertices.data(), vertexSize);
    vkUnmapMemory(mVulkanDevice->device, stagingBuffer.bufferMemory);

    mBboxVertexBuffer = std::make_unique<VulkanBuffer>(mVulkanDevice, vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mBboxVertexBuffer->CopyIntoBuffer(stagingBuffer, vertexSize);

    // Index buffer
    VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
    VulkanBuffer stagingIndexBuffer(mVulkanDevice, indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkMapMemory(mVulkanDevice->device, stagingIndexBuffer.bufferMemory, 0, indexSize, 0, &data);
    memcpy(data, indices.data(), indexSize);
    vkUnmapMemory(mVulkanDevice->device, stagingIndexBuffer.bufferMemory);

    mBboxIndexBuffer = std::make_unique<VulkanBuffer>(mVulkanDevice, indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mBboxIndexBuffer->CopyIntoBuffer(stagingIndexBuffer, indexSize);
}
void Bbox::CreateBboxPipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent)
{
    CreateBboxBuffers();

    // Previous cleanup
	mBboxPipeline.destroy(mVulkanDevice->device);

    // 1. Shaders
    auto vertCode = CompileShaderRuntime("Resources/Shaders/Model/model_bbox.vert");
    auto fragCode = CompileShaderRuntime("Resources/Shaders/Model/model_bbox.frag");
    VkShaderModule vertModule = CreateShaderModule(mVulkanDevice->device, vertCode);
    VkShaderModule fragModule = CreateShaderModule(mVulkanDevice->device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main" }
    };

    // 2. Vertex input (position only)
    auto bindingDescription = sBboxVertex::getBindingDescription();
    auto attributeDescriptions = sBboxVertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();


    // 3. DescriptorSetLayout (sMatrixColorUBO)
    array<VkDescriptorSetLayoutBinding, 1> bindings = { {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr }
    } };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(mVulkanDevice->device, &layoutInfo, nullptr, &mBboxPipeline.descSetLayout);

    // 4. PipelineLayout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mBboxPipeline.descSetLayout;
    vkCreatePipelineLayout(mVulkanDevice->device, &pipelineLayoutInfo, nullptr, &mBboxPipeline.pipelineLayout);

    // 5. Input assembly (lines)
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 6. Viewport & scissor
    VkViewport viewport{ 0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, swapChainExtent };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // 7. Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE; 
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // 8. Multisampling 8x
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

    // 10. Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

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
    pipelineInfo.layout = mBboxPipeline.pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    vkCreateGraphicsPipelines(mVulkanDevice->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mBboxPipeline.pipeline);

    vkDestroyShaderModule(mVulkanDevice->device, vertModule, nullptr);
    vkDestroyShaderModule(mVulkanDevice->device, fragModule, nullptr);

    CreateBboxDescriptors();
}
void Bbox::CreateBboxDescriptors()
{
    mBboxPipeline.descSet.resize(g_FramesInFlight);
    mBboxPipeline.ubo.resize(g_FramesInFlight);
    
    for (size_t i = 0; i < g_FramesInFlight; i++)
        mBboxPipeline.ubo[i] = make_unique<VulkanUBO>(mVulkanDevice, sizeof(sMatrixColorUBO));

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(g_FramesInFlight) }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = static_cast<uint32_t>(g_FramesInFlight);// 2 sets
    vkCreateDescriptorPool(mVulkanDevice->device, &poolInfo, nullptr, &mBboxPipeline.descPool);

    // Allocate 1 descriptor set per frame in flight
    vector<VkDescriptorSetLayout> layouts(g_FramesInFlight, mBboxPipeline.descSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mBboxPipeline.descPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(g_FramesInFlight);// 2 sets
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(mVulkanDevice->device, &allocInfo, mBboxPipeline.descSet.data());

    UpdateBboxDescriptors();
}
void Bbox::UpdateBboxDescriptors()
{
    for (size_t i = 0; i < g_FramesInFlight; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mBboxPipeline.ubo[i]->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = mBboxPipeline.ubo[i]->GetSize();

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mBboxPipeline.descSet[i];  // Set par frame
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(mVulkanDevice->device, 1, &write, 0, nullptr);
    }
    mHasBboxPipeline = true;
}
// Rendering of the bounding box
void Bbox::RenderBbox(VkCommandBuffer cmd, int iCurrentFrame, Camera& camera, mat4 model, vec4 color)
{
    if (!bVisible) return;

    // Writing to the UBO of the current frame
    sMatrixColorUBO* ubo = static_cast<sMatrixColorUBO*>(mBboxPipeline.ubo[iCurrentFrame]->data);
    *ubo = { model, camera.GetView(), camera.GetProjection(), color };

    // Bind the pipeline and the descriptor set of the current frame
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mBboxPipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mBboxPipeline.pipelineLayout, 0, 1, &mBboxPipeline.descSet[iCurrentFrame], 0, nullptr);  // Set par frame !

    // Bind vertex/index buffers
    VkBuffer vertexBuffers[] = { mBboxVertexBuffer->buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, mBboxIndexBuffer->buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, 24, 1, 0, 0, 0);
}

// Recreate all pipelines if SwapChain has changed
void Bbox::RecreatePipelines(VkRenderPass renderPassScene, VkRenderPass renderPassShadow, VkRenderPass renderPassReflection,  VkRenderPass renderPassBridgeMask, VkExtent2D swapChainExtent)
{
    vkDeviceWaitIdle(mVulkanDevice->device); 
    
    if (mHasBboxPipeline)
        CreateBboxPipeline(renderPassScene, swapChainExtent);

}