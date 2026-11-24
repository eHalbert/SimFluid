/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#pragma once

// 1. PROJECT
#include "vulkan_device.hpp"
#include "vulkan_swapchain.hpp"
#include "vulkan_texture.hpp"
#include "Structures.h"
#include "Utility.h"
#include "Camera.h"

// 2. LIB   
#include <vulkan/vulkan.h>
// glm
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
using namespace glm;

// 3. WIN
#include <iostream>
#include <vector>
using namespace std;


// LINE MESH (Simple lines) ////////////////////////////////////////////////////////

struct sVertexLine
{
    float pos[3];   // x, y, z only
    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(sVertexLine);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding;
    }

    static array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions()
    {
        array<VkVertexInputAttributeDescription, 1> attrs{};
        // layout(location = 0) vec3 aPos;
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(sVertexLine, pos);
        return attrs;
    }
};

class LineMesh
{
public:
    LineMesh(shared_ptr<VulkanDevice> vulkanDevice, const vector<vec3>& vertices);
    ~LineMesh();

    void CreatePipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
    void UpdateUBO(uint32_t currentFrame, Camera& camera, mat4& model, vec4 color);
    void UpdateVertices(const vector<vec3>& vertices);
    void Render(VkCommandBuffer cmd, uint32_t currentFrame);

    void RecreatePipelines(VkRenderPass renderPass, VkExtent2D newExtent, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);

private:
    shared_ptr<VulkanDevice>    mVulkanDevice;
    uint32_t                    mFramesInFlight;
    uint32_t                    mVertexCount;

    unique_ptr<VulkanBuffer>    mVertexBuffer;
    unique_ptr<VulkanBuffer>    mStagingVertexBuffer;

    sPipeline_x                  mPipeline;

    void CreateVertexBuffer(const vector<vec3>& vertices);
    void CreateDescriptor();
    void UpdateDescriptor();
};


// GRID MESH //////////////////////////////////////////////////////////////////

class GridMesh 
{
public:
    GridMesh(shared_ptr<VulkanDevice> vulkanDevice, VkRenderPass renderPass, VkExtent2D swapChainExtent, uint32_t gridSize, float cellSize);
    ~GridMesh();

    void UpdateUBO(uint32_t currentFrame, Camera& camera, const mat4& model, const vec4& color);
    void Render(VkCommandBuffer cmd, uint32_t currentFrame);
    void RecreatePipeline(VkRenderPass renderPassScene, VkExtent2D newExtent);

    bool                        bVisible = false;

private:
    void CreateVertices();
    void CreateVertexBuffer();
    void CreatePipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent);
    void CreateDescriptor();

private:
    shared_ptr<VulkanDevice>    mVulkanDevice;
    uint32_t                    mFramesInFlight;

    // Geometry
    vector<sVertexLine>         mVertices;
    unique_ptr<VulkanBuffer>    mVertexBuffer;
    unique_ptr<VulkanBuffer>    mStagingVertexBuffer;

    // Pipeline
    sPipeline_x                 mPipeline;

    uint32_t                    mGridSize;
    float                       mCellSize;
};