/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#pragma once

// 1. PROJECT
#include "Structures.h"
#include "Utility.h"
#include "Camera.h"
#include "vulkan_device.hpp"
#include "vulkan_swapchain.hpp"
#include "vulkan_texture.hpp"
#include "vulkan_ubo.hpp"

// 2. LIB
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
// glm
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
using namespace glm;
// stb
#include <stb/stb_image.h>
// assimp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// 3. WIN
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <array>
#include <chrono>
#include <unordered_map>
using namespace std;



class Bbox
{
public:
    Bbox(shared_ptr<VulkanDevice>& vulkanDevice);
    ~Bbox();
	
	void SetBbox(const sBBox& bbox) { mBbox = bbox; }
    void CreateBboxPipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent);
    void RenderBbox(VkCommandBuffer cmd, int iCurrentFrame, Camera& camera, mat4 model, vec4 color);

    void RecreatePipelines(VkRenderPass renderPassScene, VkRenderPass renderPassShadow, VkRenderPass renderPassReflection, VkRenderPass renderPassBridgeMask, VkExtent2D swapChainExtent);
   
    bool                        bVisible        = true;

private:
    
    struct sBboxVertex {
        vec3 pos;

        static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(sBboxVertex);
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            return bindingDescription;
        }

        static array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
            array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
            attributeDescriptions[0].binding = 0;
            attributeDescriptions[0].location = 0;
            attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[0].offset = 0;
            return attributeDescriptions;
        }
    };

    sBBox				        mBbox;				// Bounding box

    shared_ptr<VulkanDevice>	mVulkanDevice;
    VkFrontFace                 mFrontFace;
    VkCullModeFlags             mCullMode;

	// For bounding box rendering (wireframe colored)
    sPipeline_x                     mBboxPipeline;
    unique_ptr<VulkanBuffer>        mBboxVertexBuffer;
    unique_ptr<VulkanBuffer>        mBboxIndexBuffer;
    bool                            mHasBboxPipeline = false;
    void                CreateBboxBuffers();
    void                CreateBboxDescriptors();
    void                UpdateBboxDescriptors();

};