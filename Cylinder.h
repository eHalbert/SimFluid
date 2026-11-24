#pragma once

// Wireframe vertical cylinder (visualization of the obstacle of the SPH solver).
// Same pipeline as Bbox (line list, model_bbox shaders, sMatrixColorUBO).

// 1. PROJECT
#include "Structures.h"
#include "Utility.h"
#include "Camera.h"
#include "vulkan_device.hpp"
#include "vulkan_ubo.hpp"

// 2. LIB
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
using namespace glm;

// 3. WIN
#include <array>
#include <memory>
#include <vector>
using namespace std;


class Cylinder
{
public:
    Cylinder(shared_ptr<VulkanDevice>& vulkanDevice);
    ~Cylinder();

    // Vertical cylinder: axis at (centerXZ.x, y, centerXZ.y), from yMin to yMax. Call before CreateCylinderPipeline().
    void SetCylinder(const vec2& centerXZ, float radius, float yMin, float yMax);

    void CreateCylinderPipeline(VkRenderPass renderPass, VkExtent2D swapChainExtent);
    void Render(VkCommandBuffer cmd, int iCurrentFrame, Camera& camera, mat4 model, vec4 color);
    void RecreatePipelines(VkRenderPass renderPass, VkExtent2D swapChainExtent);

    bool bVisible = true;

private:
    struct sCylinderVertex {
        vec3 pos;

        static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription b{};
            b.binding   = 0;
            b.stride    = sizeof(sCylinderVertex);
            b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            return b;
        }
        static array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
            array<VkVertexInputAttributeDescription, 1> a{};
            a[0].binding  = 0;
            a[0].location = 0;
            a[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            a[0].offset   = 0;
            return a;
        }
    };

    void CreateBuffers();
    void CreateDescriptors();
    void UpdateDescriptors();

    // Geometry
    vec2    mCenter{ 0.0f };
    float   mRadius = 0.5f;
    float   mYMin   = 0.0f;
    float   mYMax   = 1.0f;
    int     mSegments    = 48;      // points per circle
    float   mRingSpacing = 0.5f;    // distance between two horizontal rings (m), bottom and top always drawn
    int     mVerticalEvery = 6;     // one vertical line every N points of a circle

    uint32_t mIndexCount = 0;

    shared_ptr<VulkanDevice>    mVulkanDevice;
    sPipeline_x                 mPipeline;
    unique_ptr<VulkanBuffer>    mVertexBuffer;
    unique_ptr<VulkanBuffer>    mIndexBuffer;
    bool                        mHasPipeline = false;
};
