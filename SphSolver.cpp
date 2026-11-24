// SphSolver.cpp - implementation of the GPU DFSPH solver (see SphSolver.h for the overview).

#include "SphSolver.h"

namespace {

using Clock = chrono::steady_clock;

constexpr uint32_t QPF = 9;     // timestamps per frame slot: 0 = start, 1 = end, 2..8 = stage boundaries of the last sub-step

// Uniform block "Params" (std140) of every compute shader. Keep it identical in the .comp files.
struct GpuParams {
    vec4     boxMin;
    vec4     boxMax;
    vec4     bLo;
    vec4     gravity;       // xyz = gravity, w = maxSpeed
    ivec4    gridDim;       // xyz = fluid grid size, w = number of cells
    ivec4    bGridDim;      // xyz = boundary grid size
    uint32_t numFluid;
    uint32_t numBoundary;
    uint32_t maxNeighbors;
    uint32_t minNbrDiv;
    float    h, h2, invH, kernelK;
    float    kernelL, W0, mass, sFull;
    float    invCell, restDensity, viscCoef, viscEps;
    float    wallDamping, surfaceNbrThreshold, pcgRegularization, pad;
    vec4     cyl;           // xy = axis (x, z), z = radius, w = enabled
};
static_assert(sizeof(GpuParams) == 192, "GpuParams must match the std140 block of the shaders");

// Graphics UBO (same layout in sph.vert / sph.frag)
struct sSphUBO {
    mat4    view;
    mat4    proj;
    float   pointSize;
    float   restDensity;
    float   exposure;
    int     colorMode;
    float   rangeMin;
    float   rangeMax;
    float   viewportHeight;
    float   particleDiameter;
};

// Bindings of descriptor set 0 (same numbers in the shaders)
enum Binding {
    B_PARAMS = 0, B_PARTICLES = 1, B_SCRATCH = 2, B_META = 3, B_CELLSTART = 4, B_NBRIDX = 5, B_NBRGRAD = 6,
    B_BPOS = 7, B_BCELLSTART = 8, B_ACC = 9, B_SOLS = 10, B_PARTIALS = 11, B_CTRL = 12, B_STATS = 13, B_WARM = 14, B_PCG = 15,
    B_COUNT = 16
};

} // namespace


// ============================================================================
// Construction
// ============================================================================

SphSolver::SphSolver() : SphSolver(vec3(0.0f), vec3(1.0f), SphParams())
{
}
SphSolver::SphSolver(const vec3& boxMin, const vec3& boxMax, const SphParams& params)
    : m_BoxMin(boxMin), m_BoxMax(boxMax), m_SphParams(params)
{
    Reconfigure();
}
SphSolver::~SphSolver()
{
    DestroyVulkan();
}

// ============================================================================
// Scene
// ============================================================================

bool SphSolver::Inside(const SpawnShape& shape, const vec3& p)
{
    if (shape.type == SpawnShape::Sphere)
        return glm::length(p - shape.a) <= shape.b.x;
    return glm::all(glm::greaterThanEqual(p, shape.a)) && glm::all(glm::lessThanEqual(p, shape.b));
}
void SphSolver::Build(const vector<SpawnShape>& shapes, float jitter)
{
    // Fill the shapes with particles on the lattice of the boundary particles (anchored on the box corner,
    // so every wall is half a spacing away from the first particles). Where shapes overlap, the first one wins.
    // jitter = fraction of spacing applied as random offset (0 = perfect lattice).

    const float s = m_SphParams.spacing;

    m_vInit.clear();
    m_Count         = 0;
    m_bNeedPrepare  = true;
    m_VmaxGpu       = 0.0f;
    m_PistonPhase   = 0.0;                  // the piston restarts at rest
    m_StrokeCur     = m_Wave.stroke;

    const ivec3 latticeDim = ivec3(glm::floor((m_BoxMax - m_BoxMin) / s));
    mt19937 rng(12345u);
    uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int z = 0; z < latticeDim.z; ++z)
    {
        for (int y = 0; y < latticeDim.y; ++y)
        {
            for (int x = 0; x < latticeDim.x; ++x)
            {
                const vec3 c = m_BoxMin + (vec3(x, y, z) + 0.5f) * s;

                const SpawnShape* pHit = nullptr;
                for (const SpawnShape& shape : shapes)
                {
                    if (Inside(shape, c)) { pHit = &shape; break; }
                }
                if (!pHit)
                    continue;

                // Keep the spawn out of the cylinder
                if (m_CylRadius > 0.0f && glm::length(vec2(c.x, c.z) - m_CylCenter) < m_CylRadius + 0.25f * s)
                    continue;

                vec3 pos = c;
                if (jitter > 0.0f)
                    pos += vec3(dist(rng), dist(rng), dist(rng)) * (jitter * s);
                pos = glm::clamp(pos, m_BoxMin, m_BoxMax);

                Particle p{};
                p.posDensity = vec4(pos, m_SphParams.restDensity);
                p.velInvS    = vec4(pHit->velocity, 0.0f);
                m_vInit.push_back(p);
            }
        }
    }

    m_Count = m_vInit.size();
    if (m_Count == 0)
    {
        m_bParticlesDirty = false;
        if (m_VulkanDevice)
        {
            vkDeviceWaitIdle(m_VulkanDevice->device);
            UploadConfig();
        }
        return;
    }
    m_bParticlesDirty = true;
    if (m_VulkanDevice)
        FlushInitialParticles();
}
void SphSolver::Build(const vec3& spawnMin, const vec3& spawnMax, float jitter)
{
    Build(vector<SpawnShape>{ SpawnShape::MakeBox(spawnMin, spawnMax) }, jitter);
}
void SphSolver::SetCylinder(const vec2& centerXZ, float radius)
{
    // Vertical cylinder from floor to ceiling (radius <= 0 disables it). Call BEFORE Build().
    if (centerXZ == m_CylCenter && radius == m_CylRadius)
        return;

    m_CylCenter = centerXZ;
    m_CylRadius = radius;
    Reconfigure();      // rebuilds the boundary particles
}
void SphSolver::SetWaveMaker(bool bEnabled, float stroke, float period)
{
    // Safe to call at any time: the period changes the phase speed (the phase stays continuous) and the stroke
    // moves toward its target at a limited rate (see RecordStep), so the piston never jumps.
    m_Wave.stroke = std::max(stroke, 0.0f);
    m_Wave.period = std::max(period, 0.2f);
    if (bEnabled != m_Wave.enabled)
    {
        m_Wave.enabled = bEnabled;
        Reconfigure();                  // the static wall behind the piston is added / removed (see BuildBoundary)
    }
}
void SphSolver::PistonEval(double phase, float stroke, float w, float strokeRate, float& x, float& v) const
{
    // x = x0 + (S/2) (1 - cos(phase)): starts at rest, never goes behind the wall.
    // v = d x / dt = (S/2) w sin(phase) + (dS/dt / 2) (1 - cos(phase))
    x = m_BoxMin.x;
    v = 0.0f;
    if (!m_Wave.enabled)
        return;
    const float c = (float)std::cos(phase);
    x = m_BoxMin.x + 0.5f * stroke * (1.0f - c);
    v = 0.5f * stroke * w * (float)std::sin(phase) + 0.5f * strokeRate * (1.0f - c);
}
void SphSolver::PushPiston(VkCommandBuffer cmd, double phase, float stroke, float w, float strokeRate)
{
    // Push constants, bytes 64..79: x = face position, y = velocity, z = enabled
    float x, v;
    PistonEval(phase, stroke, w, strokeRate, x, v);
    const vec4 p(x, v, m_Wave.enabled ? 1.0f : 0.0f, m_SphParams.surfaceTension);
    vkCmdPushConstants(cmd, m_CompLayout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(PushData) + sizeof(ForceCursor), sizeof(vec4), &p);
}
void SphSolver::SetCursorForce(const vec3& p, float radius, const vec3& v, float stiffness)
{
    m_Cursor.posRadius = vec4(p, radius);
    m_Cursor.velStiff  = vec4(v, stiffness);
}
void SphSolver::SetGravityEnabled(bool bEnabled)
{
    if (bEnabled == m_bGravity)
        return;

    m_bGravity = bEnabled;
    if (m_VulkanDevice)
    {
        vkDeviceWaitIdle(m_VulkanDevice->device);   // the GPU reads the uniform block while simulating
        UploadConfig();
    }
}

// ============================================================================
// Colors
// ============================================================================

const char* const* SphSolver::ColorModeNames()
{
    static const char* const names[] = { "Density", "Speed", "Vertical velocity", "Height", "Pressure (PCG)", "Neighbor count" };
    return names;
}
void SphSolver::SetColorMode(int nMode)
{
    m_nColorMode = std::clamp(nMode, 0, ColorModeCount() - 1);
    ResetColorRange();
}
void SphSolver::ResetColorRange()
{
    // Sensible default range for each quantity
    const float g = glm::length(m_SphParams.gravity);
    switch ((ColorMode)m_nColorMode)
    {
    case ColorMode::Density:   m_ColorMin = 0.5f * m_SphParams.restDensity; m_ColorMax = 1.5f * m_SphParams.restDensity; break;
    case ColorMode::Speed:     m_ColorMin = 0.0f; m_ColorMax = m_SphParams.maxSpeed; break;
    case ColorMode::VelocityY: m_ColorMin = -0.5f * m_SphParams.maxSpeed; m_ColorMax = 0.5f * m_SphParams.maxSpeed; break;
    case ColorMode::Height:    m_ColorMin = m_BoxMin.y; m_ColorMax = m_BoxMax.y; break;
    case ColorMode::Pressure:  m_ColorMin = 0.0f; m_ColorMax = m_SphParams.restDensity * g * (m_BoxMax.y - m_BoxMin.y); break;  // hydrostatic
    case ColorMode::Neighbors: m_ColorMin = 0.0f; m_ColorMax = (float)m_nExp; break;
    default: break;
    }
}

// ============================================================================
// Frame: simulation
// ============================================================================

void SphSolver::Update(VkCommandBuffer cmd, uint32_t nFrame, float dt)
{
    // Record the simulation of 'dt' seconds (pass the REAL wall clock time elapsed since the previous call).
    // Must be called outside a render pass, once per frame. Internally split in CFL sub-steps.

    const auto t0 = Clock::now();
    m_nLastSubSteps = 0;
    m_LastSimTime   = 0.0f;

    if (!m_VulkanDevice || m_CompLayout == VK_NULL_HANDLE)
        return;

    ++m_UpdateCounter;
    RetireStaging();
    ReadBack(nFrame);                   // reads the results of the previous use of this frame slot

    // Reset AFTER the read-back (ReadBack needs the values of the previous use)
    if (nFrame < m_vRecSteps.size())        m_vRecSteps[nFrame] = 0;
    if (nFrame < m_vStageRecorded.size())   m_vStageRecorded[nFrame] = 0;

    // Previous frame(s): vertex reads of the particle buffer and compute writes must be finished
    {
        VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &b, 0, nullptr, 0, nullptr);
    }

    RecordPendingUploads(cmd);

    // Timestamp 0 = start of this Update (QPF queries per frame slot)
    if (m_QueryPool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd, m_QueryPool, QPF * nFrame, QPF);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_QueryPool, QPF * nFrame);
    }

    if (m_Count > 0 && dt > 0.0f)
    {
        const float total = std::min(dt, m_SphParams.maxUpdateDt);      // ignore hitches (breakpoint, window drag)
        m_ClippedSeconds += (double)(dt - total);                       // simulated time dropped by the clip

        const float dtCfl = ComputeCflDt(total);
        int   nSteps = std::max(1, (int)std::ceil(total / dtCfl));
        float h;
        if (nSteps > m_nSubStepLimit)
        {
            nSteps = m_nSubStepLimit;   // too slow: simulate less than 'total' (slower than real time, intended)
            h = dtCfl;
        }
        else
            h = total / (float)nSteps;  // equal split, never a tiny last step

        if (nSteps >= m_nSubStepLimit && (int)std::ceil(total / dtCfl) > m_nSubStepLimit)
            m_bCapHitWindow = true;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompLayout, 0, 1, &m_Sets[0], 0, nullptr);
        for (int s = 0; s < nSteps; ++s)
        {
            // Only the last sub-step of the frame writes the per-stage timestamps (queries 2..8)
            const bool bLast = (s == nSteps - 1) && m_QueryPool != VK_NULL_HANDLE;
            RecordStep(cmd, h, bLast ? (int)(QPF * nFrame + 2) : -1);
        }

        m_nLastSubSteps = nSteps;
        m_LastSimTime   = (float)nSteps * h;
        m_LastDt        = h;
        m_bStatsWritten = true;

        if (nFrame < m_vRecSteps.size())
            m_vRecSteps[nFrame] = nSteps;                               // used by ReadBack to get the ms per sub-step
        if (m_QueryPool != VK_NULL_HANDLE && nFrame < m_vStageRecorded.size())
            m_vStageRecorded[nFrame] = 1;
    }

    // Copy the statistics (max speed, iteration counts) to a host visible buffer of this frame slot
    if (nFrame < m_vStatsHost.size())
    {
        VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);

        VkBufferCopy r{ 0, 0, 4 * sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_Stats.buffer, m_vStatsHost[nFrame].buffer, 1, &r);

        VkMemoryBarrier hb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        m_vStatsCopied[nFrame] = m_bStatsWritten ? 1 : 0;
    }

    // Compute writes -> vertex fetch of Render()
    {
        VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // Timestamp 1 = end of this Update
    if (m_QueryPool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_QueryPool, QPF * nFrame + 1);
        m_vQueryRecorded[nFrame] = 1;
    }

    // Cumulative requested / simulated time (read by the display, per-second differences)
    if (dt > 0.0f)
    {
        m_TotRequested += dt;
        m_TotSimulated += m_LastSimTime;
    }

    m_bCapHitLast   = m_bCapHitWindow;
    m_bCapHitWindow = false;

    const auto t1 = Clock::now();
    const double lastUpdateMs = chrono::duration<double, milli>(t1 - t0).count();     // CPU time to RECORD the commands
    m_AvgUpdateMs = (m_AvgUpdateMs == 0.0) ? lastUpdateMs : m_AvgUpdateMs * 0.95 + lastUpdateMs * 0.05;
}
void SphSolver::ReadBack(uint32_t nFrame)
{
    // Results of the previous use of this frame slot (the application has waited its fence).

    if (nFrame >= m_vStatsHost.size())
        return;

    if (m_vStatsCopied[nFrame] && m_vStatsHost[nFrame].mapped)
    {
        uint32_t aStats[4];
        memcpy(aStats, m_vStatsHost[nFrame].mapped, sizeof(aStats));
        float v;
        memcpy(&v, &aStats[0], sizeof(float));
        if (std::isfinite(v))
            m_VmaxGpu = std::min(v, m_SphParams.maxSpeed);
        m_Stage.iterPressure   = (int)aStats[1];
        m_Stage.iterDivergence = (int)aStats[2];
        float compression;
        memcpy(&compression, &aStats[3], sizeof(float));
        if (std::isfinite(compression))
            m_MaxCompression = compression;
    }

    if (m_QueryPool != VK_NULL_HANDLE && m_vQueryRecorded[nFrame])
    {
        uint64_t aResults[2 * QPF] = {};
        
        // aResults[2*i] = value, aResults[2*i+1] = availability of query i. VK_NOT_READY is expected here: the stage queries stay unavailable in frames without sub-steps.
        vkGetQueryPoolResults(m_VulkanDevice->device, m_QueryPool, QPF * nFrame, QPF, sizeof(aResults), aResults,
            2 * sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (aResults[1] != 0 && aResults[3] != 0 && aResults[2] >= aResults[0])
        {
            const double ms = (double)(aResults[2] - aResults[0]) * (double)m_TimestampPeriod * 1e-6;
            m_TotGpuMs += ms;

            if (m_vRecSteps[nFrame] > 0)
            {
                const double perStep = ms / (double)m_vRecSteps[nFrame];
                m_MsPerStep = (m_MsPerStep == 0.0) ? perStep : m_MsPerStep * 0.8 + perStep * 0.2;
                m_nSubStepLimit = std::clamp((int)std::floor(m_SphParams.maxFrameMs / m_MsPerStep), 1, std::max(1, m_SphParams.maxSubSteps));
            }
        }

        if (m_vStageRecorded[nFrame])
        {
            bool bOk = true;
            for (uint32_t i = 2; i < QPF; ++i)
                bOk = bOk && (aResults[2 * i + 1] != 0);
            if (bOk)
            {
                for (uint32_t s = 0; s < 6; ++s)
                {
                    if (aResults[2 * (3 + s)] < aResults[2 * (2 + s)])
                        continue;
                    const double t = (double)(aResults[2 * (3 + s)] - aResults[2 * (2 + s)]) * (double)m_TimestampPeriod * 1e-6;
                    m_StageMs[s] = (m_StageMs[s] == 0.0) ? t : m_StageMs[s] * 0.8 + t * 0.2;
                }
            }
        }
    }
}

// ============================================================================
// Vulkan: setup and rendering
// ============================================================================

void SphSolver::InitVulkan(shared_ptr<VulkanDevice> vulkanDevice, VkRenderPass renderPass, VkExtent2D extent, size_t maxParticles)
{
    // Call once, after Build() (or give maxParticles if Build() is called later with a bigger count).

    m_VulkanDevice = vulkanDevice;
    m_GpuCapacity  = std::max(maxParticles, m_Count);
    if (m_GpuCapacity == 0)
        m_GpuCapacity = 1;      // a Vulkan buffer cannot have a size of 0

    CreateComputeObjects();
    AllocateParticleBuffers();
    AllocateNeighborBuffers();
    AllocateGridBuffers();
    UpdateComputeDescriptors();
    UploadConfig();
    FlushInitialParticles();

    m_nSubStepLimit = std::max(1, m_SphParams.maxSubSteps);
    m_bNeedPrepare  = true;

    CreatePipeline(renderPass, extent);
    
    m_pSurface = make_unique<SphSurface>(m_VulkanDevice);
    m_pSurface->Create(renderPass, extent);
}
void SphSolver::DestroyVulkan()
{
    if (!m_VulkanDevice)
        return;

    vkDeviceWaitIdle(m_VulkanDevice->device);
    m_Pipeline.destroy(m_VulkanDevice->device);
    DestroyComputeObjects();
    m_VulkanDevice.reset();

    m_pSurface.reset();
}
void SphSolver::RecreatePipelines(VkRenderPass renderPass, VkExtent2D newExtent)
{
    m_Pipeline.destroy(m_VulkanDevice->device);
    CreatePipeline(renderPass, newExtent);

    if (m_pSurface)
        m_pSurface->Create(renderPass, newExtent);
}
void SphSolver::Render(VkCommandBuffer commandBuffer, uint32_t nFrame, Camera& camera, float pointSize, float exposure)
{
    (void)pointSize;    // the size comes from m_PointScale (SetPointScale); the parameter is kept for the existing call sites

    if (!m_bVisible || m_Count == 0 || m_Particles.buffer == VK_NULL_HANDLE)
        return;

    if (m_nRenderMode == (int)RenderMode::Surface && m_pSurface)
    {
        m_pSurface->RecordComposite(commandBuffer, nFrame);
        return;
    }

    // Update UBO
    sSphUBO ubo{};
    ubo.view             = camera.GetView();
    ubo.proj             = camera.GetProjection();
    ubo.pointSize        = m_PointScale;
    ubo.restDensity      = m_SphParams.restDensity;
    ubo.exposure         = exposure;
    ubo.colorMode        = m_nColorMode;
    ubo.rangeMin         = m_ColorMin;
    ubo.rangeMax         = m_ColorMax;
    ubo.viewportHeight   = (float)m_nViewportH;
    ubo.particleDiameter = m_SphParams.spacing;
    memcpy(m_Pipeline.ubo[nFrame]->data, &ubo, sizeof(sSphUBO));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline.pipeline);

    // Descriptor set
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline.pipelineLayout, 0, 1, &m_Pipeline.descSet[nFrame], 0, nullptr);

    // Vertex buffers: the simulation buffer itself (one instance = one point), the pressure and the neighbor counts
    VkBuffer     aVertexBuffers[3] = { m_Particles.buffer, m_Warm.buffer, m_Meta.buffer };
    VkDeviceSize aOffsets[3]       = { 0, 0, 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 3, aVertexBuffers, aOffsets);

    vkCmdDraw(commandBuffer, 1, (uint32_t)m_Count, 0, 0);
}

// ============================================================================
// Low level Vulkan helpers
// ============================================================================

uint32_t SphSolver::FindMemoryType(uint32_t nTypeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) const
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_VulkanDevice->physicalDevice, &mp);
    const VkMemoryPropertyFlags aTries[2] = { want, fallback };
    for (VkMemoryPropertyFlags flags : aTries)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((nTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
    return UINT32_MAX;
}
SphSolver::GpuBuffer SphSolver::CreateGpuBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool bHostVisible, bool bPreferDeviceLocal)
{
    // bHostVisible = false: DEVICE_LOCAL memory. bHostVisible = true: mapped, coherent memory
    // (bPreferDeviceLocal asks for the DEVICE_LOCAL | HOST_VISIBLE heap (BAR) first, for small data read by the GPU).

    GpuBuffer b;
    b.size = std::max<VkDeviceSize>(size, 16);

    VkBufferCreateInfo ci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size        = b.size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(m_VulkanDevice->device, &ci, nullptr, &b.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_VulkanDevice->device, b.buffer, &req);

    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags want, fallback;
    if (!bHostVisible)
    {
        want     = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        fallback = 0;
    }
    else
    {
        want     = bPreferDeviceLocal ? (hv | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : hv;
        fallback = hv;
    }

    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, want, fallback);
    vkAllocateMemory(m_VulkanDevice->device, &ai, nullptr, &b.memory);
    vkBindBufferMemory(m_VulkanDevice->device, b.buffer, b.memory, 0);

    if (bHostVisible)
        vkMapMemory(m_VulkanDevice->device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped);
    return b;
}
void SphSolver::DestroyGpuBuffer(GpuBuffer& buffer)
{
    if (!m_VulkanDevice)
        return;
    if (buffer.mapped)
        vkUnmapMemory(m_VulkanDevice->device, buffer.memory);
    if (buffer.buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(m_VulkanDevice->device, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE)
        vkFreeMemory(m_VulkanDevice->device, buffer.memory, nullptr);
    buffer = GpuBuffer();
}
void SphSolver::StageUpload(GpuBuffer& dst, const void* pData, VkDeviceSize size)
{
    // Copy to a staging buffer now; the copy to 'dst' is recorded by the next Update().
    if (size == 0)
        return;
    GpuBuffer staging = CreateGpuBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    memcpy(staging.mapped, pData, (size_t)size);
    m_vPending.push_back({ staging, &dst, size });
}
void SphSolver::RecordPendingUploads(VkCommandBuffer cmd)
{
    if (m_vPending.empty())
        return;

    for (PendingCopy& p : m_vPending)
    {
        if (p.dst->buffer != VK_NULL_HANDLE)
        {
            VkBufferCopy r{ 0, 0, std::min(p.size, p.dst->size) };
            vkCmdCopyBuffer(cmd, p.staging.buffer, p.dst->buffer, 1, &r);
        }
        m_vRetired.push_back({ p.staging, m_UpdateCounter });
    }
    m_vPending.clear();

    VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}
void SphSolver::RetireStaging()
{
    // A staging buffer is safe to free once the frame slot that used it has been recycled.
    for (size_t i = 0; i < m_vRetired.size(); )
    {
        if (m_UpdateCounter > m_vRetired[i].stamp + g_FramesInFlight)
        {
            DestroyGpuBuffer(m_vRetired[i].buffer);
            m_vRetired[i] = m_vRetired.back();
            m_vRetired.pop_back();
        }
        else
            ++i;
    }
}

// ============================================================================
// Compute objects: layouts, pipelines, descriptors, buffers
// ============================================================================

const char* SphSolver::KernelPath(int nKernel)
{
    static const char* aPaths[K_COUNT] = {
        "Resources/Shaders/Sph/sph_grid_clear.comp",
        "Resources/Shaders/Sph/sph_grid_count.comp",
        "Resources/Shaders/Sph/sph_grid_scatter.comp",
        "Resources/Shaders/Sph/sph_grid_copy.comp",
        "Resources/Shaders/Sph/sph_neighbors.comp",
        "Resources/Shaders/Sph/sph_forces_acc.comp",
        "Resources/Shaders/Sph/sph_forces_apply.comp",
        "Resources/Shaders/Sph/sph_solver_error.comp",
        "Resources/Shaders/Sph/sph_solver_correct.comp",
        "Resources/Shaders/Sph/sph_solver_check.comp",
        "Resources/Shaders/Sph/sph_advect.comp",
        "Resources/Shaders/Sph/sph_finalize.comp",
        "Resources/Shaders/Sph/sph_scan_local.comp",
        "Resources/Shaders/Sph/sph_scan_add.comp",
        "Resources/Shaders/Sph/sph_pcg_prep.comp",
        "Resources/Shaders/Sph/sph_pcg_dir.comp",
        "Resources/Shaders/Sph/sph_pcg_op.comp",
        "Resources/Shaders/Sph/sph_pcg_scalar.comp",
        "Resources/Shaders/Sph/sph_pcg_update.comp",
        "Resources/Shaders/Sph/sph_pcg_dirupdate.comp",
        "Resources/Shaders/Sph/sph_pcg_finish.comp",
        "Resources/Shaders/Sph/sph_surface_normal.comp",
    };
    return aPaths[nKernel];
}
void SphSolver::CreateComputeObjects()
{
    VkDevice dev = m_VulkanDevice->device;

    // Set 0: parameters + every buffer of the simulation
    array<VkDescriptorSetLayoutBinding, B_COUNT> aMainBindings{};
    for (uint32_t i = 0; i < B_COUNT; ++i)
    {
        aMainBindings[i].binding         = i;
        aMainBindings[i].descriptorType  = (i == B_PARAMS) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        aMainBindings[i].descriptorCount = 1;
        aMainBindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = (uint32_t)aMainBindings.size();
    li.pBindings    = aMainBindings.data();
    vkCreateDescriptorSetLayout(dev, &li, nullptr, &m_MainSetLayout);

    // Set 1: prefix sum (data, sums). Three sets are used, one per level of the scan.
    array<VkDescriptorSetLayoutBinding, 2> aScanBindings{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        aScanBindings[i].binding         = i;
        aScanBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        aScanBindings[i].descriptorCount = 1;
        aScanBindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    li.bindingCount = 2;
    li.pBindings    = aScanBindings.data();
    vkCreateDescriptorSetLayout(dev, &li, nullptr, &m_ScanSetLayout);

    // Pipeline layout shared by all the kernels: PushData + force cursor + piston
    VkDescriptorSetLayout aSetLayouts[2] = { m_MainSetLayout, m_ScanSetLayout };
    VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushData) + sizeof(ForceCursor) + sizeof(vec4) };
    VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl.setLayoutCount         = 2;
    pl.pSetLayouts            = aSetLayouts;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(dev, &pl, nullptr, &m_CompLayout);

    // Pipelines
    for (int k = 0; k < K_COUNT; ++k)
    {
        auto vCode = CompileShaderRuntime(KernelPath(k));
        VkShaderModule mod = CreateShaderModule(dev, vCode);

        VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main" };
        ci.layout = m_CompLayout;
        vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &m_Kernels[k]);
        vkDestroyShaderModule(dev, mod, nullptr);
    }

    // Descriptor pool: 1 main set + 3 scan sets
    VkDescriptorPoolSize aSizes[2] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (B_COUNT - 1) + 3 * 2 },
    };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 2;
    pi.pPoolSizes    = aSizes;
    pi.maxSets       = 4;
    vkCreateDescriptorPool(dev, &pi, nullptr, &m_CompPool);

    VkDescriptorSetLayout aLayouts[4] = { m_MainSetLayout, m_ScanSetLayout, m_ScanSetLayout, m_ScanSetLayout };
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool     = m_CompPool;
    ai.descriptorSetCount = 4;
    ai.pSetLayouts        = aLayouts;
    vkAllocateDescriptorSets(dev, &ai, m_Sets);

    // Fixed size buffers
    m_Params = CreateGpuBuffer(sizeof(GpuParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, true);
    m_Ctrl   = CreateGpuBuffer(8 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
    m_Stats  = CreateGpuBuffer(4 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);

    m_vStatsHost.resize(g_FramesInFlight);
    for (GpuBuffer& b : m_vStatsHost)
        b = CreateGpuBuffer(4 * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    m_vStatsCopied.assign(g_FramesInFlight, 0);
    m_vQueryRecorded.assign(g_FramesInFlight, 0);
    m_vStageRecorded.assign(g_FramesInFlight, 0);
    m_vRecSteps.assign(g_FramesInFlight, 0);

    // GPU timing (QPF timestamps per frame in flight)
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_VulkanDevice->physicalDevice, &props);
    m_TimestampPeriod = props.limits.timestampPeriod;
    if (m_TimestampPeriod > 0.0f)
    {
        VkQueryPoolCreateInfo qi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = QPF * g_FramesInFlight;
        vkCreateQueryPool(dev, &qi, nullptr, &m_QueryPool);
    }
}
void SphSolver::DestroyComputeObjects()
{
    VkDevice dev = m_VulkanDevice->device;

    for (PendingCopy& p : m_vPending)
        DestroyGpuBuffer(p.staging);
    m_vPending.clear();
    for (RetiredBuffer& r : m_vRetired)
        DestroyGpuBuffer(r.buffer);
    m_vRetired.clear();

    DestroyParticleBuffers();
    DestroyNeighborBuffers();
    DestroyGridBuffers();
    DestroyGpuBuffer(m_Params);
    DestroyGpuBuffer(m_Ctrl);
    DestroyGpuBuffer(m_Stats);

    for (GpuBuffer& b : m_vStatsHost)
        DestroyGpuBuffer(b);
    m_vStatsHost.clear();

    if (m_QueryPool != VK_NULL_HANDLE)
        vkDestroyQueryPool(dev, m_QueryPool, nullptr);
    m_QueryPool = VK_NULL_HANDLE;

    for (int k = 0; k < K_COUNT; ++k)
    {
        if (m_Kernels[k] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, m_Kernels[k], nullptr);
        m_Kernels[k] = VK_NULL_HANDLE;
    }
    if (m_CompLayout != VK_NULL_HANDLE)     vkDestroyPipelineLayout(dev, m_CompLayout, nullptr);
    if (m_MainSetLayout != VK_NULL_HANDLE)  vkDestroyDescriptorSetLayout(dev, m_MainSetLayout, nullptr);
    if (m_ScanSetLayout != VK_NULL_HANDLE)  vkDestroyDescriptorSetLayout(dev, m_ScanSetLayout, nullptr);
    if (m_CompPool != VK_NULL_HANDLE)       vkDestroyDescriptorPool(dev, m_CompPool, nullptr);
    m_CompLayout    = VK_NULL_HANDLE;
    m_MainSetLayout = m_ScanSetLayout = VK_NULL_HANDLE;
    m_CompPool      = VK_NULL_HANDLE;
}
void SphSolver::AllocateParticleBuffers()
{
    // Buffers whose size depends on the particle capacity only

    DestroyParticleBuffers();
    const VkBufferUsageFlags st = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkDeviceSize cap = (VkDeviceSize)m_GpuCapacity;

    m_Particles = CreateGpuBuffer(cap * sizeof(Particle), st | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    m_Scratch   = CreateGpuBuffer(cap * sizeof(Particle), st, false);
    m_Acc       = CreateGpuBuffer(cap * sizeof(vec4), st, false);
    m_SolS      = CreateGpuBuffer(cap * sizeof(float), st, false);
    m_Partials  = CreateGpuBuffer((VkDeviceSize)(Groups((uint32_t)m_GpuCapacity) + 1) * 4 * sizeof(float), st, false);   // 4 sums per work group
    m_Pcg       = CreateGpuBuffer(16 * sizeof(float) + cap * sizeof(vec2), st, false);
    m_Meta      = CreateGpuBuffer(cap * 4 * sizeof(uint32_t), st | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);
    m_Warm      = CreateGpuBuffer(cap * sizeof(float), st | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);
}
void SphSolver::DestroyParticleBuffers()
{
    DestroyGpuBuffer(m_Particles);
    DestroyGpuBuffer(m_Scratch);
    DestroyGpuBuffer(m_Meta);
    DestroyGpuBuffer(m_Acc);
    DestroyGpuBuffer(m_SolS);
    DestroyGpuBuffer(m_Partials);
    DestroyGpuBuffer(m_Warm);
    DestroyGpuBuffer(m_Pcg);
}
void SphSolver::AllocateNeighborBuffers()
{
    // Cached neighbor lists: depend on the capacity and on m_nMaxNeighbors

    DestroyNeighborBuffers();
    const VkDeviceSize cells = (VkDeviceSize)m_GpuCapacity * m_nMaxNeighbors;
    m_NbrIdx  = CreateGpuBuffer(cells * sizeof(uint32_t),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    m_NbrGrad = CreateGpuBuffer(cells * 3 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
}
void SphSolver::DestroyNeighborBuffers()
{
    DestroyGpuBuffer(m_NbrIdx);
    DestroyGpuBuffer(m_NbrGrad);
}
void SphSolver::AllocateGridBuffers()
{
    // Grid (cell starts + scan levels) and static boundary particles: depend on the box and on h

    DestroyGridBuffers();
    const VkBufferUsageFlags st = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    m_CellStart  = CreateGpuBuffer((VkDeviceSize)m_nScanN0 * sizeof(uint32_t), st, false);
    m_Sums1      = CreateGpuBuffer((VkDeviceSize)m_nScanN1 * sizeof(uint32_t), st, false);
    m_Sums2      = CreateGpuBuffer((VkDeviceSize)m_nScanN2 * sizeof(uint32_t), st, false);
    m_Sums3      = CreateGpuBuffer(sizeof(uint32_t), st, false);
    m_BPos       = CreateGpuBuffer((VkDeviceSize)m_vBoundaryPos.size() * sizeof(vec4), st | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    m_BCellStart = CreateGpuBuffer((VkDeviceSize)m_vBoundaryCellStart.size() * sizeof(uint32_t), st | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
}
void SphSolver::DestroyGridBuffers()
{
    DestroyGpuBuffer(m_CellStart);
    DestroyGpuBuffer(m_Sums1);
    DestroyGpuBuffer(m_Sums2);
    DestroyGpuBuffer(m_Sums3);
    DestroyGpuBuffer(m_BPos);
    DestroyGpuBuffer(m_BCellStart);
}
void SphSolver::UpdateComputeDescriptors()
{
    array<VkDescriptorBufferInfo, 32> aInfos{};
    vector<VkWriteDescriptorSet> vWrites;
    vWrites.reserve(32);
    uint32_t nInfos = 0;

    auto add = [&](VkDescriptorSet set, uint32_t nBinding, VkDescriptorType type, const GpuBuffer& b) {
        assert(b.buffer != VK_NULL_HANDLE && "descriptor written with a buffer that was never created");
        aInfos[nInfos] = { b.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet          = set;
        w.dstBinding      = nBinding;
        w.descriptorCount = 1;
        w.descriptorType  = type;
        w.pBufferInfo     = &aInfos[nInfos];
        vWrites.push_back(w);
        ++nInfos;
    };
    const VkDescriptorType ST = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    add(m_Sets[0], B_PARAMS,     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_Params);
    add(m_Sets[0], B_PARTICLES,  ST, m_Particles);
    add(m_Sets[0], B_SCRATCH,    ST, m_Scratch);
    add(m_Sets[0], B_META,       ST, m_Meta);
    add(m_Sets[0], B_CELLSTART,  ST, m_CellStart);
    add(m_Sets[0], B_NBRIDX,     ST, m_NbrIdx);
    add(m_Sets[0], B_NBRGRAD,    ST, m_NbrGrad);
    add(m_Sets[0], B_BPOS,       ST, m_BPos);
    add(m_Sets[0], B_BCELLSTART, ST, m_BCellStart);
    add(m_Sets[0], B_ACC,        ST, m_Acc);
    add(m_Sets[0], B_SOLS,       ST, m_SolS);
    add(m_Sets[0], B_PARTIALS,   ST, m_Partials);
    add(m_Sets[0], B_CTRL,       ST, m_Ctrl);
    add(m_Sets[0], B_STATS,      ST, m_Stats);
    add(m_Sets[0], B_WARM,       ST, m_Warm);
    add(m_Sets[0], B_PCG,        ST, m_Pcg);

    // Scan levels: (data, sums)
    add(m_Sets[1], 0, ST, m_CellStart); add(m_Sets[1], 1, ST, m_Sums1);
    add(m_Sets[2], 0, ST, m_Sums1);     add(m_Sets[2], 1, ST, m_Sums2);
    add(m_Sets[3], 0, ST, m_Sums2);     add(m_Sets[3], 1, ST, m_Sums3);

    vkUpdateDescriptorSets(m_VulkanDevice->device, (uint32_t)vWrites.size(), vWrites.data(), 0, nullptr);
}
void SphSolver::UploadConfig()
{
    // Parameters block (host visible, the device is idle when this is called) + boundary particles.

    GpuParams g{};
    const float nu        = m_SphParams.viscosity / m_SphParams.restDensity;
    g.boxMin              = vec4(m_BoxMin, 0.0f);
    g.boxMax              = vec4(m_BoxMax, 0.0f);
    g.bLo                 = vec4(m_BoundaryLo, 0.0f);
    g.gravity             = vec4(EffectiveGravity(), m_SphParams.maxSpeed);
    g.gridDim             = ivec4(m_GridDim, (int)m_NumCells);
    g.bGridDim            = ivec4(m_BoundaryGridDim, 0);
    g.numFluid            = (uint32_t)m_Count;
    g.numBoundary         = (uint32_t)m_vBoundaryPos.size();
    g.maxNeighbors        = m_nMaxNeighbors;
    g.minNbrDiv           = m_nMinNbrDiv;
    g.h                   = m_SphParams.smoothingRadius;
    g.h2                  = m_H2;
    g.invH                = m_InvH;
    g.kernelK             = m_KernelK;
    g.kernelL             = m_KernelL;
    g.W0                  = m_W0;
    g.mass                = m_Mass;
    g.sFull               = m_SFull;
    g.invCell             = m_InvCell;
    g.restDensity         = m_SphParams.restDensity;
    g.viscCoef            = 2.0f * (3.0f + 2.0f) * nu;     // 2 (d + 2) nu
    g.viscEps             = 0.01f * m_H2;
    g.wallDamping         = m_SphParams.wallDamping;
    g.surfaceNbrThreshold = (float)m_nExp * m_SphParams.pcgSurfaceRatio;
    g.pcgRegularization   = m_SphParams.pcgRegularization;
    g.cyl                 = vec4(m_CylCenter, m_CylRadius, m_CylRadius > 0.0f ? 1.0f : 0.0f);
    g.pad                 = m_SphParams.spacing;           // read by sph_neighbors.comp (piston lattice)

    memcpy(m_Params.mapped, &g, sizeof(GpuParams));

    if (m_bBoundaryDirty)
    {
        StageUpload(m_BPos,       m_vBoundaryPos.data(),       (VkDeviceSize)m_vBoundaryPos.size() * sizeof(vec4));
        StageUpload(m_BCellStart, m_vBoundaryCellStart.data(), (VkDeviceSize)m_vBoundaryCellStart.size() * sizeof(uint32_t));
        m_bBoundaryDirty = false;
    }
}
void SphSolver::FlushInitialParticles()
{
    // Upload the particles generated by Build() (the device is idle while the parameters block changes).

    if (!m_bParticlesDirty || !m_VulkanDevice)
        return;

    vkDeviceWaitIdle(m_VulkanDevice->device);
    if (m_Count > m_GpuCapacity)
    {
        m_GpuCapacity = m_Count;
        AllocateParticleBuffers();
        AllocateNeighborBuffers();
        UpdateComputeDescriptors();
    }
    UploadConfig();
    StageUpload(m_Particles, m_vInit.data(), (VkDeviceSize)m_Count * sizeof(Particle));

    const vector<float> vZeros(m_Count, 0.0f);
    StageUpload(m_Warm, vZeros.data(), (VkDeviceSize)m_Count * sizeof(float));

    m_vInit.clear();
    m_vInit.shrink_to_fit();
    m_bParticlesDirty = false;
}

// ============================================================================
// Graphics pipeline (points)
// ============================================================================

void SphSolver::CreatePipeline(VkRenderPass renderPass, VkExtent2D extent)
{
    m_nViewportH = extent.height;

    // 1. Shaders
    auto vVertCode = CompileShaderRuntime("Resources/Shaders/Sph/sph.vert");
    auto vFragCode = CompileShaderRuntime("Resources/Shaders/Sph/sph.frag");
    VkShaderModule vertModule = CreateShaderModule(m_VulkanDevice->device, vVertCode);
    VkShaderModule fragModule = CreateShaderModule(m_VulkanDevice->device, vFragCode);

    VkPipelineShaderStageCreateInfo aShaderStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main" },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main" }
    };

    // 2. Vertex input: binding 0 = Particle, binding 1 = warm (pressure s), binding 2 = meta (uvec4)
    array<VkVertexInputBindingDescription, 3> aBindings = {};
    aBindings[0] = { 0, sizeof(Particle),     VK_VERTEX_INPUT_RATE_INSTANCE };
    aBindings[1] = { 1, sizeof(float),        VK_VERTEX_INPUT_RATE_INSTANCE };
    aBindings[2] = { 2, 4 * sizeof(uint32_t), VK_VERTEX_INPUT_RATE_INSTANCE };

    array<VkVertexInputAttributeDescription, 5> aAttribs = {};
    aAttribs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0 };                   // position
    aAttribs[1] = { 1, 0, VK_FORMAT_R32_SFLOAT,          3 * sizeof(float) };   // density
    aAttribs[2] = { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 4 * sizeof(float) };   // velocity + invS
    aAttribs[3] = { 3, 1, VK_FORMAT_R32_SFLOAT,          0 };                   // pressure s
    aAttribs[4] = { 4, 2, VK_FORMAT_R32_UINT,            2 * sizeof(uint32_t) };// nbrTotal (meta.z)

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = (uint32_t)aBindings.size();
    vertexInputInfo.pVertexBindingDescriptions      = aBindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)aAttribs.size();
    vertexInputInfo.pVertexAttributeDescriptions    = aAttribs.data();

    // 3. Descriptor Set Layout (1 UBO)
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding         = 0;
    layoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &layoutBinding;
    vkCreateDescriptorSetLayout(m_VulkanDevice->device, &layoutInfo, nullptr, &m_Pipeline.descSetLayout);

    // 4. Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts    = &m_Pipeline.descSetLayout;
    vkCreatePipelineLayout(m_VulkanDevice->device, &pipelineLayoutInfo, nullptr, &m_Pipeline.pipelineLayout);

    // 5. Input assembly (POINTS)
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // 6. Viewport & scissors
    VkViewport viewport = { 0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f };
    VkRect2D scissor = { {0, 0}, extent };

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
    viewportState.scissorCount  = 1; viewportState.pScissors  = &scissor;

    // 7. Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable        = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_NONE;
    rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // 8. Multisampling
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = m_VulkanDevice->msaaSamples;

    // 9. Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // 10. Color blending (alpha)
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable         = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    colorBlendAttachment.colorWriteMask      = 0xF;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable   = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    // 11. Pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = aShaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.layout              = m_Pipeline.pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;
    vkCreateGraphicsPipelines(m_VulkanDevice->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline.pipeline);

    vkDestroyShaderModule(m_VulkanDevice->device, vertModule, nullptr);
    vkDestroyShaderModule(m_VulkanDevice->device, fragModule, nullptr);

    CreateDescriptors();
}
void SphSolver::CreateDescriptors()
{
    m_Pipeline.descSet.resize(g_FramesInFlight);
    m_Pipeline.ubo.resize(g_FramesInFlight);

    // 1. UBO Buffer (1 per frame in flight)
    for (size_t i = 0; i < g_FramesInFlight; i++)
        m_Pipeline.ubo[i] = make_unique<VulkanUBO>(m_VulkanDevice, sizeof(sSphUBO));

    // 2. Descriptor Pool
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * g_FramesInFlight };
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = g_FramesInFlight;
    vkCreateDescriptorPool(m_VulkanDevice->device, &poolInfo, nullptr, &m_Pipeline.descPool);

    // 3. Allocate 1 descriptor set per frame in flight
    vector<VkDescriptorSetLayout> vLayouts(g_FramesInFlight, m_Pipeline.descSetLayout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_Pipeline.descPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(vLayouts.size());
    allocInfo.pSetLayouts        = vLayouts.data();
    vkAllocateDescriptorSets(m_VulkanDevice->device, &allocInfo, m_Pipeline.descSet.data());

    UpdateDescriptors();
}
void SphSolver::UpdateDescriptors()
{
    for (size_t i = 0; i < g_FramesInFlight; ++i)
    {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = m_Pipeline.ubo[i]->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(sSphUBO);

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet          = m_Pipeline.descSet[i];
        descriptorWrite.dstBinding      = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.pBufferInfo     = &bufferInfo;

        vkUpdateDescriptorSets(m_VulkanDevice->device, 1, &descriptorWrite, 0, nullptr);
    }
}

// ============================================================================
// Setup (CPU side, identical to the CPU solver)
// ============================================================================

float SphSolver::KernelW(float r) const
{
    const float q = r * m_InvH;
    if (q >= 1.0f)
        return 0.0f;
    if (q <= 0.5f)
        return m_KernelK * (6.0f * (q * q * q - q * q) + 1.0f);
    const float t = 1.0f - q;
    return m_KernelK * 2.0f * t * t * t;
}
float SphSolver::KernelDWdr(float r) const
{
    // dW/dr (negative for 0 < r < h)
    const float q = r * m_InvH;
    if (q >= 1.0f)
        return 0.0f;
    if (q <= 0.5f)
        return m_KernelL * q * (3.0f * q - 2.0f) * m_InvH;
    const float t = 1.0f - q;
    return -m_KernelL * t * t * m_InvH;
}
void SphSolver::Reconfigure()
{
    const float h  = m_SphParams.smoothingRadius;
    const float s  = m_SphParams.spacing;
    const float pi = 3.14159265358979f;

    m_H2      = h * h;
    m_InvH    = 1.0f / h;
    m_KernelK = 8.0f  / (pi * h * h * h);
    m_KernelL = 48.0f / (pi * h * h * h);
    m_W0      = m_KernelK;

    // Calibrate the particle mass on a simple cubic lattice of step 'spacing': a particle surrounded by a full lattice must get exactly restDensity.
    const int nR = std::max(1, (int)std::ceil(h / s));
    double sumW = 0.0, sumG2 = 0.0;
    uint32_t nExp = 0;
    for (int k = -nR; k <= nR; ++k)
    {
        for (int j = -nR; j <= nR; ++j)
        {
            for (int i = -nR; i <= nR; ++i)
            {
                const float r = s * std::sqrt((float)(i * i + j * j + k * k));
                if (r >= h)
                    continue;
                sumW += KernelW(r);
                if (r > 0.0f)
                {
                    const double g = KernelDWdr(r);
                    sumG2 += g * g;
                    ++nExp;
                }
            }
        }
    }
    m_Mass         = (float)((double)m_SphParams.restDensity / sumW);
    m_nExp         = nExp;
    m_SFull        = (float)((double)m_Mass * m_Mass * sumG2);   // DFSPH denominator of a full lattice particle
    m_nMaxNeighbors = std::clamp(2u * nExp + 8u, 32u, 256u);
    m_nMinNbrDiv   = (uint32_t)(0.7f * (float)nExp);             // below: free surface, no divergence solve

    // Grid. Cell size >= h keeps the 3x3x3 neighborhood valid. It is enlarged if the grid would be too big (protects memory and the per-step clear of cellStart).
    const int   nLayers = std::max(1, (int)std::ceil(h / s));
    const float pad     = (float)(nLayers + 1) * s;
    m_BoundaryLo        = m_BoxMin - vec3(pad);
    const vec3  boundaryHi = m_BoxMax + vec3(pad);

    const vec3  boxSize      = glm::max(m_BoxMax - m_BoxMin, vec3(h));
    const vec3  boundarySize = glm::max(boundaryHi - m_BoundaryLo, vec3(h));
    const float maxCells = 2.0e6f;
    float cell = h;
    const float cells = (boundarySize.x / cell) * (boundarySize.y / cell) * (boundarySize.z / cell);
    if (cells > maxCells)
        cell *= std::cbrt(cells / maxCells);

    m_InvCell   = 1.0f / cell;
    m_GridDim   = glm::max(ivec3(glm::ceil(boxSize * m_InvCell)), ivec3(1));
    m_NumCells  = (size_t)m_GridDim.x * m_GridDim.y * m_GridDim.z;

    m_BoundaryGridDim  = glm::max(ivec3(glm::ceil(boundarySize * m_InvCell)), ivec3(1));
    m_NumBoundaryCells = (size_t)m_BoundaryGridDim.x * m_BoundaryGridDim.y * m_BoundaryGridDim.z;

    // Sizes of the three levels of the GPU prefix sum (256 elements per work group)
    m_nScanN0 = (uint32_t)m_NumCells + 1u;
    m_nScanN1 = Groups(m_nScanN0);
    m_nScanN2 = Groups(m_nScanN1);
    assert(m_nScanN2 <= 256 && "grid too large for the 3-level scan");

    BuildBoundary(nLayers);
    m_bBoundaryDirty = true;
    m_bNeedPrepare   = true;

    if (m_VulkanDevice)
    {
        vkDeviceWaitIdle(m_VulkanDevice->device);
        AllocateGridBuffers();
        AllocateNeighborBuffers();
        UpdateComputeDescriptors();
        UploadConfig();
    }
}
void SphSolver::BuildBoundary(int nLayers)
{
    // Static boundary particles on the lattice continuing the fluid lattice outside the box
    // (same spacing, same mass as a fluid particle). A particle is a boundary particle if at least
    // one of its coordinates is outside the box. Sorted by cell, uploaded once to the GPU.

    const float s = m_SphParams.spacing;

    struct Axis { vector<float> c; vector<uint8_t> outside; };
    Axis aAxis[3];
    for (int a = 0; a < 3; ++a)
    {
        for (int k = nLayers - 1; k >= 0; --k)
        {
            aAxis[a].c.push_back(m_BoxMin[a] - ((float)k + 0.5f) * s);
            aAxis[a].outside.push_back(1);
        }
        for (int i = 0; ; ++i)
        {
            const float c = m_BoxMin[a] + ((float)i + 0.5f) * s;
            if (c > m_BoxMax[a])
                break;
            aAxis[a].c.push_back(c);
            aAxis[a].outside.push_back(0);
        }
        for (int k = 0; k < nLayers; ++k)
        {
            aAxis[a].c.push_back(m_BoxMax[a] + ((float)k + 0.5f) * s);
            aAxis[a].outside.push_back(1);
        }
    }

    vector<vec3> vRaw;
    for (size_t z = 0; z < aAxis[2].c.size(); ++z)
    {
        for (size_t y = 0; y < aAxis[1].c.size(); ++y)
        {
            for (size_t x = 0; x < aAxis[0].c.size(); ++x)
            {
                if (!(aAxis[0].outside[x] || aAxis[1].outside[y] || aAxis[2].outside[z]))
                    continue;

                // Wave maker: the piston replaces the x-min wall (its particles are generated in sph_neighbors.comp).
                // The first 'nLayers' entries of aAxis[0] are the layers behind x-min; edges and corners are kept.
                if (m_Wave.enabled && x < (size_t)nLayers && !aAxis[1].outside[y] && !aAxis[2].outside[z])
                    continue;

                vRaw.push_back(vec3(aAxis[0].c[x], aAxis[1].c[y], aAxis[2].c[z]));
            }
        }
    }

    // Vertical cylinder: concentric rings of boundary particles, 'nLayers' rings deep below the surface.
    // Same vertical rows as the fluid lattice, so the wall behaves like the box walls.
    if (m_CylRadius > 0.0f)
    {
        const float pi = 3.14159265358979f;
        for (int k = 0; k < nLayers; ++k)
        {
            const float rk = m_CylRadius - ((float)k + 0.5f) * s;
            if (rk <= 0.0f)
                break;
            const int nPoints = std::max(1, (int)std::lround(2.0f * pi * rk / s));    // ~ s apart along the ring
            for (int i = 0; ; ++i)
            {
                const float y = m_BoxMin.y + ((float)i + 0.5f) * s;
                if (y > m_BoxMax.y)
                    break;
                for (int a = 0; a < nPoints; ++a)
                {
                    // Offset per row and per layer avoids perfectly aligned columns
                    const float ang = 2.0f * pi * ((float)a + 0.5f * (float)(i & 1) + 0.25f * (float)k) / (float)nPoints;
                    vRaw.push_back(vec3(m_CylCenter.x + rk * std::cos(ang), y, m_CylCenter.y + rk * std::sin(ang)));
                }
            }
        }
    }

    // Counting sort by cell (same layout as the fluid grid: x is the fastest index)
    m_vBoundaryCellStart.assign(m_NumBoundaryCells + 1, 0u);
    vector<uint32_t> vCellOf(vRaw.size());
    for (size_t i = 0; i < vRaw.size(); ++i)
    {
        const vec3  f = glm::clamp((vRaw[i] - m_BoundaryLo) * m_InvCell, vec3(0.0f), vec3(m_BoundaryGridDim - ivec3(1)));
        const ivec3 c = ivec3(f);
        vCellOf[i] = (uint32_t)(c.x + m_BoundaryGridDim.x * (c.y + m_BoundaryGridDim.y * c.z));
        ++m_vBoundaryCellStart[vCellOf[i] + 1];
    }
    std::partial_sum(m_vBoundaryCellStart.begin(), m_vBoundaryCellStart.end(), m_vBoundaryCellStart.begin());

    vector<uint32_t> vCursor(m_vBoundaryCellStart.begin(), m_vBoundaryCellStart.end() - 1);
    m_vBoundaryPos.assign(vRaw.size(), vec4(0.0f));
    for (size_t i = 0; i < vRaw.size(); ++i)
        m_vBoundaryPos[vCursor[vCellOf[i]]++] = vec4(vRaw[i], 0.0f);
}

// ============================================================================
// Command recording
// ============================================================================

float SphSolver::ComputeCflDt(float total) const
{
    // dt = 0.4 * h / vmax, clamped to [minDt, maxDt]. vmax comes from the GPU with a latency of a few frames: safety margin + velocity gained under gravity.

    const float g    = glm::length(EffectiveGravity());
    const float vmax = 1.1f * m_VmaxGpu + g * total;

    float dt = m_SphParams.maxDt;
    dt = std::min(dt, 0.4f * m_SphParams.smoothingRadius / std::max(vmax, 1e-6f));

    if (g > 1e-6f)
        dt = std::min(dt, 0.25f * std::sqrt(m_SphParams.spacing / g));

    const float lo = std::min(m_SphParams.minDt, m_SphParams.maxDt);
    return std::max(dt, std::max(lo, 1e-6f));
}
void SphSolver::Barrier(VkCommandBuffer cmd)
{
    // Compute -> compute (also makes the indirect dispatch arguments visible)
    VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0, 1, &b, 0, nullptr, 0, nullptr);
}
void SphSolver::Dispatch(VkCommandBuffer cmd, Kernel kernel, uint32_t nGroups)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Kernels[kernel]);
    vkCmdDispatch(cmd, nGroups, 1, 1);
}
void SphSolver::DispatchIndirect(VkCommandBuffer cmd, Kernel kernel)
{
    // The number of groups is written by sph_solver_check.comp / sph_pcg_scalar.comp (0 once the solver has converged)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Kernels[kernel]);
    vkCmdDispatchIndirect(cmd, m_Ctrl.buffer, 0);
}
void SphSolver::Push(VkCommandBuffer cmd, const PushData& pc)
{
    vkCmdPushConstants(cmd, m_CompLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushData), &pc);
}
void SphSolver::RecordScanLevel(VkCommandBuffer cmd, Kernel kernel, int nSetIndex, uint32_t nElements, uint32_t nGroups)
{
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompLayout, 1, 1, &m_Sets[nSetIndex], 0, nullptr);
    vkCmdPushConstants(cmd, m_CompLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &nElements);
    Dispatch(cmd, kernel, nGroups);
    Barrier(cmd);
}
void SphSolver::RecordPrepare(VkCommandBuffer cmd)
{
    // Counting sort of the particles by cell (physically reordered), then neighbor lists, density, DFSPH factor.
    const uint32_t nGroups = Groups((uint32_t)m_Count);

    Dispatch(cmd, K_GRID_CLEAR, m_nScanN1);     Barrier(cmd);       // zero the m_nScanN0 cell counters
    Dispatch(cmd, K_GRID_COUNT, nGroups);       Barrier(cmd);       // cell id + rank inside the cell (atomicAdd)

    // Exclusive prefix sum of the counters: 3 levels of 256 elements per work group
    RecordScanLevel(cmd, K_SCAN_LOCAL, 1, m_nScanN0, m_nScanN1);
    RecordScanLevel(cmd, K_SCAN_LOCAL, 2, m_nScanN1, m_nScanN2);
    RecordScanLevel(cmd, K_SCAN_LOCAL, 3, m_nScanN2, 1);
    RecordScanLevel(cmd, K_SCAN_ADD,   2, m_nScanN1, m_nScanN2);
    RecordScanLevel(cmd, K_SCAN_ADD,   1, m_nScanN0, m_nScanN1);

    Dispatch(cmd, K_GRID_SCATTER, nGroups);     Barrier(cmd);       // particles -> scratch (sorted)
    Dispatch(cmd, K_GRID_COPY, nGroups);        Barrier(cmd);       // scratch -> particles
    Dispatch(cmd, K_NEIGHBORS, nGroups);        Barrier(cmd);
}
void SphSolver::RecordPcg(VkCommandBuffer cmd, float dt)
{
    // Density solver by preconditioned conjugate gradient. The iterations are recorded, the shaders stop on the GPU
    // (indirect dispatch with 0 groups once the mean density error is under the tolerance).

    const uint32_t nGroups = Groups((uint32_t)m_Count);

    PushData pc;
    pc.dt    = dt;
    pc.tol   = m_SphParams.maxDensityError * 0.01f * m_SphParams.restDensity;
    pc.minIt = 0;
    pc.maxIt = (uint32_t)std::max(1, m_SphParams.pcgMaxIterations);
    pc.first = m_SphParams.pcgSurfaceMask ? 1u : 0u;
    pc.warm  = m_SphParams.pcgWarm;

    // First guess, residual, first direction
    pc.mode = 0;  Push(cmd, pc);
    Dispatch(cmd, K_PCG_PREP, nGroups);     Barrier(cmd);
    Dispatch(cmd, K_PCG_DIR, nGroups);      Barrier(cmd);    // D(x0)
    Dispatch(cmd, K_PCG_OP, nGroups);       Barrier(cmd);    // b, r0, p0
    Dispatch(cmd, K_PCG_SCALAR, 1);         Barrier(cmd);

    for (uint32_t it = 0; it < pc.maxIt; ++it)
    {
        pc.mode = 1;  Push(cmd, pc);
        DispatchIndirect(cmd, K_PCG_DIR);       Barrier(cmd);    // D(p)
        DispatchIndirect(cmd, K_PCG_OP);        Barrier(cmd);    // Ap
        Dispatch(cmd, K_PCG_SCALAR, 1);         Barrier(cmd);    // alpha
        DispatchIndirect(cmd, K_PCG_UPDATE);    Barrier(cmd);    // x, r
        pc.mode = 2;  Push(cmd, pc);
        Dispatch(cmd, K_PCG_SCALAR, 1);         Barrier(cmd);    // beta + convergence
        DispatchIndirect(cmd, K_PCG_DIRUPDATE); Barrier(cmd);    // p
    }

    // Velocity correction with the final pressure (clamped to >= 0)
    pc.mode = 2;  Push(cmd, pc);
    Dispatch(cmd, K_PCG_DIR, nGroups);      Barrier(cmd);
    Dispatch(cmd, K_PCG_FINISH, nGroups);   Barrier(cmd);
}
void SphSolver::RecordSolver(VkCommandBuffer cmd, float dt, bool bDivergence)
{
    // Jacobi iterations of the DFSPH solvers, recorded as: error, check, then maxIt x (correct, error, check).
    // sph_solver_check.comp decides on the GPU whether the following passes still run (indirect dispatch).

    if (!bDivergence && m_SphParams.usePcg)
    {
        RecordPcg(cmd, dt);
        return;
    }

    const float rho0 = m_SphParams.restDensity;

    PushData pc;
    pc.dt         = dt;
    pc.divergence = bDivergence ? 1u : 0u;
    pc.tol        = bDivergence ? m_SphParams.maxDivError * 0.01f * rho0 / dt
                                : m_SphParams.maxDensityError * 0.01f * rho0;
    pc.minIt      = (uint32_t)std::max(bDivergence ? 1 : 2, m_SphParams.minIterations);
    const int nCap = bDivergence ? std::min(m_SphParams.maxDivIterations, m_SphParams.maxIterations) : m_SphParams.maxIterations;
    pc.maxIt      = (uint32_t)std::max((int)pc.minIt, nCap);
    pc.first      = 1;

    const uint32_t nGroups = Groups((uint32_t)m_Count);

    Push(cmd, pc);
    Dispatch(cmd, K_SOLVER_ERROR, nGroups); Barrier(cmd);
    Dispatch(cmd, K_SOLVER_CHECK, 1);       Barrier(cmd);

    pc.first = 0;
    Push(cmd, pc);
    for (uint32_t it = 0; it < pc.maxIt; ++it)
    {
        DispatchIndirect(cmd, K_SOLVER_CORRECT);    Barrier(cmd);
        DispatchIndirect(cmd, K_SOLVER_ERROR);      Barrier(cmd);
        Dispatch(cmd, K_SOLVER_CHECK, 1);           Barrier(cmd);
    }
}
void SphSolver::Stamp(VkCommandBuffer cmd, int nBase, uint32_t nIndex)
{
    if (nBase >= 0)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_QueryPool, (uint32_t)nBase + nIndex);
}
void SphSolver::RecordStep(VkCommandBuffer cmd, float dt, int nStampBase)
{
    const uint32_t nGroups = Groups((uint32_t)m_Count);

    // Piston motion over this sub-step
    const float  w         = 6.283185307179586f / m_Wave.period;
    const float  dS        = std::clamp(m_Wave.stroke - m_StrokeCur, -kStrokeRate * dt, kStrokeRate * dt);
    const float  strokeRate = dS / dt;
    const double phaseEnd  = m_PistonPhase + (double)w * (double)dt;
    const float  strokeEnd = m_StrokeCur + dS;

    // Ranges above PushData: set before the first dispatch that may read them
    vkCmdPushConstants(cmd, m_CompLayout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(PushData), sizeof(ForceCursor), &m_Cursor);
    PushPiston(cmd, m_PistonPhase, m_StrokeCur, w, strokeRate);

    if (m_bNeedPrepare)
    {
        RecordPrepare(cmd);
        m_bNeedPrepare = false;
    }

    Stamp(cmd, nStampBase, 0);
    PushData pc;
    pc.dt = dt;
    Push(cmd, pc);
    if (m_SphParams.surfaceTension > 0.0f)
    {
        Dispatch(cmd, K_SURF_NORMAL, nGroups);  Barrier(cmd);   // normals for the surface tension
    }
    Dispatch(cmd, K_FORCES_ACC, nGroups);   Barrier(cmd);
    Dispatch(cmd, K_FORCES_APPLY, nGroups); Barrier(cmd);
    Stamp(cmd, nStampBase, 1);

    RecordSolver(cmd, dt, false);
    Stamp(cmd, nStampBase, 2);

    PushPiston(cmd, phaseEnd, strokeEnd, w, strokeRate);    // piston at the end of the sub-step: advect, prepare, divergence solver
    Push(cmd, pc);
    Dispatch(cmd, K_ADVECT, nGroups);       Barrier(cmd);
    Stamp(cmd, nStampBase, 3);

    RecordPrepare(cmd);
    Stamp(cmd, nStampBase, 4);

    RecordSolver(cmd, dt, true);
    Stamp(cmd, nStampBase, 5);

    Dispatch(cmd, K_FINALIZE, nGroups);     Barrier(cmd);
    Stamp(cmd, nStampBase, 6);

    // Commit the piston state (wrapped so that the phase keeps its precision over long runs)
    m_PistonPhase = std::fmod(phaseEnd, 6.283185307179586);
    m_StrokeCur   = strokeEnd;
}
const char* const* SphSolver::RenderModeNames()
{
    static const char* const names[] = { "Particles", "Surface" };
    return names;
}
void SphSolver::RecordSurface(VkCommandBuffer cmd, uint32_t nFrame, Camera& camera)
{
    if (m_nRenderMode != (int)RenderMode::Surface || !m_pSurface || m_Count == 0)
        return;
    m_pSurface->RecordOffscreen(cmd, nFrame, m_Particles.buffer, (uint32_t)m_Count, m_SphParams.spacing,
        camera.GetView(), camera.GetProjection(), m_SurfaceSettings);
}
