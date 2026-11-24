#pragma once
// SphSolver.h - GPU DFSPH (Divergence-Free SPH) fluid solver confined in an axis-aligned box.
// Requires C++17, GLM and Vulkan 1.0+ (compute shaders, no extension needed). Implementation in SphSolver.cpp.
//
// OVERVIEW
//   Method    : Bender & Koschier, "Divergence-Free Smoothed Particle Hydrodynamics" (2015),
//               GPU port of the CPU solver (same parameters, same walls).
//   Walls     : static boundary particles (Akinci et al.), uploaded once to the GPU.
//               They also build the optional vertical cylinder obstacle (SetCylinder()).
//   Wave maker: optional piston on the x-min wall (SetWaveMaker()). Its boundary particles are generated
//               on the fly in sph_neighbors.comp, with the piston velocity used by the solvers.
//   Density   : preconditioned conjugate gradient (default, SphParams::usePcg) or Jacobi iterations.
//               The PCG works on the pressure s = p / rho^2, with a Jacobi preconditioner, a Tikhonov
//               regularization, a warm start from the previous sub-step and a free-surface mask.
//   Divergence: Jacobi iterations (same shaders as the Jacobi density solver, other mode).
//
// PIPELINE OF A SUB-STEP
//   All of it is recorded in the command buffer, nothing is read back to decide.
//   1. forces_acc / forces_apply   gravity + viscosity (explicit) + force cursor
//   2. density solver              PCG: prep -> dir -> op -> scalar, then (dir, op, scalar, update, scalar, dirupdate)
//                                  per iteration, then dir (final pressure) -> finish (velocity correction)
//                                  Jacobi: error -> check, then (correct, error, check) per iteration
//   3. advect                      positions + safety clamp inside the box, around the cylinder and the piston
//   4. prepare                     grid clear -> count -> scan (3 levels) -> scatter -> copy -> neighbor lists
//                                  (particles and the PCG first guess are physically reordered by cell)
//   5. divergence solver           removes the residual divergence
//   6. finalize                    velocity sanitizing + max speed reduction (used for the CFL of the next frames)
//
// DATA LAYOUT
//   One 32-byte Particle { vec4 posDensity; vec4 velInvS; } per particle in ONE storage buffer.
//   The same buffer is the vertex buffer of the point renderer (stride 32): no CPU->GPU copy per frame.
//   The pressure (warm[]) and the neighbor counts (meta[]) are also bound as vertex buffers, so the
//   renderer can color the particles by density, speed, height, pressure or neighbor count (ColorMode).
//   Neighbor lists are cached in a neighbor-major layout (index k * numFluid + i) for coalesced reads.
//
// TIME STEPPING
//   - Update(dt) splits the frame time in k equal sub-steps. The sub-step comes from the CFL condition
//     (dt = 0.4 * h / vmax, clamped to [minDt, maxDt]), with vmax measured on the GPU a few frames ago
//     (safety margin included). Velocities are clamped to SphParams::maxSpeed anyway.
//   - The number of sub-steps is capped by a GPU budget: maxFrameMs / (measured ms per sub-step, from
//     timestamp queries), and by maxSubSteps. Above the cap the simulation runs slower than real time.
//   - Frames longer than maxUpdateDt are clipped (the lost time is counted in ClippedSeconds()).
//   - The solvers stop on the GPU (indirect dispatch with 0 groups once converged). The loops are recorded
//     with the maximum number of iterations: converged iterations only cost an empty dispatch.
//
// FRAME USAGE (the application owns the command buffer and the synchronization)
//   - Build() then InitVulkan(...) once (or give MaxParticles to InitVulkan if Build() comes later).
//   - Update(cmd, nFrame, dt) once per frame, OUTSIDE any render pass, before vkCmdBeginRenderPass.
//     It records the whole simulation (compute) and the barrier compute -> vertex input.
//   - Render(cmd, nFrame, camera, ...) inside the render pass.
//   - Compute and graphics must be submitted to the SAME queue (the barriers rely on submission order).
//   - nFrame is the frame-in-flight index. The application must have waited the fence of this slot before
//     calling Update(): the results of the previous use of this slot (max speed, iteration counts, GPU
//     times) are read back at the beginning of Update().
//   - After a swapchain / render pass recreation, call RecreatePipelines().
//
// NAMING
//   Private members: m_ prefix + Hungarian type prefix: b = bool, n = int / uint32_t, v = std::vector,
//   a = C array. size_t, float, double, glm types and objects have no type prefix.
//   The members of the structs are not renamed.

// 1. PROJECT
#include "Utility.h"
#include "Camera.h"
#include "vulkan_device.hpp"
#include "vulkan_pipeline.hpp"
#include "vulkan_ubo.hpp"
#include "SphSurface.h"

// 2. LIB
#include <vulkan/vulkan.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
using namespace glm;

// 3. WIN
#define NOMINMAX
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;

extern uint32_t g_FramesInFlight;

// Same memory layout on the CPU (Build) and in the shaders (std430 struct { vec4 pd; vec4 vi; }).
struct Particle {
    vec4 posDensity;    // xyz = position, w = density rho_i
    vec4 velInvS;       // xyz = velocity, w = DFSPH factor 1 / (|sum m grad W|^2 + sum |m grad W|^2)
};
static_assert(sizeof(Particle) == 32, "Particle must match the std430 layout of the shaders");

struct SphParams {
    float   spacing             = 0.05f;    // initial distance between particles (m)
    float   smoothingRadius     = 0.1f;     // h = kernel support radius, must be about 2 * spacing (m)
    float   restDensity         = 1000.0f;  // rho0 (kg/m^3)
    float   viscosity           = 0.5f;     // dynamic viscosity mu (Pa.s), kinematic nu = mu / restDensity
    vec3    gravity             = vec3(0.0f, -9.81f, 0.0f);
    float   wallDamping         = 0.3f;     // safety clamp only (fraction of normal velocity kept if a particle escapes)
    float   maxSpeed            = 10.0f;    // velocity clamp for stability (m/s)

    // Adaptive time step: dt = 0.4 * h / vmax, clamped to [minDt, maxDt]
    float   cfl                 = 0.6f;
    float   maxDt               = 0.01f;    // upper bound of a sub-step (s)
    float   minDt               = 0.0002f;  // lower bound of a sub-step (s)
    double  maxFrameMs          = 33.0;     // keep: GPU budget of one Update()
    float   maxUpdateDt         = 0.05f;    // keep: must stay > 1/30 s, otherwise frames at 30 fps get clipped
    int     maxSubSteps         = 64;       // keep: 33 ms / ~2.7 ms (CFL) = ~13 sub-steps needed, so 64 never bites

    // Pressure solvers. The loops are RECORDED with maxIterations iterations (empty dispatches once converged), so keep it moderate on the GPU.
    int     minIterations       = 1;        // minimum number of iterations of each solver
    int     maxIterations       = 6;        // cap of iterations of each solver
    int     maxDivIterations    = 1;        // divergence solver cap: it only has to remove residual divergence
    float   maxDensityError     = 0.1f;     // density solver stops when the mean density error < this % of restDensity
    float   maxDivError         = 0.1f;     // divergence solver stops when the mean |d rho / dt| * dt < this % of restDensity

    bool    usePcg              = false;    // true: preconditioned conjugate gradient for the density solver (false: Jacobi)
    int     pcgMaxIterations    = 4;        // cap of PCG iterations (each recorded iteration = 6 dispatches)
    float   pcgWarm             = 1.0f;     // fraction of the previous pressure used as first guess (0 = start from zero)
    bool    pcgSurfaceMask      = true;     // p = 0 on free-surface particles (incomplete neighborhood)
    float   pcgSurfaceRatio     = 0.85f;    // free surface under this fraction of the neighbors of a full lattice
    float   pcgRegularization   = 0.3f;

    float   surfaceTension      = 0.0f;     // gamma of the Akinci 2013 cohesion + curvature model (0 = off)
};

class SphSolver {
public:
    // ---- Types ----

    // Initial shape of the fluid (see Build())
    struct SpawnShape {
        enum Type { Box, Sphere } type = Box;
        vec3 a{ 0.0f }, b{ 0.0f };      // Box: min / max corners. Sphere: a = center, b.x = radius
        vec3 velocity{ 0.0f };          // initial velocity of the particles of this shape

        static SpawnShape MakeBox(const vec3& lo, const vec3& hi, const vec3& v = vec3(0.0f))
        {
            SpawnShape s; s.type = Box; s.a = lo; s.b = hi; s.velocity = v; return s;
        }
        static SpawnShape MakeSphere(const vec3& c, float r, const vec3& v = vec3(0.0f))
        {
            SpawnShape s; s.type = Sphere; s.a = c; s.b = vec3(r, 0.0f, 0.0f); s.velocity = v; return s;
        }
    };

    struct StageTimes {
        int iterPressure   = 0;     // iterations of the density solver (last sub-step, measured on the GPU)
        int iterDivergence = 0;     // iterations of the divergence solver (last sub-step)
    };

    enum class ColorMode : int { Density, Speed, VelocityY, Height, Pressure, Neighbors, Count };

    // ---- Construction ----
    SphSolver();
    SphSolver(const vec3& boxMin, const vec3& boxMax, const SphParams& params = SphParams());
    ~SphSolver();

    // Not copyable (owns Vulkan resources), but movable: allows g_SphSolver = SphSolver(...).
    // Do the assignment BEFORE InitVulkan(): a moved-from object no longer owns any Vulkan resource.
    SphSolver(const SphSolver&)            = delete;
    SphSolver& operator=(const SphSolver&) = delete;
    SphSolver(SphSolver&&)                 = default;
    SphSolver& operator=(SphSolver&&)      = default;

    // ---- Scene ----
    void Build(const vector<SpawnShape>& shapes, float jitter = 0.01f);
    void Build(const vec3& spawnMin, const vec3& spawnMax, float jitter = 0.01f);
    void SetCylinder(const vec2& centerXZ, float radius);                  // vertical cylinder, radius <= 0 disables it
    void SetWaveMaker(bool bEnabled, float stroke, float period);          // piston on the x-min wall, safe to call at any time

    // ---- Vulkan ----
    void InitVulkan(shared_ptr<VulkanDevice> vulkanDevice, VkRenderPass renderPass, VkExtent2D extent, size_t maxParticles = 0);
    void RecreatePipelines(VkRenderPass renderPass, VkExtent2D newExtent);
    void Update(VkCommandBuffer cmd, uint32_t nFrame, float dt);            // outside a render pass, once per frame
    void Render(VkCommandBuffer commandBuffer, uint32_t nFrame, Camera& camera, float pointSize, float exposure);

    // ---- Interaction ----
    void SetCursorForce(const vec3& p, float radius, const vec3& v, float stiffness);
    void ClearCursorForce() { m_Cursor = ForceCursor(); }

    // ---- Settings (safe to change between two frames) ----
    void SetUsePcg(bool bOn)            { m_SphParams.usePcg = bOn; }
    void SetPointScale(float scale)     { m_PointScale = std::max(scale, 0.05f); }
    void SetColorMode(int nMode);                                           // also resets the range
    void SetColorRange(float lo, float hi) { m_ColorMin = lo; m_ColorMax = hi; }
    void ResetColorRange();
    void  SetSurfaceTension(float gamma) { m_SphParams.surfaceTension = std::max(gamma, 0.0f); }
    float SurfaceTension() const { return m_SphParams.surfaceTension; }
    void SetGravityEnabled(bool bEnabled);      // waits for the GPU (the uniform block is rewritten)
    bool GravityEnabled() const { return m_bGravity; }

    static const char* const*   ColorModeNames();
    static int                  ColorModeCount() { return (int)ColorMode::Count; }

	// ---- Surface ----
    enum class RenderMode : int { Particles, Surface, Count };
    static const char* const* RenderModeNames();
    static int  RenderModeCount() { return (int)RenderMode::Count; }
    void        SetRenderMode(int nMode) { m_nRenderMode = std::clamp(nMode, 0, RenderModeCount() - 1); }
    int         RenderModeIndex() const { return m_nRenderMode; }
    SphSurface::Settings& SurfaceSettings() { return m_SurfaceSettings; }
    void        RecordSurface(VkCommandBuffer cmd, uint32_t nFrame, Camera& camera);    // outside a render pass, after Update()

    // ---- Queries ----
    size_t              Count()             const { return m_Count; }
    vec3                BoxMin()            const { return m_BoxMin; }
    vec3                BoxMax()            const { return m_BoxMax; }
    const SphParams&    Params()            const { return m_SphParams; }
    bool                UsePcg()            const { return m_SphParams.usePcg; }
    float               PointScale()        const { return m_PointScale; }
    int                 ColorModeIndex()    const { return m_nColorMode; }
    float               ColorMin()          const { return m_ColorMin; }
    float               ColorMax()          const { return m_ColorMax; }
    bool                WaveMakerEnabled()  const { return m_Wave.enabled; }
    float               PistonPosition()    const { float x, v; PistonEval(m_PistonPhase, m_StrokeCur, 0.0f, 0.0f, x, v); return x; }   // for the display
    float               MaxCompression()    const { return m_MaxCompression; }     // fraction (0.0001 = 0.01 %), last sub-step, a few frames old
    
    // Monitoring
    double              AvgUpdateMs()       const { return m_AvgUpdateMs; }     // CPU time spent recording Update()
    double              TotalGpuMs()        const { return m_TotGpuMs; }
    double              MsPerStep()         const { return m_MsPerStep; }       // smoothed GPU cost of one sub-step (ms)
    const double*       StageMs()           const { return m_StageMs; }
    int                 LastSubSteps()      const { return m_nLastSubSteps; }   // sub-steps recorded by the last Update()
    int                 SubStepLimit()      const { return m_nSubStepLimit; }
    float               LastDt()            const { return m_LastDt; }          // last sub-step (s)
    float               MaxSpeed()          const { return m_VmaxGpu; }         // measured on the GPU, a few frames old
    double              RequestedSeconds()  const { return m_TotRequested; }
    double              SimulatedSeconds()  const { return m_TotSimulated; }
    double              ClippedSeconds()    const { return m_ClippedSeconds; }
    bool                CapHit()            const { return m_bCapHitLast; }
    StageTimes          LastStageTimes()    const { return m_Stage; }

private:
    // ---- Private types ----

    // Push constants (32 bytes) of the main compute pipeline layout.
    struct PushData {
        float    dt         = 0.0f;
        float    tol        = 0.0f;     // convergence threshold on the mean error
        uint32_t divergence = 0;        // 0 = density solver, 1 = divergence solver
        uint32_t minIt      = 0;
        uint32_t maxIt      = 0;
        uint32_t first      = 0;        // Jacobi: 1 = first check. PCG: bit 0 = surface mask enabled
        uint32_t mode       = 0;        // PCG kernels: kernel specific mode
        float    warm       = 0.0f;     // PCG: fraction of the previous pressure used as first guess
    };
    static_assert(sizeof(PushData) == 32, "PushData must match the push constant block");

    // Force cursor, pushed at offset sizeof(PushData) (read by sph_forces_acc.comp)
    struct ForceCursor {
        vec4 posRadius = vec4(0.0f);    // xyz = interaction point, w = radius (0 = disabled)
        vec4 velStiff  = vec4(0.0f);    // xyz = cursor velocity, w = stiffness (1/s)
    };
    static_assert(sizeof(PushData) + sizeof(ForceCursor) <= 128, "128 bytes of push constants are guaranteed by Vulkan");

    struct WaveMaker { bool enabled = false; float stroke = 0.3f; float period = 1.5f; };

    struct GpuBuffer {
        VkBuffer        buffer = VK_NULL_HANDLE;
        VkDeviceMemory  memory = VK_NULL_HANDLE;
        VkDeviceSize    size   = 0;
        void*           mapped = nullptr;
    };

    struct PendingCopy {
        GpuBuffer       staging;
        GpuBuffer*      dst;
        VkDeviceSize    size;
    };

    struct RetiredBuffer {
        GpuBuffer       buffer;
        uint64_t        stamp;
    };

    enum Kernel {
        K_GRID_CLEAR, K_GRID_COUNT, K_GRID_SCATTER, K_GRID_COPY, K_NEIGHBORS,
        K_FORCES_ACC, K_FORCES_APPLY,
        K_SOLVER_ERROR, K_SOLVER_CORRECT, K_SOLVER_CHECK,
        K_ADVECT, K_FINALIZE,
        K_SCAN_LOCAL, K_SCAN_ADD,
        K_PCG_PREP, K_PCG_DIR, K_PCG_OP, K_PCG_SCALAR, K_PCG_UPDATE, K_PCG_DIRUPDATE, K_PCG_FINISH, K_SURF_NORMAL,
        K_COUNT
    };

    static constexpr float kStrokeRate = 0.3f;     // m/s, maximum speed of the stroke change of the piston

    static const char*  KernelPath(int nKernel);
    static uint32_t     Groups(uint32_t nCount) { return (nCount + 255u) / 256u; }
    static bool         Inside(const SpawnShape& shape, const vec3& p);
    vec3                EffectiveGravity() const { return m_bGravity ? m_SphParams.gravity : vec3(0.0f); }

    // ---- Low level Vulkan helpers ----
    uint32_t    FindMemoryType(uint32_t nTypeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) const;
    GpuBuffer   CreateGpuBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool bHostVisible, bool bPreferDeviceLocal = false);
    void        DestroyGpuBuffer(GpuBuffer& buffer);
    void        StageUpload(GpuBuffer& dst, const void* pData, VkDeviceSize size);
    void        RecordPendingUploads(VkCommandBuffer cmd);
    void        RetireStaging();

    // ---- Compute objects: layouts, pipelines, descriptors, buffers ----
    void        CreateComputeObjects();
    void        DestroyComputeObjects();
    void        AllocateParticleBuffers();
    void        DestroyParticleBuffers();
    void        AllocateNeighborBuffers();
    void        DestroyNeighborBuffers();
    void        AllocateGridBuffers();
    void        DestroyGridBuffers();
    void        UpdateComputeDescriptors();
    void        UploadConfig();
    void        FlushInitialParticles();
    void        ReadBack(uint32_t nFrame);

    // ---- Graphics pipeline (points) ----
    void        CreatePipeline(VkRenderPass renderPass, VkExtent2D extent);
    void        CreateDescriptors();
    void        UpdateDescriptors();
    void        DestroyVulkan();

    // ---- Setup (CPU side) ----
    float       KernelW(float r) const;
    float       KernelDWdr(float r) const;
    void        Reconfigure();
    void        BuildBoundary(int nLayers);

    // ---- Wave maker ----
    void        PistonEval(double phase, float stroke, float w, float strokeRate, float& x, float& v) const;
    void        PushPiston(VkCommandBuffer cmd, double phase, float stroke, float w, float strokeRate);

    // ---- Command recording ----
    float       ComputeCflDt(float total) const;
    void        Barrier(VkCommandBuffer cmd);
    void        Dispatch(VkCommandBuffer cmd, Kernel kernel, uint32_t nGroups);
    void        DispatchIndirect(VkCommandBuffer cmd, Kernel kernel);
    void        Push(VkCommandBuffer cmd, const PushData& pc);
    void        RecordScanLevel(VkCommandBuffer cmd, Kernel kernel, int nSetIndex, uint32_t nElements, uint32_t nGroups);
    void        RecordPrepare(VkCommandBuffer cmd);
    void        RecordPcg(VkCommandBuffer cmd, float dt);
    void        RecordSolver(VkCommandBuffer cmd, float dt, bool bDivergence);
    void        Stamp(VkCommandBuffer cmd, int nBase, uint32_t nIndex);
    void        RecordStep(VkCommandBuffer cmd, float dt, int nStampBase);


private:
    // Container
    vec3                m_BoxMin, m_BoxMax;
    SphParams           m_SphParams;

    // Obstacles and wave maker
    vec2                m_CylCenter{ 0.0f };
    float               m_CylRadius     = 0.0f;
    WaveMaker           m_Wave;
    double              m_PistonPhase   = 0.0;      // rad, accumulated: continuous when the period changes
    float               m_StrokeCur     = 0.3f;     // stroke currently applied (follows m_Wave.stroke at a limited rate)
    ForceCursor         m_Cursor;

    // Particles (CPU side: only the initial state; the simulation lives on the GPU)
    vector<Particle>    m_vInit;
    size_t              m_Count             = 0;
    bool                m_bParticlesDirty   = false;
    bool                m_bNeedPrepare      = true;     // grid / neighbors / density must be rebuilt before the next step

    // Grid sizes
    ivec3               m_GridDim{ 1 };
    size_t              m_NumCells          = 1;
    float               m_InvCell           = 1.0f;
    uint32_t            m_nScanN0 = 1, m_nScanN1 = 1, m_nScanN2 = 1;    // elements of the 3 levels of the prefix sum

    // Boundary particles (static) and their grid
    vec3                m_BoundaryLo{ 0.0f };
    ivec3               m_BoundaryGridDim{ 1 };
    size_t              m_NumBoundaryCells  = 1;
    vector<vec4>        m_vBoundaryPos;
    vector<uint32_t>    m_vBoundaryCellStart;
    bool                m_bBoundaryDirty    = true;

    uint32_t            m_nMaxNeighbors     = 64;
    uint32_t            m_nMinNbrDiv        = 18;
    uint32_t            m_nExp              = 26;       // neighbors of a particle in a full lattice

    // Derived constants
    float               m_H2                = 0.0f;
    float               m_InvH              = 0.0f;
    float               m_KernelK           = 0.0f;     // cubic spline: W
    float               m_KernelL           = 0.0f;     // cubic spline: dW/dq
    float               m_W0                = 0.0f;     // W(0)
    float               m_Mass              = 0.0f;     // fluid and boundary particle mass
    float               m_SFull             = 0.0f;     // DFSPH denominator of a particle with a full lattice neighborhood

    // Timing / monitoring
    double              m_AvgUpdateMs       = 0.0;
    int                 m_nLastSubSteps     = 0;
    int                 m_nSubStepLimit     = 64;
    float               m_LastSimTime       = 0.0f;
    float               m_LastDt            = 0.0f;
    float               m_VmaxGpu           = 0.0f;
    StageTimes          m_Stage;
    float               m_TimestampPeriod   = 0.0f;
    bool                m_bStatsWritten     = false;
    vector<uint8_t>     m_vStatsCopied;
    vector<uint8_t>     m_vQueryRecorded;
    vector<uint8_t>     m_vStageRecorded;
    vector<int>         m_vRecSteps;                    // sub-steps recorded in each frame slot (to convert GPU ms into ms per sub-step)
    double              m_ClippedSeconds    = 0.0;
    double              m_TotRequested      = 0.0;
    double              m_TotSimulated      = 0.0;
    double              m_TotGpuMs          = 0.0;
    double              m_MsPerStep         = 0.0;
    double              m_StageMs[6]        = {};       // forces, density solver, advect, prepare, divergence solver, finalize
    bool                m_bCapHitWindow     = false;    // the sub-step cap limited the simulation during the current update
    bool                m_bCapHitLast       = false;    // same, for the previous update

    // Staging
    uint64_t                m_UpdateCounter = 0;
    vector<PendingCopy>     m_vPending;
    vector<RetiredBuffer>   m_vRetired;

    // Vulkan: compute
    shared_ptr<VulkanDevice>    m_VulkanDevice;
    VkDescriptorSetLayout       m_MainSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout       m_ScanSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout            m_CompLayout    = VK_NULL_HANDLE;
    VkDescriptorPool            m_CompPool      = VK_NULL_HANDLE;
    VkDescriptorSet             m_Sets[4]       = {};   // 0 = main, 1..3 = scan levels
    VkPipeline                  m_Kernels[K_COUNT] = {};
    VkQueryPool                 m_QueryPool     = VK_NULL_HANDLE;

    GpuBuffer           m_Params, m_Particles, m_Scratch, m_Meta, m_Acc, m_SolS, m_Partials;
    GpuBuffer           m_NbrIdx, m_NbrGrad;
    GpuBuffer           m_CellStart, m_Sums1, m_Sums2, m_Sums3, m_BPos, m_BCellStart;
    GpuBuffer           m_Ctrl, m_Stats, m_Warm, m_Pcg;
    vector<GpuBuffer>   m_vStatsHost;
    size_t              m_GpuCapacity       = 0;        // number of particles the GPU buffers can hold

    // Vulkan: graphics
    sPipeline_x         m_Pipeline;
    bool                m_bVisible          = true;
    int                 m_nColorMode        = 0;
    float               m_ColorMin          = 500.0f;
    float               m_ColorMax          = 1500.0f;
    uint32_t            m_nViewportH        = 1080;
    float               m_PointScale        = 1.0f;

	// Vulkan: surface
    unique_ptr<SphSurface>  m_pSurface;
    SphSurface::Settings    m_SurfaceSettings;
    int                     m_nRenderMode   = 0;

    bool                m_bGravity          = true;
    float               m_MaxCompression    = 0.0f;
};
