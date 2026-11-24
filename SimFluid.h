/* SimFluid by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

// 1. PROJECT
#include "Utility.h"
#include "Timer.h"
#include "Camera.h"
#include "Model.h"
#include "ScreenQuad.h"
#include "vulkan_debug.hpp"
#include "vulkan_buffer.hpp"
#include "vulkan_swapchain.hpp"
#include "Ini.h"
#include "Overlay.h"
#include "SphSolver.h"
#include "Bbox.h"
#include "Cylinder.h"

// 2. LIB

// glfw
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
// imgui
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
// pugixml
#include "pugixml/pugixml.hpp"
#ifdef _DEBUG
#pragma comment(lib, "pugixml/Debug/pugixml.lib")
#else
#pragma comment(lib, "pugixml/Release/pugixml.lib")
#endif

// 3. WIN
#include <vector>
#include <optional>
#include <array>
#include <stdexcept>
#include <set>
#include <Windows.h>
#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <chrono>  // For high_resolution_clock
#include <thread>  // For sleep_for
#include <algorithm>
using namespace std;

#ifdef _DEBUG
#define CONSOLE
#endif

#define SAFE_DELETE(ptr) if (ptr) { delete ptr; ptr = nullptr; }

#pragma warning(push)
#pragma warning(disable:6031)  // Ignore return value of function
#pragma warning(disable:4244)  // Ignore bad conversion warnings

sChrono						Chronos[10];

// WINDOW GLFW /////////////////////////////////////////
GLFWwindow                * g_hWindow                   = nullptr;      // InitWindow()
GLFWmonitor               * g_Monitor                   = nullptr;
bool                        g_IsFullscreen              = false;
uint32_t                    g_WindowX                   = 0;
uint32_t                    g_WindowY                   = 0;
uint32_t                    g_WindowW                   = 1920;
uint32_t                    g_WindowH                   = 1080;
uint32_t			        g_WindowW_2                 = 960;
uint32_t			        g_WindowH_2                 = 540;
bool						g_PendingFullscreen			= false; 
bool						g_IsRecreating				= false;
bool                        g_bPause                    = false;
wstring				        g_CaptureName;
float                       g_CaptureDisplayTime        = 0.0f;         // Remaining display time
const float                 g_CaptureMaxTime            = 3.0f;         // 3 seconds
eh::Timer			        g_TimerScene;
eh::Timer					g_Chrono;
bool						g_bChrono					= false;
int							g_ChronoStep				= -1;

// VULKAN //////////////////////////////////////////////
VkInstance                  g_Instance                  = nullptr;       // CreateInstance()
shared_ptr<VulkanDevice>    g_Device					= nullptr;
unique_ptr<VulkanSwapChain> g_SwapChain					= nullptr;
uint32_t                    g_FramesInFlight			= 2;

VkRenderPass				g_RenderPass;								// CreateRenderPass()
vector<VkFramebuffer>       g_vFramebuffersSwapChain;					// CreateFramebuffers()

unique_ptr<VulkanTexture>   g_ColorImage				= nullptr;		// CreateImages()
unique_ptr<VulkanTexture>   g_DepthImage				= nullptr;
unique_ptr<VulkanTexture>   g_ColorImageResolve			= nullptr;
unique_ptr<VulkanTexture>   g_DepthImageResolve			= nullptr;

vector<VkCommandBuffer>     g_vCommandBuffers;							// CreateCommandBuffers()
vector<VkSemaphore>         g_vImageAvailableSemaphores;				// CreateSyncObjects()
vector<VkSemaphore>         g_vRenderFinishedSemaphores;
vector<VkFence>             g_vImageFences;
uint32_t                    g_iCurrentFrame             = 0;			// Render()

// DEBUG VULKAN ////////////////////////////////////////
VkDebugUtilsMessengerEXT_T* g_DebugMessenger            = nullptr;

// FPS COUNTER /////////////////////////////////////////
double                      g_FpsTime                   = 0.0;
int                         g_FpsFrameCount             = 0;
int                         g_nFps                      = 0;
bool					    g_bTargetFps                = false;
const double                g_FpsTarget                 = 120.0;

// MODELS //////////////////////////////////////////////
unique_ptr<Model>			g_Axis                      = nullptr;
unique_ptr<Bbox>			g_Bbox						= nullptr;
unique_ptr<Cylinder>		g_Cylinder					= nullptr;
unique_ptr<Bbox>			g_Piston					= nullptr;

// FORCE CURSOR ////////////////////////////////////////
bool						g_ForceActive				= false;
bool						g_ForceHasPrev				= false;
vec3						g_ForcePrev					= vec3(0.0f);
vec3						g_ForceVel					= vec3(0.0f);
float						g_ForceRadius				= 1.0f;			// m
float						g_ForceStiffness			= 30.0f;		// 1/s (keep stiffness * dt < 1, so under ~80 with dt <= 0.01)
float						g_ForceDepth				= 0.0f;			// m, offset of the interaction plane along the view direction

// SCENARIOS ///////////////////////////////////////////
enum class eScenario { DamBreak, FallingDrop, WaveMaker, Count };
static const char*			g_ScenarioNames[]			= { "Dam break", "Falling drop", "Wave maker (piston)" };
int							g_Scenario					= 0;
float						g_WaveStroke				= 0.3f;			// m, total stroke of the piston
float						g_WavePeriod				= 1.5f;			// s

// VIEWS ///////////////////////////////////////////////
vec2						g_InitialPosition			= vec2(-2.94097114, 47.38162231);
Camera                      g_Camera;

// SKY /////////////////////////////////////////////////
unique_ptr<Sky>             g_Sky						= nullptr;

// SPH ////////////////////////////////////////////////
SphSolver					g_SphSolver;
string						g_SphSolverInfo;
static float				g_SphDt						= 0.0f;
static bool					g_PendingSphReset			= false;

// IMGUI ///////////////////////////////////////////////
VkDescriptorPool            g_ImGuiDescriptorPool		= nullptr;
ImFont					  * g_FontArial08				= nullptr;
ImFont					  * g_FontArial10				= nullptr;
ImFont					  * g_FontArial12				= nullptr;
ImFont					  * g_FontArial14				= nullptr;
ImFont					  * g_FontArial16				= nullptr;
ImFont					  * g_FontArial20				= nullptr;
ImFont					  * g_FontArial24				= nullptr;
ImFont					  * g_FontArial36				= nullptr;
ImFont					  * g_FontCaveat36				= nullptr;
ImFont					  * g_FontCaveat72				= nullptr;

bool						g_bShowShortcuts			= false;		// [ F1 ]
bool						g_bShowSceneWindow			= true;			// [ F2 ]
bool						g_bShowStatusBar			= true;			// [ F4 ]


ImU32						g_ColorBlack				= IM_COL32(0, 0, 0, 255);
ImU32						g_ColorGray192				= IM_COL32(192, 192, 192, 255);
ImU32						g_ColorWhite				= IM_COL32(255, 255, 255, 255);

ImU32						g_ColorRed					= IM_COL32(200, 0, 0, 255);
ImU32						g_ColorGreen				= IM_COL32(0, 200, 0, 255);
ImU32						g_ColorBlue					= IM_COL32(0, 0, 200, 255);
ImU32						g_ColorAmbre				= IM_COL32(255, 165, 0, 255);
ImU32						g_ColorCyan					= IM_COL32(0, 255, 255, 255);
ImU32						g_ColorMagenta				= IM_COL32(255, 0, 255, 255);
ImU32						g_ColorYellow				= IM_COL32(255, 255, 0, 255);

ImU32						g_ColorSimShip1				= IM_COL32(37, 87, 131, 255);
ImU32						g_ColorSimShip2				= IM_COL32(44, 129, 169, 255);
ImU32						g_ColorSimShip3				= IM_COL32(118, 150, 175, 255);
ImU32						g_ColorSimShip4				= IM_COL32(184, 199, 208, 255);


////////////////////////////////////////////////////////
vector<pair<string, string>> vShortcuts = {
	{ "C", "Orbital camera" },
	{ "F", "FPS camera" },
	{ "I", "Cycle interpolation function" },
	{ "F2", "Scene settings" },
	{ "F4", "Status bar" },
	{ "F8", "Window capture" },
	{ "F11", "Full screen" },
	{ "Space", "Pause / Resume" },
	{ "G", "Change theme" },
	{ "Esc", "Quit" },
};

void ApplyTheme();
void PartialRecreate();