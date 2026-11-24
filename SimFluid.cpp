#include "SimFluid.h"

int IMGUI_STYLE     = 4;


void SwitchToFullScreen()
{
    vkDeviceWaitIdle(g_Device->device);

    static uint32_t oldWindowX;
    static uint32_t oldWindowY;
    static uint32_t oldWindowW;
    static uint32_t oldWindowH;

    if (!g_IsFullscreen)    // -> Switch to fullscreen
    {
        // Storing the window position BEFORE fullscreen, glfwGetWindowPos returns the top-left corner of the client area
        int wx, wy;
        glfwGetWindowPos(g_hWindow, &wx, &wy);
        oldWindowX = wx;
        oldWindowY = wy;
        oldWindowW = g_WindowW;
        oldWindowH = g_WindowH;

        GLFWmonitor* targetMonitor = get_current_monitor(g_hWindow);
        const GLFWvidmode* mode = glfwGetVideoMode(targetMonitor);
        glfwSetWindowMonitor(g_hWindow, targetMonitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    }
    else
    {
        // Restoring directly with the saved coordinates without adjustment by GetWindowFrameSize (unreliable post-fullscreen)
        glfwSetWindowMonitor(g_hWindow, nullptr, oldWindowX, oldWindowY, oldWindowW, oldWindowH, 0);
    }

    g_IsFullscreen = !g_IsFullscreen;
}
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // Callback function for window resizing
    if (width == 0 || height == 0) return;
    if (g_IsRecreating) return;

    g_WindowW = width;
    g_WindowH = height;
    g_WindowW_2 = width / 2;
    g_WindowH_2 = height / 2;

    g_Camera.SetViewportSize(g_WindowW, g_WindowH);
    
    g_IsRecreating = true; 
    PartialRecreate();
    g_IsRecreating = false;
}
void CursorPosCallback(GLFWwindow* window, double xposIn, double yposIn)
{
    g_Camera.MousePosUpdate(xposIn, yposIn);
}
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_RELEASE)
            g_ForceActive = false;
        else if (action == GLFW_PRESS && !ImGui::GetIO().WantCaptureMouse)
            g_ForceActive = true;
        g_ForceHasPrev = false;
        return;
    }

    if (!ImGui::GetIO().WantCaptureMouse)
        g_Camera.MouseButtonUpdate(button, action, mods);
}
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    vec2 mouse(mouseX, mouseY);
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (g_Camera.GetMode() == eCameraMode::ORBITAL)
            g_Camera.AdjustOrbitRadius(-yoffset * 0.1f * g_Camera.GetOrbitRadius());
    }
}
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    g_Camera.KeyboardUpdate(key, scancode, action, mods);
    
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
            // Pause
        case GLFW_KEY_SPACE:
            g_bPause = !g_bPause;
            if (g_bPause)   g_TimerScene.stop();
            else            g_TimerScene.start();
            break;
        case GLFW_KEY_G:
            IMGUI_STYLE++;
			ApplyTheme();
			break;
        case GLFW_KEY_I:    // Cycle the interpolation function to the next type
        {
            int current = static_cast<int>(g_Camera.GetInterpolation());
            int next = (current + 1) % static_cast<int>(eInterpolation::COUNT);
            g_Camera.SetInterpolation(static_cast<eInterpolation>(next));
        }
        break;
		case GLFW_KEY_F1:
            g_bShowShortcuts = !g_bShowShortcuts;
            break;
        case GLFW_KEY_F2:
            g_bShowSceneWindow = !g_bShowSceneWindow;
            break;
        case GLFW_KEY_F4:
            g_bShowStatusBar = !g_bShowStatusBar;
            break;
        case GLFW_KEY_F8:
        {
            HWND hWnd = glfwGetWin32Window(g_hWindow);
            g_CaptureName = SaveClientArea(hWnd);
            g_CaptureDisplayTime = g_CaptureMaxTime;
        }
        break;
        case GLFW_KEY_F11:
            g_PendingFullscreen = true;
            break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        // A key has just been released
    }
}

ImVec2 GetLeftAlignedTextPos(const char* text, ImVec2 box_left, float padding_x, float padding_y, ImFont* font)
{
    ImGui::PushFont(font);

    // Calculate text size
    ImVec2 text_size = ImGui::CalcTextSize(text);

    // Text position: left edge of the box + padding (left-aligned)
    ImVec2 text_pos = ImVec2(box_left.x + padding_x, box_left.y - text_size.y * 0.5f);

    ImGui::PopFont();
    return text_pos;
}
ImVec2 GetCenteredTextPos(const char* text, ImVec2 box_center, float padding_x, float padding_y, ImFont* font)
{
    ImGui::PushFont(font);

    // Calculate text size
    ImVec2 text_size = ImGui::CalcTextSize(text);

    // Text position: center of the box - half text size
    ImVec2 text_pos = ImVec2( box_center.x - text_size.x * 0.5f, box_center.y - text_size.y * 0.5f );

    ImGui::PopFont();
    return text_pos;
}
ImVec2 GetRightAlignedTextPos(const char* text, ImVec2 box_right, float padding_x, float padding_y, ImFont* font)
{
    ImGui::PushFont(font);

    // Calculate text size
    ImVec2 text_size = ImGui::CalcTextSize(text);

    // Text position: right edge of the box - text size (right-aligned)
    ImVec2 text_pos = ImVec2(box_right.x - text_size.x - padding_x, box_right.y - text_size.y * 0.5f);

    ImGui::PopFont();
    return text_pos;
}

static bool CursorToWorld(vec3& outPoint)
{
    double mx, my;
    glfwGetCursorPos(g_hWindow, &mx, &my);
    int ww, wh;
    glfwGetWindowSize(g_hWindow, &ww, &wh);
    if (ww <= 0 || wh <= 0)
        return false;

    // Ray through the pixel (two points of the ray, whatever the depth convention of the projection)
    const float nx = 2.0f * (float)mx / (float)ww - 1.0f;
    const float ny = 2.0f * (float)my / (float)wh - 1.0f;
    const mat4  view = g_Camera.GetView();
    const mat4  invVP = glm::inverse(g_Camera.GetProjection() * view);
    const vec4  a = invVP * vec4(nx, ny, 0.0f, 1.0f);
    const vec4  b = invVP * vec4(nx, ny, 1.0f, 1.0f);
    const vec3  o = vec3(a) / a.w;
    const vec3  dir = glm::normalize(vec3(b) / b.w - o);

    // Plane perpendicular to the view direction, through the center of the tank
    const vec3 fwd = -vec3(view[0][2], view[1][2], view[2][2]);
    const vec3 center = 0.5f * (g_SphSolver.BoxMin() + g_SphSolver.BoxMax()) + fwd * g_ForceDepth;
    const float denom = glm::dot(dir, fwd);
    if (std::abs(denom) < 1e-4f)
        return false;
    const float t = glm::dot(center - o, fwd) / denom;
    if (t < 0.0f)
        return false;

    outPoint = glm::clamp(o + t * dir, g_SphSolver.BoxMin(), g_SphSolver.BoxMax());
    return true;
}
static void UpdateForceCursor(double dt)
{
    vec3 p;
    if (!g_ForceActive || dt <= 0.0 || !CursorToWorld(p))
    {
        g_SphSolver.ClearCursorForce();
        g_ForceHasPrev = false;
        g_ForceVel = vec3(0.0f);
        return;
    }

    if (g_ForceHasPrev)
    {
        vec3 v = (p - g_ForcePrev) / (float)dt;
        const float maxV = g_SphSolver.Params().maxSpeed;
        const float sp = glm::length(v);
        if (sp > maxV)
            v *= maxV / sp;
        g_ForceVel = glm::mix(g_ForceVel, v, 0.3f);     // smoothing: mouse events are jerky
    }
    g_ForcePrev = p;
    g_ForceHasPrev = true;
    g_SphSolver.SetCursorForce(p, g_ForceRadius, g_ForceVel, g_ForceStiffness);
}

// Imgui
void RenderStatusbar()
{
    if (!g_bShowStatusBar)
        return;

    ImGui::PushFont(g_FontArial16);

    char text[256];
    vec3 position = g_Camera.GetPosition();
    float distCameraShip = glm::length(position);
    vec2 lonlat = opengl_to_lonlat(position.x, position.z);
    sprintf_s(text, "CAMERA             x = %.2f y = %.2f z = %.2f          Heading = %03d\xc2\xb0 Roll = %d\xc2\xb0 Pitch = %03d\xc2\xb0           Distance to centre = %d m",
        position.x, position.y, position.z,
        (int)g_Camera.GetNorthAngleDEG(), (int)g_Camera.GetAttitudeDEG(), (int)g_Camera.GetRollDEG(),
        int(distCameraShip));

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();

    // White semi-transparent background rectangle (at the bottom of the screen)
    draw_list->AddRectFilled(ImVec2(0, g_WindowH - 20), ImVec2(g_WindowW, g_WindowH), IM_COL32(255, 255, 255, 64));

    // Black text vertically centered
    ImVec2 box_left = ImVec2(0, g_WindowH - 20 + 10);
    ImVec2 text_pos = GetLeftAlignedTextPos(text, box_left, 5.0f, 0.0f, g_FontArial16);

    draw_list->AddText(text_pos, IM_COL32(0, 0, 0, 255), text);

    ImGui::PopFont();
}
void RenderTitle()
{
    float x = g_WindowW - 55.0f;
    float y = g_WindowH - 45.0f;
    
    ImGui::PushFont(g_FontCaveat36);  // "Caveat" font equivalent

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    const char* text = "SimFluid";

    // Adjust position according to alignment (simulates nvgTextAlign)
    ImVec2 text_size = ImGui::CalcTextSize(text);
    float textX = x;

    textX -= text_size.x * 0.5f;

    // White title
    draw_list->AddText(ImVec2(textX, y), g_ColorWhite, text);

    ImGui::PopFont();
}
void RenderInfo()
{
    ImGui::PushFont(g_FontArial36);

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();

    // Screenshot name
    if (g_CaptureName.length() > 0 && g_CaptureDisplayTime > 0.0f)
    {
        std::string s(g_CaptureName.begin(), g_CaptureName.end());

        // Get viewport (window size)
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 window_center = viewport->GetCenter();

        // Calculate text size for precise offset
        ImVec2 text_size = ImGui::CalcTextSize(s.c_str());

        // Final position: center - (text size / 2)
        ImVec2 text_pos = ImVec2(window_center.x - text_size.x * 0.5f, window_center.y - text_size.y * 0.5f);

        // Transparent black rectangle (padding 10px around text)
        ImVec2 rect_min = ImVec2(text_pos.x - 10.0f, text_pos.y - 5.0f);
        ImVec2 rect_max = ImVec2(text_pos.x + text_size.x + 10.0f, text_pos.y + text_size.y + 5.0f);

        // Black color 70% transparent
        draw_list->AddRectFilled(rect_min, rect_max, IM_COL32(0, 0, 0, 180), 4.0f);

        // White text on top
        draw_list->AddText(text_pos, g_ColorWhite, s.c_str());

        g_CaptureDisplayTime -= ImGui::GetIO().DeltaTime;
    }
    ImGui::PopFont();
}
void RenderSphSolverInfo()
{
    ImGui::PushFont(g_FontArial20);

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();

    if (g_SphSolverInfo.length() > 0)
    {
        std::string s(g_SphSolverInfo.begin(), g_SphSolverInfo.end());

        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        // Marge depuis le bas de la fenêtre
        const float bottom_margin = 40.0f;

        ImVec2 text_size = ImGui::CalcTextSize(s.c_str());

        // Centré horizontalement, proche du bas verticalement
        float x = viewport->GetCenter().x - text_size.x * 0.5f;
        float y = viewport->WorkPos.y + viewport->WorkSize.y
            - bottom_margin - text_size.y;

        ImVec2 text_pos = ImVec2(x, y);

        // Rectangle de fond avec padding
        ImVec2 rect_min = ImVec2(text_pos.x - 10.0f, text_pos.y - 5.0f);
        ImVec2 rect_max = ImVec2(text_pos.x + text_size.x + 10.0f, text_pos.y + text_size.y + 5.0f);

        draw_list->AddRectFilled(rect_min, rect_max, IM_COL32(0, 0, 0, 180), 4.0f);
        draw_list->AddText(text_pos, g_ColorWhite, s.c_str());
    }

    ImGui::PopFont();
}
void RenderShortcuts()
{
    if (!g_bShowShortcuts)
        return;

    ImGui::PushFont(g_FontArial20);

    // Split : from "W" to "O" → left col
    size_t splitIndex = 24;

    float padding = 20.0f;
    float lineHeight = 22.0f;
    float boxWidth = 560.0f;
    float boxHeight = lineHeight * std::max(22, (int)(vShortcuts.size() - 22)) + padding * 2;

    // Center the box
    float x = (g_WindowW - boxWidth) / 2.0f;
    float y = (g_WindowH - boxHeight) / 2.0f;

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImVec2 pos = ImVec2(x, y);

    // Semi-transparent background
    draw_list->AddRectFilled(pos, ImVec2(x + boxWidth, y + boxHeight), IM_COL32(0, 0, 0, 160));

    // Central dividing line
    float separatorX = x + padding + (boxWidth - 2 * padding) * 0.5f;
    draw_list->AddLine(ImVec2(separatorX, y + padding),
        ImVec2(separatorX, y + boxHeight - padding),
        IM_COL32(255, 255, 255, 180), 2.0f);

    // Width of a column
    float columnWidth = (boxWidth - 2 * padding) * 0.5f;

    // Column positions
    float colLeftXKey = x + padding;
    float colLeftXDesc = x + padding + columnWidth - 20;
    float colRightXKey = x + padding + columnWidth + 20;
    float colRightXDesc = x + boxWidth - padding;

    // Left column
    float textYLeft = y + padding + lineHeight * 0.8f;
    for (size_t i = 0; i <= splitIndex && i < vShortcuts.size(); ++i)
    {
        // Key (left-aligned)
        draw_list->AddText(ImVec2(colLeftXKey, textYLeft), g_ColorWhite, vShortcuts[i].first.c_str());
        // Description (right-aligned)
        float descWidth = ImGui::CalcTextSize(vShortcuts[i].second.c_str()).x;
        draw_list->AddText(ImVec2(colLeftXDesc - descWidth, textYLeft), g_ColorWhite, vShortcuts[i].second.c_str());
        textYLeft += lineHeight;
    }

    // Right column
    float textYRight = y + padding + lineHeight * 0.8f;
    for (size_t i = splitIndex + 1; i < vShortcuts.size(); ++i)
    {
        // Key (left-aligned)
        draw_list->AddText(ImVec2(colRightXKey, textYRight), g_ColorWhite, vShortcuts[i].first.c_str());
        // Description (right-aligned)
        float descWidth = ImGui::CalcTextSize(vShortcuts[i].second.c_str()).x;
        draw_list->AddText(ImVec2(colRightXDesc - descWidth, textYRight), g_ColorWhite, vShortcuts[i].second.c_str());
        textYRight += lineHeight;
    }

    ImGui::PopFont();
}
void RenderForceCursor()
{
    if (!g_ForceActive || !g_ForceHasPrev)
        return;

    const mat4 proj = g_Camera.GetProjection();
    const vec4 c = proj * g_Camera.GetView() * vec4(g_ForcePrev, 1.0f);
    if (c.w <= 1e-3f)
        return;

    int ww, wh;
    glfwGetWindowSize(g_hWindow, &ww, &wh);
    double mx, my;
    glfwGetCursorPos(g_hWindow, &mx, &my);
    const float rpx = g_ForceRadius * std::abs(proj[1][1]) * (float)wh * 0.5f / c.w;   // same formula as the point size
    ImGui::GetBackgroundDrawList()->AddCircle(ImVec2((float)mx, (float)my), rpx, IM_COL32(255, 160, 30, 220), 48, 2.0f);
}
void FrameImGui(VkCommandBuffer commandBuffer)
{
    auto RightAlignText = [&](const char* fmt, ...) {
        char buf[64];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        float text_w = ImGui::CalcTextSize(buf).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - text_w - ImGui::GetStyle().ItemSpacing.x);
        ImGui::Text("%s", buf);
        };

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static string resultAnalyse;
    static vector<vector<sResultData>> vResults;

#pragma region Scene F2
    if (g_bShowSceneWindow)
    {
        // Fixed position at the bottom right of the window
        ImVec2 window_size(300, 930);
        ImVec2 window_pos(5, 5);
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_FirstUseEver);

        /////////////////////////////////
        if (ImGui::Begin("Scene [F2]", &g_bShowSceneWindow))
        {
            ImGui::Text("FPS = %d", g_nFps);
            ImGui::SameLine();
            double elapsed = glfwGetTime();  // seconds elapsed since the start of the program or ImGui
            int hours = static_cast<int>(elapsed) / 3600;
            int minutes = (static_cast<int>(elapsed) % 3600) / 60;
            int seconds = static_cast<int>(elapsed) % 60;
            char buffer[12];
            snprintf(buffer, sizeof(buffer), "  %02d:%02d:%02d", hours, minutes, seconds);
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 128, 255));
            ImGui::SetWindowFontScale(1.5f);
            ImGui::Text("%s", buffer);
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopStyleColor();
            // ------------
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            // Camera type
            switch (g_Camera.GetMode())
            {
            case eCameraMode::ORBITAL:  ImGui::Text("[ ORBITAL ]"); break;
            case eCameraMode::BRIDGE:   ImGui::Text("[ BRIDGE ]"); break;
            case eCameraMode::FPS:      ImGui::Text("[ FPS ]");     break;
            }
            ImGui::SameLine();
            // Interpolation function
            switch (g_Camera.GetInterpolation())
            {
            case eInterpolation::Linear:    ImGui::Text("( Linear )");      break;
            case eInterpolation::SmoothStep:ImGui::Text("( SmoothStep )");  break;
            case eInterpolation::EaseIn:    ImGui::Text("( EaseIn )");      break;
            case eInterpolation::EaseOut:   ImGui::Text("( EaseOut )");     break;
            case eInterpolation::EaseInOut: ImGui::Text("( EaseInOut )");   break;
            }
            ImGui::PopStyleColor(1);
            // ------------
            if (ImGui::Button(" PAUSE "))
            {
                g_bPause = !g_bPause;
                if (g_bPause)
                    g_TimerScene.stop();
                else
                    g_TimerScene.start();
            }
            ImGui::SameLine();
            if (ImGui::Button(" FULLSCREEN "))
                g_PendingFullscreen = true;
            // ------------
            if (ImGui::CollapsingHeader("OBJECTS", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Checkbox("Axis", &g_Axis->bVisible);
                ImGui::SameLine();
                ImGui::Checkbox("Bbox", &g_Bbox->bVisible);
                ImGui::SameLine();
                ImGui::Checkbox("Cyl", &g_Cylinder->bVisible);
            }
            // ------------
            if (ImGui::CollapsingHeader("SPH SIMULATION", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
                if (ImGui::Button("   RESET SPH   "))
                    g_PendingSphReset = true;
                ImGui::PopStyleColor();
                // ------------
                bool usePcg = g_SphSolver.UsePcg();
                if (ImGui::Checkbox("usePcg", &usePcg))
                    g_SphSolver.SetUsePcg(usePcg);
                ImGui::SameLine();
                bool bGravity = g_SphSolver.GravityEnabled();
                if (ImGui::Checkbox("Gravity", &bGravity))
                    g_SphSolver.SetGravityEnabled(bGravity);
                float tension = g_SphSolver.SurfaceTension();
                if (ImGui::SliderFloat("Surface tension", &tension, 0.0f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
                    g_SphSolver.SetSurfaceTension(tension);
                // ------------
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
                if (ImGui::Combo("Scenario", &g_Scenario, g_ScenarioNames, IM_ARRAYSIZE(g_ScenarioNames)))
                    g_PendingSphReset = true;                       // same path as the "Reset SPH" button
                ImGui::PopStyleColor();
                if (g_Scenario == (int)eScenario::WaveMaker)
                {
                    bool changed = ImGui::SliderFloat("Stroke (m)", &g_WaveStroke, 0.05f, 0.8f, "%.1f");
                    changed |= ImGui::SliderFloat("Period (s)", &g_WavePeriod, 0.8f, 4.0f, "%.1f");
                    if (changed && g_SphSolver.WaveMakerEnabled())
                        g_SphSolver.SetWaveMaker(true, g_WaveStroke, g_WavePeriod);     // live: no Reset needed
                }
            }
            // ------------
            if (ImGui::CollapsingHeader("RENDERING", ImGuiTreeNodeFlags_DefaultOpen))
            {
                // Pressure only exists with the PCG solver (warm[] is never written by the Jacobi solver)
                const bool pcgOn = g_SphSolver.UsePcg();
                const int  pressureIdx = (int)SphSolver::ColorMode::Pressure;
                
                int renderMode = g_SphSolver.RenderModeIndex();
                ImGui::Text("Render");
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
                if (ImGui::RadioButton("Particles", &renderMode, 0))
                    g_SphSolver.SetRenderMode(0);
                ImGui::SameLine();
                if (ImGui::RadioButton("Surface", &renderMode, 1))
                    g_SphSolver.SetRenderMode(1);
                ImGui::PopStyleColor();
                ImGui::Separator();
                // ------------
                ImGui::Text("Surface");
                SphSurface::Settings& s = g_SphSolver.SurfaceSettings();
                ImGui::SliderFloat("Sphere radius", &s.radiusScale, 0.5f, 2.0f);   // (x spacing)
                ImGui::SliderInt("Blur iterations", &s.nBlurIterations, 1, 4);
                ImGui::SliderFloat("Filter radius", &s.filterRadius, 0.5f, 6.0f);
                ImGui::SliderFloat("Depth sigma", &s.depthSigma, 0.2f, 3.0f);
                ImGui::SliderInt("Max filter (px)", &s.nMaxFilterPixels, 4, 48);
                ImGui::ColorEdit3("Water color", &s.waterColor.x);
                ImGui::SliderFloat("Absorption", &s.absorption, 0.0f, 10.0f);
                ImGui::SliderFloat("Fresnel F0", &s.fresnelF0, 0.0f, 0.2f);
                ImGui::SliderFloat("Specular power", &s.specularPower, 8.0f, 600.0f);
                ImGui::SliderFloat("Light azimuth", &s.lightAzimuth, -180.0f, 180.0f);
                ImGui::SliderFloat("Light elevation", &s.lightElevation, 5.0f, 90.0f);
                ImGui::SliderFloat("Exposure", &s.exposure, 0.3f, 2.0f);
                // ------------
                ImGui::Separator();
                ImGui::Text("Particles");
                int colorMode = g_SphSolver.ColorModeIndex();
                if (!pcgOn && colorMode == pressureIdx)
                {
                    colorMode = (int)SphSolver::ColorMode::Density;     // fall back to a mode that works
                    g_SphSolver.SetColorMode(colorMode);
                }

                // Combo built by hand to hide the unavailable entry
                const char* const* names = SphSolver::ColorModeNames();
                if (ImGui::BeginCombo("Color by", names[colorMode]))
                {
                    for (int n = 0; n < SphSolver::ColorModeCount(); ++n)
                    {
                        if (n == pressureIdx && !pcgOn)
                            continue;

                        const bool selected = (n == colorMode);
                        if (ImGui::Selectable(names[n], selected))
                            g_SphSolver.SetColorMode(n);                // also resets the range to a sensible default
                        if (selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                float lo = g_SphSolver.ColorMin();
                float hi = g_SphSolver.ColorMax();
                const float step = std::max((hi - lo) * 0.005f, 1e-4f);
                bool changed = ImGui::DragFloat("Min", &lo, step, 0.0f, 0.0f, "%.1f");
                changed |= ImGui::DragFloat("Max", &hi, step, 0.0f, 0.0f, "%.1f");
                if (changed)
                    g_SphSolver.SetColorRange(lo, hi);
                if (ImGui::Button("Reset range"))
                    g_SphSolver.ResetColorRange();

                // Legend (same ramp as sph.frag: blue -> cyan -> green -> yellow -> red)
                {
                    ImVec2 p = ImGui::GetCursorScreenPos();
                    float w = ImGui::GetContentRegionAvail().x, bar = 12.0f;
                    ImDrawList* dl = ImGui::GetWindowDrawList();

                    const ImU32 stops[5] = {
                        IM_COL32(0,   0, 255, 255),     // blue
                        IM_COL32(0, 255, 255, 255),     // cyan
                        IM_COL32(0, 255,   0, 255),     // green
                        IM_COL32(255, 255,   0, 255),   // yellow
                        IM_COL32(255,   0,   0, 255)    // red
                    };
                    const float seg = w * 0.25f;
                    for (int s = 0; s < 4; ++s)
                    {
                        ImVec2 a(p.x + seg * s, p.y), b(p.x + seg * (s + 1), p.y + bar);
                        dl->AddRectFilledMultiColor(a, b, stops[s], stops[s + 1], stops[s + 1], stops[s]);
                    }
                    ImGui::Dummy(ImVec2(w, bar));
                    ImGui::Text("%.0f", lo); ImGui::SameLine(w - 40.0f); ImGui::Text("%.0f", hi);
                }
                // ------------
                float ps = g_SphSolver.PointScale();
                if (ImGui::SliderFloat("Point size", &ps, 0.2f, 3.0f, "%0.1f"))
                    g_SphSolver.SetPointScale(ps);
            }
            // ------------
            if (ImGui::CollapsingHeader("FORCE CURSOR", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Text("Force cursor (right mouse button)");
                ImGui::SliderFloat("Radius", &g_ForceRadius, 0.1f, 1.5f, "%.1f");
                ImGui::SliderFloat("Stiffness", &g_ForceStiffness, 0.0f, 80.0f, "%.1f");
                ImGui::SliderFloat("Depth", &g_ForceDepth, -1.3f, 1.3f, "%.1f");
            }
        }
        ImGui::End();
    }
#pragma endregion

    //RenderEnv(g_WindowW - 85, 5);
    RenderSphSolverInfo();
    RenderShortcuts();  // F1
	RenderStatusbar();  // F4
    RenderInfo();
    RenderTitle();
    RenderForceCursor();

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
}

// Render
void FrameRender(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // SPH compute: must be recorded outside the render pass (also handles the initial upload, even when paused with dt = 0)
    float simDt = 0.0f;
    if (g_SphDt >= 0.004f)      // about maxDt: do not run the solver for tiny steps
    {
        simDt = g_SphDt;
        g_SphDt = 0.0f;
    }
    g_SphSolver.Update(commandBuffer, g_iCurrentFrame, simDt);
    g_SphSolver.RecordSurface(commandBuffer, g_iCurrentFrame, g_Camera);

    float width = (float)g_SwapChain->extent.width;
    float height = (float)g_SwapChain->extent.height;
    VkViewport viewport{ 0, 0, width, height, 0, 1 };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor{ { 0, 0 }, g_SwapChain->extent };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // UBO updates must happen outside the render pass
    g_Axis->UpdateMsUBOs(imageIndex, g_Camera, g_Sky.get(), mat4(1.0f));

    // One entry per attachment up to the last one that is cleared (index 1); index 2 (swapchain) uses DONT_CARE so its value is ignored
    VkClearValue clearValues[3]{};
    clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 1.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo passInfo{};
    passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    passInfo.renderPass = g_RenderPass;
    passInfo.framebuffer = g_vFramebuffersSwapChain[imageIndex];
    passInfo.renderArea.offset = { 0, 0 };
    passInfo.renderArea.extent = g_SwapChain->extent;
    passInfo.clearValueCount = 3;
    passInfo.pClearValues = clearValues;
    vkCmdBeginRenderPass(commandBuffer, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Subpass 0: MSAA scene
    g_Axis->RenderMs(commandBuffer, g_iCurrentFrame);
    g_Bbox->RenderBbox(commandBuffer, g_iCurrentFrame, g_Camera, mat4(1.0f), vec4(1.0f, 1.0f, 1.0f, 1.0f));
    if (g_SphSolver.WaveMakerEnabled())
    {
        const float dx = g_SphSolver.PistonPosition() - g_SphSolver.BoxMin().x;
        g_Piston->RenderBbox(commandBuffer, g_iCurrentFrame, g_Camera, glm::translate(mat4(1.0f), vec3(dx, 0.0f, 0.0f)), vec4(1.0f, 0.4f, 0.2f, 1.0f));
    }
	g_Cylinder->Render(commandBuffer, g_iCurrentFrame, g_Camera, mat4(1.0f), vec4(1.0f, 1.0f, 1.0f, 1.0f));
    g_SphSolver.Render(commandBuffer, g_iCurrentFrame, g_Camera, 1.0f, 1.0f);   // size = 1 x spacing

    // Subpass 1: ImGui
    vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
    FrameImGui(commandBuffer);

    vkCmdEndRenderPass(commandBuffer);
    vkEndCommandBuffer(commandBuffer);
}
void Render()
{
    // Wait for the current frame's fence before reusing its resources
    vkWaitForFences(g_Device->device, 1, &g_vImageFences[g_iCurrentFrame], VK_TRUE, UINT64_MAX);

    // Acquire the next swapchain image; signal imageAvailable[currentFrame] when ready
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(g_Device->device, g_SwapChain->swapChain, UINT64_MAX, g_vImageAvailableSemaphores[g_iCurrentFrame],   // indexed on currentFrame
        VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return; // Handle swapchain recreation here if needed

    // Reset fence and command buffer only once we know this frame will be submitted
    vkResetFences(g_Device->device, 1, &g_vImageFences[g_iCurrentFrame]);
    vkResetCommandBuffer(g_vCommandBuffers[g_iCurrentFrame], 0);

    // Record all rendering commands for this frame
    FrameRender(g_vCommandBuffers[g_iCurrentFrame], imageIndex);

    // ── Build wait semaphore list ────────────────────────────────────────
    VkSemaphore          waitSemaphores[3];
    VkPipelineStageFlags waitStages[3];
    uint32_t             waitCount = 0;

    // Always wait for the swapchain image to be available
    waitSemaphores[waitCount] = g_vImageAvailableSemaphores[g_iCurrentFrame];
    waitStages[waitCount] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitCount++;

    // ── Submit graphics command buffer ───────────────────────────────────
    
    // Signal semaphore indexed on currentFrame (NOT imageIndex) to avoid signaling a semaphore still pending from a previous present cycle
    VkSemaphore signalSemaphores[] = { g_vRenderFinishedSemaphores[g_iCurrentFrame] };

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = waitCount;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &g_vCommandBuffers[g_iCurrentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // Fence signals when this frame's GPU work is complete
    vkQueueSubmit(g_Device->graphicsQueue, 1, &submitInfo, g_vImageFences[g_iCurrentFrame]);

    // ── Present ──────────────────────────────────────────────────────────
    
    // Wait on the same renderFinished[currentFrame] semaphore before presenting
    VkSwapchainKHR swapchains[] = { g_SwapChain->swapChain };

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;  // currentFrame ✅
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(g_Device->presentQueue, &presentInfo);

    g_FpsFrameCount++;
    double currentTime = glfwGetTime();
    if (currentTime - g_FpsTime >= 1.0)
    {
        g_nFps = static_cast<int>(g_FpsFrameCount / (currentTime - g_FpsTime));
        g_FpsFrameCount = 0;
        g_FpsTime = currentTime;
    }

    // Advance to the next frame slot (wraps around MAX_FRAMES_IN_FLIGHT)
    g_iCurrentFrame = (g_iCurrentFrame + 1) % g_vImageFences.size();
}

// Initialization
void InitConsole()
{
    // Create a Windows console attached to the process

    // UTF-8 codepage for Windows console
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    // Force UTF-16 for streams
    _setmode(_fileno(stdout), _O_U16TEXT);
    _setmode(_fileno(stderr), _O_U16TEXT);

    // Locales + sync
    ios::sync_with_stdio(false);
    locale::global(locale(".65001"));  // UTF-8 locale

    // Fix wcout specific to Windows
    wcout.imbue(locale(".65001"));
    wcerr.imbue(locale(".65001"));

    AllocConsole();
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    freopen("CONIN$", "r", stdin);

    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    SetConsoleTitle(L"SimFluid Console");
    SetWindowPos(GetConsoleWindow(), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE);
}
void InitWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    g_hWindow = glfwCreateWindow(g_WindowW, g_WindowH, "SimFluid (Vulkan)", nullptr, nullptr);
    if (!g_hWindow)
        throw runtime_error("Failed to create GLFW window");

    // Center the window on the primary monitor
    g_Monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* videoMode = glfwGetVideoMode(g_Monitor);
    glfwSetWindowPos(g_hWindow, (videoMode->width - g_WindowW) / 2, (videoMode->height - g_WindowH) / 2);

    glfwSetFramebufferSizeCallback(g_hWindow, framebuffer_size_callback);
    glfwSetCursorPosCallback(g_hWindow, CursorPosCallback);
    glfwSetScrollCallback(g_hWindow, ScrollCallback);
    glfwSetMouseButtonCallback(g_hWindow, MouseButtonCallback);
    glfwSetKeyCallback(g_hWindow, KeyCallback);
    glfwShowWindow(g_hWindow);
}
void CreateInstance()
{
    // Create a Vulkan instance (the global context)

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 4, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 4, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Validation layers
    if (g_bEnableValidationLayers) 
    {
        if (!checkValidationLayerSupport()) 
            throw runtime_error("Validation layers are not available!");

        createInfo.enabledLayerCount = static_cast<uint32_t>(g_vValidationLayers.size());
        createInfo.ppEnabledLayerNames = g_vValidationLayers.data();
#ifdef INFO_INIT    // Structures.h
        wcout << L"Validation layers enabled\n";
#endif
    }
    else
        createInfo.enabledLayerCount = 0;

    // Extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    if (g_bEnableValidationLayers)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &g_Instance) != VK_SUCCESS)
        throw runtime_error("Vulkan instance creation failed");

    if (g_bEnableValidationLayers) 
        setupDebugMessenger(g_Instance);

#ifdef INFO_INIT    // Structures.h
    wcout << L"Instance OK\n";
#endif
}

void CreateRenderPass()
{
    VkFormat colorFormat = g_SwapChain->imageFormat;
    VkFormat depthFormat = FindDepthFormat();

    // Attachment 0: MSAA color (render target, never stored)
    VkAttachmentDescription colorMsaa{};
    colorMsaa.format = colorFormat;
    colorMsaa.samples = g_Device->msaaSamples;
    colorMsaa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorMsaa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorMsaa.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorMsaa.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorMsaa.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorMsaa.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Attachment 1: MSAA depth (never stored)
    VkAttachmentDescription depthMsaa{};
    depthMsaa.format = depthFormat;
    depthMsaa.samples = g_Device->msaaSamples;
    depthMsaa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthMsaa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthMsaa.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthMsaa.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthMsaa.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthMsaa.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Attachment 2: swapchain image (MSAA resolve target, then ImGui target)
    VkAttachmentDescription swapchain{};
    swapchain.format = colorFormat;
    swapchain.samples = VK_SAMPLE_COUNT_1_BIT;
    swapchain.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // fully overwritten by the resolve
    swapchain.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    swapchain.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    swapchain.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    swapchain.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchain.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    array<VkAttachmentDescription, 3> attachments = { colorMsaa, depthMsaa, swapchain };

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkAttachmentReference swapRef{ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpasses[2]{};

    // Subpass 0: MSAA scene, resolved into the swapchain image at the end of the subpass
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = 1;
    subpasses[0].pColorAttachments = &colorRef;
    subpasses[0].pResolveAttachments = &swapRef;
    subpasses[0].pDepthStencilAttachment = &depthRef;

    // Subpass 1: ImGui, drawn 1x on top of the resolved image
    subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[1].colorAttachmentCount = 1;
    subpasses[1].pColorAttachments = &swapRef;

    VkSubpassDependency deps[2]{};

    // External -> subpass 0: wait for swapchain acquire (imageAvailable semaphore)
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Subpass 0 -> subpass 1: resolve must finish before ImGui blends over it
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = 1;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = (uint32_t)attachments.size();
    info.pAttachments = attachments.data();
    info.subpassCount = 2;
    info.pSubpasses = subpasses;
    info.dependencyCount = 2;
    info.pDependencies = deps;

    vkCreateRenderPass(g_Device->device, &info, nullptr, &g_RenderPass);
}
void CreateImages()
{
    // 8x MSAA color (render target only, resolved into the swapchain)
    g_ColorImage = make_unique<VulkanTexture>(
        g_Device, g_SwapChain->extent.width, g_SwapChain->extent.height, 1,
        1, g_Device->msaaSamples, g_SwapChain->imageFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // 8x MSAA depth
    g_DepthImage = make_unique<VulkanTexture>(
        g_Device, g_SwapChain->extent.width, g_SwapChain->extent.height, 1,
        1, g_Device->msaaSamples, FindDepthFormat(), VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
}
void CreateFramebuffers()
{
    g_vFramebuffersSwapChain.resize(g_SwapChain->imageCount);

    for (size_t i = 0; i < g_SwapChain->imageCount; i++)
    {
        array<VkImageView, 3> attachments = {
            g_ColorImage->imageView,        // 0: MSAA color
            g_DepthImage->imageView,        // 1: MSAA depth
            g_SwapChain->vImageViews[i]     // 2: swapchain (resolve + ImGui)
        };

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = g_RenderPass;
        fb.attachmentCount = (uint32_t)attachments.size();
        fb.pAttachments = attachments.data();
        fb.width = g_SwapChain->extent.width;
        fb.height = g_SwapChain->extent.height;
        fb.layers = 1;

        vkCreateFramebuffer(g_Device->device, &fb, nullptr, &g_vFramebuffersSwapChain[i]);
    }
}
void CreateCommandBuffers()
{
    uint32_t framebufferCount = static_cast<uint32_t>(g_vFramebuffersSwapChain.size());
    if (framebufferCount == 0)
        throw runtime_error("No framebuffer!");

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = g_Device->graphicsCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = framebufferCount;

    g_vCommandBuffers.resize(framebufferCount);
    if (vkAllocateCommandBuffers(g_Device->device, &allocInfo, g_vCommandBuffers.data()) != VK_SUCCESS) 
        throw runtime_error("Failed to allocate command buffers");

#ifdef INFO_INIT    // Structures.h
    wcout << framebufferCount << L" command buffers OK\n";
#endif // INFO_INIT
}
void CreateSyncObjects()
{
    uint32_t imageCount = static_cast<uint32_t>(g_SwapChain->imageCount);  // 2

    g_vImageAvailableSemaphores.resize(imageCount);
    g_vRenderFinishedSemaphores.resize(imageCount);
    g_vImageFences.resize(imageCount);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < imageCount; i++) 
    {  
        vkCreateSemaphore(g_Device->device, &semaphoreInfo, nullptr, &g_vImageAvailableSemaphores[i]);
        vkCreateSemaphore(g_Device->device, &semaphoreInfo, nullptr, &g_vRenderFinishedSemaphores[i]);
        vkCreateFence(g_Device->device, &fenceInfo, nullptr, &g_vImageFences[i]);
    }

#ifdef INFO_INIT    // Structures.h
    wcout << imageCount << L" sync sets (1 par image) OK\n";
#endif // INFO_INIT
}

void InitVulkan()
{
    int width, height;
    glfwGetFramebufferSize(g_hWindow, &width, &height);
    g_SwapChain = make_unique<VulkanSwapChain>(g_Device, width, height);
    g_FramesInFlight = g_SwapChain->imageCount;

    // RenderPass (defines the expected formats)
    CreateRenderPass();

    // Create all images + their individual views
    CreateImages();
   
    // Framebuffers (uses all views)
    CreateFramebuffers();          // Render pass 3 (3D)

    CreateCommandBuffers();
    CreateSyncObjects();
}
void ApplyTheme()
{
    if (IMGUI_STYLE > 4)
		IMGUI_STYLE = 0;

    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    switch (IMGUI_STYLE)
    {
    case 0: ImGui::StyleColorsDark(); break;
    case 1: // Nord
    {
        // Corners
        style.WindowRounding = 0.0f;

        // Borders
        style.WindowBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;   // 0 border on frames (like Charcoal)
        style.PopupBorderSize = 1.0f;

        // Padding and spacing
        style.WindowPadding = ImVec2(8.0f, 8.0f);
        style.FramePadding = ImVec2(5.0f, 3.0f);
        style.ItemSpacing = ImVec2(8.0f, 4.0f);
        style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        style.IndentSpacing = 21.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 10.0f;

        // Nord palette
        const ImVec4 polarNight0 = ImVec4(0.16f, 0.18f, 0.22f, 1.00f); // #2e3440
        const ImVec4 polarNight1 = ImVec4(0.18f, 0.20f, 0.25f, 1.00f); // #3b4252
        const ImVec4 polarNight2 = ImVec4(0.20f, 0.23f, 0.29f, 1.00f); // #434c5e
        const ImVec4 polarNight3 = ImVec4(0.23f, 0.26f, 0.32f, 1.00f); // #4c566a
        const ImVec4 snowStorm1 = ImVec4(0.85f, 0.87f, 0.91f, 1.00f); // #d8dee9
        const ImVec4 snowStorm2 = ImVec4(0.90f, 0.91f, 0.94f, 1.00f); // #e5e9f0
        const ImVec4 snowStorm3 = ImVec4(0.93f, 0.95f, 0.97f, 1.00f); // #eceff4
        const ImVec4 frost1 = ImVec4(0.55f, 0.70f, 0.79f, 1.00f); // #8fbcbb
        const ImVec4 frost2 = ImVec4(0.53f, 0.75f, 0.82f, 1.00f); // #88c0d0
        const ImVec4 frost3 = ImVec4(0.37f, 0.51f, 0.67f, 1.00f); // #81a1c1
        const ImVec4 frost4 = ImVec4(0.36f, 0.51f, 0.72f, 1.00f); // #5e81ac
        const ImVec4 auroraRed = ImVec4(0.75f, 0.38f, 0.42f, 1.00f); // #bf616a
        const ImVec4 auroraOrange = ImVec4(0.82f, 0.53f, 0.39f, 1.00f); // #d08770
        const ImVec4 auroraYellow = ImVec4(0.92f, 0.80f, 0.55f, 1.00f); // #ebcb8b
        const ImVec4 auroraGreen = ImVec4(0.64f, 0.75f, 0.55f, 1.00f); // #a3be8c
        const ImVec4 auroraPurple = ImVec4(0.71f, 0.56f, 0.74f, 1.00f); // #b48ead
        const ImVec4 border = ImVec4(0.27f, 0.30f, 0.35f, 1.00f);

        colors[ImGuiCol_Text] = snowStorm3;
        colors[ImGuiCol_TextDisabled] = snowStorm1;
        colors[ImGuiCol_WindowBg] = polarNight1;
        colors[ImGuiCol_ChildBg] = polarNight2;
        colors[ImGuiCol_PopupBg] = polarNight2;
        colors[ImGuiCol_Border] = border;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg] = polarNight3;
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.30f, 0.35f, 0.42f, 1.00f);
        colors[ImGuiCol_FrameBgActive] = frost3;
        colors[ImGuiCol_TitleBg] = polarNight0;
        colors[ImGuiCol_TitleBgActive] = frost3;
        colors[ImGuiCol_TitleBgCollapsed] = polarNight1;
        colors[ImGuiCol_MenuBarBg] = polarNight2;
        colors[ImGuiCol_ScrollbarBg] = polarNight0;
        colors[ImGuiCol_ScrollbarGrab] = frost3;
        colors[ImGuiCol_ScrollbarGrabHovered] = frost2;
        colors[ImGuiCol_ScrollbarGrabActive] = frost2;
        colors[ImGuiCol_CheckMark] = auroraGreen;
        colors[ImGuiCol_SliderGrab] = frost4;
        colors[ImGuiCol_SliderGrabActive] = frost2;
        colors[ImGuiCol_Button] = polarNight3;
        colors[ImGuiCol_ButtonHovered] = frost3;
        colors[ImGuiCol_ButtonActive] = frost2;
        colors[ImGuiCol_Header] = ImVec4(0.37f, 0.51f, 0.67f, 0.40f);
        colors[ImGuiCol_HeaderHovered] = frost3;
        colors[ImGuiCol_HeaderActive] = frost2;
        colors[ImGuiCol_Separator] = border;
        colors[ImGuiCol_SeparatorHovered] = auroraPurple;
        colors[ImGuiCol_SeparatorActive] = auroraPurple;
        colors[ImGuiCol_ResizeGrip] = frost3;
        colors[ImGuiCol_ResizeGripHovered] = frost2;
        colors[ImGuiCol_ResizeGripActive] = frost2;
        colors[ImGuiCol_Tab] = polarNight3;
        colors[ImGuiCol_TabHovered] = frost2;
        colors[ImGuiCol_TabActive] = frost2;
        colors[ImGuiCol_TabUnfocused] = polarNight1;
        colors[ImGuiCol_TabUnfocusedActive] = polarNight3;
        colors[ImGuiCol_PlotLines] = frost2;
        colors[ImGuiCol_PlotLinesHovered] = auroraOrange;
        colors[ImGuiCol_PlotHistogram] = frost3;
        colors[ImGuiCol_PlotHistogramHovered] = auroraGreen;
        colors[ImGuiCol_TableHeaderBg] = polarNight3;
        colors[ImGuiCol_TableBorderStrong] = border;
        colors[ImGuiCol_TableBorderLight] = polarNight3;
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.37f, 0.51f, 0.67f, 0.35f);
        colors[ImGuiCol_DragDropTarget] = auroraYellow;
        colors[ImGuiCol_NavHighlight] = frost1;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.85f, 0.90f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.35f);
    }
    break;
    case 2: // Carbon
    {
        // Corners
        style.WindowRounding = 0.0f;
        style.ChildRounding = 0.0f;
        style.FrameRounding = 0.0f;
        style.PopupRounding = 0.0f;
        style.ScrollbarRounding = 0.0f;
        style.GrabRounding = 0.0f;
        style.TabRounding = 0.0f;

        // Borders
        style.WindowBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.PopupBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;

        // Padding and spacing
        style.WindowPadding = ImVec2(8.0f, 8.0f);
        style.FramePadding = ImVec2(5.0f, 3.0f);
        style.ItemSpacing = ImVec2(8.0f, 4.0f);
        style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        style.IndentSpacing = 21.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 10.0f;

        // Carbon palette
        const ImVec4 bg0 = ImVec4(0.08f, 0.08f, 0.08f, 1.00f); // darkest background
        const ImVec4 bg1 = ImVec4(0.12f, 0.12f, 0.12f, 1.00f); // bgVeryDark
        const ImVec4 bg2 = ImVec4(0.16f, 0.16f, 0.16f, 1.00f); // bgDark
        const ImVec4 bg3 = ImVec4(0.22f, 0.22f, 0.22f, 1.00f); // bgLight (frame bg)
        const ImVec4 bg4 = ImVec4(0.27f, 0.27f, 0.27f, 1.00f); // frame bg hover discreet
        const ImVec4 overlay0 = ImVec4(0.35f, 0.35f, 0.35f, 1.00f); // overlay discreet
        const ImVec4 overlay1 = ImVec4(0.45f, 0.45f, 0.45f, 1.00f); // overlay medium
        const ImVec4 textLight = ImVec4(0.96f, 0.96f, 0.96f, 1.00f); // main text
        const ImVec4 textDisabled = ImVec4(0.55f, 0.55f, 0.55f, 1.00f); // disabled text
        const ImVec4 accentDark = ImVec4(0.20f, 0.48f, 0.65f, 1.00f); // dark cyan-blue
        const ImVec4 accent = ImVec4(0.30f, 0.60f, 0.80f, 1.00f); // main cyan-blue
        const ImVec4 accentLight = ImVec4(0.40f, 0.65f, 0.85f, 1.00f); // light cyan-blue
        const ImVec4 accentBright = ImVec4(0.55f, 0.75f, 0.92f, 1.00f); // bright cyan-blue
        const ImVec4 cyan = ImVec4(0.30f, 0.80f, 0.85f, 1.00f); // complementary cyan
        const ImVec4 yellow = ImVec4(0.90f, 0.80f, 0.35f, 1.00f); // drag&drop yellow
        const ImVec4 orange = ImVec4(0.85f, 0.55f, 0.25f, 1.00f); // orange plot hover
        const ImVec4 purple = ImVec4(0.60f, 0.45f, 0.80f, 1.00f); // separator hover
        const ImVec4 green = ImVec4(0.45f, 0.80f, 0.45f, 1.00f); // checkmark
        const ImVec4 border = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);

        colors[ImGuiCol_Text] = textLight;
        colors[ImGuiCol_TextDisabled] = textDisabled;
        colors[ImGuiCol_WindowBg] = bg1;
        colors[ImGuiCol_ChildBg] = bg2;
        colors[ImGuiCol_PopupBg] = bg2;
        colors[ImGuiCol_Border] = border;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg] = bg3;
        colors[ImGuiCol_FrameBgHovered] = bg4;
        colors[ImGuiCol_FrameBgActive] = overlay0;
        colors[ImGuiCol_TitleBg] = bg0;
        colors[ImGuiCol_TitleBgActive] = accent;
        colors[ImGuiCol_TitleBgCollapsed] = bg1;
        colors[ImGuiCol_MenuBarBg] = bg2;
        colors[ImGuiCol_ScrollbarBg] = bg0;
        colors[ImGuiCol_ScrollbarGrab] = accent;
        colors[ImGuiCol_ScrollbarGrabHovered] = accentLight;
        colors[ImGuiCol_ScrollbarGrabActive] = accentBright;
        colors[ImGuiCol_CheckMark] = green;
        colors[ImGuiCol_SliderGrab] = accentDark;
        colors[ImGuiCol_SliderGrabActive] = accent;
        colors[ImGuiCol_Button] = bg3;
        colors[ImGuiCol_ButtonHovered] = accent;
        colors[ImGuiCol_ButtonActive] = accentLight;
        colors[ImGuiCol_Header] = ImVec4(0.30f, 0.60f, 0.80f, 0.40f);
        colors[ImGuiCol_HeaderHovered] = accent;
        colors[ImGuiCol_HeaderActive] = accentLight;
        colors[ImGuiCol_Separator] = border;
        colors[ImGuiCol_SeparatorHovered] = purple;
        colors[ImGuiCol_SeparatorActive] = purple;
        colors[ImGuiCol_ResizeGrip] = accent;
        colors[ImGuiCol_ResizeGripHovered] = accentLight;
        colors[ImGuiCol_ResizeGripActive] = accentBright;
        colors[ImGuiCol_Tab] = bg2;
        colors[ImGuiCol_TabHovered] = accentLight;
        colors[ImGuiCol_TabActive] = accentLight;
        colors[ImGuiCol_TabUnfocused] = bg1;
        colors[ImGuiCol_TabUnfocusedActive] = bg3;
        colors[ImGuiCol_PlotLines] = accentBright;
        colors[ImGuiCol_PlotLinesHovered] = orange;
        colors[ImGuiCol_PlotHistogram] = cyan;
        colors[ImGuiCol_PlotHistogramHovered] = green;
        colors[ImGuiCol_TableHeaderBg] = bg2;
        colors[ImGuiCol_TableBorderStrong] = border;
        colors[ImGuiCol_TableBorderLight] = bg3;
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.30f, 0.60f, 0.80f, 0.35f);
        colors[ImGuiCol_DragDropTarget] = yellow;
        colors[ImGuiCol_NavHighlight] = accentBright;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.35f);
    }
    break;
	case 3: // Mocha
    {
        // Corners
        style.WindowRounding = 6.0f;
        style.ChildRounding = 6.0f;
        style.FrameRounding = 4.0f;
        style.PopupRounding = 4.0f;
        style.ScrollbarRounding = 9.0f;
        style.GrabRounding = 4.0f;
        style.TabRounding = 4.0f;

        // Borders
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.TabBorderSize = 0.0f;

        // Padding and spacing
        style.WindowPadding = ImVec2(8.0f, 8.0f);
        style.FramePadding = ImVec2(5.0f, 3.0f);
        style.ItemSpacing = ImVec2(8.0f, 4.0f);
        style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        style.IndentSpacing = 21.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 10.0f;

        // Catppuccin Mocha Palette
        const ImVec4 base = ImVec4(0.117f, 0.117f, 0.172f, 1.0f);
        const ImVec4 mantle = ImVec4(0.109f, 0.109f, 0.156f, 1.0f);
        const ImVec4 surface0 = ImVec4(0.200f, 0.207f, 0.286f, 1.0f);
        const ImVec4 surface1 = ImVec4(0.247f, 0.254f, 0.337f, 1.0f);
        const ImVec4 surface2 = ImVec4(0.290f, 0.301f, 0.388f, 1.0f);
        const ImVec4 overlay0 = ImVec4(0.396f, 0.403f, 0.486f, 1.0f);
        const ImVec4 overlay2 = ImVec4(0.576f, 0.584f, 0.654f, 1.0f);
        const ImVec4 text = ImVec4(0.803f, 0.815f, 0.878f, 1.0f);
        const ImVec4 subtext0 = ImVec4(0.639f, 0.658f, 0.764f, 1.0f);
        const ImVec4 mauve = ImVec4(0.796f, 0.698f, 0.972f, 1.0f);
        const ImVec4 peach = ImVec4(0.980f, 0.709f, 0.572f, 1.0f);
        const ImVec4 yellow = ImVec4(0.980f, 0.913f, 0.596f, 1.0f);
        const ImVec4 green = ImVec4(0.650f, 0.890f, 0.631f, 1.0f);
        const ImVec4 teal = ImVec4(0.580f, 0.886f, 0.819f, 1.0f);
        const ImVec4 sapphire = ImVec4(0.458f, 0.784f, 0.878f, 1.0f);
        const ImVec4 blue = ImVec4(0.533f, 0.698f, 0.976f, 1.0f);
        const ImVec4 lavender = ImVec4(0.709f, 0.764f, 0.980f, 1.0f);

        colors[ImGuiCol_Text] = text;
        colors[ImGuiCol_TextDisabled] = subtext0;
        colors[ImGuiCol_WindowBg] = base;
        colors[ImGuiCol_ChildBg] = base;
        colors[ImGuiCol_PopupBg] = surface0;
        colors[ImGuiCol_Border] = surface1;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        colors[ImGuiCol_FrameBg] = surface0;
        colors[ImGuiCol_FrameBgHovered] = surface1;
        colors[ImGuiCol_FrameBgActive] = surface2;
        colors[ImGuiCol_TitleBg] = mantle;
        colors[ImGuiCol_TitleBgActive] = surface0;
        colors[ImGuiCol_TitleBgCollapsed] = mantle;
        colors[ImGuiCol_MenuBarBg] = mantle;
        colors[ImGuiCol_ScrollbarBg] = surface0;
        colors[ImGuiCol_ScrollbarGrab] = surface2;
        colors[ImGuiCol_ScrollbarGrabHovered] = overlay0;
        colors[ImGuiCol_ScrollbarGrabActive] = overlay2;
        colors[ImGuiCol_CheckMark] = green;
        colors[ImGuiCol_SliderGrab] = sapphire;
        colors[ImGuiCol_SliderGrabActive] = blue;
        colors[ImGuiCol_Button] = surface0;
        colors[ImGuiCol_ButtonHovered] = surface1;
        colors[ImGuiCol_ButtonActive] = surface2;
        colors[ImGuiCol_Header] = surface0;
        colors[ImGuiCol_HeaderHovered] = surface1;
        colors[ImGuiCol_HeaderActive] = surface2;
        colors[ImGuiCol_Separator] = surface1;
        colors[ImGuiCol_SeparatorHovered] = mauve;
        colors[ImGuiCol_SeparatorActive] = mauve;
        colors[ImGuiCol_ResizeGrip] = surface2;
        colors[ImGuiCol_ResizeGripHovered] = mauve;
        colors[ImGuiCol_ResizeGripActive] = mauve;
        colors[ImGuiCol_Tab] = surface0;
        colors[ImGuiCol_TabHovered] = surface2;
        colors[ImGuiCol_TabActive] = surface1;
        colors[ImGuiCol_TabUnfocused] = surface0;
        colors[ImGuiCol_TabUnfocusedActive] = surface1;
        colors[ImGuiCol_PlotLines] = blue;
        colors[ImGuiCol_PlotLinesHovered] = peach;
        colors[ImGuiCol_PlotHistogram] = teal;
        colors[ImGuiCol_PlotHistogramHovered] = green;
        colors[ImGuiCol_TableHeaderBg] = surface0;
        colors[ImGuiCol_TableBorderStrong] = surface1;
        colors[ImGuiCol_TableBorderLight] = surface0;
        colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.06f);
        colors[ImGuiCol_TextSelectedBg] = surface2;
        colors[ImGuiCol_DragDropTarget] = yellow;
        colors[ImGuiCol_NavHighlight] = lavender;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.7f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.2f);
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.35f);
    }
    break;
    case 4: // SimFluid
    {
        // Corners
        style.WindowRounding = 6.0f;
        style.ChildRounding = 6.0f;
        style.FrameRounding = 4.0f;
        style.PopupRounding = 4.0f;
        style.ScrollbarRounding = 9.0f;
        style.GrabRounding = 4.0f;
        style.TabRounding = 4.0f;

        // Borders
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.TabBorderSize = 0.0f;

        // Padding and spacing
        style.WindowPadding = ImVec2(8.0f, 8.0f);
        style.FramePadding = ImVec2(5.0f, 3.0f);
        style.ItemSpacing = ImVec2(8.0f, 4.0f);
        style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        style.IndentSpacing = 21.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 10.0f;

        // SimFluid palette
        const ImVec4 bg0 = ImVec4(0.05f, 0.07f, 0.09f, 1.00f); // darkest background
        const ImVec4 bg1 = ImVec4(0.12f, 0.15f, 0.19f, 1.00f); // bgVeryDark
        const ImVec4 bg2 = ImVec4(0.16f, 0.20f, 0.25f, 1.00f); // bgDark
        const ImVec4 bg3 = ImVec4(0.21f, 0.25f, 0.31f, 1.00f); // bgLight (frame bg)
        const ImVec4 bg4 = ImVec4(0.26f, 0.31f, 0.37f, 1.00f); // frame bg hover discreet
        const ImVec4 overlay0 = ImVec4(0.28f, 0.33f, 0.39f, 1.00f); // overlay discreet
        const ImVec4 overlay1 = ImVec4(0.36f, 0.42f, 0.49f, 1.00f); // overlay medium
        const ImVec4 textLight = ImVec4(0.93f, 0.94f, 0.95f, 1.00f); // main text
        const ImVec4 textDisabled = ImVec4(0.52f, 0.56f, 0.60f, 1.00f); // disabled text
        const ImVec4 accentDark = ImVec4(0.00f, 0.20f, 0.30f, 1.00f); // dark cyan-blue 
        const ImVec4 accent = ImVec4(0.00f, 0.28f, 0.40f, 1.00f); // main cyan-blue
        const ImVec4 accentLight = ImVec4(0.20f, 0.50f, 0.65f, 1.00f); // light cyan-blue
        const ImVec4 accentBright = ImVec4(0.35f, 0.65f, 0.80f, 1.00f); // bright cyan-blue
        const ImVec4 teal = ImVec4(0.20f, 0.70f, 0.75f, 1.00f); // complementary teal
        const ImVec4 yellow = ImVec4(0.85f, 0.78f, 0.30f, 1.00f); // yellow drag&drop
        const ImVec4 orange = ImVec4(0.80f, 0.50f, 0.20f, 1.00f); // orange plot hover
        const ImVec4 purple = ImVec4(0.45f, 0.40f, 0.75f, 1.00f); // separator hover
        const ImVec4 border = ImVec4(0.20f, 0.25f, 0.30f, 1.00f);

        colors[ImGuiCol_Text] = textLight;
        colors[ImGuiCol_TextDisabled] = textDisabled;
        colors[ImGuiCol_WindowBg] = bg1;
        colors[ImGuiCol_ChildBg] = bg2;
        colors[ImGuiCol_PopupBg] = bg2;
        colors[ImGuiCol_Border] = border;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg] = bg3;
        colors[ImGuiCol_FrameBgHovered] = bg4;
        colors[ImGuiCol_FrameBgActive] = overlay0;
        colors[ImGuiCol_TitleBg] = bg0;
        colors[ImGuiCol_TitleBgActive] = accent;
        colors[ImGuiCol_TitleBgCollapsed] = bg1;
        colors[ImGuiCol_MenuBarBg] = bg2;
        colors[ImGuiCol_ScrollbarBg] = bg0;
        colors[ImGuiCol_ScrollbarGrab] = accent;
        colors[ImGuiCol_ScrollbarGrabHovered] = accentLight;
        colors[ImGuiCol_ScrollbarGrabActive] = accentBright;
        colors[ImGuiCol_CheckMark] = accentBright;
        colors[ImGuiCol_SliderGrab] = accentBright;
        colors[ImGuiCol_SliderGrabActive] = accentBright;
        colors[ImGuiCol_Button] = bg3;
        colors[ImGuiCol_ButtonHovered] = accent;
        colors[ImGuiCol_ButtonActive] = accentLight;
        colors[ImGuiCol_Header] = ImVec4(0.00f, 0.28f, 0.40f, 0.40f);
        colors[ImGuiCol_HeaderHovered] = accent;
        colors[ImGuiCol_HeaderActive] = accentLight;
        colors[ImGuiCol_Separator] = border;
        colors[ImGuiCol_SeparatorHovered] = purple;
        colors[ImGuiCol_SeparatorActive] = purple;
        colors[ImGuiCol_ResizeGrip] = accent;
        colors[ImGuiCol_ResizeGripHovered] = accentLight;
        colors[ImGuiCol_ResizeGripActive] = accentBright;
        colors[ImGuiCol_Tab] = bg2;
        colors[ImGuiCol_TabHovered] = accentLight;
        colors[ImGuiCol_TabActive] = accentLight;
        colors[ImGuiCol_TabUnfocused] = bg1;
        colors[ImGuiCol_TabUnfocusedActive] = bg3;
        colors[ImGuiCol_PlotLines] = accentBright;
        colors[ImGuiCol_PlotLinesHovered] = orange;
        colors[ImGuiCol_PlotHistogram] = teal;
        colors[ImGuiCol_PlotHistogramHovered] = accentBright;
        colors[ImGuiCol_TableHeaderBg] = bg2;
        colors[ImGuiCol_TableBorderStrong] = border;
        colors[ImGuiCol_TableBorderLight] = bg3;
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 0.28f, 0.40f, 0.35f);
        colors[ImGuiCol_DragDropTarget] = yellow;
        colors[ImGuiCol_NavHighlight] = accentBright;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.35f);
    }
    break;
    }
}
void InitImGui()
{
    // Create the descriptor pool for ImGui
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * (uint32_t)(IM_ARRAYSIZE(pool_sizes));
    pool_info.poolSizeCount = (uint32_t)(IM_ARRAYSIZE(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;

    vkCreateDescriptorPool(g_Device->device, &pool_info, nullptr, &g_ImGuiDescriptorPool);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ApplyTheme();

    // Fonts
    io.FontDefault = io.Fonts->AddFontDefault();
    g_FontArial08 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 8.0f);
    g_FontArial10 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 10.0f);
    g_FontArial12 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 12.0f);
    g_FontArial14 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 14.0f);
    g_FontArial16 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 16.0f);
    g_FontArial20 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 20.0f); 
    g_FontArial24 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 24.0f);
    g_FontArial36 = io.Fonts->AddFontFromFileTTF("C://Windows/Fonts/Arial.ttf", 36.0f);
    g_FontCaveat36 = io.Fonts->AddFontFromFileTTF("Resources/Interface/Caveat.ttf", 36.0f);
    g_FontCaveat72 = io.Fonts->AddFontFromFileTTF("Resources/Interface/Caveat.ttf", 72.0f);
    io.Fonts->Build();

    // Backend GLFW
    ImGui_ImplGlfw_InitForVulkan(g_hWindow, true);

    // Backend Vulkan
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = g_Instance;
    init_info.PhysicalDevice = g_Device->physicalDevice;
    init_info.Device = g_Device->device;
    init_info.QueueFamily = g_Device->familyIndices.graphicsFamily.value();
    init_info.Queue = g_Device->graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = g_ImGuiDescriptorPool;
    init_info.MinImageCount = g_SwapChain->minImageCount;
    init_info.ImageCount = g_SwapChain->imageCount;
    init_info.Allocator = nullptr;
    init_info.PipelineInfoMain.RenderPass = g_RenderPass;
    init_info.PipelineInfoMain.Subpass = 1;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&init_info);

#ifdef INFO_INIT    // Structures.h
    wcout << L"-> ImGui Vulkan OK\n";
#endif // INFO_INIT
}

void ResetSph()
{
    // Rebuilds the particles for the selected scenario (and the obstacles, which rebuild the boundary particles)
    using Shape = SphSolver::SpawnShape;
    const vec3  bmin = g_SphSolver.BoxMin(), bmax = g_SphSolver.BoxMax();
    const vec3  size = bmax - bmin;
    const vec2  cylCenter(0.0f, 0.0f);
    const float cylRadius = 0.25f * size.z;                  // same as the display in InitScene()

    vector<Shape> shapes;
    bool cylinder = true;
    bool wave = false;

    switch ((eScenario)g_Scenario)
    {
    case eScenario::DamBreak:
        // Column against the left wall: half of the length, 83% of the height
        shapes.push_back(Shape::MakeBox(bmin, vec3(bmin.x + 0.5f * size.x, bmin.y + 0.83f * size.y, bmax.z)));
        break;

    case eScenario::FallingDrop:
        // Shallow pool + a sphere of water falling on it (off-center: no cylinder in this scenario)
        shapes.push_back(Shape::MakeBox(bmin, vec3(bmax.x, bmin.y + 0.6f, bmax.z)));
        shapes.push_back(Shape::MakeSphere(vec3(bmin.x + 0.5f * size.x + 1.5f, bmin.y + 2.0f, 0.5f * (bmin.z + bmax.z)), 0.4f));
        cylinder = false;
        break;

    case eScenario::WaveMaker:
        // Pool at rest, the piston on the left wall makes waves toward the cylinder
        shapes.push_back(Shape::MakeBox(bmin, vec3(bmax.x, bmin.y + 0.8f, bmax.z)));
        wave = true;
        break;

    default: break;
    }

    g_SphSolver.SetCylinder(cylCenter, cylinder ? cylRadius : 0.0f);
    g_SphSolver.SetWaveMaker(wave, g_WaveStroke, g_WavePeriod);
    g_SphSolver.Build(shapes);

    if (g_Cylinder)
        g_Cylinder->bVisible = cylinder;
}
void InitScene()
{
    // Camera
    g_Camera.SetProjection(45.0f, g_WindowW, g_WindowH, 0.1f, 300.0f);
    g_Camera.LookAt(vec3(0.0f, 3.5f, 11.0f), vec3(0.0f, 3.0f, 0.0f));
    g_Camera.SetSpeeds(0.001f, 0.0025f);
    g_Camera.SetMode(eCameraMode::ORBITAL);

    // Time
    g_Sky = make_unique<Sky>(g_InitialPosition, g_WindowW, g_WindowH);
    g_Sky->SetTime(12, 0);

    // Models
    g_Axis = make_unique<Model>(g_Device);
    g_Axis->LoadModel("Resources/Interface/Axis.glb");
    g_Axis->CreateMsPipeline(g_RenderPass, g_SwapChain->extent);
    g_Axis->bVisible = false;
    
    // Bbox: tank 8 m long, floor at y = 0
    g_Bbox = make_unique<Bbox>(g_Device);
    sBBox bbox(vec3(-4.0f, 0.0f, -1.3f), vec3(4.0f, 3.0f, 1.3f));
    g_Bbox->SetBbox(bbox);
    g_Bbox->CreateBboxPipeline(g_RenderPass, g_SwapChain->extent);
    g_Bbox->bVisible = true;

	// Cylinder: radius = 0.25 * height, height = bbox.max.z - bbox.min.z
	g_Cylinder = make_unique<Cylinder>(g_Device);
    const vec2  cylCenter(0.0f, 0.0f);
    const float cylRadius = 0.25f * (bbox.max.z - bbox.min.z);      // diameter = 20% of the width
    g_Cylinder->SetCylinder(cylCenter, cylRadius, bbox.min.y, bbox.max.y);
    g_Cylinder->CreateCylinderPipeline(g_RenderPass, g_SwapChain->extent);

    // Piston
    g_Piston = make_unique<Bbox>(g_Device);
    g_Piston->SetBbox(sBBox(vec3(bbox.min.x - 0.1f, bbox.min.y, bbox.min.z), vec3(bbox.min.x, bbox.max.y, bbox.max.z)));
    g_Piston->CreateBboxPipeline(g_RenderPass, g_SwapChain->extent);

    // Sph
    g_SphSolver = SphSolver(bbox.min, bbox.max);
    ResetSph();                                         // scenario + Build (no GPU object yet)
    g_SphSolver.InitVulkan(g_Device, g_RenderPass, g_SwapChain->extent);

    g_TimerScene.start();
}

// Cleanup
void PartialCleanup()
{
    // Wait for the device to be idle
    vkDeviceWaitIdle(g_Device->device);

    // Command buffers: InitVulkan() -> CreateCommandBuffers() allocates new ones after a resize
    if (!g_vCommandBuffers.empty())
    {
        vkFreeCommandBuffers(g_Device->device, g_Device->graphicsCommandPool, (uint32_t)g_vCommandBuffers.size(), g_vCommandBuffers.data());
        g_vCommandBuffers.clear();
    }

    // Destroy ImGui
    if (ImGui::GetCurrentContext())
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    if (g_ImGuiDescriptorPool)
    {
        vkDestroyDescriptorPool(g_Device->device, g_ImGuiDescriptorPool, nullptr);
        g_ImGuiDescriptorPool = VK_NULL_HANDLE;
    }

    // Destroy objects
    g_Sky.reset();

    // Framebuffers
    for (auto& fb : g_vFramebuffersSwapChain)
        vkDestroyFramebuffer(g_Device->device, fb, nullptr);
    g_vFramebuffersSwapChain.clear();
    
    // RenderPasses
    vkDestroyRenderPass(g_Device->device, g_RenderPass, nullptr);
    g_RenderPass = nullptr;

    // Images
    g_ColorImage.reset();
    g_DepthImage.reset();
    g_ColorImageResolve.reset();
    g_DepthImageResolve.reset();

    // Swapchain
    for (auto imageView : g_SwapChain->vImageViews)
        vkDestroyImageView(g_Device->device, imageView, nullptr);
    vkDestroySwapchainKHR(g_Device->device, g_SwapChain->swapChain, nullptr);

    // Sync objects
    for (size_t i = 0; i < g_vImageAvailableSemaphores.size(); i++)
    {
        vkDestroySemaphore(g_Device->device, g_vImageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(g_Device->device, g_vRenderFinishedSemaphores[i], nullptr);
        vkDestroyFence(g_Device->device, g_vImageFences[i], nullptr);
    }
    g_vImageAvailableSemaphores.clear();
    g_vRenderFinishedSemaphores.clear();
    g_vImageFences.clear();

    g_SwapChain.reset();
}
void RecreateScene()
{
    // Time
    g_Sky = make_unique<Sky>(g_InitialPosition, g_WindowW, g_WindowH);

    // Models
    g_Axis->RecreatePipelines(g_RenderPass, nullptr, nullptr, nullptr, g_SwapChain->extent);
    g_Bbox->RecreatePipelines(g_RenderPass, nullptr, nullptr, nullptr, g_SwapChain->extent);
    g_Cylinder->RecreatePipelines(g_RenderPass, g_SwapChain->extent);
    g_Piston->RecreatePipelines(g_RenderPass, nullptr, nullptr, nullptr, g_SwapChain->extent);
    g_SphSolver.RecreatePipelines(g_RenderPass, g_SwapChain->extent);

    g_TimerScene.start();
}
void PartialRecreate()
{
	/* If resize, only these objects must be recreated:
    Swapchain
    ├── ImageViews
    ├── Images (color, depth, resolve)
    ├── RenderPasses
    ├── Framebuffers
    ├── Pipelines (depend on the extent and the render pass)
    └── Sync objects / QueryPool

    The device, physical device, queues, command pools are never recreated.
    */

    PartialCleanup();

    InitVulkan();
    InitImGui();
    RecreateScene();

    g_iCurrentFrame = 0;
}
void FinalCleanup()
{
    glfwSetWindowShouldClose(g_hWindow, GLFW_TRUE);
    glfwPollEvents();

    if (g_Device) {
        vkQueueWaitIdle(g_Device->graphicsQueue);
        vkDeviceWaitIdle(g_Device->device);
    }

    // Scene objects (contain VkBuffer, etc.)
    g_Axis.reset();
	g_Bbox.reset();
	g_Cylinder.reset();
    g_Piston.reset();

    // Command buffers (free VkCommandBuffer)
    if (g_Device && !g_vCommandBuffers.empty()) {
        vkFreeCommandBuffers(g_Device->device, g_Device->graphicsCommandPool, (uint32_t)g_vCommandBuffers.size(), g_vCommandBuffers.data());
        g_vCommandBuffers.clear();
    }

    PartialCleanup();
    g_SwapChain.reset();

    // Surface (before destroying the device)
    if (g_Device && g_Device->surface) {
        vkDestroySurfaceKHR(g_Instance, g_Device->surface, nullptr);
        g_Device->surface = VK_NULL_HANDLE;
    }

    g_Device.reset();

    // Debug messenger (before destroying the instance)
    if (g_DebugMessenger) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func)
            func(g_Instance, g_DebugMessenger, nullptr);
        g_DebugMessenger = nullptr;
    }

    //vkDestroyInstance(g_Instance, nullptr);
    //glfwDestroyWindow(g_hWindow);
    //glfwTerminate();
    //return EXIT_SUCCESS;
   
    exit(0);
}

const char* SimBottleneck(double rt, double load, double dClip, bool capHit)
{
    // Which factor limits the real-time ratio? Only meaningful below 100%. Order matters: the first matching cause is reported.
   
    if (rt >= 98.0)
        return nullptr;                                 // real time reached, nothing to report
    if (dClip > 0.001)
        return "long frames clipped (maxUpdateDt)";     // frames longer than maxUpdateDt
    if (load > 85.0)
        return "GPU saturated";                         // the GPU cannot simulate faster
    if (capHit)
        return "sub-step cap (raise maxFrameMs)";       // cap reached while the GPU still has headroom
    return "unknown";
}
int main()
{
#ifdef CONSOLE
    InitConsole();
#endif

    try {
        InitWindow();
        CreateInstance();
        g_Device = make_shared<VulkanDevice>(g_Instance, g_hWindow);
        InitVulkan();
        InitImGui();
        InitScene();

        int prevWidth = 0, prevHeight = 0;
        glfwGetFramebufferSize(g_hWindow, &prevWidth, &prevHeight);

        static double prevTime = g_TimerScene.getTime();

        // Wall-clock reference, independent of g_TimerScene (detects a wrong timer scale)
        using Clock = std::chrono::steady_clock;
        static auto wallPrev = Clock::now();

        while (!glfwWindowShouldClose(g_hWindow))
        {
            glfwPollEvents();

            // Time
            double time = g_TimerScene.getTime();
            double dt = time - prevTime;
            prevTime = time;

            auto   wallNow = Clock::now();
            double wallDt = std::chrono::duration<double>(wallNow - wallPrev).count();
            wallPrev = wallNow;

            // Camera update
            dvec3 pos = dvec3(0.0, 0.0, 0.0);
            dvec3 p = dvec3(0.0, 0.0, 0.0);
            dvec3 t = dvec3(0.0, 0.0, 0.0);
            g_Camera.Animate(time, pos, p, t);

            UpdateForceCursor(g_bPause ? 0.0 : wallDt);

            if (g_PendingSphReset)
            {
                g_PendingSphReset = false;
                ResetSph();
                g_SphDt = 0.0f;   // do not simulate the time spent in the reset
            }

            if (!g_bPause)
            {
                static double wallAcc = 0.0;
                wallAcc += wallDt;

                // Real elapsed time: consumed by Update() when the frame is recorded in FrameRender()
                g_SphDt += (float)wallDt;

                if (wallAcc >= 1.0)
                {
                    wallAcc = 0.0;

                    static double prevReq = 0.0, prevSim = 0.0, prevClip = 0.0, prevGpu = 0.0;
                    const double req = g_SphSolver.RequestedSeconds();
                    const double sim = g_SphSolver.SimulatedSeconds();
                    const double clip = g_SphSolver.ClippedSeconds();
                    const double gpu = g_SphSolver.TotalGpuMs();
                    const double dReq = req - prevReq, dSim = sim - prevSim, dClip = clip - prevClip;
                    const double dGpu = gpu - prevGpu;
                    prevReq = req; prevSim = sim; prevClip = clip; prevGpu = gpu;

                    // Real-time ratio over the last second (simulated seconds per requested second)
                    const double rt = (dReq > 1e-6) ? 100.0 * dSim / dReq : 100.0;
                    // Fraction of the real time spent by the GPU on the simulation
                    const double load = (dReq > 1e-6) ? 100.0 * dGpu / (1000.0 * dReq) : 0.0;

                    const auto    st = g_SphSolver.LastStageTimes();
                    const double* sm = g_SphSolver.StageMs();
                    char line[256];

                    // Line 1: simulation state
                    snprintf(line, sizeof(line), "Simulation: N = %zu | realtime %.0f%% | %d fps",
                        g_SphSolver.Count(), rt, g_nFps);
                    g_SphSolverInfo = line;

                    // Limiting factor (only shown when the real-time ratio is below 100%)
                    if (const char* why = SimBottleneck(rt, load, dClip, g_SphSolver.CapHit()))
                    {
                        snprintf(line, sizeof(line), " | limit: %s", why);
                        g_SphSolverInfo += line;
                    }

                    // Simulated time lost because of long frames (maxUpdateDt clip)
                    if (dClip > 0.001)
                    {
                        snprintf(line, sizeof(line), " | clipped %.2f s", dClip);
                        g_SphSolverInfo += line;
                    }

                    // Line 2: time stepping and solver
                    snprintf(line, sizeof(line),
                        "\nSolver %s: substeps %d/%d%s | dt %.4f s | vmax %.1f m/s | iter dens %d div %d",
                        g_SphSolver.UsePcg() ? "PCG" : "Jacobi",
                        g_SphSolver.LastSubSteps(), g_SphSolver.SubStepLimit(),
                        g_SphSolver.CapHit() ? " (CAP)" : "",
                        g_SphSolver.LastDt(), g_SphSolver.MaxSpeed(), st.iterPressure, st.iterDivergence);
                    g_SphSolverInfo += line;

                    // Volume compression (the DFSPH paper reports a maximum of 0.01 %)
                    snprintf(line, sizeof(line), "\nDensity: max compression %.3f%%", 100.0f * g_SphSolver.MaxCompression());
                    g_SphSolverInfo += line;

                    // Line 3: everything about the GPU (load, cost per sub-step, breakdown by stage)
                    snprintf(line, sizeof(line),
                        "\nGPU: load %.0f%% | %.1f ms/step | forces %.2f | dens %.2f | adv %.2f | prep %.2f | div %.2f | fin %.2f (ms)",
                        load, g_SphSolver.MsPerStep(),
                        sm[0], sm[1], sm[2], sm[3], sm[4], sm[5]);
                    g_SphSolverInfo += line;

                    // Line 4: CPU cost of recording the commands
                    snprintf(line, sizeof(line), "\nCPU: record %.2f ms", g_SphSolver.AvgUpdateMs());
                    g_SphSolverInfo += line;
                }
            }

            // Pending actions
            if (g_PendingFullscreen)
            {
                g_PendingFullscreen = false;
                SwitchToFullScreen();
            }

            Render();
        }
    }
    catch (const exception& e) 
    { 
        cerr << e.what() << endl; 
        glfwDestroyWindow(g_hWindow);
        glfwTerminate(); 
        return EXIT_FAILURE; 
    }

    FinalCleanup();
}

#pragma warning(pop)

