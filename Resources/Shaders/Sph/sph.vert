#version 450

layout(location = 0) in vec3  inPosition;
layout(location = 1) in float inDensity;
layout(location = 2) in vec4  inVelInvS;     // xyz = velocity, w = DFSPH factor
layout(location = 3) in float inPressureS;   // s = p / rho^2 (warm[])
layout(location = 4) in uint  inNbrTotal;    // meta[i].z

layout(binding = 0) uniform SphUBO {
    mat4  view;
    mat4  proj;
    float pointSize;        // multiplier of the particle diameter (1 = spacing)
    float restDensity;
    float exposure;
    int   colorMode;
    float rangeMin;
    float rangeMax;
    float viewportHeight;   // pixels
    float particleDiameter; // world units (spacing)
} ubo;

layout(location = 0) out float fragT;       // normalized value in [0, 1]

float colorValue()
{
    switch (ubo.colorMode)
    {
    case 0: return inDensity;
    case 1: return length(inVelInvS.xyz);
    case 2: return inVelInvS.y;
    case 3: return inPosition.y;
    case 4: return inPressureS * inDensity * inDensity;
    case 5: return float(inNbrTotal);
    }
    return 0.0;
}

void main()
{
    gl_Position  = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    // Perspective size: world diameter projected on the screen, clamped to the device range
    float px = ubo.pointSize * ubo.particleDiameter * abs(ubo.proj[1][1]) * ubo.viewportHeight * 0.5 / max(gl_Position.w, 1e-3);
    gl_PointSize = clamp(px, 1.0, 64.0);
    fragT = clamp((colorValue() - ubo.rangeMin) / max(ubo.rangeMax - ubo.rangeMin, 1e-6), 0.0, 1.0);
}