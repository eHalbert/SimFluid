#version 450

layout(location = 0) in float fragT;

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

layout(location = 0) out vec4 outColor;

// Hue ramp: blue (t = 0) -> cyan -> green -> yellow -> red (t = 1)
vec3 rampBlueRed(float t)
{
    float hue = (1.0 - clamp(t, 0.0, 1.0)) * (2.0 / 3.0);     // 240 deg -> 0 deg
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(hue) + K.xyz) * 6.0 - K.www);
    return clamp(p - K.xxx, 0.0, 1.0);                         // full saturation, full value
}

void main()
{
    // Round point
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0)
        discard;

    vec3 col = rampBlueRed(fragT);

    // Soft shading like a sphere
    float shade = sqrt(1.0 - r2);
    outColor = vec4(col * shade * ubo.exposure, 1.0);
}
