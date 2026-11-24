#version 450
// Surface rendering, passes 1 and 2: one point per particle (sphere impostor), sized like the sphere on the screen.

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform SurfaceUBO {
    mat4 view;
    mat4 proj;
    mat4 invView;
    vec4 light;         // xyz = direction towards the light (world space)
    vec4 screen;        // x = width, y = height, z = 1 / width, w = 1 / height
    vec4 params0;       // x = impostor radius (m)
    vec4 absorption;    // rgb = absorption coefficients (1/m)
    vec4 waterColor;    // rgb
    vec4 params1;       // x = F0, y = specular power, z = exposure, w = thickness scale
} u;

layout(location = 0) out float vDepth;      // view-space distance of the center (positive)

void main()
{
    vec4 vp = u.view * vec4(inPosition, 1.0);
    vDepth      = -vp.z;
    gl_Position = u.proj * vp;

    // Diameter of the sphere in pixels
    float px = 2.0 * u.params0.x * abs(u.proj[1][1]) * u.screen.y * 0.5 / max(vDepth, 1e-3);
    gl_PointSize = clamp(px, 1.0, 128.0);
}
