#version 450
// Surface rendering, pass 2: thickness of the fluid along the view ray (chord of each sphere, additive blending).

layout(location = 0) in float vDepth;

layout(binding = 0) uniform SurfaceUBO {
    mat4 view;
    mat4 proj;
    mat4 invView;
    vec4 light;
    vec4 screen;
    vec4 params0;       // x = impostor radius (m)
    vec4 absorption;
    vec4 waterColor;
    vec4 params1;       // w = thickness scale
} u;

layout(location = 0) out float outThickness;

void main()
{
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0)
        discard;

    outThickness = 2.0 * u.params0.x * sqrt(1.0 - r2) * u.params1.w;
}
