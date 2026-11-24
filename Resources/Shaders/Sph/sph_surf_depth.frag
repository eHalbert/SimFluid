#version 450
// Surface rendering, pass 1: linear view depth of the front of the sphere + z-test (gl_FragDepth).

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
    vec4 params1;
} u;

layout(location = 0) out float outDepth;

void main()
{
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0)
        discard;

    float nz = sqrt(1.0 - r2);
    float d  = vDepth - nz * u.params0.x;       // distance of the sphere surface along the view axis
    outDepth = d;

    // Same depth as the rasterizer would give for the point (view-space z = -d)
    float zv = -d;
    gl_FragDepth = (u.proj[2][2] * zv + u.proj[3][2]) / (u.proj[2][3] * zv);
}
