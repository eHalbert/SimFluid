#version 450
// Surface rendering, pass 4 (inside the main render pass): normals from the smoothed depth, then shading:
// Fresnel (Schlick) between the reflected sky and the refracted background attenuated by the thickness (Beer-Lambert),
// plus a specular highlight. gl_FragDepth is written so that the scene geometry still occludes the fluid.

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

layout(binding = 1) uniform sampler2D uDepth;       // smoothed linear view depth (0 = no fluid)
layout(binding = 2) uniform sampler2D uThickness;   // thickness (m)

layout(location = 0) out vec4 outColor;

float fetchDepth(ivec2 p)
{
    ivec2 hi = textureSize(uDepth, 0) - ivec2(1);
    return texelFetch(uDepth, clamp(p, ivec2(0), hi), 0).r;
}

// View-space position of the pixel p whose linear depth is d
vec3 viewPos(ivec2 p, float d)
{
    vec2 ndc = (vec2(p) + 0.5) * u.screen.zw * 2.0 - 1.0;
    return vec3(ndc.x * d / u.proj[0][0], ndc.y * d / u.proj[1][1], -d);
}

// Procedural environment seen through / reflected by the water
vec3 sky(vec3 dir)
{
    vec3 horizon = vec3(0.70, 0.80, 0.92);
    vec3 zenith  = vec3(0.20, 0.40, 0.80);
    vec3 ground  = vec3(0.25);
    float t = sqrt(clamp(abs(dir.y), 0.0, 1.0));
    return (dir.y > 0.0) ? mix(horizon, zenith, t) : mix(horizon, ground, t);
}

void main()
{
    ivec2 c = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(c);
    if (d <= 0.0)
        discard;

    // Normal: finite differences on the side where the depth changes the least (keeps the silhouettes clean)
    vec3 P = viewPos(c, d);
    float dr = fetchDepth(c + ivec2(1, 0)), dl = fetchDepth(c - ivec2(1, 0));
    float du = fetchDepth(c + ivec2(0, 1)), dd = fetchDepth(c - ivec2(0, 1));

    vec3 dx = vec3(1.0, 0.0, 0.0);
    if (dr > 0.0 && dl > 0.0)
        dx = (abs(dr - d) < abs(d - dl)) ? viewPos(c + ivec2(1, 0), dr) - P : P - viewPos(c - ivec2(1, 0), dl);
    else if (dr > 0.0)
        dx = viewPos(c + ivec2(1, 0), dr) - P;
    else if (dl > 0.0)
        dx = P - viewPos(c - ivec2(1, 0), dl);

    vec3 dy = vec3(0.0, 1.0, 0.0);
    if (du > 0.0 && dd > 0.0)
        dy = (abs(du - d) < abs(d - dd)) ? viewPos(c + ivec2(0, 1), du) - P : P - viewPos(c - ivec2(0, 1), dd);
    else if (du > 0.0)
        dy = viewPos(c + ivec2(0, 1), du) - P;
    else if (dd > 0.0)
        dy = P - viewPos(c - ivec2(0, 1), dd);

    vec3 N = normalize(cross(dx, dy));
    vec3 V = normalize(-P);
    if (dot(N, V) < 0.0)
        N = -N;                                     // always face the camera

    // Fresnel (Schlick)
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    float F    = u.params1.x + (1.0 - u.params1.x) * pow(1.0 - cosT, 5.0);

    // Reflection and refraction (air -> water), directions converted to world space
    vec3 reflW = normalize((u.invView * vec4(reflect(-V, N), 0.0)).xyz);
    vec3 refrW = normalize((u.invView * vec4(refract(-V, N, 1.0 / 1.33), 0.0)).xyz);
    vec3 refl  = sky(reflW);
    vec3 bg    = sky(refrW);

    // Absorption along the path in the water: the deeper, the closer to the water color
    float thick = texelFetch(uThickness, c, 0).r;
    vec3  trans = exp(-u.absorption.rgb * thick);
    vec3  Lv    = normalize(mat3(u.view) * u.light.xyz);
    float ndl   = max(dot(N, Lv), 0.0);
    vec3  body  = mix(u.waterColor.rgb * (0.7 + 0.3 * ndl), bg, trans);

    // Specular highlight (Blinn-Phong)
    vec3  H    = normalize(Lv + V);
    float spec = pow(max(dot(N, H), 0.0), u.params1.y) * (0.3 + F);

    vec3 col = mix(body, refl, F) + vec3(spec);
    outColor = vec4(col * u.params1.z, 1.0);

    // Depth of the (smoothed) surface, for the occlusion by the scene geometry
    float zv = -d;
    gl_FragDepth = (u.proj[2][2] * zv + u.proj[3][2]) / (u.proj[2][3] * zv);
}
