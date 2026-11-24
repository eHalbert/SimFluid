#version 450
// Surface rendering, pass 3: separable bilateral filter of the linear depth.
// The radius in pixels follows the world radius (it shrinks with the distance); the range weight keeps the silhouettes.
// A depth of 0 means "no fluid": such texels are ignored and stay 0.

layout(binding = 0) uniform sampler2D uSrc;

layout(push_constant) uniform PC {
    vec2  dir;              // (1, 0) or (0, 1)
    float filterRadius;     // m
    float projScale;        // pixels per meter at a distance of 1 m
    float depthSigma;       // m
    float maxRadius;        // pixels
    vec2  pad;
} pc;

layout(location = 0) out float outDepth;

void main()
{
    ivec2 p  = ivec2(gl_FragCoord.xy);
    ivec2 hi = textureSize(uSrc, 0) - ivec2(1);
    float dc = texelFetch(uSrc, p, 0).r;
    if (dc <= 0.0)
    {
        outDepth = 0.0;
        return;
    }

    float rpx   = clamp(pc.filterRadius * pc.projScale / dc, 1.0, pc.maxRadius);
    float sigS  = max(rpx * 0.5, 0.5);
    int   n     = int(ceil(rpx));
    ivec2 step_ = ivec2(pc.dir);

    float sum = 0.0, wsum = 0.0;
    for (int i = -n; i <= n; ++i)
    {
        float dn = texelFetch(uSrc, clamp(p + step_ * i, ivec2(0), hi), 0).r;
        if (dn <= 0.0)
            continue;
        float ws = exp(-0.5 * float(i * i) / (sigS * sigS));
        float dd = (dn - dc) / pc.depthSigma;
        float w  = ws * exp(-0.5 * dd * dd);
        sum  += dn * w;
        wsum += w;
    }
    outDepth = sum / max(wsum, 1e-8);
}
