#version 330 core

layout(location = 0) in vec2 inPos;   // position (x,y)
layout(location = 1) in float inRho;  // density

uniform mat4 uProj;       // projection matrix
uniform float uRhoMin;    // e.g. 800.0
uniform float uRhoMax;    // e.g. 1200.0

out float vRhoNorm;       // normalized density → [0,1]

void main()
{
    // 1) Project position:
    gl_Position = uProj * vec4(inPos, 0.0, 1.0);

    // 2) Normalize density to [0..1]
    float t = clamp((inRho - uRhoMin) / (uRhoMax - uRhoMin), 0.0, 1.0);
    t = pow(t, 0.5);
    vRhoNorm = t;
}
