#version 330 core

layout(location = 0) in vec2 inPos;
uniform mat4 uProj;

void main() {
    // project position into clip‐space
    gl_Position = uProj * vec4(inPos, 0.0, 1.0);
    // draw as little squares/points
    gl_PointSize = 10.0;
}
