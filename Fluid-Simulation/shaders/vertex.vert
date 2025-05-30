#version 330 core
layout(location = 0) in vec2 inPos;
uniform mat4 uProj;
uniform float pointSize;    

void main() {
    gl_Position = uProj * vec4(inPos, 0.0, 1.0);
    gl_PointSize = pointSize;
}
