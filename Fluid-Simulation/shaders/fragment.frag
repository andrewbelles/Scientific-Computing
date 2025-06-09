#version 330 core

in float vRhoNorm;    // from vertex shader
out vec4 fragColor;

void main()
{
    vec3 col;
    if (vRhoNorm < 0.5) {
       // map [0..0.5] → [blue → green]
       float t = vRhoNorm / 0.5;
       col = mix(vec3(0.0, 0.0, 1.0),  // blue
                 vec3(0.0, 1.0, 0.0),  // green
                 t);
    } else {
       // map [0.5..1] → [green → red]
       float t = (vRhoNorm - 0.5) / 0.5;
       col = mix(vec3(0.0, 1.0, 0.0),  // green
                 vec3(1.0, 0.0, 0.0),  // red
                 t);
    }
    fragColor = vec4(col, 1.0); 
}
