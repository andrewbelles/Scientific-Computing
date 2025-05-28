#version 330 core

uniform vec3 light_position;
uniform vec3 view_position;
uniform vec3 light_color;
uniform vec3 object_color;

in vec3 frag_position;
in vec3 normal;

out vec4 frag_color;

void main() {
  float ambient_strength = 1.1;
  vec3 ambient = ambient_strength * light_color;

  vec3 norm = normalize(normal);
  vec3 light_direction = normalize(light_position - frag_position);
  float diff = max(dot(norm, light_direction), 0.0);
  vec3 diffuse = diff * light_color;

  float specular_strength = 0.6;
  vec3 view_direction = normalize(view_position - frag_position);
  vec3 reflect_direction = reflect(-light_direction, norm);
  float shininess = 35.0;
  float spec = pow(max(dot(view_direction, reflect_direction), 0.0), shininess);
  vec3 specular = specular_strength * spec * light_color;

  vec3 result = (ambient + diffuse + specular) * object_color;
  frag_color = vec4(result, 1.0);
}