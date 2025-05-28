#ifndef __SHADER_HPP__
#define __SHADER_HPP__

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include "glm/gtc/type_ptr.hpp"

extern "C" {
 #include "GL/glew.h"
 #include "GL/gl.h"
}

namespace shdr {

/**** Lighting Class to set lighting ****/
class Lighting {
 public:

  Lighting(glm::vec3 position, glm::vec3 color) : position_(position), color_(color) {}

  // Get
  glm::vec3 GetPosition() const { return position_; }
  glm::vec3 GetColor() const { return color_; }

  // Set
  void SetPosition(const glm::vec3& position) { position_ = position; } 
  void SetColor(const glm::vec3& color) { color_ = color; } 

 private:
  glm::vec3 position_, color_; 
};

/**** Camera Class composed of Lighting and Shader class for dynamic camera ****/
class Camera {
 public:
  Lighting lighting_;

  Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up, glm::vec3 light_position, glm::vec3 light_color) 
    : lighting_(light_position, light_color), position_(position), target_(target), up_(up) {}

  // Get
  glm::vec3 GetPosition() const { return position_; }
  glm::vec3 GetTarget() const { return target_; }
  glm::vec3 GetUp() const { return up_; }
  glm::mat4 GetProjection() const { return projection_; }
  glm::mat4 GetView() const { return view_; }

  // Set
  void SetPosition(const glm::vec3& position) { position_ = position; } 
  void SetTarget(const glm::vec3& target) { target_ = target; } 
  void SetUp(const glm::vec3& up) { up_ = up; } 

  void SetViewMatrix() {
    view_ = glm::lookAt(position_, target_, up_);
  }

  void SetProjectionMatrix(float fov, float aspect_ratio, float near_plane, float far_plane) {
    projection_ = glm::perspective(glm::radians(fov), aspect_ratio, near_plane, far_plane);
  }

 private:
  glm::vec3 position_, target_, up_;
  glm::mat4 projection_, view_;
};

/**** Class for shader program required to draw shapes ****/
class Shader {
 public:
  // Shader program should be the only public variable
  Camera camera_;
  GLuint shader_program; 

  // No empty constructor: Constructor should generate shader_program from src files
  Shader(const char *vertex_path, const char *fragment_path, glm::vec3 position, glm::vec3 target, glm::vec3 up, glm::vec3 light_position, glm::vec3 light_color) 
    : camera_(position, target, up, light_position, light_color) {
    const char *vertex_code, *fragment_code;
    std::string vertex_src = pull_shaders(vertex_path);
    std::string fragment_src = pull_shaders(fragment_path);
    GLint success = 0;
    GLchar infoLog[512];

    vertex_code = vertex_src.c_str();
    fragment_code = fragment_src.c_str();

    // std::cout << "Shaders Pulled" << std::endl;

    vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_code);
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_code);

    // std::cout << "Shaders Compiled" << std::endl;

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);

    // std::cout << "Shader Program Created" << std::endl;

    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
      std::cerr << "Shader linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // std::cout << "Shaders Deleted" << std::endl;
  }

  void SetCamera(Camera camera) { camera_ = camera; }

  // Object method to use program
  void Use() {
    glUseProgram(shader_program);
  }

  void SetMat4(const std::string &name, const glm::mat4 &mat) {
    glUniformMatrix4fv(glGetUniformLocation(shader_program, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat)); 
  }

  void SetVec3(const std::string &name, const glm::vec3 &vec) {
    glUniform3f(glGetUniformLocation(shader_program, name.c_str()), vec.x, vec.y, vec.z);
  }

  void Render(float fov, float aspect_ratio, float near_plane, float far_plane, glm::vec3& object_color, glm::mat4& model) {
    Use();

    SetVec3("lighting_position", camera_.lighting_.GetPosition());
    SetVec3("view_position", camera_.GetPosition());
    SetVec3("light_color", camera_.lighting_.GetColor());
    SetVec3("object_color", object_color);

    camera_.SetViewMatrix();
    camera_.SetProjectionMatrix(fov, aspect_ratio, near_plane, far_plane);

    SetMat4("model", model);
    SetMat4("view", camera_.GetView());
    SetMat4("projection", camera_.GetProjection());
  }

 // Vertex and fragment shaders should be inaccesible outside class individually 
 private:
  GLuint vertex_shader, fragment_shader;

  /**** Simple function to read shader source code from filepath and return the string output */
  std::string pull_shaders(const char *src_filepath) {
    std::ifstream file(src_filepath);
    std::stringstream ss;

    // Checks for open file
    if (!file.is_open()) {
      std::cerr << "Invalid File Path: " << src_filepath << std::endl;
      return "";
    }                                                   // Return gracefully

    // Fill stream stream
    ss << file.rdbuf();
    // Return source code
    return ss.str();
  }

  // Compiles individually passed shader
  GLuint compile_shader(GLenum type, const char* shader_code) {
    GLint success = 0;
    GLchar infoLog[512];

    // std::cout << "In Shader Compile" << std::endl;

    GLuint shader = glCreateShader(type);
    if (shader == 0) {
      std::cerr << "Shader Creation Failed" << std::endl;
      exit(1);
    }

    // std::cout << "Shader Created" << std::endl;
    
    // std::cout << "Shader precompile" << std::endl;

    glShaderSource(shader, 1, &shader_code, nullptr);

    // std::cout << "Shader sourced" << std::endl;

    glCompileShader(shader);

    // std::cout << "Shader Compiled" << std::endl;

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 512, NULL, infoLog);
      std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }

    return shader;              // Returns compiled shader
  }
};

} // namespace shdr

#endif // __SHADER_HPP__
