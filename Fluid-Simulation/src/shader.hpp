#include <iostream>
#include <fstream>
#include <GL/glew.h>
#include <raylib.h>
#include <cuda_gl_interop.h>

namespace shdr {

// Read all data from shader file into return value 
// Credit ChatGPT o4-mini-high
std::string load_shader(const std::string& path)
{
  std::ifstream shader_file(path, std::ios::in | std::ios::binary);
  if (!shader_file)
    throw std::runtime_error("File doesn't exist");

  std::string shader; 
  shader_file.seekg(0, std::ios::end);
  shader.resize(shader_file.tellg());
  shader_file.seekg(0, std::ios::beg);
  shader_file.read(&shader[0], shader.size());
  shader_file.close();

  return shader;
}


// Compile the shader
GLuint compile_shader(GLenum type, const std::string& src)
{
  char error[512];
  GLint status;
  GLuint shader = glCreateShader(type);
  const char* cstr = src.c_str();

  glShaderSource(shader, 1, &cstr, nullptr);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status)
  {
    glGetShaderInfoLog(shader, 512, nullptr, error); 
    throw std::runtime_error(error);
  }

  return shader;
}


// Creates the program from file paths to fragment and vertex shaders 
GLuint create_program(const std::string& fragment_path, const std::string& vertex_path)
{
  // Get shaders from source and compile 
  std::string vertex_src   = load_shader(vertex_path);
  std::string fragment_src = load_shader(fragment_path);
  GLuint vertex_shader   = compile_shader(GL_VERTEX_SHADER, vertex_src);
  GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_src);
  GLuint program = glCreateProgram();

  GLint status;
  char error[512];

  // Attach shaders to program 
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (!status)
  {
    glGetProgramInfoLog(program, 512, nullptr, error);
    glDeleteProgram(program);
    throw std::runtime_error(error);
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return program;
}

}
