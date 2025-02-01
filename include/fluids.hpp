#ifndef __FLUIDS_HPP__
#define __FLUIDS_HPP__

/** Cpp headers to include **/
#include <iostream>
#include <cmath>
#include <memory>

#include "spatial.hpp"
#include "boundary.hpp"
#include "integrate.hpp"
#include "iterate.hpp"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "shader.hpp"
#include "meshes.hpp"
#include "SDL2/SDL.h"
#include "GL/glew.h"
#include "GL/gl.h"

/* Fluids namespace for initializations */
namespace flds {

/* Context class to set up SDL and OpenGL */
class Context {
public:
  SDL_Window *window_;
  SDL_GLContext context_;
  std::unique_ptr<shdr::Shader> shader_;

  // Initializes SDL2 and OpenGl and generates shaders from glsl files 
  Context(float cube_size) {
    // Initialize SDL2 Context
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
      std::cerr << "SDL2 Initialization Error: " << SDL_GetError() << std::endl;
      exit(1);
    }

    // std::cout << "SDL2 Initialized" << std::endl;

    // Create window, context, and use GLEW
    window_ = SDL_CreateWindow(
      "Particle Simulation",
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      1920,
      1080,
      SDL_WINDOW_OPENGL
    );

    // Check if window was created 
    if (window_ == nullptr) {
      std::cerr << "SDL2 Window Creation Error: " << SDL_GetError() << std::endl;
      exit(1);
    }

    // std::cout << "Window Created" << std::endl;

    // Create and check OpenGL context
    context_ = SDL_GL_CreateContext(window_);
    if (context_ == nullptr) {
      std::cerr << "OpenGL context failure: " << SDL_GetError() << std::endl;
      exit(1);
    }

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
      std::cerr << "Failed to initialize GLEW" << std::endl;
      exit(1);
    }

    // std::cout << "OpenGL Context Created" << std::endl;

    // Sets openGL calls for window
    glSetAttributes();

    // std::cout << "OpenGL Attributes Set" << std::endl;

    shader_ = std::make_unique<shdr::Shader>(
      "shaders/vertex_shader.glsl",
      "shaders/fragment_shader.glsl",
      glm::vec3(cube_size / 2.0, cube_size / 2.0, cube_size * 3.0),
      glm::vec3(cube_size / 2.0, cube_size / 2.0, cube_size / 2.0),
      glm::vec3(0.0, 1.0, 0.0),
      glm::vec3(0.0, cube_size + 1.0, 0.0),
      glm::vec3(1.0, 1.0, 1.0)
    );
  }

  /**** Simply returns the float value from command-line arg ****/
  static float parse(char *argv[], const int index) {
    return atof(argv[index]);   
  }

  /* Sets attributes required for SDL context */
  static void glSetAttributes() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    glEnable(GL_DEPTH_TEST);
  }

  /* Default destructor for context */
  ~Context() {
    SDL_GL_DeleteContext(context_);
    SDL_DestroyWindow(window_);
    SDL_Quit();
  }
};

} // End namespace flds

#endif // __FLUIDS_HPP__
