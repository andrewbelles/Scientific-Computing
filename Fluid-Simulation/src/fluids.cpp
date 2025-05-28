#include "fluids.hpp"
#include "integrate.hpp"

#define alpha 1.5
#define rho0 1000

// #define __debug
//#define _step

glm::vec3 densityColor(float density, float target_density);

// Start of main
int main(int argc, char *argv[]) 
{
  // Check argc 
  if (argc != 3) {
    std::cerr << "Invalid Input count" << std::endl;
    std::cerr << "Usage: [particle count] [cube size]" << std::endl;
    return 1;
  }

  // Set command line inputs 
  int particle_count = static_cast<int>(flds::Context::parse(argv, 1));
  float cube_size = flds::Context::parse(argv, 2);

  // std::cout << "Particle Count: " << particle_count << "\nCube Size: " << cube_size << std::endl;

  // Start context
  flds::Context context_(cube_size);
  // std::cout << "Context Created" << std::endl;

  // Local variables
  SDL_Event event;
  glm::vec3 center(cube_size / 2.0, cube_size / 2.0, cube_size / 2.0), position(1.0);
  glm::vec3 view_position = glm::vec3(cube_size / 2.0, cube_size / 2.0, cube_size * 10.0);
  glm::vec3 density_derived_color(0);
  glm::mat4 model(1.0), cube_model(1.0), sphere_model(1.0);
  bool lmb_down = false, initial_mouse = true;
 
  // Set initial values
  float yaw = 0.0, pitch = 0.0, fov = 60.0;
  float near_plane = 0.1, far_plane = 100.0, aspect_ratio = 1280.0 / 720.0;
  float last_mouse_x = 0.0, last_mouse_y = 0.0, x_offset = 0.0, y_offset = 0.0, sensititivity = 5e-5;
  float h;
  float radius;

  // Set the smoothing_radius and model radius 

  // Create shape meshes
  // Container defn
  std::vector<float> container(3, cube_size);

  h = 1.2 * (pow(static_cast<float>(container[0] * container[1] * container[2] / particle_count), 1.0/3.0));
  h *= scale_factor;  
  std::cout << "Smoothing Radius: " << h << '\n';

  radius = h;
  mesh::Sphere_GL sphere_gl(40, 40, radius);
  mesh::Cube_GL cube_gl(cube_size);

  // Initialize the simulation
  uint32_t partition_count = 0;
  Lookup *d_lookup_ = nullptr;
  particleContainer *d_objs_ = nullptr;

  // Unified positions
  float *u_positions = nullptr;
  float *u_densities = nullptr;

  initOffsetTable();

  // Call to initialization
  initalizeSimulation(
    &d_lookup_,
    &d_objs_,
    container,
    &partition_count,
    particle_count,
    h
  ); 
#ifdef __debug 
  std::cout << "Simulation Initialized\n";
#endif

  // Update loop
  int iter = 0;
  bool first = true;
  while (true) {
    // Clear color
    glClearColor(0.0, 0.0, 0.0, 1.0);
#ifdef _step 
    char step;
    std::cout << "step...\n";
    std::cin >> step; 
#endif
    // std::cout << "Clear Color" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // std::cout << "Pre-Iterator" << std::endl;
    if (iter == 0) {
      first = true;
      iter++;
    } else {
      first = false;
    }

    // Update Position State
    particleIterator(
      d_objs_,
      u_positions,
      u_densities,
      d_lookup_,
      container,
      particle_count,
      partition_count,
      h,
      first
    );

#ifdef __debug
    std::cout << "Iterated over particles\n";
#endif
    // Polls for events in SDL instance
    while (SDL_PollEvent(&event)) {
      // Quit program out
      if (event.type == SDL_QUIT) {
        exit(0);
      }
      // Checks for specific keypresses
      switch (event.type) {
        case SDL_KEYDOWN:
          // Handle Key events
          switch (event.key.keysym.sym) {
            // Escape exit
            case SDLK_ESCAPE:
              exit(0);
              break;
            // FOV sliders c and v keys
            case SDLK_c:
              fov += 1;
              break;
            case SDLK_v:
              fov -= 1;
              break;
            default:
              break;
          }
        case SDL_MOUSEBUTTONDOWN:
          if (event.button.button == SDL_BUTTON_LEFT) {
            lmb_down = true;
          }
          break;
        case SDL_MOUSEBUTTONUP:
          if (event.button.button == SDL_BUTTON_LEFT) {
            lmb_down = false;
            initial_mouse = true;
          }
          break;
        // Handle Moving Mouse
        case SDL_MOUSEMOTION:
          // Only handle if left mouse down is true
          if (lmb_down) {
            // Set initial mouse position
            if (initial_mouse) {
              last_mouse_x = event.motion.x;
              last_mouse_y = event.motion.y;
              initial_mouse = false;
            }

            // Calculate Mouse offsets
            x_offset = (event.motion.x - last_mouse_x) * sensititivity;
            y_offset = (event.motion.y - last_mouse_y) * sensititivity;

            // Find yaw and pitch 
            yaw -= x_offset;
            pitch += y_offset;
            pitch = (pitch > 89.0) ? 89.0 : pitch;
            pitch = (pitch < -89.0) ? -89.0 : pitch;

            // Set View Matrix
            cube_model = glm::translate(model, center);
            cube_model = glm::rotate(cube_model, yaw, glm::vec3(0.0, 1.0, 0.0));
            cube_model = glm::rotate(cube_model, pitch, glm::vec3(1.0, 0.0, 0.0));
            cube_model = glm::translate(cube_model, -center);
          }
          break;
        // Default case
        default:
          break;
      }

    }

    // std::cout << "Pre-render Call" << std::endl;

    context_.shader_->Render(fov, aspect_ratio, near_plane, far_plane, cube_gl.object_color_, cube_model);
    cube_gl.DrawCube();
#ifdef __debug
    std::cout << "Cube Draw Call" << std::endl;
#endif
    for (int idx = 0; idx < particle_count; ++idx) {
      //std::cout << "Test: " << u_positions[idx * 3];

      position = glm::vec3(
        u_positions[idx * 3],
        u_positions[idx * 3 + 1],
        u_positions[idx * 3 + 2]
      );

      sphere_model = glm::translate(cube_model, position);
      // std::cout << "Density: " << u_densities[idx] << '\n';

      density_derived_color = densityColor(u_densities[idx], rho0);
      sphere_gl.setColor(density_derived_color);

      context_.shader_->camera_.lighting_.SetPosition(center + glm::vec3(0.0, 5.0 + center.y, 0.0));
      context_.shader_->Render(fov, aspect_ratio, near_plane, far_plane, sphere_gl.object_color_, sphere_model);
      sphere_gl.DrawSphere();
    }
#ifdef __debug
    std::cout << "Sphere Draw Call" << std::endl;
#endif
    // Swap buffers
    SDL_GL_SwapWindow(context_.window_);
  }

  // Free gpu memory
  delete (d_objs_);
  cudaFree(u_positions);
  cudaFree(d_lookup_);
  
  return 0;
}
// End of main

glm::vec3 densityColor(float density, float target_density) {
  float red = 0.0, blue = 0.0, green = 0.0;
  float interp = 0.0;

  // Set high low flags 
  bool high = (density > 10.0 * target_density);
  bool low  = (density < 0.1 * target_density);

  // Set fully to blue or red if exceedinly high or low
  blue = (density < low) ? 1.0 : blue;
  red = (density > high) ? 1.0 : red;

  // If exceedingly high or low valued return early
  if (low || high) return glm::vec3(red, green, blue);

  // Find normalizing value
  interp = (density - (0.75 * target_density)) / (2 * 0.75);

  // Generate an interpolated color for closer ranges
  if (interp < 0.5) {
    blue = 1.0 - 2.0 * interp;
    green = 2.0 * interp;
  } else {
    green = 2.0 * (1.0 - interp);
    red   = 2.0 * (interp - 0.5);
  }

  return glm::vec3(red, blue, green);
}
