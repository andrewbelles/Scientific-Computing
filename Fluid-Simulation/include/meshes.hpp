#ifndef MESHES_HPP
#define MESHES_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "shader.hpp"

extern "C" {
 #include "SDL2/SDL.h"
 #include "GL/glew.h"
 #include "GL/gl.h"
}

namespace mesh {

/**** Cube Mesh Class from cube size ****/
class Cube_GL {
 public:
  float size;
  std::vector<GLfloat> vertices;
  std::vector<GLuint> indices, buffers;
  glm::vec3 object_color_;

  // Empty constructor and instantiating constructor
  Cube_GL() : size(0), vertices(24, 0), indices(24, 0), buffers(3, 0) {}
  Cube_GL(float cube_size) : size(cube_size), vertices(24, 0), indices(24, 0), buffers(3, 0) {
    create_cube_vertices(size);
    create_cube_indices();
    cube_buffers();
    object_color_ = {1.0, 1.0, 1.0};
  }

  /**** Draws wireframe cube given defn ****/
  void DrawCube() {
    glLineWidth(2.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(buffers[0]);
    glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }
 
 // Vertices, Indices, and buffers can only be modified within the Cube_GL class
 private:
  /**** Creates the vertices from <0,0,0> to <size,size,size> ****/
  void create_cube_vertices(float size) {
    vertices = {
      0.0F, 0.0F, 0.0F,
      size, 0.0F, 0.0F,
      size, size, 0.0F,
      0.0F, size, 0.0F,
      0.0F, 0.0F, size,
      size, 0.0F, size,
      size, size, size,
      0.0F, size, size
    };
  }

  /**** Standard cubic indices ****/
  void create_cube_indices() {
    indices = {
      0, 1, 1, 2, 2, 3, 3, 0, // bottom face
      4, 5, 5, 6, 6, 7, 7, 4, // top face
      0, 4, 1, 5, 2, 6, 3, 7  // connecting edges
    };
  }
  
  /**** Sets buffers appropriately ****/
  void cube_buffers() {
    glGenVertexArrays(1, &buffers[0]);
    glGenBuffers(1, &buffers[1]);
    glGenBuffers(1, &buffers[2]);

    glBindVertexArray(buffers[0]);

    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
  }
};

class Sphere_GL {
 public:
  float radius;
  std::vector<GLfloat> vertices;
  std::vector<GLuint> indices, buffers;
  glm::vec3 object_color_;

  Sphere_GL(int stack, int slice, float radius) : radius(radius), buffers(3, 0) {
    create_sphere_vertices(stack, slice);
    create_sphere_indices(stack, slice);
    sphere_buffers();
    object_color_ = glm::vec3(0.1, 0.0, 0.75);
  }

  // Set the color based on density
  void setColor(glm::vec3 density_derived_color) {
    object_color_ = density_derived_color;
  }

  void DrawSphere() {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindVertexArray(buffers[0]);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

 private:
  void create_sphere_vertices(int stack, int slice) {
    float V, phi, U, theta, x, y, z;
    glm::vec3 normal;

    // Calculate the x, y, z coordinate of vertex based on spherical coordinates
    for (int i = 0; i < stack; i++) {
      
      V = i / (float)stack;
      phi = V * glm::pi<float>();

      for (int j = 0; j < slice; j++) {
      
        U = j / (float)slice;
        theta = U * (glm::pi<float>() * 2);

        // Spherical jacobian applied to cartesian coordinates
        x = cos(theta) * sin(phi);
        y = cos(phi);
        z = sin(theta) * sin(phi);

        // Store the vertex position
        vertices.push_back(radius * x);
        vertices.push_back(radius * y);
        vertices.push_back(radius * z);

        // Store normals
        normal = glm::normalize(glm::vec3(x, y, z));
        vertices.push_back(normal.x);
        vertices.push_back(normal.y);
        vertices.push_back(normal.z);
      }
    }
  }

  void create_sphere_indices(int stack, int slice) {
    int first, second;
    for (int i = 0; i < stack; i++) {
      for (int j = 0; j < slice; j++) {
        first = (i * (slice + 1)) + j;
        second = first + slice + 1;
        
        // Fill indices for each iteration
        indices.push_back(first);
        indices.push_back(second);
        indices.push_back(first + 1);
        indices.push_back(second);
        indices.push_back(second + 1);
        indices.push_back(first + 1);
      }
    }
  }

  void sphere_buffers() {
    // Generate vao, vbo, ebo buffers
    glGenVertexArrays(1, &buffers[0]);
    glGenBuffers(1, &buffers[1]);
    glGenBuffers(1, &buffers[2]);

    // Bind vao, vbo, ebo buffers
    glBindVertexArray(buffers[0]);

    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind
    glBindVertexArray(0);
  }
};

} // namespace mesh

#endif // MESHES_HPP
