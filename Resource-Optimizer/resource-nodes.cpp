/*
 * Graph Theory Optimizer from Source Node to Output 
 * Some Rate of Resource is required. 
 * This program checks if a valid graph exists from source to Output
 * Then Finds the most optimal graph to maximize the output 
 *
 * Graph follows a directed acyclic graph 
 */
#include "swo.hpp"

// Testing 
int main(void)
{
  // OptimizedHypergraph takes significant time for even simple supply chains 
  // swo.hpp uses a different (better) class setup than hypergraph.hpp (Among numerous other improvements)
  
  prod::Graph reinforced_graph;
  reinforced_graph.set_target("Reinforced Iron Plate");
  reinforced_graph.add_resource("Iron Ore", true, 120.0);
  reinforced_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  reinforced_graph.add_functor("Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  reinforced_graph.add_functor("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  reinforced_graph.add_functor("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});

  reinforced_graph.add_functor("Reinforced Plate Assembler", 
                                  {{"Iron Plate", 30.0}, {"Screw", 60.0}},
                                  {"Reinforced Iron Plate", 5.0});

  SWO::SpiderWaspOptimizer reinforced(reinforced_graph, 30, {0.3, 0.1}, {0.8, 0.4, 0.4, 0.3}, {0.4, 0.7, 0.6, 0.1}, 500, 250);
  prod::Solution reinforced_solution = reinforced.optimize(true);

  reinforced.print(reinforced_solution);

  // Easy Supply Chain Problems 
  // Rotors
  prod::Graph rotor_graph;

  // Create source node and functors 
  rotor_graph.set_target("Rotor");
  rotor_graph.add_resource("Iron Ore", true, 120.0);
  rotor_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  rotor_graph.add_functor("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  rotor_graph.add_functor("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});
  rotor_graph.add_functor("Rotor Assembler", {{"Iron Rod", 20.0}, {"Screw", 100.0}}, {"Rotor", 4.0});

  // Just using default hyperparameters for the moment
  SWO::SpiderWaspOptimizer rotors(rotor_graph, 20, {0.25, 0.1}, {0.8, 0.4, 0.4, 0.3}, {0.35, 0.7, 0.6, 0.1}, 500, 250);
  prod::Solution rotor_solution = rotors.optimize(true);

  // rotors.print_stats();
  rotors.print(rotor_solution); 

  // Modular Frame Graph 
  prod::Graph modular_frame_graph;
  modular_frame_graph.set_target("Modular Frame");
  modular_frame_graph.add_resource("Iron Ore", true, 480.0);

  // Basic processing 
  modular_frame_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  modular_frame_graph.add_functor("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  modular_frame_graph.add_functor("Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  modular_frame_graph.add_functor("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});

  // Intermediate processing 
  modular_frame_graph.add_functor("Reinforced Plate Assembler", 
                                  {{"Iron Plate", 30.0}, {"Screw", 60.0}},
                                  {"Reinforced Iron Plate", 5.0});

  // Final assembly
  modular_frame_graph.add_functor("Modular Frame Assembler",
                                  {{"Reinforced Iron Plate", 3.0}, {"Iron Rod", 12.0}},
                                  {"Modular Frame", 2.0});

  SWO::SpiderWaspOptimizer modular_frames(modular_frame_graph, 30, {0.25, 0.1}, {0.8, 0.4, 0.4, 0.3}, {0.35, 0.7, 0.6, 0.1}, 1500, 500);
  prod::Solution frame_solution = modular_frames.optimize(true);

  // modular_frames.print_stats();
  modular_frames.print(frame_solution);

  // Advanced Supply Chain Problems
  // Heavy Modular Frames 
  prod::Graph hmf_graph;
  hmf_graph.set_target("Heavy Modular Frame");

  hmf_graph.add_resource("Iron Ore", true, 640.0);
  hmf_graph.add_resource("Coal", true, 205.0);
  hmf_graph.add_resource("Limestone", true, 240.0);
  
  // Basic processing 
  hmf_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  hmf_graph.add_functor("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  hmf_graph.add_functor("Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  hmf_graph.add_functor("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});
  hmf_graph.add_functor("Steel Smelter", {{"Iron Ore", 45.0}, {"Coal", 45.0}}, {"Steel Ingot", 45.0});
  hmf_graph.add_functor("Steel Pipe Constructor", {{"Steel Ingot", 30.0}}, {"Steel Pipe", 20.0});
  hmf_graph.add_functor("Steel Beam Constructor", {{"Steel Ingot", 60.0}}, {"Steel Beam", 15.0});
  hmf_graph.add_functor("Concrete Constructor", {{"Limestone", 45.0}}, {"Concrete", 15.0});

  // Intermediate processing
  hmf_graph.add_functor("Reinforced Plate Assembler", 
                        {{"Iron Plate", 30.0}, {"Screw", 60.0}}, 
                        {"Reinforced Iron Plate", 5.0});

  hmf_graph.add_functor("Modular Frame Assembler",
                        {{"Reinforced Iron Plate", 3.0}, {"Iron Rod", 12.0}},
                        {"Modular Frame", 2.0});

  hmf_graph.add_functor("Encased Beam Assembler", 
                        {{"Steel Beam", 24.0}, {"Concrete", 30.0}},
                        {"Encased Industrial Beam", 6.0});

  // Final Manufacturer 
  hmf_graph.add_functor("Heavy Modular Frame Manufacturer", 
                        {{"Modular Frame", 10.0}, {"Steel Pipe", 30.0},
                         {"Encased Industrial Beam", 10.0}, {"Screw", 200.0}},
                        {"Heavy Modular Frame", 2.0});

  SWO::SpiderWaspOptimizer heavy_modular_frames(hmf_graph, 50, {0.3, 0.2}, {0.8, 0.6, 0.3, 0.4}, {0.3, 0.75, 0.65, 0.15}, 1500, 500);
  prod::Solution hmf_solution = heavy_modular_frames.optimize(true);

  heavy_modular_frames.print_stats();
  heavy_modular_frames.print(hmf_solution);

prod::Graph computer_graph;
computer_graph.set_target("Computer");

// Raw resources
computer_graph.add_resource("Iron Ore", true, 360.0);
computer_graph.add_resource("Copper Ore", true, 360.0);
computer_graph.add_resource("Coal", true, 240.0);
computer_graph.add_resource("Plastic", true, 120.0);   
computer_graph.add_resource("Rubber", true, 120.0);
computer_graph.add_resource("Quartz", true, 120.0);  
computer_graph.add_resource("Caterium Ore", true, 120.0);  
  
// Basic processing
computer_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
computer_graph.add_functor("Copper Smelter", {{"Copper Ore", 30.0}}, {"Copper Ingot", 30.0});
computer_graph.add_functor("Steel Smelter", {{"Iron Ore", 45.0}, {"Coal", 45.0}}, {"Steel Ingot", 45.0});
computer_graph.add_functor("Caterium Smelter", {{"Caterium Ore", 30.0}}, {"Caterium Ingot", 15.0});
computer_graph.add_functor("Quartz Crystal Constructor", {{"Quartz", 37.5}}, {"Quartz Crystal", 22.5});

// Component manufacturing
computer_graph.add_functor("Wire Constructor", {{"Copper Ingot", 15.0}}, {"Wire", 30.0});
computer_graph.add_functor("Cable Constructor", {{"Wire", 60.0}}, {"Cable", 30.0});
computer_graph.add_functor("Iron Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
computer_graph.add_functor("Steel Beam Constructor", {{"Steel Ingot", 60.0}}, {"Steel Beam", 15.0});
computer_graph.add_functor("Quickwire Constructor", {{"Caterium Ingot", 12.0}}, {"Quickwire", 60.0});

// Circuit boards
computer_graph.add_functor("Circuit Board Assembler", 
                          {{"Copper Sheet", 15.0}, {"Plastic", 30.0}}, 
                          {"Circuit Board", 7.5});
computer_graph.add_functor("Copper Sheet Constructor", 
                          {{"Copper Ingot", 20.0}}, 
                          {"Copper Sheet", 10.0});

// High-level components
computer_graph.add_functor("AI Limiter Assembler", 
                          {{"Copper Sheet", 5.0}, {"Quickwire", 20.0}}, 
                          {"AI Limiter", 5.0});

computer_graph.add_functor("Crystal Oscillator Assembler",
                          {{"Quartz Crystal", 36.0}, {"Cable", 28.0}, {"Reinforced Iron Plate", 5.0}},
                          {"Crystal Oscillator", 2.0});

computer_graph.add_functor("Reinforced Iron Plate Assembler",
                          {{"Iron Plate", 30.0}, {"Screw", 60.0}},
                          {"Reinforced Iron Plate", 5.0});

computer_graph.add_functor("Screw Constructor",
                          {{"Iron Rod", 10.0}},
                          {"Screw", 40.0});

computer_graph.add_functor("Iron Rod Constructor",
                          {{"Iron Ingot", 15.0}},
                          {"Iron Rod", 15.0});

// Final assembly
computer_graph.add_functor("Computer Manufacturer", 
                          {{"Circuit Board", 10.0}, {"Cable", 9.0}, 
                           {"Plastic", 18.0}, {"Steel Beam", 5.0}}, 
                          {"Computer", 1.0});

// Correct alternative recipe for Crystal Computers
computer_graph.add_functor("Crystal Computer Manufacturer", 
                          {{"Circuit Board", 8.0}, {"Crystal Oscillator", 3.0}}, 
                          {"Computer", 3.0});

SWO::SpiderWaspOptimizer computer_factory(computer_graph, 40, {0.3, 0.2}, 
                                          {0.8, 0.5, 0.4, 0.3}, 
                                          {0.4, 0.7, 0.6, 0.2}, 
                                          2000, 500);

prod::Solution computer_solution = computer_factory.optimize(true);

computer_factory.print_stats();
computer_factory.print(computer_solution);

  return 0;
}
