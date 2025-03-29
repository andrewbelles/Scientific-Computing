/*
 * Graph Theory Optimizer from Source Node to Output 
 * Some Rate of Resource is required. 
 * This program checks if a valid graph exists from source to Output
 * Then Finds the most optimal graph to maximize the output 
 *
 * Graph follows a directed acyclic graph 
 */
#define nohypr_
#ifndef nohypr_
#include "hypergraph.hpp"
#endif
#include "swo.hpp"

// Testing 
int main(void)
{
#ifndef nohypr_
  // Define our source and functors  
  hypr::Hypergraph graph = hypr::Hypergraph("SI-Simple Graph");
  // Produces 120 of X 
  int Source = graph.addResourceNode("X", true, 120.0);
  int XtrY = graph.addTransformation("G", {{"X", 60.0}}, {"Y", 30.0}); 
  int YtrA = graph.addTransformation("H", {{"Y", 30.0}}, {"A", 10.0});

  // Print the simplest graph that solves our problem space 
  std::vector<hypr::PathStep> simple_path = graph.search_simple();
  print_graph(graph, simple_path, "Unoptimal Solution");

  // Multiple Input Graph 
  hypr::Hypergraph migraph = hypr::Hypergraph("MI-Intermediate Graph");

  int Source_X = migraph.addResourceNode("X", true, 240.0);
  int Source_Y = migraph.addResourceNode("Y", true, 120.0);

  int XtrZ = migraph.addTransformation("G", {{"X", 60.0}}, {"W", 15.0});
  int YtrZ = migraph.addTransformation("H", {{"W", 30.0}, {"Y", 60.0}}, {"A", 10.0});

  std::vector<hypr::PathStep> misimple_path = migraph.search_simple();
  print_graph(migraph, misimple_path, "Unoptimal Solution");

  std::vector<hypr::PathStep> optimal_graph = migraph.search_optimal();
  print_graph(migraph, optimal_graph, "Optimal Solution");

  // Hypergraph cannot solve these problems 

  hypr::OptimizedHypergraph optimized_rotor = hypr::OptimizedHypergraph("OptimizedHypergraph - Rotor Factory");

  optimized_rotor.addResourceNode("Iron Ore", true, 480.0);  
  
  optimized_rotor.addTransformation("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  
  optimized_rotor.addTransformation("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  optimized_rotor.addTransformation("Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  optimized_rotor.addTransformation("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});
  
  optimized_rotor.addTransformation("Rotor Assembler", {{"Iron Rod", 20.0}, {"Screw", 100.0}}, {"A", 4.0});    

  std::vector<hypr::PathStep> optimized_simple = optimized_rotor.search_simple();
  print_graph(optimized_rotor, optimized_simple, "Simple Solution");

  std::vector<hypr::PathStep> optimized_optimal = optimized_rotor.search_optimal();
  print_graph(optimized_rotor, optimized_optimal, "Optimal Solution");
#endif
  // OptimizedHypergraph takes significant time for even simple supply chains 
  // swo.hpp uses a different (better) class setup than hypergraph.hpp (Among numerous other improvements)
  
  // Easy Supply Chain Problems 
  // Rotors
  prod::Graph rotor_graph;

  // Create source node and functors 
  rotor_graph.set_target("Rotor");
  rotor_graph.add_resource("Iron Ore", true, 480.0);
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

  hmf_graph.add_resource("Iron Ore", true, 240.0);
  hmf_graph.add_resource("Coal", true, 90.0);
  hmf_graph.add_resource("Limestone", true, 90.0);
  
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
                        {{"Steel Beam", 4.0}, {"Concrete", 5.0}},
                        {"Encased Industrial Beam", 1.0});

  // Final Manufacturer 
  hmf_graph.add_functor("Heavy Modular Frame Manufacturer", 
                        {{"Modular Frame", 5.0}, {"Steel Pipe", 15.0},
                         {"Encased Industrial Beam", 5.0}, {"Screw", 100.0}},
                        {"Heavy Modular Frame", 1.0});

  SWO::SpiderWaspOptimizer heavy_modular_frames(hmf_graph, 50, {0.3, 0.2}, {0.8, 0.6, 0.3, 0.4}, {0.3, 0.75, 0.65, 0.15}, 4500, 1500);
  prod::Solution hmf_solution = heavy_modular_frames.optimize(true);

  heavy_modular_frames.print_stats();
  heavy_modular_frames.print(hmf_solution);

  /*
  OptimizedHypergraph computer_factory = OptimizedHypergraph("Computer Factory");
  
  // Add source nodes (raw resources)
  int iron_ore = computer_factory.addResourceNode("Iron Ore", true, 240.0);
  int copper_ore = computer_factory.addResourceNode("Copper Ore", true, 120.0);
  int coal = computer_factory.addResourceNode("Coal", true, 120.0);
  int caterium_ore = computer_factory.addResourceNode("Caterium Ore", true, 120.0);
  int crude_oil = computer_factory.addResourceNode("Crude Oil", true, 120.0);
  
  computer_factory.addTransformation("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  computer_factory.addTransformation("Copper Smelter", {{"Copper Ore", 30.0}}, {"Copper Ingot", 30.0});
  computer_factory.addTransformation("Steel Foundry", {{"Iron Ore", 45.0}, {"Coal", 45.0}}, {"Steel Ingot", 45.0});
  computer_factory.addTransformation("Caterium Smelter", {{"Caterium Ore", 30.0}}, {"Caterium Ingot", 15.0});
  computer_factory.addTransformation("Plastic Refinery", {{"Crude Oil", 30.0}}, {"Plastic", 20.0});
  computer_factory.addTransformation("Rubber Refinery", {{"Crude Oil", 30.0}}, {"Rubber", 20.0});
  
  computer_factory.addTransformation("Iron Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  computer_factory.addTransformation("Iron Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  computer_factory.addTransformation("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});
  computer_factory.addTransformation("Copper Sheet Constructor", {{"Copper Ingot", 20.0}}, {"Copper Sheet", 10.0});
  computer_factory.addTransformation("Wire Constructor", {{"Copper Ingot", 15.0}}, {"Wire", 30.0});
  computer_factory.addTransformation("Cable Constructor", {{"Wire", 20.0}}, {"Cable", 10.0});
  computer_factory.addTransformation("Quickwire Constructor", {{"Caterium Ingot", 12.0}}, {"Quickwire", 60.0});
  
  computer_factory.addTransformation("Reinforced Plate Assembler", {{"Iron Plate", 30.0}, {"Screw", 60.0}}, {"Reinforced Iron Plate", 5.0});
  computer_factory.addTransformation("Steel Pipe Constructor", {{"Steel Ingot", 15.0}}, {"Steel Pipe", 20.0});
  computer_factory.addTransformation("Steel Beam Constructor", {{"Steel Ingot", 20.0}}, {"Steel Beam", 10.0});
  computer_factory.addTransformation("Circuit Board Assembler", {{"Copper Sheet", 15.0}, {"Plastic", 30.0}}, {"Circuit Board", 7.5});

  computer_factory.addTransformation("Computer Manufacturer", 
                                    {{"Circuit Board", 25.0}, {"Cable", 22.5}, {"Plastic", 45.0}, {"Screw", 130.0}}, 
                                    {"A", 2.5});  

  std::vector<PathStep> simple_computer_path = computer_factory.search_simple();
  print_graph(computer_factory, simple_computer_path, "Simple Path");

  std::vector<PathStep> optimal_computer_path = computer_factory.search_optimal();
print_graph(computer_factory, simple_computer_path, "Optimal Path");
*/
  return 0;
}
