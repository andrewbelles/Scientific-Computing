/*
 * Graph Theory Optimizer from Source Node to Output 
 * Some Rate of Resource is required. 
 * This program checks if a valid graph exists from source to Output
 * Then Finds the most optimal graph to maximize the output 
 *
 * Graph follows a directed acyclic graph 
 */

#include "hypergraph.hpp"
#include "swo.hpp"

// Testing 
int main(void)
{
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
  
  // OptimizedHypergraph fails to even compute the simple path of the computer problem. 
  // Lets define our SWO system now.
  
  // swo.hpp uses a different (better) class setup than hypergraph.hpp
  prod::Graph intermediate_graph;

  // Create source node and functors 
  intermediate_graph.set_target("Rotor");
  intermediate_graph.add_resource("Iron Ore", true, 480.0);
  intermediate_graph.add_functor("Iron Smelter", {{"Iron Ore", 30.0}}, {"Iron Ingot", 30.0});
  intermediate_graph.add_functor("Rod Constructor", {{"Iron Ingot", 15.0}}, {"Iron Rod", 15.0});
  intermediate_graph.add_functor("Plate Constructor", {{"Iron Ingot", 30.0}}, {"Iron Plate", 20.0});
  intermediate_graph.add_functor("Screw Constructor", {{"Iron Rod", 10.0}}, {"Screw", 40.0});
  intermediate_graph.add_functor("Rotor Assembler", {{"Iron Rod", 20.0}, {"Screw", 100.0}}, {"Rotor", 4.0});

  // Just using default hyperparameters for the moment
  SWO::SpiderWaspOptimizer rotors(intermediate_graph, 30);

  std::cout << "Seg fault isn't in constructor\n";

  prod::Solution solution = rotors.optimize(true);

  std::cout << "Seg fault isn't in optimize\n";

  rotors.print(solution); 
  std::cout << "End of source file\n";

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
