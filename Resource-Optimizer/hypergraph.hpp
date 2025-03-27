#ifndef HYPERGRAPH_HPP
#define HYPERGRAPH_HPP

#include <sstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <iostream>

namespace hypr {

// Forward class declarations 
class ResourceGraph;
class Hypergraph;

struct ResourceAmount
{
  std::string resource_type;
  double amount;
};

// Defines a connection from one node to another
struct Edge
{
  // src / dest node ids 
  int src;
  int dest;
  // Define the amount of resources and what kinds 
  std::vector<ResourceAmount> inputs_required;
  ResourceAmount output;
};

// Defines a single node in the graph
struct Node 
{
  int id; 
  std::string resource_type;
  std::vector<Edge> connections;
  // Unique to if a Node generates some resource 
  bool is_source;
  double source_production;
};

// Defines some functor that can map n inputs onto a single output from nodes  
struct Transformation
{
  int id; 
  std::string functor_name;
  std::vector<std::pair<int, double>> inputs;
  std::pair<int, double> output;
};

struct PathStep
{
  // Stores raw ids of inputs, what functor applied to them, and the output node id 
  int transformation_id; 
  std::vector<int> input_nodes; 
  int output_node;
};

class ResourceGraph
{
private:
  std::vector<Node> nodes;
  std::unordered_map<std::string, std::vector<int>> type_to_nodes;
  std::unordered_set<int> source_ids;   // Unique set of resource nodes 
  int target;

public:
  ResourceGraph() { target = addNode("A", false, 0.0); }

  int addNode(const std::string& resource_type, const bool& is_source, const double& source_production)
  {
    // nth node 
    int id = nodes.size();
    nodes.push_back({id, resource_type, {}, is_source, source_production});
    type_to_nodes[resource_type].push_back(id);

    // Insert to set if node is a source node
    if (is_source)
    {
      source_ids.insert(id);
    }
  
    return id;
  }

  void add_edge(int& src, int& dest, std::vector<ResourceAmount>& inputs, ResourceAmount& output)
  {
    // Add a connection using the src id to the dest id 
    nodes[src].connections.push_back({src, dest, inputs, output});
  }

  // Get from map 
  std::vector<int> get_nodes_of_type(const std::string& resource_type)
  {
    return type_to_nodes[resource_type];   // From map 
  }

  // Simple Getters 
  int get_target() const { return target; }
  const Node& get_node(const int& id) const { return nodes[id]; }
  bool is_source_node(const int& id) const { return nodes[id].is_source; }
  std::vector<Node> get_all_nodes() const { return nodes; }

  std::vector<int> get_nodes_of_type(const std::string& type) const
  {
    auto it = type_to_nodes.find(type);
    if (it != type_to_nodes.end())
    {
      return it->second;
    }
    return {};
  }

  std::vector<int> get_source_ids() const { return std::vector<int>(source_ids.begin(), source_ids.end()); }
};

// Define the hypergraph class to be optimized 
class Hypergraph
{
protected:
  const double tol{1e-6};
  ResourceGraph resources_;
  std::vector<Transformation> transformations;

  // Initialization 
  // Create map of resource pool 
  std::unordered_map<int, double> initialize_search() const
  {
    std::unordered_map<int, double> resources;
    const auto& source_ids = resources_.get_source_ids();

    // Get each source node and enter it into map 
    for (const int& sid : source_ids)
    {
      const Node& source_node = resources_.get_node(sid);
      resources[sid] = source_node.source_production;
    }
    return resources;
  }

  // Generate a key for a unique key based on a resources state/quantity left 
  virtual std::string generate_resource_key(const std::unordered_map<int, double>& resources) const
  {
    std::string key;
    for (const auto& [node, quantity] : resources)
    {
      if (quantity > tol)
        key += std::to_string(node) + ":" + std::to_string(quantity) + "|";
    }
    return key;
  }

  // Boolean check to see if T can map resource X to Y under T: X -> Y
  bool can_apply_transformations(const Transformation& T,
                                 const std::unordered_map<int, double>& resources)
  {
    for (const auto& [input, amount] : T.inputs)
    {
      auto it = resources.find(input);
      if (it == resources.end() || it->second < amount - tol)
        return false;
    }
    return true;
  }

  // Apply transformation T to some resource X to map to Y; T: X -> Y  
  std::unordered_map<int, double> apply_transformation(const Transformation& T,
                                                       const std::unordered_map<int, double>& resources) const
  {
    std::unordered_map<int, double> new_resources(resources);

    // Remove resources from pool
    for (const auto& [input, amount] : T.inputs)
      new_resources[input] -= amount;

    // Insert the type of resource output by a functor into pool
    int output_id = T.output.first;
    double output_amount = T.output.second;
    new_resources[output_id] += output_amount;

    return new_resources;
  }

  // Create a single path step for a transformation 
  PathStep create_step(size_t T_idx, const Transformation& T) const
  {
    std::vector<int> inputs;
    inputs.reserve(T.inputs.size());

    // Add every input to the path step that were used in transformation T 
    for (const auto& [input, _] : T.inputs)
      inputs.push_back(input);

    // Return the full PathStep
    return {static_cast<int>(T_idx), inputs, T.output.first};
  }

public:
  const std::string name;
  // Explicit call to empty constructor to composed ResourceGraph class
  Hypergraph(const std::string& name) : resources_(), name(name) {}

  int addResourceNode(const std::string& resource_type, const bool& is_source = false, const double& production = 0.0) 
  { 
    return resources_.addNode(resource_type, is_source, production);
  }

  int get_or_create_node(const std::string& resource_type)
  {
    std::vector<int> existing_nodes = resources_.get_nodes_of_type(resource_type);
    // If existing_nodes is a non-empty vector then return the first id 
    if (existing_nodes.empty() == false)
    {
      return existing_nodes[0];
    }
    // If there are no nodes of the specified type return the id to the newly create node
    return addResourceNode(resource_type);
  }

  int addTransformation(const std::string& name, 
                        const std::vector<std::pair<std::string, double>>& inputs,
                        std::pair<std::string, double> output)
  {
    int id = transformations.size();
    
    // Convert string resource types to their raw id 
    std::vector<std::pair<int, double>> input_node_ids;
    for (const auto& [resource_type, amount] : inputs)
    {
      int node_id = get_or_create_node(resource_type);
      input_node_ids.push_back({node_id, amount});
    }

    int output_id = get_or_create_node(output.first);

    transformations.push_back({id, name, input_node_ids, {output_id, output.second}});
    return id;
  }

  // Getters 
  const Node& get_node(const int& id) const { return resources_.get_node(id); }
  int get_target() const { return resources_.get_target(); }
  std::vector<int> get_source_ids() const { return resources_.get_source_ids(); }
  const std::vector<Node> get_all_nodes() const { return resources_.get_all_nodes(); }
  const Transformation& get_transformation(const int& id) const { return transformations[id]; }
  const std::vector<Transformation> get_transformations() const { return transformations; }

  // Graph optimization methods 
    
  virtual std::vector<PathStep> search_simple()
  {
    std::unordered_map<int, double> initial_resources = initialize_search();

    using state_tuple = std::tuple<int, std::vector<PathStep>, std::unordered_map<int, double>>;
    std::queue<state_tuple> queue;

    queue.push({0, {}, initial_resources});

    std::unordered_map<std::string, int> best_state;

    const int target = resources_.get_target();
    std::vector<PathStep> best_path;
    int best_total = std::numeric_limits<int>::max();

    while (!queue.empty())
    {
      auto [total_transformations, current_path, current_resources] = queue.front();
      queue.pop();

      if (total_transformations >= best_total) continue;

      if (current_resources[target] > tol)
      {
        best_path = current_path;
        best_total = total_transformations;
        continue;
      }

      std::string resource_key = generate_resource_key(current_resources);

      if (best_state.find(resource_key) != best_state.end() &&
          best_state[resource_key] <= total_transformations)
        continue;

      best_state[resource_key] = total_transformations;

      for (size_t i = 0; i < transformations.size(); i++)
      {
        const auto& transformation = transformations[i];

        if (!can_apply_transformations(transformation, current_resources))
          continue;

        std::unordered_map<int, double> new_resources = apply_transformation(transformation, current_resources);

        std::vector<PathStep> new_path = current_path;
        new_path.push_back(create_step(i, transformation));

        queue.push({total_transformations + 1, new_path, new_resources});
      }
    }

    return best_path;
  }

  virtual std::vector<PathStep> search_optimal()
  {
    // Check if solution exists 
    if (search_simple().empty())
      return {};
    
    std::unordered_map<int, double> initial_resources = initialize_search();
    using state_tuple = std::tuple<double, int, std::vector<PathStep>, std::unordered_map<int, double>>;

    // Custom comparator for map within tuple 
    auto map_comparator = [](const state_tuple& a, const state_tuple& b)
      { return std::get<0>(a) < std::get<0>(b); };

    std::priority_queue<state_tuple, std::vector<state_tuple>, decltype(map_comparator)> queue(map_comparator);

    queue.push({0.0, 0, {}, initial_resources});

    std::unordered_map<std::string, std::pair<double, int>> best_state;

    const int target = resources_.get_target();

    std::vector<PathStep> best_path;
    double best_output = 0.0;
    int best_total_transformations = std::numeric_limits<int>::max();

    // Constrain to prevent infinite loops 
    const int max_transformations = 250; 

    while (!queue.empty())
    {
      auto [output_quantity, total_transformations, current_path, current_resources] = queue.top();
      queue.pop();

      if (total_transformations >= max_transformations) continue;

      if (current_resources[target] > best_output + tol)
      {
        best_output = current_resources[target];
        best_path   = current_path; 
        best_total_transformations = total_transformations;
      }
      else if (std::abs(current_resources[target] - best_output) < tol &&
          total_transformations < best_total_transformations)
      {
        best_path = current_path;
        best_total_transformations = total_transformations;
      }

      std::string resource_key = generate_resource_key(current_resources);

      auto it = best_state.find(resource_key);
      if (it != best_state.end())
      {
        auto [prev_output, prev_transformations] = it->second;
        if (prev_output > output_quantity + tol ||
            (std::abs(prev_output - output_quantity) < tol &&
             prev_transformations <= total_transformations))
          continue;
      }

      best_state[resource_key] = {output_quantity, total_transformations};

      for (size_t i = 0; i < transformations.size(); i++)
      {
        const auto& transformation = transformations[i];

        double max_applications = std::numeric_limits<double>::max();
        for (const auto& [input, amount] : transformation.inputs)
        {
          if (amount > tol)
            max_applications = std::min(max_applications, current_resources[input] / amount);
        }
      
        int applications = static_cast<int>(max_applications);
        if (applications < 1) continue;

        for (int application_count : {1, applications})
        {
          if (application_count > applications) continue;

          std::unordered_map<int, double> new_resources = current_resources;

          for (const auto& [input, amount] : transformation.inputs)
            new_resources[input] -= amount * application_count;    

          int output_id = transformation.output.first;
          double output_amount = transformation.output.second * applications;
          new_resources[output_id] += output_amount;

          std::vector<PathStep> new_path = current_path;

          for (int j = 0; j < application_count; j++)
            new_path.push_back(create_step(i, transformation));

          double new_output_quantity = new_resources[target];

          queue.push({new_output_quantity, total_transformations + application_count, new_path, new_resources});
        }
      }
    }

    return best_path;
  }
};

class OptimizedHypergraph : public Hypergraph
{
private:
  std::unordered_map<std::string, std::pair<double, std::vector<PathStep>>> cache_optimal;
  std::unordered_map<std::string, std::pair<int, std::vector<PathStep>>> cache_simple;

  std::string generate_resource_key(const std::unordered_map<int, double>& resources) const override 
  {
    std::vector<std::pair<int, double>> sorted_resources;
    for (const auto& [id, amount] : resources)
    {
      if (amount > tol)
        sorted_resources.push_back({id, std::round(amount * 1000)});

    }
    std::sort(sorted_resources.begin(), sorted_resources.end());

    std::ostringstream oss;
    for (const auto& [id, amount] : sorted_resources)
    {
      oss << id << ":" << amount << "|";
    }
    return oss.str();
  }

  void prune(std::unordered_map<int, double>& resources)
  {
    // Remove all near tol values from map
    for (auto it = resources.begin(); it != resources.end();)
    {
      if (it->second < tol)
      {
        it = resources.erase(it);
      } else 
        it++;
    }
  }

  std::pair<int, std::vector<PathStep>> recursive_helper_simple(const std::unordered_map<int, double>& resources, int depth, int current_transformations)
  {
    if (depth > 50)
    {
      return {
        resources.count(resources_.get_target()) && resources.at(resources_.get_target()) > tol ? 
        current_transformations : std::numeric_limits<int>::max(),
        {}
      };
    }

    std::string key = generate_resource_key(resources);

    auto it = cache_simple.find(key);
    if (it != cache_simple.end())
    {
      return it->second;
    }

    if (resources.count(resources_.get_target()) && resources.at(resources_.get_target()) > tol)
      return {current_transformations, {}};

    int best_transformations = std::numeric_limits<int>::max();
    std::vector<PathStep> best_path;

    for (size_t i = 0; i < transformations.size(); i++)
    {
      const auto& transformation = transformations[i];

      bool can_transform = true;
      for (const auto& [input, amount] : transformation.inputs)
      {
        auto it = resources.find(input);
        if (it == resources.end() || it->second < amount - tol)
        {
          can_transform = false;
          break;
        }
      }
      
      if (!can_transform) continue;

      std::unordered_map<int, double> new_resources = resources;
      for (const auto& [input, amount] : transformation.inputs)
        new_resources[input] -= amount;

      int output_id = transformation.output.first;
      double output_amount = transformation.output.second;
      new_resources[output_id] += output_amount;

      prune(new_resources);
      
      auto [child_transformations, child_path] = recursive_helper_simple(new_resources, depth + 1, current_transformations + 1);

      if (child_transformations < best_transformations)
      {
        best_transformations = child_transformations;
        best_path = child_path;

        best_path.insert(best_path.begin(), create_step(i, transformation));
      }
    }

    cache_simple[key] = {best_transformations, best_path};
    return {best_transformations, best_path};
  }

  std::pair<double, std::vector<PathStep>> recursive_helper_optimal(const std::unordered_map<int, double>& resources, int depth)
  {
    // If max depth is reached 
    if (depth > 50) 
      return {resources.count(resources_.get_target()) ? resources.at(resources_.get_target()) : 0.0, {}};

    std::string key = generate_resource_key(resources);

    auto it = cache_optimal.find(key);
    if (it != cache_optimal.end())
      return it->second;

    // Set current (and "best") output/path
    double current_output = resources.count(resources_.get_target()) ? resources.at(resources_.get_target()) : 0.0;

    double best_output = current_output; 
    std::vector<PathStep> best_path;

    // Iterate over all transformations checking if we can apply one 
    for (size_t i = 0; i < transformations.size(); i++)
    {
      const auto& transformation = transformations[i];

      bool can_transform = true;
      double max_applications = std::numeric_limits<double>::max();

      // Get the maximum number of applications we can apply with a functor 
      for (const auto& [input, amount] : transformation.inputs)
      {
        if (amount > tol)
        {
          auto it = resources.find(input);
          if (it == resources.end() || it->second < amount)
          {
            can_transform = false;
            break;
          }
          max_applications = std::min(max_applications, it->second / amount);
        }
      }

      if (!can_transform) continue;   // Skip if can't apply transformation 

      for (int applications : {1, static_cast<int>(max_applications)})
      {
        if (applications < 1 || applications > max_applications) continue;

        // Subtract used resources from map of resource pool 
        std::unordered_map<int, double> new_resources(resources);
        for (const auto& [input, amount] : transformation.inputs)
          new_resources[input] -= amount * applications;

        // Calculate output of resource from transformations applied 
        int output_id = transformation.output.first;
        double output_amount = transformation.output.second * applications;
        new_resources[output_id] += output_amount;

        // < tol -> Erase 
        prune(new_resources);

        // Recursive call
        auto [child_output, child_path] = recursive_helper_optimal(new_resources, depth + 1);

        // If child's output exceeded parents 
        if (child_output > best_output + tol)
        {
          // Update best output and path
          best_output = child_output;
          best_path   = child_path;

          // Add number of times transformation used to path 
          for (int j = 0; j < applications; j++)
            best_path.insert(best_path.begin(), create_step(i, transformation));
        }
      }
    }

    // Cache result and return 
    cache_optimal[key] = {best_output, best_path};
    return {best_output, best_path};
  }

// Constructor 
public:
  OptimizedHypergraph(const std::string& name) : Hypergraph(name) {}

  std::vector<PathStep> search_simple() override 
  {
    cache_simple.clear();
    // Get starting resource pool and trigger recursive call
    std::unordered_map<int, double> initial_resources = initialize_search();
    auto [min_transformations, path] = recursive_helper_simple(initial_resources, 0, 0);

    // For min to still equal INT_MAX means no solution was found 
    if (min_transformations == std::numeric_limits<int>::max())
      return {};

    return path;
  }

  std::vector<PathStep> search_optimal() override
  {
    if (search_simple().empty()) 
      return {};    // Empty path for no solution 

    cache_optimal.clear();
    std::unordered_map<int, double> initial_resources = initialize_search();
    auto [max_output, path] = recursive_helper_optimal(initial_resources, 0);
    return path;
  }
};

// Prints a vector<PathStep> output to display a graph 
void print_graph(const Hypergraph& graph, const std::vector<PathStep>& path, const std::string& tag)
{
  std::cout << "\n" << graph.name << "\n";
  std::cout << tag << "\n\n";
  if (path.empty())
  {
    std::cout << "No solutions\n";
    return;
  }

  std::unordered_set<int> nodes_in_path;
  for (const auto& step : path)
  {
    for (int input : step.input_nodes)
    {
      nodes_in_path.insert(input);
    }
    nodes_in_path.insert(step.output_node);
  }

  std::cout << "Sources:\n";
  for (int id : graph.get_source_ids())
  {
    if (nodes_in_path.find(id) != nodes_in_path.end())
    {
      const Node& node = graph.get_node(id);
      std::cout << " " <<  node.resource_type << " (" << node.source_production << " units)\n";
    }
  }
  std::cout << '\n';

  std::cout << "Functors:\n";
  for (const auto& functor : graph.get_transformations())
  {
    std::cout << " " << functor.functor_name << ": ";

    for (size_t i = 0; i < functor.inputs.size(); i++)
    {
      const auto& [input_id, amount] = functor.inputs[i];
      std::cout << amount << " " << graph.get_node(input_id).resource_type;

      if (i < functor.inputs.size() - 1)
        std::cout << " + ";
    }

    std::cout << " -> " << functor.output.second << " "
              << graph.get_node(functor.output.first).resource_type;

    std::cout << '\n';
  }
    std::cout << '\n';

  std::cout << "Production:\n";

  for (const auto& step : path)
  {
    for (int i = 0; i < step.input_nodes.size(); i++)
    {
      int input = step.input_nodes[i];
      std::cout << " " << graph.get_node(input).resource_type;

      for (const auto& [in_id, amount] : graph.get_transformation(step.transformation_id).inputs)
      {
        if (in_id == input)
        {
          std::cout << " (" << amount << " units)";
          break;
        }
      }

      if (i < step.input_nodes.size() - 1)
        std::cout << " + "; 
    }

    std::cout << " -> " << graph.get_transformation(step.transformation_id).functor_name
              << " -> " << graph.get_node(step.output_node).resource_type
              << " ("   << graph.get_transformation(step.transformation_id).output.second << " units)\n";
  }
}

}

#endif // HYPERGRAPH_HPP
