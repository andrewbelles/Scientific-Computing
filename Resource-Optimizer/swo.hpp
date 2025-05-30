#ifndef SWO_HPP 
#define SWO_HPP

// Spider Wasp Optimizer Solution to Satisfactory DAG problem 

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>

#define tol 1e-6

namespace prod {

// Define a resource that can be pulled or added to 
struct Resource 
{
  int id;
  std::string name;
  bool is_source{false};
  float production{0.0};
};

// Define some mapping from X -> Y 
class Functor
{
public:
  int id;
  std::string name;
  std::vector<std::pair<int, float>> inputs; // (id, amount)
  std::pair<int, float> output; // Likewise 
  
  Functor(int id, 
          const std::string& name, 
          const std::vector<std::pair<int, float>>& inputs, 
          const std::pair<int, float>& output)
    : id(id), name(name), inputs(inputs), output(output) {}

  // Check if a resource is large enough to have some functor act on it 
  bool can_apply(const std::unordered_map<int, float>& resource_pool) const 
  {
    for (const auto& [resource_id, required_amount] : inputs)
    {
      auto it = resource_pool.find(resource_id);
      if (it == resource_pool.end() || it->second < required_amount - tol)
        return false;

    }
    return true;
  }

  // Apply a specific functor to a resource
  void apply(std::unordered_map<int, float>& resource_pool) const
  {
    for (const auto& [resource_id, required_amount] : inputs)
    {
      // Subtract functor's required amount 
      resource_pool[resource_id] -= required_amount;
      // If resource is exhausted remove 
      if (resource_pool[resource_id] <= tol)
        resource_pool.erase(resource_id);
    }
    // Add new output from functor application 
    resource_pool[output.first] += output.second;
  }
};

class Graph 
{
private:
  std::vector<Resource> resources;
  std::vector<Functor> functors;

  // Convert name string to id for Nodes
  std::unordered_map<std::string, int> resource_name_to_id;
  std::unordered_map<std::string, int> functor_name_to_id;

  int target_id{-1};

public:
  Graph() {}

  int add_resource(const std::string& name, bool is_source = false, float production = 0.0)
  {
    // Check if pre-existing 
    auto it = resource_name_to_id.find(name);
    if (it != resource_name_to_id.end())
      return it->second; 

    // Get id, add it to pool and lookup table, return di  
    size_t id = resources.size();
    resources.emplace_back(id, name, is_source, production);
    resource_name_to_id[name] = id;
    return static_cast<int>(id);
  }

  int add_functor(const std::string& name,
                  const std::vector<std::pair<std::string, float>>& input_resources,
                  const std::pair<std::string, float>& output_resource)
  {
    std::vector<std::pair<int, float>> inputs;
    for (const auto& [resource_name, amount] : input_resources)
    {
      int resource_id = add_resource(resource_name);    // This either creates the resource or just gets the id
      inputs.emplace_back(resource_id, amount);
    }

    int output_id = add_resource(output_resource.first);

    size_t id = functors.size();
    functors.push_back(Functor(id, name, inputs, std::make_pair(output_id, output_resource.second)));
    functor_name_to_id[name] = id;
    return static_cast<int>(id);
  } 

  void set_target(const std::string& target_name)
  {
    auto it = resource_name_to_id.find(target_name);
    if (it != resource_name_to_id.end())
      target_id = it->second;
    else 
      target_id = add_resource(target_name);
  }

  std::unordered_map<int, float> initialize_resource_pool() const 
  {
    std::unordered_map<int, float> pool;
    for (const auto& resource : resources)
    {
      if (resource.is_source)
        pool[resource.id] = resource.production;
    }
    return pool;
  }

  // Update the state using a partial sequence and parital pool 
  void update(std::unordered_map<int, float>& pool, std::unordered_map<int, int> map) 
  {
    if (map.empty())
      return;

    std::vector<int> ordering = sort(map);
    
    for (int functor_id : ordering)
    {
      int count = map[functor_id];
      const auto& F = get_functor(functor_id);

      int max_applications = count; 
      for (const auto& [resource_id, required_amount] : F.inputs)
      {
        auto it = pool.find(resource_id);
        if (it == pool.end())
        {
          max_applications = 0;
          break;
        }

        int possible = static_cast<int>(it->second / required_amount);
        max_applications = std::min(max_applications, possible);
      }

      for (int i = 0; i < max_applications; i++)
        F.apply(pool);
    }
  }

  // Perform a topological sort on functor solution map 
  // Return the indexes functors should be applied
  std::vector<int> sort(const std::unordered_map<int, int>& functor_counts) const 
  {
    std::unordered_map<int, std::vector<int>> dependencies;
    std::unordered_map<int, int> in_degree;

    // Initialize helper maps
    for (const auto& [functor_id, _] : functor_counts)
    {
      in_degree[functor_id]    = 0;
      dependencies[functor_id] = std::vector<int>();
    }

    // Build dependency graph 
    for (const auto& [functor_id, _] : functor_counts)
    {
      const auto& F = get_functor(functor_id);
      for (const auto& [input_id, _] : F.inputs)
      {
        // Iterate over Functors. Skip if self 
        for (const auto& [other_functor_id, _] : functor_counts)
        {
          if (other_functor_id == functor_id)
            continue;

          // Check if another functor makes the input that F requires
          const auto& Fp = get_functor(other_functor_id);
          if (Fp.output.first == input_id)
          {
            dependencies[other_functor_id].push_back(functor_id);
            in_degree[functor_id]++;
          }
        }
      }
    }

    std::vector<int> sorted_functors;
    std::queue<int> zero_in_degree;   // Functors that don't currently have a dependency

    for (const auto& [functor_id, degree] : in_degree)
    {
      if (degree == 0)
        zero_in_degree.push(functor_id);
    }

    while (!zero_in_degree.empty())
    {
      int current = zero_in_degree.front();
      zero_in_degree.pop();
      sorted_functors.push_back(current);

      for (int neighbor : dependencies[current])
      {
        // Reduce dependency degree for functors 
        in_degree[neighbor]--;
        // Add new 0 degree functors to queue
        if (in_degree[neighbor] == 0)
          zero_in_degree.push(neighbor);
      }
    }

    // This won't be a problem there are no graph cycles 
    if (sorted_functors.size() != functor_counts.size())
      return {};

    return sorted_functors;
  }

  const std::vector<Resource>& get_resources() const { return resources; }
  const std::vector<Functor>& get_functors() const { return functors; }
  int get_target() const { return target_id; }

  // By id 
  const Resource& get_resource(int id) const { return resources[id]; }
  const Functor& get_functor(int id) const { return functors[id]; }

  // Get id by name 
  int get_resource_id(const std::string& name) const 
  {
    auto it = resource_name_to_id.find(name);
    return (it != resource_name_to_id.end()) ? it->second : -1; // Invalid id if not in set
  }

  int get_functor_id(const std::string& name) const 
  {
    auto it = functor_name_to_id.find(name);
    return (it != functor_name_to_id.end()) ? it->second : -1; // Invalid id if not in set
  }

  // Static method as this in general can be used independetly by a hypergraph, swo, or whatever 
  static void prune_pool(std::unordered_map<int, float>& pool)
  {
    for (auto it = pool.begin(); it != pool.end();)
    {
      if (it->second <= tol)
        it = pool.erase(it);
      else 
        it++;
    }
  }

};

// Class to take/store sequences and evaluate them to determine/store efficacy
class Solution
{
private:
  // Fitness function that scores biased towards output then inputs to output then any intermediate resource
  float calculate_fitness(const Graph& graph, std::unordered_map<int, float>& pool)
  {
    const size_t target_id = graph.get_target();
    float fitness = 0.0;

    if (pool.count(target_id) && pool.at(target_id) > tol)
      fitness = 100.0 * pool.at(target_id); 
    
    
    // Find all inputs that direct to the output and give some reward for producing them
    // Find all inputs that direct to the output and their required amounts
    std::unordered_map<int, float> direct_inputs_required;
    std::unordered_map<int, float> direct_inputs_output_rates;
    
    for (const auto& F : graph.get_functors())
    {
      if (F.output.first == target_id)
      {
        for (const auto& [input_id, amount] : F.inputs)
        {
          direct_inputs_required[input_id] = amount;
        }
      }
    }
    
    // Find the output rates for each intermediate product
    for (const auto& F : graph.get_functors())
    {
      if (direct_inputs_required.find(F.output.first) != direct_inputs_required.end())
        direct_inputs_output_rates[F.output.first] = F.output.second;
    }

    // Reward for producing penultimate products, scaled by their output rates
    for (const auto& [input_id, required_amount] : direct_inputs_required)
    {
      auto it = pool.find(input_id);
      if (it != pool.end() && it->second > tol)
      {
        float output_rate = 1.0; 
        auto rate_it = direct_inputs_output_rates.find(input_id);
        if (rate_it != direct_inputs_output_rates.end() && rate_it->second > tol)
          output_rate = rate_it->second;

        float scaled_output = 0.1 * std::exp(-0.05 * output_rate);
        fitness += (it->second * scaled_output);
      }
    }

    // Reward for using all source nodes 
    std::unordered_set<int> used_sources;
    std::unordered_set<int> all_sources; 

    for (const auto& resource : graph.get_resources())
    {
      if (resource.is_source)
        all_sources.insert(resource.id);
    }

    std::unordered_map<int, float> initial_pool = graph.initialize_resource_pool();
    for (int source_id : all_sources)
    {
      auto it_initial = initial_pool.find(source_id);
      auto it_current = pool.find(source_id);

      float initial_amount = (it_initial != initial_pool.end()) ? it_initial->second : 0.0;
      float current_amount = (it_current != pool.end()) ? it_current->second : 0.0;

      if (current_amount < initial_amount + tol)
        used_sources.insert(source_id);
    }

    if (!all_sources.empty())
    {
      float source_usage_ratio = static_cast<float>(used_sources.size() / all_sources.size());
      float source_usage_bonus = 30.0 * source_usage_ratio;
      fitness += source_usage_bonus;

      if (used_sources.size() == all_sources.size())
        fitness += 15.0;
    }


    return fitness;
  }

public:
  std::unordered_map<int, int> map; // (functor_id -> count)
  float fitness{0.0};
  std::unordered_map<int, float> final_resource_pool;

  // Default 
  Solution() {}

  // Evalute graph with functors solution sequence 
  Solution(const std::unordered_map<int, int>& sol, const Graph& graph)
    : map(sol) { evaluate(graph); }

  // Evaluates graph with solution map
  void evaluate(const Graph& graph)
  {
    std::unordered_map<int, float> pool = graph.initialize_resource_pool();
    std::vector<int> ordering = graph.sort(map);
    
    for (int functor_id : ordering)
    {
      int count = map.at(functor_id);
      const auto& F = graph.get_functor(functor_id);

      int max_applications = count; 
      for (const auto& [resource_id, required_amount] : F.inputs)
      {
        auto it = pool.find(resource_id);
        if (it == pool.end() || required_amount <= tol)
        {
          max_applications = 0;
          break;
        }

        int possible = static_cast<int>(it->second / required_amount);
        max_applications = std::min(max_applications, possible);
      }

      for (int i = 0; i < max_applications; i++)
        F.apply(pool);
    }

    // Update evaluation result  
    final_resource_pool = pool;
    fitness = calculate_fitness(graph, pool);
  }
};

}

namespace SWO {

// Utility classes that will assist in hyperparameter tuning 

// Solution repository which serves to store good solutions in memory 
// Reduces penalty for wasps for "making bad decisions" 
class Repository 
{
private:
  // RNG 
  mutable std::mt19937 rng{std::random_device{}()};

  std::vector<prod::Solution> solutions;
  size_t capacity;
  float diversity;
  
public:
  Repository(size_t capacity = 15, float diversity = 0.1) : capacity(capacity), diversity(diversity) {}

  float distance(const prod::Solution& solution_one, const prod::Solution& solution_two) const
  {
    std::set<int> functors_one, functors_two;
    
    // Get all unique functors used in both solutions 
    for (const auto& [functor_id, _] : solution_one.map)
      functors_one.insert(functor_id);

    for (const auto& [functor_id, _] : solution_two.map)
      functors_two.insert(functor_id);

    // Count common elemnts
    size_t common(0);
    for (int id : functors_one)
    {
      if (functors_two.count(id) > 0)
        common++;
    }

    // return Jaccard distance computation 
    size_t union_size = functors_one.size() + functors_two.size() - common;
    return union_size > 0 ? 1.0 - static_cast<float>(common) / union_size : 0.0;  // 1.0 - similarity 
  }

  bool add(const prod::Solution& solution)
  {
    // Auto reject failed solutions 
    if (solution.fitness <= tol)
      return false;

    // If not full 
    if (solutions.size() < capacity)
    {
      solutions.push_back(solution);
      std::sort(solutions.begin(), solutions.end(),
          [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
      return true;
    }     // Assume a full solutions vector from here on 

    for (size_t i = 0; i < solutions.size(); i++)
    {
      // Mutable reference 
      auto& existing = solutions[i];
      if (distance(existing, solution) < diversity) 
      {
        // Similar but new solution is actually better 
        if (existing.fitness < solution.fitness)
        {
          existing = solution;
          std::sort(solutions.begin(), solutions.end(),
            [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
          return true;
        }
        return false;
      }
    }

    // Replace worst 
    if (solution.fitness > solutions.back().fitness)
    {
      solutions.back() = solution;
      std::sort(solutions.begin(), solutions.end(),
          [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
      return true;
    }

    return false;
  }

  prod::Solution get_random() const 
  {
    if (solutions.empty())
      return prod::Solution();
    
    std::uniform_int_distribution<size_t> d(0, solutions.size() - 1);
    return solutions[d(rng)];
  }

  prod::Solution get_biased() const 
  {
    if (solutions.empty())
      return prod::Solution();

    const size_t size(solutions.size());

    // Biased towards better (front) solutions 
    float total = size * (size + 1) / 2.0;
    std::uniform_real_distribution<float> d(0.0, total);
    float random  = d(rng);
    float sum = 0.0;
    for (size_t i = 0; i < size; i++)
    {
      float rank = static_cast<float>(size - i);
      sum += rank;

      if (random <= sum)
        return solutions[i];
    }

    return solutions.back();
  }

  prod::Solution get_front() const 
  {
    if (solutions.empty())
      return prod::Solution();
    else
     return solutions.front();
  }

  prod::Solution get_diverse_from(const prod::Solution& ref) const 
  {
    if (solutions.empty())
      return prod::Solution();

    size_t idx = 0;
    float max = -1.0;

    for (size_t i = 0; i < solutions.size(); i++)
    {
      float dist = distance(solutions[i], ref);
      if (dist > max)
      {
        max = dist;
        idx = i;
      }
    }
    return solutions[idx];
  }

  const std::vector<prod::Solution>& get_solutions() const { return solutions; }
  size_t size() const { return solutions.size(); }
  void clear() { solutions.clear(); }
};

class Statistics
{
private:
  // Timings 
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  float runtime;

  size_t max_iterations{0};
  size_t max_stagnate_iterations{0};

  float best_fitness{0.0};
  size_t iteration_best_found{0};


  float avg_distance{0.0};
  float min_distance{1.0};

  // Performance metrics 
  size_t total_evaluations{0};
  size_t cache_hits{0};

  // History metrics
  std::vector<std::pair<size_t, float>> convergence;
  std::vector<std::pair<size_t, float>> diversity_history;

public:
  // Incremental stats 
  size_t current_iteration{0};
  size_t stagnate_iterations{0};   // Iter stats

  size_t hunting_operations{0};
  size_t parasite_operations{0};
  size_t successful_hunt{0};
  size_t successful_parasite{0};   // Wasp stats

  size_t repository_additions{0};
  size_t repository_rejections{0}; // Repository

  Statistics(size_t max_iter = 500, size_t max_stagnate = 200)
    : max_iterations(max_iter), max_stagnate_iterations(max_stagnate) {}

  void reset()
  {
    *this = Statistics(max_iterations, max_stagnate_iterations);
  }

  void start() { start_time = std::chrono::high_resolution_clock::now(); }
  void stop() 
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);  
    runtime = static_cast<float>(diff.count());
  }

  void increment_iterations()
  {
    current_iteration++;
    stagnate_iterations++;
    if (current_iteration % 10 == 0)
      convergence.emplace_back(current_iteration, best_fitness);
  }

  bool update_best_fitness(float fitness)
  {
    if (fitness > best_fitness)
    {
      best_fitness = fitness;
      iteration_best_found = current_iteration;
      stagnate_iterations = 0;
      return true;
    }
    return false;
  }

  // Find average distance between solutions and minimum distance 
  void update_diversity_metrics(const Repository& repo)
  {
    const auto& solutions = repo.get_solutions();
    const size_t size = solutions.size();
    if (size < 2)
    {
      avg_distance = 0.0;
      min_distance = 1.0;
      return;
    }

    float total_distance = 0.0;
    float minimum = 1.0;
    size_t count = 0;

    for (size_t i = 0; i < size; i++)
    {
      for (size_t j = i + 1; j < size; j++)
      {
        float d = repo.distance(solutions[i], solutions[j]);
        total_distance += d;
        min_distance = std::min(min_distance, d);
        count++;
      }
    }

    avg_distance = total_distance / count;

    if (current_iteration % 10 == 0)
      diversity_history.emplace_back(current_iteration, avg_distance);
  }

  // Check against iteration caps
  bool should_end() const 
  {
    return current_iteration >= max_iterations || 
      stagnate_iterations >= max_stagnate_iterations;
  }

  // Getters 
  float get_runtime() const { return runtime; }
  float get_best_fitness() const { return best_fitness; }
  size_t get_best_iteration() const { return iteration_best_found; }
  float get_avg_distance() const { return avg_distance; }
  
  // Convergence rate
  float get_convergence_rate() const 
  {
    if (iteration_best_found == 0) return 0.0;
    return best_fitness / iteration_best_found;
  }

  void print() const 
  {
    std::cout << "\n======== Statistics ========\n\n";

    std::cout << "Runtime: " << runtime << " ms\n";
    std::cout << "Iterations: " << current_iteration << " / " << max_iterations;
    if (current_iteration >= max_iterations)
      std::cout << " (Max Iterations reached)";
    std::cout << '\n';

    std::cout << "Stagnant Iterations: " << stagnate_iterations;
    if (stagnate_iterations >= max_stagnate_iterations)
      std::cout << " (Stagnation Limit reached)";
    std::cout << "\n\n";

    std::cout << "Best Solution:\n"
              << "  Fitness: " << best_fitness << '\n'
              << "  Iteration: " << iteration_best_found << '\n';

    if (iteration_best_found > 0)
    {
      float improvement_rate = best_fitness / iteration_best_found;
      std::cout << "  Improvement Rate: " << improvement_rate << " per iteration\n";
    }
    std::cout << '\n';

    std::cout << "Wasp Behavior:\n"
              << "  Hunting Operations: " << hunting_operations << '\n'
              << "  Parasite Operations: " << parasite_operations << "\n\n";
    
    float hunt_success_rate = hunting_operations > 0 ?
      (static_cast<float>(successful_hunt) / hunting_operations) * 100.0 : 0.0;
    float parasite_success_rate = parasite_operations > 0 ?
      (static_cast<float>(successful_parasite) / parasite_operations) * 100.0 : 0.0;

    std::cout << "  Hunting success rate: " << hunt_success_rate << "%\n"
              << "  Parasite success rate: " << parasite_success_rate << "%\n\n";


    size_t total_repo_operations = repository_additions + repository_rejections;
    float acceptance_rate = total_repo_operations > 0 ?
      (static_cast<float>(repository_additions) / total_repo_operations) * 100.0 : 0.0;

    std::cout << "Repository:\n"
              << "  Additions: " << repository_additions << '\n'
              << "  Rejections: " << repository_rejections << '\n'
              << "  Acceptance rate: " << acceptance_rate << "%\n\n";

    std::cout << "Diversity:\n"
              << "  Average Distance: " << avg_distance << '\n'
              << "  Minimum Distance: " << min_distance << "\n\n";

    std::cout << "=================================================\n";
  }
};

struct OperationProbabilities
{
  float add;
  float remove;
  float modify;
  float direct;
};

struct TransitionProbabilities
{
  float to_hunter;
  float to_parasite;
};

class SpiderWaspOptimizer
{
private:

  // Class specific data structures 
  enum Phase : uint8_t
  {
    HUNTING,
    PARASITISM
  };

  struct Wasp 
  {
    prod::Solution current_solution;
    prod::Solution best_solution;    // Wasps independent best
    Phase current_phase{HUNTING};
    float hunting_radius;
    float transition_probability;

    Wasp() {}
  };


  // Composed classes 
  prod::Graph graph;                 // Entire Resource Graph passed in constructor
  Repository repo;
  Statistics stats;

  int wasp_count{0};
  float radius_decay{0.98};
  float min_hunting_radius{0.05};
  float hunting_radius{0.4};

  TransitionProbabilities transitions;

  std::vector<Wasp> wasps;
  prod::Solution global_best;

  mutable std::mt19937 rng{std::random_device{}()};     // TODO: Use thread safe
  OperationProbabilities hunting_probabilities{0.8, 0.2, 0.65, 0.3};
  OperationProbabilities parasite_probabilities{0.4, 0.65, 0.5, 0.1};

  void initialize_population()
  {
    wasps.clear();
    wasps.reserve(wasp_count);

    for (size_t i = 0; i < wasp_count; i++)
    {
      Wasp wasp;
      wasp.hunting_radius = hunting_radius;
      wasp.transition_probability = transitions.to_parasite;
      wasp.current_solution = prod::Solution({}, graph); 
      wasp.best_solution = wasp.current_solution;

      wasps.push_back(wasp);
    }

    global_best = prod::Solution({}, graph);
  }

  // Hunt and Parasite helper functions 
  // Add some functor to a sequence 
  void add(std::unordered_map<int, int>& map, std::unordered_map<int, float>& pool)
  {
    const auto& functors = graph.get_functors(); 
    std::vector<int> applicable_functors;
    for (size_t i = 0; i < functors.size(); i++)
    {
      // Store id of functors that can apply to our current resource state 
      if (functors[i].can_apply(pool))
        applicable_functors.push_back(i);
    }
    
    // Don't add a functor yet if none can be applied to pool
    if (applicable_functors.empty())
      return;

    std::uniform_int_distribution<size_t> f(0, applicable_functors.size() - 1);
    int functor_id = functors[applicable_functors[f(rng)]].id;

    // Ensure that the range of count is limited to the maximum number of times functor can apply itself
    const auto& F = functors[functor_id];
    int max_applications = std::numeric_limits<int>::max(); 
    for (const auto& [resource_id, required_amount] : F.inputs)
    {
      auto it = pool.find(resource_id);
      if (it == pool.end() || required_amount <= tol)
      {
        max_applications = 0;
        break;
      }

      int possible = static_cast<int>(it->second / required_amount);
      max_applications = std::min(max_applications, possible);
    }

    if (max_applications == 0)
      return;

    std::uniform_int_distribution<int> c(1, max_applications);
    int count = c(rng);

    map[functor_id] += count;
  }

  // Remove some functor from a map 
  void remove(std::unordered_map<int, int>& map)
  {
    if (map.empty()) return;

    // Get vector of ids in map to remove one randomly 
    std::vector<int> functor_ids;
    functor_ids.reserve(map.size());
    for (const auto& [id, _] : map)
      functor_ids.push_back(id);

    // Get random idx
    std::uniform_int_distribution<size_t> f(0, functor_ids.size() - 1);
    int functor_id = functor_ids[f(rng)];

    map.erase(functor_id);    // Remove
  }

  // Randomly modify some functor 
  void modify(std::unordered_map<int, int>& map, std::unordered_map<int, float>& pool)
  {
    if (map.empty()) return;

    // Get vector of ids in map to remove one randomly 
    std::vector<int> functor_ids;
    functor_ids.reserve(map.size());
    for (const auto& [id, _] : map)
      functor_ids.push_back(id);

    // Get random idx
    std::uniform_int_distribution<size_t> f(0, functor_ids.size() - 1);
    int functor_id = functor_ids[f(rng)];

    // Get the maximum number of times the functor can be applied to one of its inputs 
    const auto& F = graph.get_functor(functor_id);
    int max_applications = std::numeric_limits<int>::max();
    for (const auto& [resource_id, min_amount] : F.inputs)
    {
      auto it = pool.find(resource_id);
      if (it != pool.end() && min_amount > tol)
      {
        int possible_applications = static_cast<int>(it->second / min_amount);
        max_applications = std::min(max_applications, possible_applications);
      }
      else if (min_amount < tol)
      {
        max_applications = 0;
        break;
      }
    }

    // Don't modify functor  
    if (max_applications == 0 || max_applications == std::numeric_limits<int>::max()) return;

    // Modify to use some range of valid applications 
    std::uniform_int_distribution<int> c(1, max_applications);
    int new_count = c(rng);
    map[functor_id] = new_count;
  }

  // Mechanism that adds a random functor it can from source nodes. 
  // Prioritizes source nodes that take multiple inputs 
  void direct(std::unordered_map<int, int>& map, std::unordered_map<int, float>& pool)
  {
    const auto& functors  = graph.get_functors();
    const auto& resources = graph.get_resources();

    std::vector<int> source_consuming_functors;
    std::vector<int> multi_source_input_functors;

    for (size_t i = 0; i < functors.size(); i++)
    {
      const auto& F = functors[i];
      bool can_apply = true;
      std::unordered_set<int> used_sources; 

      for (const auto& [input_id, required_amount] : F.inputs)
      {
        if (input_id < resources.size() && resources[input_id].is_source)
        {
          auto it = pool.find(input_id);
          if (it == pool.end() || it->second < required_amount + tol)
          {
            can_apply = false;
            break;
          }

          used_sources.insert(input_id);
        }
      }

      if (can_apply && !used_sources.empty())
      {
        source_consuming_functors.push_back(i);

        if (used_sources.size() > 1)
          multi_source_input_functors.push_back(i);
      }
    }

    if (source_consuming_functors.empty())
      return;

    int functor_id;
    std::uniform_real_distribution<float> d(0.0, 1.0);

    if (!multi_source_input_functors.empty() && d(rng) < 0.75)
    {
      std::uniform_int_distribution<size_t> p(0, multi_source_input_functors.size() - 1);
      functor_id = functors[multi_source_input_functors[p(rng)]].id;
    }
    else 
    {
      std::uniform_int_distribution<size_t> p(0, source_consuming_functors.size() - 1);
      functor_id = functors[source_consuming_functors[p(rng)]].id;
    }

    // Ensure that the range of count is limited to the maximum number of times functor can apply itself
    const auto& F = functors[functor_id];
    int max_applications = std::numeric_limits<int>::max(); 
    for (const auto& [resource_id, required_amount] : F.inputs)
    {
      auto it = pool.find(resource_id);
      if (it == pool.end() || required_amount <= tol)
      {
        max_applications = 0;
        break;
      }

      int possible = static_cast<int>(it->second / required_amount);
      max_applications = std::min(max_applications, possible);
    }

    if (max_applications == 0)
      return;

    std::uniform_int_distribution<int> c(1, max_applications);
    int count = c(rng);

    map[functor_id] += count;
  }

  void chain(std::unordered_map<int, int>& map)
  {
    const auto& functors = graph.get_functors();

    std::uniform_real_distribution<float> t(0.0, 1.0);

    int target_id = graph.get_target();
    std::vector<int> target_producers;

    for (const auto& F : functors)
    {
      if (F.output.first == target_id)
        target_producers.push_back(F.id);
    }

    
    int start_functor_id;
    if (t(rng) < 0.2 && !target_producers.empty())
    {
      std::uniform_int_distribution<size_t> p(0, target_producers.size() - 1);
      start_functor_id = target_producers[p(rng)];
    }
    else 
    {
      std::vector<int> non_target_functors;
      for (const auto& F : functors)
      {
        bool is_target_producer = false;
        for (int id : target_producers)
        {
          if (F.id == id)
          {
            is_target_producer = true;
            break;
          }
        }

        if (!is_target_producer)
          non_target_functors.push_back(F.id);
      }

      if (non_target_functors.empty())
      {
        std::uniform_int_distribution<size_t> f(0, functors.size() - 1); 
        start_functor_id = functors[f(rng)].id;
      }
      else  
      {   
        std::uniform_int_distribution<size_t> f(0, non_target_functors.size() - 1); 
        start_functor_id = functors[f(rng)].id;
      }
    }

    map[start_functor_id] += 1;

    std::unordered_map<int, float> required_resources;
    const auto& G = graph.get_functor(start_functor_id);

    for (const auto& [input_id, amount] : G.inputs)
      required_resources[input_id] = amount;

    // Complex problems with more than 3+ depth shouldn't try to add functors near end in this fashion
    std::unordered_set<int> processed_resources;
    int chain_depth = 0;
    const int max_depth = 3;

    while (chain_depth < max_depth)
    {
      std::vector<int> resources_to_process;
      for (const auto& [resource_id, amount] : required_resources)
      {
        if (!processed_resources.count(resource_id) &&
            !graph.get_resource(resource_id).is_source)
          resources_to_process.push_back(resource_id);
      }

      if (resources_to_process.empty())
        break;

      for (int resource_id : resources_to_process)
      {
        processed_resources.insert(resource_id);

        std::vector<int> producing_functors;
        for (const auto& F : functors)
        {
          if (F.output.first == resource_id)
            producing_functors.push_back(F.id);
        }

        if (producing_functors.empty())
          continue;

        std::uniform_int_distribution<size_t> p(0, producing_functors.size() - 1);
        int functor_id = producing_functors[p(rng)]; 
        const auto& F = graph.get_functor(functor_id);

        float required_amount = required_resources[resource_id];
        float output_per_application = F.output.second;

        int applications = static_cast<int>(std::ceil(required_amount / output_per_application));

        map[functor_id] += applications;

        for (const auto& [input_id, input_amount] : F.inputs)
        {
          float total_required = input_amount * applications;

          if (required_resources.find(input_id) != required_resources.end())
            required_resources[input_id] += total_required;
          else 
            required_resources[input_id] = total_required;
        }
      }
      chain_depth++;
    }
  }

  // Creates a random solution using the four add, remove, modify, swap operations 
  // Wasps are resource aware in decision making 
  prod::Solution hunt(Wasp& wasp)
  {
    stats.hunting_operations++;
    std::unordered_map<int, int> map;

    const auto& functors = graph.get_functors();
    if (functors.empty())
      return wasp.current_solution;

    // Functors is not empty 

    // Get local resource pool 
    std::unordered_map<int, float> current_resources = graph.initialize_resource_pool();

    std::uniform_real_distribution<float> d(0.0, 1.0);
    const auto& probs = hunting_probabilities;

    // Random operations 
    
    // Hunt while resources can be expended 
    bool available_resources = true;
    //size_t iter(0);
    while (available_resources) // Some check for exhausted resources 
    {
      if (d(rng) < probs.add)
        add(map, current_resources);

      if (d(rng) < probs.remove && !map.empty())
        remove(map);

      if (d(rng) < probs.modify && !map.empty())
        modify(map, current_resources);

      if (d(rng) < probs.direct)
        direct(map, current_resources);

      //if (d(rng) < 0.1)
      //  chain(map);

      std::unordered_map<int, float> pool = graph.initialize_resource_pool();
      graph.update(pool, map);
      current_resources = pool;

      available_resources = false;
      for (const auto& F : functors)
      {
        if (F.can_apply(current_resources))
        {
          available_resources = true;
          break;
        }
      }

      // Some small chance to end early 
      if (d(rng) < 0.05)
        available_resources = false;

      //iter++;
    }

    // Constructor call handles solution evaluation
    prod::Solution new_solution(map, graph);
    return new_solution;
  }

  // Exploit a good solution in a similar manner to hunt 
  prod::Solution parasite(Wasp& wasp)
  {
    stats.parasite_operations++;  // Inc is cheap can be constantly tracked
    // Get a random (biased towards better) solution from repository
    prod::Solution host = repo.get_biased();

    // No solution to exploit so just hunt 
    if (host.map.empty())
      return hunt(wasp);

    // Get resources and new_sequence 
    std::unordered_map<int, int> map = host.map;

    const auto& functors = graph.get_functors();
    if (functors.empty())
      return wasp.current_solution;

    // Functors is not empty 

    // Get local resource pool copy and update to state defined by partial solution map 
    std::unordered_map<int, float> current_resources = graph.initialize_resource_pool();
    graph.update(current_resources, map);
    
    std::uniform_real_distribution<float> d(0.0, 1.0);
    const auto& probs = parasite_probabilities;

    // Random operations 

    bool available_resources = true;
    //size_t iter(0);
    while (available_resources) // Some check for exhausted resources 
    {
      if (d(rng) < probs.add)
        add(map, current_resources);

      if (d(rng) < probs.remove && !map.empty())
        remove(map);

      if (d(rng) < probs.modify && !map.empty())
        modify(map, current_resources);

      if (d(rng) < probs.direct)
        direct(map, current_resources);

      //if (d(rng) < 0.1)
      // chain(map);

      std::unordered_map<int, float> pool = graph.initialize_resource_pool();
      graph.update(pool, map);
      current_resources = pool;

      available_resources = false;
      for (const auto& F : functors)
      {
        if (F.can_apply(current_resources))
        {
          available_resources = true;
          break;
        }
      }

      // Some small chance to end early 
      if (d(rng) < 0.05)
        available_resources = false;

      //iter++;
    }

    // Evaluate and return sequence 
    prod::Solution new_solution(map, graph);
    return new_solution;
  }

  // Updates the phase and hunting radius for a single wasp passed from loop
  void update_phase(Wasp& wasp, float value)
  {
    if (value < wasp.transition_probability)
    {
      // Set new phase and swap transition probabilities 
      wasp.current_phase = (wasp.current_phase == HUNTING) ? PARASITISM : HUNTING;
      if (wasp.current_phase == HUNTING)
        wasp.transition_probability = transitions.to_parasite;
      else 
        wasp.transition_probability = transitions.to_hunter;
    }

    // Decay hunting radius 
    wasp.hunting_radius = std::max(min_hunting_radius, wasp.hunting_radius * radius_decay);
  }

  void prune_map(std::unordered_map<int, int>& map)
  {
    std::unordered_map<int, float> pool = graph.initialize_resource_pool();
    std::vector<int> ordering = graph.sort(map);
    std::unordered_map<int, int> reduced_map;

    for (int functor_id : ordering)
    {
      int count = map.at(functor_id);
      const auto& F = graph.get_functor(functor_id);

      int max_applications = count;
      for (const auto& [resource_id, required_amount] : F.inputs)
      {
        auto it = pool.find(resource_id);
        if (it == pool.end() || required_amount <= tol)
        {
          max_applications = 0;
          break;
        }

        int possible = static_cast<int>(it->second / required_amount);
        max_applications = std::min(max_applications, possible);
      }

      if (max_applications > 0)
      {
        reduced_map[functor_id] = max_applications;

        for (const auto& [input_id, amount] : F.inputs)
        {
          pool[input_id] -= amount * max_applications;
          if (pool[input_id] <= tol)
            pool.erase(input_id);
        }

        pool[F.output.first] += F.output.second * max_applications;
      }
    }

    map = reduced_map;
  }

  // Runs a single iteration of SWO w/ or w/o updating stats
  // Increment operations will not be limited by stat t/f
  bool run_iteration(bool check_stats)
  {
    bool improved = false;

    for (auto& wasp : wasps)
    {
      prod::Solution new_solution;
      // Run one of two strategies depending on current phase 
      new_solution = (wasp.current_phase == HUNTING) ? hunt(wasp) : parasite(wasp);
      prune_map(new_solution.map);

      if (new_solution.fitness > wasp.current_solution.fitness)
      {
        wasp.current_solution = new_solution;
        if (wasp.current_phase == HUNTING)
          stats.successful_hunt++;
        else 
          stats.successful_parasite++;

        if (new_solution.fitness > wasp.best_solution.fitness)
        {
          wasp.best_solution = new_solution;
          if (repo.add(new_solution))
            stats.repository_additions++;
          else 
            stats.repository_rejections++;

          if (new_solution.fitness > global_best.fitness)
          {
            global_best = new_solution;
            improved = true; 
            if (check_stats)
              stats.update_best_fitness(new_solution.fitness);
          }
        }
      }

      // Update single wasps phase 
      std::uniform_real_distribution<float> d(0.0, 1.0);
      update_phase(wasp, d(rng));
    }

    // More expensive storing of statistics 
    if (check_stats)
    {
      stats.update_diversity_metrics(repo);
      stats.increment_iterations(); 
    }
    else 
    {
      stats.current_iteration++;
      stats.stagnate_iterations++;
    }

    return improved;    // Return whether the global was updated 
  }

public:
  SpiderWaspOptimizer(
    prod::Graph problem_space,
    size_t wasp_count,
    TransitionProbabilities t_prob,
    OperationProbabilities hunt_probs,
    OperationProbabilities parasite_probs,
    size_t max_iter = 500,
    size_t max_stagnate = 250
  ) : 
    graph(std::move(problem_space)),
    repo(wasp_count / 2),
    stats(max_iter, max_stagnate),
    wasp_count(wasp_count)
  {
    hunting_probabilities = hunt_probs;
    parasite_probabilities = parasite_probs;
    transitions = t_prob;
  }

  prod::Solution optimize(bool check_stats = false)
  {
    repo.clear();
    global_best = prod::Solution({}, graph);
    stats.reset();
    stats.start();
    initialize_population();

    while (!stats.should_end())
      run_iteration(check_stats);

    stats.stop();
    return global_best;
  }

  // Print methods 
  void print_stats() const { stats.print(); }
  
  void print(const prod::Solution& solution, const std::string& tag = "Optimal Solution") const
  {
    std::cout << "\n======== " << tag << " ========\n\n";

    if (solution.map.empty() || solution.fitness <= tol)
    {
      std::cout << "No solution found for problem\n";
      return;
    }

    std::cout << "Target Resource: " << graph.get_resource(graph.get_target()).name;
    auto it = solution.final_resource_pool.find(graph.get_target());
    if (it != solution.final_resource_pool.end())
    {
      std::cout << " (produced: " << solution.final_resource_pool.find(graph.get_target())->second << " units)\n\n";
    }
    else 
      std::cout << " (produced: 0 units)\n\n";

    std::unordered_set<int> resources_used;
    for (const auto& [functor_id, _] : solution.map)
    {
      const auto& functor = graph.get_functor(functor_id);
      for (const auto& [input_id, _] : functor.inputs)
        resources_used.insert(input_id);

      resources_used.insert(functor.output.first);
    }

    std::cout << "Source Resources:\n";
    for (const auto& resource : graph.get_resources())
    {
      if (resource.is_source && resources_used.count(resource.id))
        std::cout << resource.name << "  "  << "(" << resource.production << " units)\n";
    }
    std::cout << '\n';

    std::cout << "Functors Used:\n";
    std::unordered_set<int> functors_used;
    for (const auto& [functor_id, _] : solution.map)
      functors_used.insert(functor_id);

    for (int functor_id : functors_used)
    {
      const auto& functor = graph.get_functor(functor_id);
      std::cout << "  " << functor.name << ": ";

      for (size_t i = 0; i < functor.inputs.size(); i++)
      {
        const auto& [input_id, amount] = functor.inputs[i];
        std::cout << amount << " " << graph.get_resource(input_id).name;

        if (i < functor.inputs.size() - 1)
          std::cout << " + ";
      }

      std::cout << " -> " << functor.output.second << " " << graph.get_resource(functor.output.first).name;
      std::cout << '\n';
    }
    std::cout << '\n';

    std::vector<int> topological_order = graph.sort(solution.map);
    std::unordered_map<int, float> pool = graph.initialize_resource_pool(); 

    std::cout << "Production Sequence:\n";
    size_t counter = 1;
    for (int functor_id : topological_order)
    {
      int count = solution.map.at(functor_id);
      const auto& F = graph.get_functor(functor_id);

      std::cout << counter << ". Apply " << F.name << " " << count << " times: ";

      // Check the number of times we can apply a functor. (This should ALWAYS match count from map)
      int max_applications = count;
      for (const auto& [input_id, required_amount] : F.inputs)
      {
        auto it = pool.find(input_id);
        if (it == pool.end())
        {
          max_applications = 0;
          break;
        }

        int possible = static_cast<int>(it->second / required_amount);
        max_applications = std::min(max_applications, possible);
      }

      for (size_t j = 0; j < F.inputs.size(); j++)
      {
        const auto& [input_id, amount] = F.inputs[j];
        float exhausted = amount * max_applications;
        std::cout << exhausted << " " << graph.get_resource(input_id).name;

        if (max_applications > 0)
        {
          pool[input_id] -= exhausted;
          if (pool[input_id] <= tol)
            pool.erase(input_id);
        }
        
        if (j < F.inputs.size() - 1)
          std::cout << " + ";
      }

      float generated = F.output.second * max_applications;
      std::cout << " -> " << generated << " "
                << graph.get_resource(F.output.first).name;

      if (max_applications > 0)
        pool[F.output.first] += generated;

      std::cout << '\n';
      counter++;
    }
    std::cout << '\n';
  }
};

}

#endif // SWO_HPP
