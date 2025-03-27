// Spider Wasp Optimizer Solution to Satisfactory DAG problem 

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <set>
#include <unordered_map>
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

  // Taking some sequence of functors determine the amount of target resource gained 
  std::pair<float, std::unordered_map<int, float>> evaluate_functor_sequence(
      const std::vector<std::pair<int, int>>& sequence) const 
  {
    // Get pool and start checking if functors be applied to a resource 
    std::unordered_map<int, float> resources = initialize_resource_pool();
    for (const auto& [functor_id, count] : sequence)
    { 
      const Functor& F = functors[functor_id];
      for (int i = 0; i < count; i++)
{
        if (!F.can_apply(resources))
          break;
        
        F.apply(resources);
      }
    }

    // Return the amount of the target resource 
    return std::make_pair((resources.count(target_id) ? resources[target_id] : 0.0), resources);
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

// Public class to take/store sequences and evaluate them to determine/store efficacy
struct PotentialSolution
{
  std::vector<std::pair<int, int>> sequence;
  float fitness{0.0};
  std::unordered_map<int, float> final_resource_pool;

  // Default 
  PotentialSolution() {}

  // Evalute graph with functors solution sequence 
  PotentialSolution(const std::vector<std::pair<int, int>>& seq, const Graph& graph)
    : sequence(seq) { evaluate(graph); }

  void evaluate(const Graph& graph)
  {
    const auto& [target_amount, final_pool] = graph.evaluate_functor_sequence(sequence);
    final_resource_pool = final_pool;
    fitness = target_amount;
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

  std::vector<prod::PotentialSolution> solutions;
  size_t capacity;
  float diversity;
  
public:
  Repository(size_t capacity = 15, float diversity = 0.1) : capacity(capacity), diversity(diversity) {}

  float distance(const prod::PotentialSolution& solution_one, const prod::PotentialSolution& solution_two) const
  {
    std::set<int> functors_one, functors_two;
    
    // Get all unique functors used in both solutions 
    for (const auto& [functor_id, _] : solution_one.sequence)
      functors_one.insert(functor_id);

    for (const auto& [functor_id, _] : solution_two.sequence)
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

  bool add(const prod::PotentialSolution& solution)
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

  prod::PotentialSolution get_random() const 
  {
    if (solutions.empty())
      return prod::PotentialSolution();
    
    std::uniform_int_distribution<size_t> d(0, solutions.size() - 1);
    return solutions[d(rng)];
  }

  prod::PotentialSolution get_biased() const 
  {
    if (solutions.empty())
      return prod::PotentialSolution();

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

  prod::PotentialSolution get_front() const 
  {
    if (solutions.empty())
      return prod::PotentialSolution();
    else
     return solutions.front();
  }

  prod::PotentialSolution get_diverse_from(const prod::PotentialSolution& ref) const 
  {
    if (solutions.empty())
      return prod::PotentialSolution();

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

  const std::vector<prod::PotentialSolution>& get_solutions() const { return solutions; }
  size_t size() const { return solutions.size(); }
  void clear() { solutions.clear(); }
};

class Statistics
{
private:
  // Timings 
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  float runtime;

  size_t current_iteration{0};
  size_t max_iterations{0};
  size_t stagnate_iterations{0};
  size_t max_stagnate_iterations{0};

  float best_fitness{0.0};
  size_t iteration_best_found{0};

  size_t hunting_operations{0};
  size_t parasite_operations{0};
  size_t successful_hunt{0};
  size_t successful_parasite{0};

  size_t repository_additions{0};
  size_t repositroy_rejections{0};

  float avg_distance{0.0};
  float min_distance{1.0};

  // Performance metrics 
  size_t total_evaluations{0};
  size_t cache_hits{0};

  // History metrics
  std::vector<std::pair<size_t, float>> convergence;
  std::vector<std::pair<size_t, float>> diversity_history;

public:

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
    min_distance = minimum;

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
  size_t get_iterations() const { return current_iteration; }
  float get_best_fitness() const { return best_fitness; }
  size_t get_best_iteration() const { return iteration_best_found; }
  float get_avg_distance() const { return avg_distance; }
  
  // Convergence rate
  float get_convergence_rate() const 
  {
    if (iteration_best_found == 0) return 0.0;
    return best_fitness / iteration_best_found;
  }
};

enum Phase : uint8_t
{
  HUNTING,
  PARASITISM
};

struct Wasp 
{
  prod::PotentialSolution current_solution;
  prod::PotentialSolution best_solution;
  Phase current_phase;
  float hunting_radius;
  float transition_probability;

  Wasp() : current_phase(HUNTING) {}
};

class SpiderWaspOptimizer
{
private:
  // Composed classes 
  prod::Graph graph;  // Entire Resource Graph predefined
  Repository repo;
  Statistics stats;

  int wasp_count{0};
  float hunting_radius{0.0};
  float transition_probability{0.0};
  
  std::vector<Wasp> wasps;

public:
  SpiderWaspOptimizer() {}

  prod::PotentialSolution optimize_with_stats()
  {
    stats.reset();
    stats.start();

    while (!stats.should_end())
    {
      stats.current_iteration++;


      if ()
      {

        if ()
        {

        }
      }


    }

  }

};

}
