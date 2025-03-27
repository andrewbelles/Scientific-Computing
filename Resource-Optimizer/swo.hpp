#ifndef SWO_HPP 
#define SWO_HPP

// Spider Wasp Optimizer Solution to Satisfactory DAG problem 

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
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
struct Solution
{
  std::vector<std::pair<int, int>> sequence;
  float fitness{0.0};
  std::unordered_map<int, float> final_resource_pool;

  // Default 
  Solution() {}

  // Evalute graph with functors solution sequence 
  Solution(const std::vector<std::pair<int, int>>& seq, const Graph& graph)
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

  std::vector<prod::Solution> solutions;
  size_t capacity;
  float diversity;
  
public:
  Repository(size_t capacity = 15, float diversity = 0.1) : capacity(capacity), diversity(diversity) {}

  float distance(const prod::Solution& solution_one, const prod::Solution& solution_two) const
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

  struct OperationProbabilities
  {
    float add;
    float remove;
    float modify;
    float swap;
  };

  // Composed classes 
  prod::Graph graph;                 // Entire Resource Graph passed in constructor
  Repository repo;
  Statistics stats;

  int wasp_count{0};
  float radius_decay{0.98};
  float min_hunting_radius{0.05};
  float hunting_radius{0.3};

  float hunting_to_parasite_transition{0.1};
  float parasite_to_hunting_transition{0.3};
  size_t max_sequence_length{50};
  
  std::vector<Wasp> wasps;
  prod::Solution global_best;

  mutable std::mt19937 rng{std::random_device{}()};     // TODO: Use thread safe
  OperationProbabilities hunting_probabilities{0.7, 0.3, 0.5, 0.3};
  OperationProbabilities parasite_probabilities{0.3, 0.4, 0.7, 0.5};

  void initialize_population()
  {
    wasps.clear();
    wasps.reserve(wasp_count);

    for (size_t i = 0; i < wasp_count; i++)
    {
      Wasp wasp;
      wasp.hunting_radius = hunting_radius;
      wasp.transition_probability = hunting_to_parasite_transition;
      wasp.current_solution = prod::Solution({}, graph); 
      wasp.best_solution = wasp.current_solution;

      wasps.push_back(wasp);
    }

    global_best = prod::Solution({}, graph);
  }

  // Creates a random solution using the four add, remove, modify, swap operations 
  prod::Solution hunt(Wasp& wasp)
  {
    stats.hunting_operations++;
    std::vector<std::pair<int, int>> new_sequence = wasp.current_solution.sequence;

    const auto& functors = graph.get_functors();
    if (functors.empty())
      return wasp.current_solution;

    std::uniform_real_distribution<float> d(0.0, 1.0);
    const auto& probs = hunting_probabilities;

    if (d(rng) < probs.add && new_sequence.size() < max_sequence_length)
    {
      std::uniform_int_distribution<size_t> f(0, functors.size() - 1);
      std::uniform_int_distribution<int> c(1, 30);

      int functor_index = f(rng);
      int functor_id    = functors[functor_index].id;
      int count         = c(rng);

      if (new_sequence.empty())
        new_sequence.emplace_back(functor_id, count);
      else 
      {
        std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
        size_t position = p(rng);
        new_sequence.insert(new_sequence.begin() + position, std::make_pair(functor_id, count));
      }
    }

    if (!new_sequence.empty())
    {
      // Remove operation 
      if (d(rng) < probs.remove)
      {
        std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
        size_t position = p(rng);

        new_sequence.erase(new_sequence.begin() + position);
      }

      if (d(rng) < probs.modify)
      {
        std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
        std::uniform_int_distribution<int> c(1, 30);

        size_t position = p(rng);
        int new_count   = c(rng);

        new_sequence[position].second = new_count;
      }

      if (d(rng) < probs.swap && new_sequence.size() >= 2)
      {
        std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
        size_t first = p(rng), second = p(rng);
        
        while (first == second)
          second = p(rng);

        std::swap(new_sequence[first], new_sequence[second]);
      }
    }

    // Constructor call handles solution evaluation
    prod::Solution new_solution(new_sequence, graph);
    return new_solution;
  }

  // Exploit a good solution in a similar manner to hunt 
  prod::Solution parasite(Wasp& wasp)
  {
    stats.parasite_operations++;  // Inc is cheap can be constantly tracked
    // Get a random (biased towards better) solution from repository
    prod::Solution host = repo.get_biased();

    // No solution to exploit so just hunt 
    if (host.sequence.empty())
      return hunt(wasp);

    std::vector<std::pair<int, int>> new_sequence = host.sequence;
    
    std::uniform_real_distribution<float> d(0.0, 1.0);
    const auto& probs = parasite_probabilities;

    // Identical impl 
    if (d(rng) < probs.modify && !new_sequence.empty())
    {
      std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
      std::uniform_int_distribution<int> c(1, 30);

      size_t position = p(rng);
      int new_count   = c(rng);

      new_sequence[position].second = new_count;
    }

    // Identical impl
    if (d(rng) < probs.remove && !new_sequence.empty())
    {
      std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
      size_t position = p(rng);

      new_sequence.erase(new_sequence.begin() + position);
    }

    // We know the sequence isn't empty by this point so no inner check 
    if (d(rng) < probs.add && new_sequence.size() < max_sequence_length)
    {
      const auto& functors = graph.get_functors();

      std::uniform_int_distribution<size_t> f(0, functors.size() - 1);
      std::uniform_int_distribution<int> c(1, 30);

      int functor_index = f(rng);
      int functor_id    = functors[functor_index].id;
      int count         = c(rng);

      std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
      size_t position = p(rng);

      if (new_sequence.empty())
        new_sequence.emplace_back(functor_id, count);
      else 
      {
        std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
        size_t position = p(rng);
        new_sequence.insert(new_sequence.begin() + position, std::make_pair(functor_id, count));
      }
    }

    // Identical impl
    if (d(rng) < probs.swap && new_sequence.size() >= 2)
    {
      std::uniform_int_distribution<size_t> p(0, new_sequence.size() - 1);
      size_t first = p(rng), second = p(rng);
      
      while (first == second)
        second = p(rng);

      std::swap(new_sequence[first], new_sequence[second]);
    }

    // Evaluate and return sequence 
    prod::Solution new_solution(new_sequence, graph);
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
        wasp.transition_probability = hunting_to_parasite_transition;
      else 
        wasp.transition_probability = parasite_to_hunting_transition;
    }

    // Decay hunting radius 
    wasp.hunting_radius = std::max(min_hunting_radius, wasp.hunting_radius * radius_decay);
  }

  // Runs a single iteration of SWO w/ or w/o updating stats
  // Increment operations will not be limited by stat t/f
  bool run_iteration(bool check_stats)
  {
    std::cout << "Starting Iteration " << stats.current_iteration << '\n';
    bool improved = false;

    for (auto& wasp : wasps)
    {
      prod::Solution new_solution;
      // Run one of two strategies depending on current phase 
      new_solution = (wasp.current_phase == HUNTING) ? hunt(wasp) : parasite(wasp);
      std::cout << "Wasp finished " << wasp.current_phase << '\n';
      
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
    float hunting_radius = 0.3,
    float hunting_transition  = 0.3,
    float parasite_transition = 0.1,
    size_t max_iter = 500,
    size_t max_stagnate = 100
  ) : 
    graph(std::move(problem_space)),
    repo(wasp_count / 2),
    stats(max_iter, max_stagnate),
    wasp_count(wasp_count),
    hunting_radius(hunting_radius),
    parasite_to_hunting_transition(hunting_transition),
    hunting_to_parasite_transition(parasite_transition)
  {}

  prod::Solution optimize(bool check_stats = false)
  {
    repo.clear();
    global_best = prod::Solution({}, graph);
    stats.reset();
    stats.start();
    initialize_population();

    while (!stats.should_end())
    {
      run_iteration(check_stats);
    }

    stats.stop();
    return global_best;
  }

  void print(const prod::Solution& solution, const std::string& tag = "Optimal Solution") const
  {
    std::cout << "\n======== " << tag << "========\n\n";

    if (solution.sequence.empty() || solution.fitness <= tol)
    {
      std::cout << "No solution found for problem\n";
      return;
    }

    std::cout << "Target Resource: " << graph.get_resource(graph.get_target()).name
              << "(produced: " << solution.fitness << " units)\n\n";

    std::unordered_set<int> resources_used;
    for (const auto& [functor_id, _] : solution.sequence)
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
        std::cout << "  " << resource.name << "(" << resource.production << " units)\n";
    }
    std::cout << '\n';

    std::cout << "Functors Used:\n";
    std::unordered_set<int> functors_used;
    for (const auto& [functor_id, _] : solution.sequence)
      functors_used.insert(functor_id);

    for (int functor_id : functors_used)
    {
      const auto& functor = graph.get_functor(functor_id);
      std::cout << "  " << functor.name << ": ";

      for (size_t i = 0; i < functor.inputs.size(); i++)
      {
        const auto& [input_id, amount] = functor.inputs[i];
        std::cout << amount << graph.get_resource(input_id).name;

        if (i < functor.inputs.size() - 1)
          std::cout << " + ";
      }

      std::cout << " -> " << functor.output.second << " " << graph.get_resource(functor.output.first).name;
      std::cout << '\n';
    }
    std::cout << '\n';

    std::cout << "Production Sequence:\n";

    std::unordered_map<int, int> functor_counts;
    for (const auto& [functor_id, count] : solution.sequence)
      functor_counts[functor_id] += count;

    for (size_t i = 0; i < solution.sequence.size(); i++)
    {
      const auto& [functor_id, count] = solution.sequence[i];
      const auto& functor = graph.get_functor(functor_id);

      std::cout << i+1 << ". Apply " << functor.name << " " << count;
      
      for (size_t j = 0; j < functor.inputs.size(); j++)
      {
        const auto& [input_id, amount] = functor.inputs[j];
        std::cout << amount * count << graph.get_resource(input_id).name;

        if (j < functor.inputs.size() - 1)
          std::cout << " + ";
      }

      std::cout << " -> " << functor.output.second * count << " " 
                << graph.get_resource(functor.output.first).name;
      std::cout << '\n';
    }
    std::cout << '\n';

    std::cout << "Final Resource Pool:\n";
    for (const auto& [resource_id, amount] : solution.final_resource_pool)
    {
      if (amount > tol)
        std::cout << " " << graph.get_resource(resource_id).name << '\n';
    }
  }

};

}

#endif // SWO_HPP
