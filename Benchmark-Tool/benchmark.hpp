#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

// Utilities
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>

// IO
#include <cstring>
#include <string>

// Simple Abstraction 
#include <iostream>
#include <utility>

// Complex Abstraction
#include <any>
#include <functional>
#include <memory>  
#include <tuple> 
#include <type_traits>
#include <typeindex>

/*
 * No Explicit Inheritance for anything but inheriting purely abstract classes
 * 
 * Benchmark is the only visible class in API
 * 
 * Benchmark only differs in template and constructor overloads
 */

namespace {

// Checks if implements begin, end, and size (which would mean its a container)
template<typename T>
concept Container = requires(T t)
{
  std::begin(t);
  std::end(t);
  std::size(t);
};

// Checks if value is pointer
template<typename T>
concept Pointer = std::is_pointer_v<T>;

// Checks if a list of arguments has a container in it 
template<typename... Args>
concept HasContainer = (Container<Args> || ...);

// Same but for pointers
template<typename... Args>
concept HasPointer = (Pointer<Args> || ...);

// Checks if a value is neither a pointer or container (simple!)
template<typename T>
concept Simple = !Pointer<T> && !Container<T>;

// Checks if a list of arguments is simple 
template<typename... Args>
concept AllSimple = (Simple<Args> && ...);

// Checks if its any integer type (int8_t to size_t; etc)
template<typename T>
concept Integer = std::is_integral_v<T>;

// Checks if the two types are a pair of some pointer and its size
template<typename T, typename U>
concept PointerSizePair = Pointer<T> && Integer<U>;

// Checks if a single type is a reference 
template<typename T>
concept Reference = std::is_reference_v<T>;

// Concept Checking to ensure function passes only by reference 
;
// Helper struct
template<typename T> 
struct function_traits;

// Specific check for function pointers
template<typename Return, typename... Args>
struct function_traits<Return(*)(Args...)> 
{
  using return_type = Return; 
  using arguments   = std::tuple<Args...>;
  static constexpr size_t nargs = sizeof...(Args);
};

// Specific check for const class member function pointers
template<typename Return, typename Class, typename... Args>
struct function_traits<Return(Class::*)(Args...)>
{
  using return_type = Return;
  using arguments   = std::tuple<Args...>;
  static constexpr size_t nargs = sizeof...(Args);
};

// Specific check for const class member function pointers
template<typename Return, typename Class, typename... Args>
struct function_traits<Return(Class::*)(Args...) const>
{
  using return_type = Return;
  using arguments   = std::tuple<Args...>;
  static constexpr size_t nargs = sizeof...(Args);
};

// Specific check for lambdas 
template<typename T>
struct function_traits : function_traits<decltype(&std::remove_reference_t<T>::operator())> {};

// Concept to check if all parameters of a function are references
template<typename Function>
concept AllReferences = requires
{
  // Get parameter types
  typename function_traits<std::decay_t<Function>>::arguments;
} 
&& []<size_t... Is>(std::index_sequence<Is...>) 
{
  using Args = typename function_traits<std::decay_t<Function>>::arguments;
  return (Reference<std::tuple_element_t<Is, Args>> && ...);
} (std::make_index_sequence<function_traits<std::decay_t<Function>>::nargs>{});

// Create runtime string with sensible units
std::string format_runtime_string(double runtime)
{
  std::ostringstream ss_result;

  // Check for empty runtime if user prints table without running 
  if (runtime == 0.0)
  {
    ss_result << "0.0000 s";
    return ss_result.str();
  }

  // Get power of ten -> order of magnitude 
  int order = static_cast<int>(std::floor(std::log10(std::abs(runtime))));
  int prefix_idx = 0;

  // Table of order prefix pairs 
  const std::pair<int, const char*> prefix_table[] = 
  {
    {0, " ns"},
    {1, " us"},
    {2, " ms"},
    {3, " s"},
    {-1, " ps"}
  };

  // Get which prefix group it falls under for specific order of magnitude 
  int group = order / 3;
  for (int i = 0; i < 5; i++)
  {
    if (group <= prefix_table[i].first)
    {
      prefix_idx = i;
      break;
    }
  }

  // Scale runtime and append prefix
  runtime *= std::pow(10.0, -prefix_table[prefix_idx].first * 3);
  ss_result << std::fixed << std::setprecision(4) << runtime << prefix_table[prefix_idx].second;
  return ss_result.str();
}

struct Data
{
  std::string id;
  double      runtime;
  float       speedup;
};

// The basic root utility used by all benchmark templates and their parents
template<typename... Args> 
class ArgManager
{
public:
  ArgManager(size_t iter, Args... args) : 
    iter(iter),
    args_(std::forward<Args>(args)...),
    args_refs(init_ref(std::make_index_sequence<sizeof...(Args)>{}))
  {
    prepare_args(std::make_index_sequence<sizeof...(Args)>{}, std::forward<Args>(args)...);
  }

  // Sets up arguments by type matching safely so that Benchmark can run without issue 
  template<size_t... Is>
  void prepare_args(std::index_sequence<Is...>, Args... args)
  {
    // Assign to class member as tuple
    args_ = std::make_tuple(args...);
    
    // Initializes array to length of tuple Args w/ value 1
    pointer_sizes.resize(sizeof...(Args), 1);
    
    // Check for pointers within args and get size of array (or 1 element for basic pointer)
    (pointer_size_pair<Is, Is+1>(args...), ...);
    // Copy safely 
    args_copy = std::make_tuple(process_argument<Is>(std::get<Is>(args_))...);

    args_refs = std::make_tuple(std::ref(std::get<Is>(args_copy))...);
  }

  // Interface for references to decayed arguments
  auto& copy()
  {
    reset_references(std::make_index_sequence<sizeof...(Args)>{});
    return args_refs;
  }

  // Benchmark indexes
  size_t iter;
  size_t start{0};
  bool has_ran{false};

private:
  // Track input data - Make copies for use in benchmark
  // None of this should be accessible outside the argument manager
  std::tuple<std::decay_t<Args>...> args_;
  std::tuple<std::decay_t<Args>...> args_copy;
  std::tuple<std::reference_wrapper<std::decay_t<Args>>...> args_refs;
  std::vector<size_t> pointer_sizes;
  std::vector<std::unique_ptr<void, std::function<void(void*)>>> ptr_copies;
  
  template<size_t... Is>
  auto init_ref(std::index_sequence<Is...>)
  {
    args_copy = std::make_tuple(std::decay_t<Args>{}...);
    return std::make_tuple(std::ref(std::get<Is>(args_copy))...);
  }

  template<size_t I>
  auto process_argument(auto&& arg)
  {
    using ArgType = std::decay_t<decltype(arg)>;

    if constexpr (Simple<ArgType>)
    {
      // Return argument unchanged 
      return std::forward<decltype(arg)>(arg);
    }
    else if constexpr (Container<ArgType>)
    {
      // Copy the container 
      return ArgType(arg);
    }
    else if constexpr (Pointer<ArgType>)
    {

    // Argument must be a pointer thus a deep copy should be enacted 
    using pointer_type = std::remove_pointer_t<ArgType>;

    size_t size = pointer_sizes[I];

    auto* ptr_copy = new pointer_type[size];
    std::memcpy(ptr_copy, arg, size * sizeof(pointer_type));

    // Push into vector the new unique pointer and a delete method
    ptr_copies.push_back(
      std::unique_ptr<void, std::function<void(void*)>>(
        ptr_copy,
        [](void* ptr) { delete[] static_cast<pointer_type*>(ptr); }
      )
    );
    return ptr_copy;
    }
    else
    {
      // Catch all for references that are still simple types 
      return std::forward<decltype(arg)>(arg);
    }
  }

  // Check for raw pointer and its number of elements proceeding 
  template<size_t I, size_t J, typename... Ts>
  void pointer_size_pair(Ts... args)
  {
    if constexpr (J < sizeof...(Ts) && I < sizeof...(Ts))
    {
      // Get types for first and second arguments passed through template
      using first_arg  = std::decay_t<std::tuple_element_t<I, std::tuple<Ts...>>>;
      using second_arg = std::decay_t<std::tuple_element_t<J, std::tuple<Ts...>>>;

      // Check for raw pointer array size pair to set the size at index
      if constexpr (PointerSizePair<first_arg, second_arg>)
      {
        // Get size from paired integer value 
        pointer_sizes[I] = std::get<J>(std::make_tuple(args...));
        // Check if negative, assume they aren't meant to be paired for simplicity
        if (pointer_sizes[I] < 0)
        {
          // Assume basic pointer in this instance and set to 1
          pointer_sizes[I] = 1;
        }
      }
      else {}   // Value is set to 1 explicitly with .resize()
    }
  }

  // Simple call to process argument for each 
  template<size_t... Is>
  void reset_references(std::index_sequence<Is...>)
  {
    ptr_copies.clear();   // Explicitly frees since unique ptrs 
    // Set args_copy here as persistent
    args_copy = std::make_tuple(process_argument<Is>(std::get<Is>(args_))...);

    // Return explicit references to args 
    args_refs = std::make_tuple(std::ref(std::get<Is>(args_copy))...);
  }
};

// Purely abstract class that implements sort based on runtime in Data
class SortManager
{
public:
  // Virtual methods for each template to implement
  virtual size_t get_count() const = 0;
  virtual Data& get_at_index(size_t index) const = 0;
  virtual void swap(size_t i, size_t j) = 0;

  // Constant selection sort implementation 
  // Since elements will in general be under 16 this is more than fine 
  void sort()
  {
    const size_t size = get_count();
    // Iterate over n^2 and skip identical indices
    for (size_t i = 1; i < size; i++)
    {
      for (size_t j = 1; j < size; j++)
      {
        // Skip identical
        if (i == j) continue;

        // Compare
        if (get_at_index(i).runtime < get_at_index(j).runtime)
          swap(i, j);   // Swap with preferred method 
      }
    }
  }
};    // Inheritable within anonymous namespace

// Root Manager for Basic Template implementation of Benchmark
template<typename Error, typename Return>
class rootSimple : public SortManager     // Purely abstract class being inherited 
{
public:
  using fn_error = std::function<Error(Return, Return)>;

  // Define Result structure for this template 
  struct Result 
  {
    Data   data; 
    Return value;
    Error  error;
  };

  // Constructor
  rootSimple(fn_error func) : error_function(func) { results.clear(); }

  // Print implementation 
  void print()
  {
    SortManager::sort();                  // Abstract class handles sorting  

    // Header
    std::cout << std::left << std::setw(32) << "ID"
              << std::setw(16) << "Runtime"
              << std::setw(16) << "Speedup"
              << std::setw(16) << "Result"
              << std::setw(16) << "Error"
              << '\n';
    std::cout << "----------------------------------------------------------------------------------------------"
              << '\n';

    for (size_t i = 0; i < results.size(); i++)
    {
      std::cout << std::left << std::setw(32) << results[i].data.id;
      
      std::string runtime_str = format_runtime_string(results[i].data.runtime);
      std::cout << std::left << std::setw(16) << runtime_str;
      
      // Formatted string to include x fast 
      std::ostringstream speedup_str;
      speedup_str << std::fixed << std::setprecision(6) << results[i].data.speedup << "x fast";
      std::cout << std::left << std::setw(16) << speedup_str.str();
      
      // Result column
      std::cout << std::setw(16) << std::fixed << std::setprecision(6) << results[i].value;

      // Error column
      std::cout << std::setw(16) << std::fixed << std::setprecision(6) << results[i].error;
      std::cout << '\n';
    }
  }

  // Members unique to this template
  fn_error error_function;
  std::vector<Result> results;

private:
  // Implementation of virtual methods 
  size_t get_count() const override { return results.size(); }
  Data& get_at_index(size_t i) const override { return const_cast<Data&>(results[i].data); }
  void swap(size_t i, size_t j) override { std::swap(results[i], results[j]); }
};

// Root Manager for Complex Error functions 
// Error function requires encapsulation of some struct or type passed through args... 
template<typename Error>
class rootComplexError : public SortManager 
{
public:
  // User is responsible for error function recasting to correct type in error function
  using fn_error = std::function<Error(void*, void*)>;
  
  struct Result 
  {
    Data  data;
    Error error;
  };

  rootComplexError(fn_error func) : error_function(func) { results.clear(); }

  // Print implementation 
  void print()
  {
    SortManager::sort();                  // Abstract class handles sorting  

    // Header
    std::cout << std::left << std::setw(32) << "ID"
              << std::setw(16) << "Runtime"
              << std::setw(16) << "Speedup"
              << std::setw(16) << "Error"
              << '\n';
    std::cout << "--------------------------------------------------------------------------------------"
              << '\n';

    for (size_t i = 0; i < results.size(); i++)
    {
      std::cout << std::left << std::setw(32) << results[i].data.id;
      
      std::string runtime_str = format_runtime_string(results[i].data.runtime);
      std::cout << std::left << std::setw(16) << runtime_str;
      
      // Formatted string to include x fast 
      std::ostringstream speedup_str;
      speedup_str << std::fixed << std::setprecision(6) << results[i].data.speedup << "x fast";
      std::cout << std::left << std::setw(16) << speedup_str.str();
      
      // Errorcolumn
      std::cout << std::setw(16) << std::fixed << std::setprecision(6) << results[i].error;
      std::cout << '\n';
    }
  }

  fn_error error_function;
  std::vector<Result> results;

private:
  size_t get_count() const override { return results.size(); }
  Data& get_at_index(size_t i) const override { return const_cast<Data&>(results[i].data); }
  void swap(size_t i, size_t j) override { std::swap(results[i], results[j]); }
};

// Simple case of no error, no return type. No template required in instantiation  
class rootVoid : public SortManager 
{
public:
   
  rootVoid(void) : results(1, Data()) {}

  // Print implementation 
  void print() 
  {
    SortManager::sort();                  // Abstract class handles sorting  

    // Header
    std::cout << std::left << std::setw(32) << "ID"
              << std::setw(16) << "Runtime"
              << std::setw(16) << "Speedup"
              << '\n';
    std::cout << "---------------------------------------------------------------------------------"
              << '\n';

    for (size_t i = 0; i < results.size(); i++)
    {
      std::cout << std::left << std::setw(32) << results[i].id;
      
      std::string runtime_str = format_runtime_string(results[i].runtime);
      std::cout << std::left << std::setw(16) << runtime_str;
      
      // Formatted string to include x fast 
      std::ostringstream speedup_str;
      speedup_str << std::fixed << std::setprecision(6) << results[i].speedup << "x fast";
      std::cout << std::left << std::setw(16) << speedup_str.str();
      
    }
  }

  std::vector<Data> results;

private:
  size_t get_count() const override { return results.size(); }
  Data& get_at_index(size_t i) const override { return const_cast<Data&>(results[i]); }
  void swap(size_t i, size_t j) override { std::swap(results[i], results[j]); }
};

}     // End of anonymous namespace

// Explicit Benchmark Implementations for templates. Abstract methods may not be inherited 

// Base case 
template<typename Error, typename Return, typename... Args>
class Benchmark 
{
public:
  // Force typename Function into passing by reference 
  using fn_benchmark = std::function<Return(std::add_lvalue_reference_t<std::decay_t<Args>>...)>;
  
  // Requiers that benchmarked functions only pass by reference
  template<typename Function>
    requires AllReferences<Function>
  Benchmark(rootSimple<Error, Return>::fn_error func_e, Function&& func_b, size_t iter, Args... args) :
    simple_(func_e),
    amgr_(iter, args...)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");

    // Setup functions vector 
    functions.clear();
    // Force into fn_benchmark
    functions.push_back(std::forward<Function>(func_b));
    // Setup argument manager
    amgr_.prepare_args(std::make_index_sequence<sizeof...(Args)>{}, std::forward<Args>(args)...);
    // Call Baseline Benchmark
    baseline();
  }

  template<typename Function>
    requires AllReferences<Function>
  void insert(Function&& func, const std::string& id)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");
    // Setup benchmark indexes
    if (amgr_.has_ran) amgr_.has_ran = false;
    amgr_.start = (amgr_.start == 0) ? functions.size() : amgr_.start;

    functions.push_back(func);

    using Result = typename rootSimple<Error, Return>::Result;

    Result init = (Result)
    {
      .data = (Data)
      {
        .id      = id,
        .runtime = 0.0,
        .speedup = 1.0
      },
      .value = Return(),
      .error = Error()
    };

    // Initialize to Result 
    simple_.results.push_back(init);
  }

  bool run()
  {
    // Conditions for run to be uneccesary - return flag for posterity
    if (amgr_.start == 0)
      return false;

    const size_t n_functions = functions.size();
    if (n_functions == 1 || !n_functions)
      return false;

    const size_t run_count = n_functions - amgr_.start;
    
    // Vectors to store information from benchmark
    std::vector<double> average_runtime(run_count, 0.0);
    std::vector<Return> local_results(run_count);       // Hold reference to avoid copy each iter
    std::vector<Error>  local_errors(run_count);
    
    for (size_t i = 0; i < amgr_.iter; i++)
    {
      for (size_t j = amgr_.start; j < n_functions; j++)
      {
        auto& arguments = amgr_.copy();
        const size_t k = j - amgr_.start; 

        // Run function with high resolution timer
        auto start = std::chrono::high_resolution_clock::now();
        local_results[k] = std::apply(functions[j], arguments);
        auto end   = std::chrono::high_resolution_clock::now();

        auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        average_runtime[k] += static_cast<double>(runtime.count());
      }
    }

    // Set results loop
    for (size_t i = amgr_.start; i < n_functions; i++)
    {
      const size_t j = i - amgr_.start;

      // Get timings
      average_runtime[j] /= amgr_.iter;
      float speedup = simple_.results[0].data.runtime / average_runtime[j];

      // Get error
      local_errors[j] = simple_.error_function(simple_.results[0].value, local_results[j]);

      // Set values of results vector 
      simple_.results[i].data.runtime = average_runtime[j];
      simple_.results[i].data.speedup = speedup;
      simple_.results[i].value        = local_results[j];
      simple_.results[i].error        = local_errors[j];
    }

    // Reset benchmark indexes
    amgr_.start   = 0;
    amgr_.has_ran = true;
    return true;
  }

  // Print results 
  void print() { simple_.print(); }

private:
  // Standard members
  std::vector<fn_benchmark> functions;
  // Composed classes 
  rootSimple<Error, Return> simple_;
  ArgManager<Args...> amgr_;

  // Baseline method called in constructor for 0th element
  void baseline()
  {
    Return baseline_value  = Return();
    Error  baseline_error  = Error();
    double average_runtime = 0.0;

    for (size_t i = 0; i < amgr_.iter; i++)
    {
      // Check if arguments require a copy 
      auto& arguments = amgr_.copy();

      // Run with timer
      auto start = std::chrono::high_resolution_clock::now();
      auto value = std::apply(functions[0], arguments);
      // Set on last iter
      if (i == amgr_.iter - 1)
        baseline_value = value;
      auto end   = std::chrono::high_resolution_clock::now();

      auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      average_runtime += static_cast<double>(runtime.count());
    }
    average_runtime /= amgr_.iter;  // Ave of accumulated
    
    // Set baseline error
    baseline_error = simple_.error_function(baseline_value, baseline_value); 

    using Result = typename rootSimple<Error, Return>::Result;
    Result result = (Result)
    {
      .data = (Data)
      {
        .id      = "Baseline",
        .runtime = average_runtime,
        .speedup = 1.0
      },
      .value = baseline_value,
      .error = baseline_error
    };
    simple_.results.push_back(result);
  }
};

// Complex Error function template 
// User is responsible for casting arguments within error function 
//
// Error function follows error func(void*, void*)
// where void* refers to some struct or value being compared within the function 
//
template<typename Error, typename... Args>
class Benchmark<Error, void, Args...>
{
public:
  using fn_benchmark = std::function<void(std::add_lvalue_reference_t<std::decay_t<Args>>...)>;
  using fn_error = typename rootComplexError<Error>::fn_error;      // Grab to simplify references

  // Constructor requires template for type to be encapsulated 
  template<typename Function, typename S>
    requires AllReferences<Function>
  Benchmark(fn_error func_e, Function&& func_b, size_t iter, Args... args) :
    stype(typeid(S)),
    complex_(func_e),
    amgr_(iter, args...)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");

    functions.clear();
    functions.push_back(std::forward<Function>(func_b));

    // Setup argument manager
    amgr_.prepare_args(std::make_index_sequence<sizeof...(Args)>{}, std::forward<Args>(args)...);

    // Call Baseline Benchmark
    baseline();
  }

  template<typename Function>
    requires AllReferences<Function>
  void insert(Function&& func, const std::string& id)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");

    // Setup benchmark indexes
    if (amgr_.has_ran) amgr_.has_ran = false;
    amgr_.start = (amgr_.start == 0) ? functions.size() : amgr_.start;

    functions.push_back(std::forward<Function>(func));

    using Result = typename rootComplexError<Error>::Result;
    Result init = (Result)
    {
      .data = (Data)
      {
        .id      = id,
        .runtime = 0.0,
        .speedup = 1.0
      },
      .error = Error()
    };

    // Initialize to Result 
    complex_.results.push_back(init);
  }

  bool run()
  {
    if (amgr_.start == 0)
      return false;

    const size_t n_functions = functions.size();
    if (n_functions == 1 || !n_functions)
      return false;

    const size_t run_count = n_functions - amgr_.start;
    
    // Vectors to store information from benchmark
    std::vector<double> average_runtime(run_count, 0.0);
    std::vector<Error>  local_errors(run_count);
    std::vector<std::any>  local_arguments(run_count);      // Stores values of requirest stype
    
    for (size_t i = 0; i < amgr_.iter; i++)
    {
      for (size_t j = amgr_.start; j < n_functions; j++)
      {
        auto& arguments = amgr_.copy();
        const size_t k = j - amgr_.start;

        // Run function with high resolution timer
        auto start = std::chrono::high_resolution_clock::now();
        (void)std::apply(functions[j], arguments);
        auto end   = std::chrono::high_resolution_clock::now();

        // Collect modified argument from each call
        if (i == amgr_.iter - 1)
          local_arguments[j] = std::forward<stype>(get_first_stype<stype>(std::make_index_sequence<sizeof...(Args)>{}, amgr_.args_copy));

        auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        average_runtime[k] += static_cast<double>(runtime.count());
      }
    }

    // Set results loop
    for (size_t i = amgr_.start; i < n_functions; i++)
    {
      const size_t j = i - amgr_.start;

      // Get timings
      average_runtime[j] /= amgr_.iter;
      float speedup = complex_.results[0].data.runtime / average_runtime[j];

      // Get error for jth local argument
      local_errors[j] = complex_.error_function(&base_argument, &local_arguments[j]);

      // Set values of results vector 
      complex_.results[i].data.runtime = average_runtime[j];
      complex_.results[i].data.speedup = speedup;
      complex_.results[i].error        = local_errors[j];
    }

    // Reset benchmark indexes
    amgr_.start   = 0;
    amgr_.has_ran = true;
    return true;
  }

  void print() { complex_.print(); }

private:
  // Standard members
  std::vector<fn_benchmark> functions;
  std::any base_argument;
  std::type_index stype;
  // Composed members
  rootComplexError<Error> complex_;
  ArgManager<Args...> amgr_;

  void baseline()
  {
    double average_runtime = 0.0;

    for (size_t i = 0; i < amgr_.iter; i++)
    {
      auto& arguments = amgr_.copy();

      // Run with timer
      auto start = std::chrono::high_resolution_clock::now();
      (void)std::apply(functions[0], arguments);
      auto end   = std::chrono::high_resolution_clock::now();

      auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      average_runtime += static_cast<double>(runtime.count());
    }
    average_runtime /= amgr_.iter;  // Ave of accumulated
    
    // Get argument matching type of type requested to be encapsulated 
    base_argument = std::forward<stype>(get_first_stype<stype>(std::make_index_sequence<sizeof...(Args)>{}, amgr_.args_copy));

    // Copy twice 
    auto a = std::any_cast<stype>(base_argument);
    auto b = std::any_cast<stype>(base_argument);

    // Call error function on baseline modified argument against itself 
    Error base_error = complex_.error_function(&a, &b);

    using Result = typename rootComplexError<Error>::Result;
    Result result = (Result)
    {
      .data = (Data)
      {
        .id      = "Baseline",
        .runtime = average_runtime,
        .speedup = 1.0
      },
      .error = base_error 
    };

    complex_.results.push_back(result);
  }

  template<typename S, size_t... Is>
  S& get_first_stype(std::index_sequence<Is...>, std::tuple<Args...>& src)
  {
    S* result = nullptr;

    [&]<size_t... Js>(std::index_sequence<Js...>) 
    {
      ((result = std::is_same_v<S, std::tuple_element_t<Js, std::tuple<Args...>>> ?
        &std::get<Js>(src) : result), ...);
    } (std::make_index_sequence<sizeof...(Args)>{});

    if (result == nullptr) 
    {
      throw std::runtime_error("Type not found within argument list");
    }
    return *result;   // Return dereferenced result
  }
};

// Simplest case where the user only cares about difference in runtime 
// Function doesn't return anything (Or is ignoring the resulting value) and ignoring error
template<typename... Args>
class Benchmark<void, void, Args...>
{
public:
  using fn_benchmark = std::function<void(std::add_lvalue_reference_t<std::decay_t<Args>>...)>;

  // No error constructor overload 
  template<typename Function>
    requires AllReferences<Function>
  Benchmark(Function&& func_b, size_t iter, Args... args) : 
    rvoid_(),
    amgr_(iter, args...)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");

    // Setup functions vector 
    functions.clear();
    functions.push_back(std::forward<Function>(func_b));
    // Setup argument manager
    amgr_.prepare_args(std::make_index_sequence<sizeof...(Args)>{}, std::forward<Args>(args)...);
    // Call Baseline Benchmark
    baseline();
  }

  template<typename Function>
    requires AllReferences<Function>
  void insert(Function&& func, const std::string& id)
  {
    static_assert(AllReferences<Function>, "Benchmarked function must pass all values by reference");

    // Setup benchmark indexes
    if (amgr_.has_ran) amgr_.has_ran = false;
    amgr_.start = (amgr_.start == 0) ? functions.size() : amgr_.start;

    functions.push_back(std::forward<Function>(func));

    Data init = (Data)
    {
      .id      = id,
      .runtime = 0.0,
      .speedup = 1.0
    };

    // Initialize to Result 
    rvoid_.results.push_back(init);
  }

  bool run()
  {
    if (amgr_.start == 0)
      return false;

    const size_t n_functions = functions.size();
    if (n_functions == 1 || !n_functions)
      return false;

    const size_t run_count = n_functions - amgr_.start;
    
    // Vectors to store information from benchmark
    std::vector<double> average_runtime(run_count, 0.0);
    
    for (size_t i = 0; i < amgr_.iter; i++)
    {
      for (size_t j = amgr_.start; j < n_functions; j++)
      {
        auto& arguments = amgr_.copy();

        const size_t k = j - amgr_.start;

        // Run function with high resolution timer
        auto start = std::chrono::high_resolution_clock::now();
        (void)std::apply(functions[j], arguments);
        auto end   = std::chrono::high_resolution_clock::now();

        auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        average_runtime[k] += static_cast<double>(runtime.count());
      }
    }

    // Set results loop
    for (size_t i = amgr_.start; i < n_functions; i++)
    {
      const size_t j = i - amgr_.start;

      // Get timings
      average_runtime[j] /= amgr_.iter;
      float speedup = rvoid_.results[0].runtime / average_runtime[j];

      // Set values of results vector 
      rvoid_.results[i].runtime = average_runtime[j];
      rvoid_.results[i].speedup = speedup;
    }

    // Reset benchmark indexes
    amgr_.start   = 0;
    amgr_.has_ran = true;
    return true;
  }

  void print() { rvoid_.print(); }

private:
  // Standard Members 
  std::vector<fn_benchmark> functions;
  // Composed Classes
  rootVoid rvoid_;
  ArgManager<Args...> amgr_;

  void baseline()
  {
    double average_runtime = 0.0;

    for (size_t i = 0; i < amgr_.iter; i++)
    {
      // Check if arguments require a copy 
      auto& arguments = amgr_.copy();

      // Run with timer
      auto start = std::chrono::high_resolution_clock::now();
      (void)std::apply(functions[0], arguments);
      auto end   = std::chrono::high_resolution_clock::now();

      auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      average_runtime += static_cast<double>(runtime.count());
    }
    average_runtime /= amgr_.iter;  // Ave of accumulated
    
    Data result = (Data)
    {
      .id      = "Baseline",
      .runtime = average_runtime,
      .speedup = 1.0
    };
    rvoid_.results.push_back(result);
  }
};

#endif // BENCHMARK_HPP
