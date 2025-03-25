#include "../benchmark.hpp"
#include <functional>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <fstream>

#define TABLE_SIZE 1009

typedef struct entry {
  size_t entries{0};      // Set to 0 initially
  std::pair<size_t, std::string> pair;
  struct entry* next{nullptr};
} hashtable_t;

void init_table(hashtable_t*& table) { table = new hashtable_t[TABLE_SIZE](); }

void table_deleter(hashtable_t*& table)
{
  // No delete on nullptr
  if (!table)
    return;

  for (size_t i = 0; i < TABLE_SIZE; i++)
  {
    // Free all nodes in each buckets llist 
    struct entry* curr = &table[i];
    while (curr != nullptr)
    {
      entry* destroy = curr;
      curr = curr->next;
      delete destroy;
    }
  }

  // Free table base ptr
  delete[] table;
  table = nullptr;
}

// Cheap hash 
size_t djb2_hash(size_t& size, std::string& id)
{
  size_t hash = 5381;
  int i = 0;

  while (id[i] != 0)
    hash = ((hash << 5) + hash) + id[i];

  // Fit to table size
  return hash % size;
}

// My naive hash impl xor of raw address and ascii value 
size_t naive_hash(size_t& size, std::string& id)
{
  size_t hash = 5381;
  int i = 0;

  while (id[i] != 0)
    hash += (id[i] ^ reinterpret_cast<size_t>(&id[i]));
  
  return hash % 100;
}

// Hash function pointer
using hash_function = std::function<size_t(size_t&, std::string&)>;

// Ensure pointer is passed with size pair proceeding 
void fill_table(hashtable_t*& table, hash_function& func, size_t& size, std::vector<std::string>& id_list)
{
  for (std::string& id : id_list)
  {
    // Get hash 
    size_t hash = func(size, id); 
    table[hash].entries++;

    // Get pointer to table[hash]
    struct entry* curr = &table[hash];

    // Advance to end of linked list in bucket 
    while (curr->next != nullptr)
      curr = curr->next;

    struct entry* new_entry = new entry();
    new_entry->pair = std::pair<size_t, std::string>(hash, id);
    new_entry->next = nullptr;

    curr->next = new_entry;
  }
}

// Count n differences in table. Capture size of table in lambda
// Table size needs to be a macro for this error function 
size_t compare_tables(void* b_ptr, void* nb_ptr)
{ 
  hashtable_t* base     = reinterpret_cast<hashtable_t*>(b_ptr);
  hashtable_t* non_base = reinterpret_cast<hashtable_t*>(nb_ptr);

  size_t difference(0);

  for (size_t i = 0; i < TABLE_SIZE; i++)
  {
    struct entry* a = &base[i];
    struct entry* b = &non_base[i];

    while (a != nullptr && b != nullptr)
    {
      if (a->pair.second != b->pair.second)
        difference++;

      a = a->next;
      b = b->next;
    }

    while (a != nullptr)
    {
      difference++;
      a = a->next;
    }

    while (b != nullptr)
    {
      difference++;
      b = b->next;
    }
  }

  return difference;
}

int main(void)
{
  std::ifstream file("words.txt");
  if (!file.is_open())
  {
    throw::std::runtime_error("File not found");
  }

  std::vector<std::string> id_list;
  std::string line;

  while (std::getline(file, line))
  {
    id_list.push_back(line);
  }

  file.close();

  hashtable_t* base_table  = nullptr;
  hashtable_t* naive_table = nullptr; 

  init_table(base_table);

  auto baseline = [&](hashtable_t*& table, size_t& size, std::vector<std::string>& id_list)
  {
    hash_function func = djb2_hash;
    (void)fill_table(table, func, size, id_list);
  };

  using Function = std::function<void(hashtable_t*&, size_t&, std::vector<std::string>&)>;
// Let the compiler deduce Function from baseline
Benchmark<size_t, void, hashtable_t*, size_t, std::vector<std::string>>
    bench(compare_tables, baseline, TypeTag<hashtable_t*>{}, 10, base_table, TABLE_SIZE, id_list);

  auto naive = [&](hashtable_t*& table, size_t& size, std::vector<std::string>& id_list)
  {
    hash_function func = naive_hash;
    (void)fill_table(table, func, size, id_list);
  };

  bench.insert(naive, "Naive Hash Function");
  bench.run();
  bench.print();

  table_deleter(base_table);

  return 0;
}

