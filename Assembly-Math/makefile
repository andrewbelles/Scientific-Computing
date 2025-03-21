compiler = clang++
assembler = nasm

# Flags to use 
cpp_flags = -std=c++20 -Wall -Iinclude/
asm_flags = -felf64

exec = mathtest

# Collect source files 
cpp_src = $(wildcard src/*.cpp)
asm_src = $(wildcard src/*.asm)

# Where to emplace obj files 
obj = $(cpp_src:src/%.cpp=bin/%.o) $(asm_src:src/%.asm=bin/%.o)

all: $(exec)

# Create executable
$(exec): $(obj)
	$(compiler) -o $@ $^

# Create cpp object files 
bin/%.o: src/%.cpp
	$(compiler) $(cpp_flags) -c -o $@ $<

# Asm obj 
bin/%.o: src/%.asm 
	$(assembler) $(asm_flags) -o $@ $<

clean:
	rm -rf $(obj) $(exec)

.PHONY: all clean
