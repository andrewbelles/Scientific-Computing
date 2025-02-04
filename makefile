CC = gcc
CXX = g++
CCU = nvcc
CFLAGS = -g -O3 -pg -Wall -Iinclude/ 
CXXFLAGS = -O3 -std=c++17 -pg -Wall -Iinclude/
CUFLAGS = -std=c++17 -pg -lineinfo -Iinclude/ -Xcompiler -Wall -rdc=true 
LDFLAGS = -lGLEW -lGL -lGLU -lSDL2
CUDA_LDFLAGS = -lcudart -lcudadevrt 

EXEC = fluidsim
C_SRC = $(wildcard src/*.c)
CU_SRC = $(wildcard src/cuda/*.cu)
CXX_SRC = $(wildcard src/*.cpp)
ASM_SRC = $(wildcard src/*.asm)
OBJ = $(C_SRC:src/%.c=bin/%.o) $(CXX_SRC:src/%.cpp=bin/%.o) $(CU_SRC:src/cuda/%.cu=bin/%.o) 
#$(ASM_SRC:src/%.asm=bin/%.o)

ARGS = 5000 25

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CCU) -rdc=true $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

bin/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: src/cuda/%.cu
	$(CCU) $(CUFLAGS) -c $< -o $@

#bin/%.o: src/%.asm
#	nasm -f elf64 $< -o $@

profile_cpu: $(EXEC)
	@echo "Profiling CPU"
	./$(EXEC) $(ARGS)
	gprof $(EXEC) gmon.out > cpu_prof.txt

profile_gpu: $(EXEC)
	@echo "Profiling GPU"
	sudo ncu --export gpu_prof.ncu-rep -f ./$(EXEC) $(ARGS) 

clean:
	@echo "Removing Files"
	rm -f bin/*.o $(EXEC) gmon.out *.prof *.txt

.PHONY: all clean profile_cpu profile_gpu
