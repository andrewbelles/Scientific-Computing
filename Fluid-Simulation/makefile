CC = gcc
CXX = g++
CCU = nvcc
CFLAGS = -g -O3 -pg -Wall -Iinclude/ 
CXXFLAGS = -O3 -std=c++20 -pg -Wall -Iinclude/
CUFLAGS = -std=c++20 -gencode arch=compute_89,code=sm_89 -pg -lineinfo -Iinclude/ -Xcompiler -Wall -rdc=true 
LDFLAGS = -lGLEW -lGL -lGLU -lSDL2
CUDA_LDFLAGS = -lcudart -lcudadevrt 

EXEC = fluidsim
C_SRC = $(wildcard src/*.c)
CU_SRC = $(wildcard src/*.cu)
CXX_SRC = $(wildcard src/*.cpp)
OBJ = $(C_SRC:src/%.c=bin/%.o) $(CXX_SRC:src/%.cpp=bin/%.o) $(CU_SRC:src/%.cu=bin/%.o) 

ARGS = 5000 25

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CCU) -gencode arch=compute_89,code=sm_89 -rdc=true $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

bin/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: src/%.cu
	$(CCU) $(CUFLAGS) -c $< -o $@

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
