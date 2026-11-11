# Compilação da APS 2 (Supercomp 2026.2)
# No Franky, compile no nó de login (os nós de computação não têm os headers
# da glibc). O gcc 14.2.0 já vem carregado por padrão.

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall
NVCC      = nvcc
NVCCFLAGS = -O3

all: legado cpu gpu

# Código legado sem alterações, compilado com -O3 (referência das medições)
legado: barenn.cpp
	$(CXX) $(CXXFLAGS) barenn.cpp -o legado

# Versão otimizada para CPU (OpenMP na leitura e na avaliação)
cpu: barenn_cpu.cpp
	$(CXX) $(CXXFLAGS) -fopenmp barenn_cpu.cpp -o cpu

# Versão CUDA (no Franky: module load cuda/12.8.1)
gpu: barenn_gpu.cu
	$(NVCC) $(NVCCFLAGS) barenn_gpu.cu -o gpu

# Experimento (resultado negativo): OpenMP dentro do laço de treino
omp_no_treino: experimentos/omp_no_treino.cpp
	$(CXX) $(CXXFLAGS) -fopenmp experimentos/omp_no_treino.cpp -o omp_no_treino

# Mesmo código sem otimização do compilador (só para contexto)
legado_O0: barenn.cpp
	$(CXX) -std=c++17 -O0 barenn.cpp -o legado_O0

# Versões com símbolos de depuração para o callgrind
legado_prof: barenn.cpp
	$(CXX) $(CXXFLAGS) -g barenn.cpp -o legado_prof

cpu_prof: barenn_cpu.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -g barenn_cpu.cpp -o cpu_prof

clean:
	rm -f legado legado_O0 legado_prof cpu cpu_prof omp_no_treino gpu

.PHONY: all clean
