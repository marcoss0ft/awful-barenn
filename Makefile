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

# Versão CUDA (no Franky: module load cuda/12.8.1). Sem contração de a*b+c em
# FMA, a saída fica idêntica à do legado (Issues #7 e #20).
gpu: barenn_gpu.cu
	$(NVCC) $(NVCCFLAGS) --fmad=false barenn_gpu.cu -o gpu

# Controle: a mesma versão com FMA (padrão do nvcc). A saída difere na 4ª casa.
gpu_fma: barenn_gpu.cu
	$(NVCC) $(NVCCFLAGS) barenn_gpu.cu -o gpu_fma

# Experimentos de controle do kernel com memória compartilhada
gpu_sem_padding: experimentos/gpu_sem_padding.cu
	$(NVCC) $(NVCCFLAGS) experimentos/gpu_sem_padding.cu -o gpu_sem_padding

gpu_pesos_global: experimentos/gpu_pesos_global.cu
	$(NVCC) $(NVCCFLAGS) experimentos/gpu_pesos_global.cu -o gpu_pesos_global

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
	rm -f legado legado_O0 legado_prof cpu cpu_prof omp_no_treino gpu gpu_fma gpu_sem_padding gpu_pesos_global

.PHONY: all clean
