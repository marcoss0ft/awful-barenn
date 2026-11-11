# Compilação da APS 2 (Supercomp 2026.2)
# No Franky, compile no nó de login (os nós de computação não têm os headers
# da glibc). O gcc 14.2.0 já vem carregado por padrão.

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall

all: legado

# Código legado sem alterações, compilado com -O3 (referência das medições)
legado: barenn.cpp
	$(CXX) $(CXXFLAGS) barenn.cpp -o legado

# Mesmo código sem otimização do compilador (só para contexto)
legado_O0: barenn.cpp
	$(CXX) -std=c++17 -O0 barenn.cpp -o legado_O0

# Versão com símbolos de depuração para o callgrind
legado_prof: barenn.cpp
	$(CXX) $(CXXFLAGS) -g barenn.cpp -o legado_prof

clean:
	rm -f legado legado_O0 legado_prof

.PHONY: all clean
