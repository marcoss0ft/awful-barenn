#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// ---------------------------------------------------------------------------
// Parte CUDA
// ---------------------------------------------------------------------------

// Interrompe o programa com uma mensagem se alguma chamada CUDA falhar.
void checar(cudaError_t erro) {
    if (erro != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(erro));
    }
}

constexpr int BLOCO = 128;
int grade(int n) { return (n + BLOCO - 1) / BLOCO; }

// Uma thread por neurônio de saída: z = viés + soma(entrada * pesos).
__global__ void forward_kernel(const float* d_in, const float* d_w, const float* d_b,
                               int n_in, int n_out, bool final,
                               float* d_out, float* d_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    float sum = d_b[i];
    for (int j = 0; j < n_in; ++j) {
        sum += d_in[j] * d_w[i * n_in + j];
    }
    d_z[i] = sum;
    d_out[i] = final ? sum : fmaxf(0.0f, sum);
}

// Uma thread por neurônio de saída: delta = gradiente * derivada da ReLU.
__global__ void delta_kernel(const float* d_grad_out, const float* d_z,
                             int n_out, bool final, float* d_delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    d_delta[i] = final ? d_grad_out[i]
                       : d_grad_out[i] * (d_z[i] > 0.0f ? 1.0f : 0.0f);
}

// Uma thread por entrada: grad_in[i] = soma(delta[j] * pesos[j][i]).
__global__ void grad_in_kernel(const float* d_delta, const float* d_w,
                               int n_in, int n_out, float* d_grad_in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_in) return;
    float sum = 0.0f;
    for (int j = 0; j < n_out; ++j) {
        sum += d_delta[j] * d_w[j * n_in + i];
    }
    d_grad_in[i] = sum;
}

// Uma thread por peso: SGD w -= taxa * delta * entrada.
// A thread da coluna 0 de cada linha também atualiza o viés.
__global__ void update_kernel(const float* d_in, const float* d_delta, int n_in,
                              int n_out, float taxa, float* d_w, float* d_b) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_in * n_out) return;
    int i = k / n_in;
    int j = k % n_in;
    d_w[k] -= taxa * (d_delta[i] * d_in[j]);
    if (j == 0) d_b[i] -= taxa * d_delta[i];
}

// ---------------------------------------------------------------------------
// Treino de uma época inteira em um único kernel, com os pesos em memória
// compartilhada. O SGD é serial (cada amostra usa os pesos atualizados pela
// anterior), então usamos UM bloco: as threads dividem o trabalho de cada
// amostra e __syncthreads() separa as etapas.
// ---------------------------------------------------------------------------
constexpr int OCULTA1 = 32;
constexpr int OCULTA2 = 16;
// Passo (número de colunas) de cada linha das matrizes na memória compartilhada.
constexpr int PASSO1 = entradas;
constexpr int PASSO2 = OCULTA1;
constexpr int PASSO3 = OCULTA2;

__global__ void treinar_epoca_kernel(const float* d_imagens, const int* d_rotulos,
                                     int n, float taxa,
                                     float* d_w1, float* d_b1, float* d_w2, float* d_b2,
                                     float* d_w3, float* d_b3, double* d_perda) {
    __shared__ float w1[OCULTA1 * PASSO1], b1[OCULTA1];
    __shared__ float w2[OCULTA2 * PASSO2], b2[OCULTA2];
    __shared__ float w3[classes * PASSO3], b3[classes];
    __shared__ float x[entradas], a1[OCULTA1], a2[OCULTA2];
    __shared__ float d1[OCULTA1], d2[OCULTA2], d3[classes];
    const int t = threadIdx.x;

    // 1) Pesos: memória global -> compartilhada (uma vez por época).
    for (int k = t; k < OCULTA1 * entradas; k += blockDim.x)
        w1[(k / entradas) * PASSO1 + k % entradas] = d_w1[k];
    for (int k = t; k < OCULTA2 * OCULTA1; k += blockDim.x)
        w2[(k / OCULTA1) * PASSO2 + k % OCULTA1] = d_w2[k];
    for (int k = t; k < classes * OCULTA2; k += blockDim.x)
        w3[(k / OCULTA2) * PASSO3 + k % OCULTA2] = d_w3[k];
    if (t < OCULTA1) b1[t] = d_b1[t];
    if (t < OCULTA2) b2[t] = d_b2[t];
    if (t < classes) b3[t] = d_b3[t];

    double perda_total = 0.0;  // usada só pela thread 0
    for (int amostra = 0; amostra < n; ++amostra) {
        if (t < entradas) x[t] = d_imagens[amostra * entradas + t];
        __syncthreads();

        // 2) Forward: uma thread por neurônio (mesma ordem de soma da CPU).
        if (t < OCULTA1) {
            float sum = b1[t];
            for (int j = 0; j < entradas; ++j) sum += x[j] * w1[t * PASSO1 + j];
            a1[t] = fmaxf(0.0f, sum);
        }
        __syncthreads();
        if (t < OCULTA2) {
            float sum = b2[t];
            for (int j = 0; j < OCULTA1; ++j) sum += a1[j] * w2[t * PASSO2 + j];
            a2[t] = fmaxf(0.0f, sum);
        }
        __syncthreads();
        // Última camada (linear): d3 = escore - alvo = gradiente da perda.
        if (t < classes) {
            float sum = b3[t];
            for (int j = 0; j < OCULTA2; ++j) sum += a2[j] * w3[t * PASSO3 + j];
            d3[t] = sum - (t == d_rotulos[amostra] ? 1.0f : 0.0f);
        }
        __syncthreads();

        // 3) Perda da amostra (thread 0, mesma ordem da função perda da CPU).
        if (t == 0) {
            float total = 0.0f;
            for (int i = 0; i < classes; ++i) total += 0.5f * d3[i] * d3[i];
            perda_total += total;
        }

        // 4) Backward: gradientes com os pesos ANTIGOS. A ReLU só deixa passar
        //    o gradiente onde a ativação é positiva (a > 0 equivale a z > 0).
        if (t < OCULTA2) {
            float sum = 0.0f;
            for (int j = 0; j < classes; ++j) sum += d3[j] * w3[j * PASSO3 + t];
            d2[t] = sum * (a2[t] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();
        if (t < OCULTA1) {
            float sum = 0.0f;
            for (int j = 0; j < OCULTA2; ++j) sum += d2[j] * w2[j * PASSO2 + t];
            d1[t] = sum * (a1[t] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();

        // 5) Atualização SGD: as threads dividem os pesos de cada camada.
        for (int k = t; k < classes * OCULTA2; k += blockDim.x) {
            int i = k / OCULTA2, j = k % OCULTA2;
            w3[i * PASSO3 + j] -= taxa * (d3[i] * a2[j]);
        }
        if (t < classes) b3[t] -= taxa * d3[t];
        for (int k = t; k < OCULTA2 * OCULTA1; k += blockDim.x) {
            int i = k / OCULTA1, j = k % OCULTA1;
            w2[i * PASSO2 + j] -= taxa * (d2[i] * a1[j]);
        }
        if (t < OCULTA2) b2[t] -= taxa * d2[t];
        for (int k = t; k < OCULTA1 * entradas; k += blockDim.x) {
            int i = k / entradas, j = k % entradas;
            w1[i * PASSO1 + j] -= taxa * (d1[i] * x[j]);
        }
        if (t < OCULTA1) b1[t] -= taxa * d1[t];
        // A próxima amostra só começa com todos os pesos atualizados.
        __syncthreads();
    }

    // 6) Pesos atualizados: compartilhada -> global (para a próxima época).
    for (int k = t; k < OCULTA1 * entradas; k += blockDim.x)
        d_w1[k] = w1[(k / entradas) * PASSO1 + k % entradas];
    for (int k = t; k < OCULTA2 * OCULTA1; k += blockDim.x)
        d_w2[k] = w2[(k / OCULTA1) * PASSO2 + k % OCULTA1];
    for (int k = t; k < classes * OCULTA2; k += blockDim.x)
        d_w3[k] = w3[(k / OCULTA2) * PASSO3 + k % OCULTA2];
    if (t < OCULTA1) d_b1[t] = b1[t];
    if (t < OCULTA2) d_b2[t] = b2[t];
    if (t < classes) d_b3[t] = b3[t];
    if (t == 0) *d_perda = perda_total;
}

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // Parâmetros e buffers ficam na GPU durante todo o programa: são alocados
    // uma única vez no construtor e liberados no destrutor.
    float* d_w = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    float* d_z = nullptr;
    float* d_delta = nullptr;
    float* d_grad_in = nullptr;
    // Entrada do último forward (memória da GPU).
    const float* d_last_input = nullptr;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final) {
        // Mesma inicialização do código legado, feita na CPU e enviada uma vez.
        vector<float> weights(output_size * input_size);
        vector<float> bias(output_size, 0);
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
        checar(cudaMalloc(&d_w, weights.size() * sizeof(float)));
        checar(cudaMalloc(&d_b, out_features * sizeof(float)));
        checar(cudaMalloc(&d_out, out_features * sizeof(float)));
        checar(cudaMalloc(&d_z, out_features * sizeof(float)));
        checar(cudaMalloc(&d_delta, out_features * sizeof(float)));
        checar(cudaMalloc(&d_grad_in, in_features * sizeof(float)));
        checar(cudaMemcpy(d_w, weights.data(), weights.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
        checar(cudaMemcpy(d_b, bias.data(), out_features * sizeof(float),
                          cudaMemcpyHostToDevice));
    }

    ~DenseLayer() {
        cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
        cudaFree(d_z); cudaFree(d_delta); cudaFree(d_grad_in);
    }
    // A camada é dona da memória da GPU: não pode ser copiada.
    DenseLayer(const DenseLayer&) = delete;
    DenseLayer& operator=(const DenseLayer&) = delete;

    // Recebe e devolve ponteiros da GPU: a saída de uma camada vai direto para
    // a próxima, sem passar pela CPU.
    // Ponteiros dos parâmetros na GPU (usados pelo kernel de treino).
    float* pesos_gpu() { return d_w; }
    float* vies_gpu() { return d_b; }

    const float* Forward(const float* d_input) {
        d_last_input = d_input;
        forward_kernel<<<grade(out_features), BLOCO>>>(
            d_input, d_w, d_b, in_features, out_features, final, d_out, d_z);
        checar(cudaGetLastError());
        return d_out;
    }

    const float* Backward(const float* d_grad_out, float learning_rate,
                          bool calcular_grad_in = true) {
        delta_kernel<<<grade(out_features), BLOCO>>>(
            d_grad_out, d_z, out_features, final, d_delta);
        checar(cudaGetLastError());
        // Os kernels rodam em ordem na mesma stream: o gradiente da entrada
        // usa os pesos antigos, antes de o update_kernel alterá-los.
        if (calcular_grad_in) {
            grad_in_kernel<<<grade(in_features), BLOCO>>>(
                d_delta, d_w, in_features, out_features, d_grad_in);
            checar(cudaGetLastError());
        }
        update_kernel<<<grade(in_features * out_features), BLOCO>>>(
            d_last_input, d_delta, in_features, out_features, learning_rate, d_w, d_b);
        checar(cudaGetLastError());
        return d_grad_in;
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos (e na GPU).
    float* d_imagem;
    checar(cudaMalloc(&d_imagem, entradas * sizeof(float)));
    vector<float> escores(classes);
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        checar(cudaMemcpy(d_imagem, amostra.imagem.data(), entradas * sizeof(float),
                          cudaMemcpyHostToDevice));
        const float* d_escores = l3.Forward(l2.Forward(l1.Forward(d_imagem)));
        checar(cudaMemcpy(escores.data(), d_escores, classes * sizeof(float),
                          cudaMemcpyDeviceToHost));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cudaFree(d_imagem);

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        // Mede também a leitura e a avaliação, que ficam fora do tempo de treino.
        const auto inicio_leitura = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_leitura;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Imagens de treino em um vetor contíguo [amostra][atributo], para
        // irem para a GPU em uma única cópia.
        vector<float> imagens(treino.size() * Barenn::entradas);
        for (size_t i = 0; i < treino.size(); ++i) {
            copy(treino[i].imagem.begin(), treino[i].imagem.end(),
                 imagens.begin() + i * Barenn::entradas);
        }
        vector<int> rotulos(treino.size());
        for (size_t i = 0; i < treino.size(); ++i) rotulos[i] = treino[i].rotulo;
        float* d_imagens;
        int* d_rotulos;
        double* d_perda;
        Barenn::checar(cudaMalloc(&d_imagens, imagens.size() * sizeof(float)));
        Barenn::checar(cudaMalloc(&d_rotulos, rotulos.size() * sizeof(int)));
        Barenn::checar(cudaMalloc(&d_perda, sizeof(double)));

        // Aquecimento: cria o contexto CUDA fora da medição.
        Barenn::checar(cudaFree(0));

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        // A cópia do conjunto de treino para a GPU entra na medição.
        Barenn::checar(cudaMemcpy(d_imagens, imagens.data(), imagens.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        Barenn::checar(cudaMemcpy(d_rotulos, rotulos.data(), rotulos.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        for (int epoca = 0; epoca < epocas; ++epoca) {
            // Um único lançamento treina a época inteira (1 bloco).
            Barenn::treinar_epoca_kernel<<<1, Barenn::BLOCO>>>(
                d_imagens, d_rotulos, static_cast<int>(treino.size()), taxa,
                l1.pesos_gpu(), l1.vies_gpu(), l2.pesos_gpu(), l2.vies_gpu(),
                l3.pesos_gpu(), l3.vies_gpu(), d_perda);
            Barenn::checar(cudaGetLastError());
            // A cópia síncrona espera o kernel terminar.
            double perda_total = 0;
            Barenn::checar(cudaMemcpy(&perda_total, d_perda, sizeof(double),
                                      cudaMemcpyDeviceToHost));
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        cudaFree(d_imagens);
        cudaFree(d_rotulos);
        cudaFree(d_perda);

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        const auto inicio_avaliacao = chrono::steady_clock::now();
        Barenn::avaliar(l1, l2, l3, teste);
        const chrono::duration<double> tempo_avaliacao =
            chrono::steady_clock::now() - inicio_avaliacao;
        // Linhas extras de tempo (ignoradas pelo scripts/validar.sh).
        cout << fixed << setprecision(4)
             << "Tempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
