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
        vector<float> gradiente(Barenn::classes);
        vector<float> escores(Barenn::classes);
        array<double, epocas> perdas{};

        // Imagens de treino em um vetor contíguo [amostra][atributo], para
        // irem para a GPU em uma única cópia.
        vector<float> imagens(treino.size() * Barenn::entradas);
        for (size_t i = 0; i < treino.size(); ++i) {
            copy(treino[i].imagem.begin(), treino[i].imagem.end(),
                 imagens.begin() + i * Barenn::entradas);
        }
        float *d_imagens, *d_escores_grad;
        Barenn::checar(cudaMalloc(&d_imagens, imagens.size() * sizeof(float)));
        Barenn::checar(cudaMalloc(&d_escores_grad, Barenn::classes * sizeof(float)));

        // Aquecimento: cria o contexto CUDA fora da medição.
        Barenn::checar(cudaFree(0));

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        // A cópia do conjunto de treino para a GPU entra na medição.
        Barenn::checar(cudaMemcpy(d_imagens, imagens.data(), imagens.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* d_x = d_imagens + i * Barenn::entradas;
                const float* d_escores = l3.Forward(l2.Forward(l1.Forward(d_x)));

                // Só os 10 escores voltam para a CPU, que calcula a perda; só o
                // gradiente da perda (10 floats) volta para a GPU.
                Barenn::checar(cudaMemcpy(escores.data(), d_escores,
                                          Barenn::classes * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                perda_total += Barenn::perda(escores, treino[i].rotulo, &gradiente);
                Barenn::checar(cudaMemcpy(d_escores_grad, gradiente.data(),
                                          Barenn::classes * sizeof(float),
                                          cudaMemcpyHostToDevice));

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const float* d_grad3 = l3.Backward(d_escores_grad, taxa);
                const float* d_grad2 = l2.Backward(d_grad3, taxa);
                l1.Backward(d_grad2, taxa, false);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        cudaFree(d_imagens);
        cudaFree(d_escores_grad);

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
