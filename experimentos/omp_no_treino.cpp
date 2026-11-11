// EXPERIMENTO (resultado negativo, ver Issue do bloco OpenMP):
// barenn_cpu.cpp da tag v0.1 + '#pragma omp parallel for' nos laços de
// neurônios do treino (forward, gradiente da entrada e atualização dos pesos).
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

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // Matriz de pesos achatada em um único vetor contíguo (linha por neurônio):
    // weights[i * in_features + j] liga a entrada j ao neurônio de saída i.
    vector<float> weights;
    vector<float> bias;
    // Buffers alocados uma única vez no construtor e reutilizados a cada amostra.
    vector<float> out;
    vector<float> last_z;
    vector<float> delta;
    vector<float> grad_in;
    // Entrada do último forward. Aponta para um vetor que continua vivo até o
    // backward (a imagem da amostra ou a saída da camada anterior).
    const vector<float>* last_input = nullptr;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size * input_size), bias(output_size, 0),
          out(output_size), last_z(output_size), delta(output_size),
          grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // A ordem de sorteio é a mesma do código legado (linha por linha).
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    // Recebe a entrada por referência (sem cópia) e devolve uma referência
    // para o buffer de saída da própria camada (sem alocação).
    const vector<float>& Forward(const vector<float>& input) {
        last_input = &input;
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        #pragma omp parallel for
        for (int i = 0; i < out_features; ++i) {
            const float* w = &weights[i * in_features];
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * w[j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : max(0.0f, sum);
        }
        return out;
    }

    // calcular_grad_in = false na primeira camada: o gradiente em relação à
    // imagem de entrada nunca é usado, então não precisa ser calculado.
    const vector<float>& Backward(const vector<float>& grad_out, float learning_rate,
                                  bool calcular_grad_in = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        // Mantém a mesma ordem de soma do legado para não mudar o resultado.
        if (calcular_grad_in) {
            #pragma omp parallel for
            for (int i = 0; i < in_features; ++i) {
                float sum = 0;
                for (int j = 0; j < out_features; ++j) {
                    sum += delta[j] * weights[j * in_features + i];
                }
                grad_in[i] = sum;
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        const vector<float>& input = *last_input;
        #pragma omp parallel for
        for (int i = 0; i < out_features; ++i) {
            float* w = &weights[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                w[j] -= learning_rate * (delta[i] * input[j]);
            }
            bias[i] -= learning_rate * delta[i];
        }
        return grad_in;
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

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
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
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const auto& grad3 = l3.Backward(gradiente, taxa);
                const auto& grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa, false);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

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
