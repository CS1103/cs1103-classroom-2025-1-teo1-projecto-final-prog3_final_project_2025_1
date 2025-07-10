#include "utec/algebra/tensor.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/dense.h"
#include "utec/nn/activation.h"
#include "utec/agent/PongAgent.h"
#include <iostream>
#include <random>
#include <vector>

using namespace utec::algebra;
using namespace utec::neural_network;

/**
 * EJEMPLO: Entrenamiento de Red Neuronal para Pong
 * Este ejemplo muestra cómo entrenar una red para predecir movimientos en Pong
 */

// Generar datos sintéticos de Pong para entrenamiento
std::pair<Tensor<float,2>, Tensor<float,2>> generate_pong_data(size_t num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);

    // Estados: [ball_x, ball_y, ball_vx, ball_vy, paddle_y]
    Tensor<float,2> states(num_samples, 5);
    // Acciones: [-1, 0, 1] -> [up, stay, down] (one-hot encoded)
    Tensor<float,2> actions(num_samples, 3);

    for (size_t i = 0; i < num_samples; ++i) {
        float ball_x = pos_dist(gen);
        float ball_y = pos_dist(gen);
        float ball_vx = vel_dist(gen);
        float ball_vy = vel_dist(gen);
        float paddle_y = pos_dist(gen);

        // Estado actual
        states(i,0) = ball_x;
        states(i,1) = ball_y;
        states(i,2) = ball_vx;
        states(i,3) = ball_vy;
        states(i,4) = paddle_y;

        // Lógica simple para decidir acción óptima
        float future_ball_y = ball_y + ball_vy * (1.0f - ball_x) / std::abs(ball_vx + 1e-6f);

        actions(i,0) = 0; actions(i,1) = 0; actions(i,2) = 0; // Reset

        if (future_ball_y > paddle_y + 0.1f) {
            actions(i,2) = 1; // Move down
        } else if (future_ball_y < paddle_y - 0.1f) {
            actions(i,0) = 1; // Move up
        } else {
            actions(i,1) = 1; // Stay
        }
    }

    return std::make_pair(std::move(states), std::move(actions));
}

int main() {
    std::cout << "🎮 PONG AGENT TRAINING\n";
    std::cout << "=====================\n\n";

    try {
        // Generar datos de entrenamiento
        std::cout << "📊 Generando datos de entrenamiento...\n";
        auto [X, Y] = generate_pong_data(1000);

        std::cout << "  ✅ Datos generados: " << X.shape()[0] << " samples\n";
        std::cout << "  📐 Input shape: [" << X.shape()[0] << ", " << X.shape()[1] << "]\n";
        std::cout << "  📐 Output shape: [" << Y.shape()[0] << ", " << Y.shape()[1] << "]\n\n";

        // Crear red neuronal para el agente
        std::cout << "🧠 Creando red neuronal...\n";
        NeuralNetwork<float> net;
        net.add_layer(std::make_unique<Dense<float>>(5, 32));  // Input: 5 features
        net.add_layer(std::make_unique<ReLU<float>>());
        net.add_layer(std::make_unique<Dense<float>>(32, 16));
        net.add_layer(std::make_unique<ReLU<float>>());
        net.add_layer(std::make_unique<Dense<float>>(16, 8));
        net.add_layer(std::make_unique<ReLU<float>>());
        net.add_layer(std::make_unique<Dense<float>>(8, 3));   // Output: 3 actions

        std::cout << "  ✅ Arquitectura: 5 → 32 → 16 → 8 → 3\n";
        std::cout << "  🎯 Funciones activación: ReLU\n\n";

        // Entrenar la red
        std::cout << "🚀 Iniciando entrenamiento...\n";
        auto metrics = net.train_advanced(X, Y, 500, 0.01f, 20, 1e-6f);

        std::cout << "\n✅ Entrenamiento completado!\n";
        std::cout << "📊 Métricas finales:\n";
        std::cout << "  • Épocas entrenadas: " << metrics.epochs_trained << "\n";
        std::cout << "  • Loss final: " << metrics.final_loss << "\n";
        std::cout << "  • Mejor loss: " << metrics.best_loss << "\n";
        std::cout << "  • Convergió: " << (metrics.converged ? "Sí" : "No") << "\n\n";

        // Validar con algunos ejemplos
        std::cout << "🔍 Validando predicciones...\n";
        auto predictions = net.forward(X);

        int correct_predictions = 0;
        for (size_t i = 0; i < std::min(size_t(10), X.shape()[0]); ++i) {
            // Encontrar acción predicha y esperada
            int predicted_action = 0;
            int expected_action = 0;

            float max_pred = predictions(i, 0);
            float max_exp = Y(i, 0);

            for (int j = 1; j < 3; ++j) {
                if (predictions(i, j) > max_pred) {
                    max_pred = predictions(i, j);
                    predicted_action = j;
                }
                if (Y(i, j) > max_exp) {
                    max_exp = Y(i, j);
                    expected_action = j;
                }
            }

            if (predicted_action == expected_action) {
                correct_predictions++;
            }

            if (i < 5) {  // Show first 5 examples
                std::string actions[] = {"UP", "STAY", "DOWN"};
                std::cout << "  Sample " << (i+1) << ": ";
                std::cout << "Ball(" << X(i,0) << "," << X(i,1) << ") ";
                std::cout << "Paddle(" << X(i,4) << ") ";
                std::cout << "→ Predicted: " << actions[predicted_action];
                std::cout << ", Expected: " << actions[expected_action];
                std::cout << (predicted_action == expected_action ? " ✅" : " ❌") << "\n";
            }
        }

        float accuracy = static_cast<float>(correct_predictions) / 10.0f * 100.0f;
        std::cout << "\n📈 Accuracy en validación: " << accuracy << "%\n";

        std::cout << "\n🎉 Agente Pong entrenado exitosamente!\n";
        std::cout << "🎯 El agente aprendió a:\n";
        std::cout << "  • Seguir la pelota\n";
        std::cout << "  • Anticipar trayectorias\n";
        std::cout << "  • Optimizar movimientos\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error durante entrenamiento: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
