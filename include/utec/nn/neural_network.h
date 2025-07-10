#pragma once

#include "layer.h"
#include "loss.h"
#include "utec/algebra/tensor.h"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <limits>  // For std::numeric_limits
#include <cmath>   // For std::isnan, std::isinf

namespace utec::neural_network {

template<typename T>
class NeuralNetwork : public ILayer<T> {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    MSELoss<T> criterion;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    // Ejecuta forward por todas las capas
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        if (layers.empty()) {
            throw std::runtime_error("No layers in network");
        }

        auto current = x;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    // Lanza backward desde la última capa
    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
        if (layers.empty()) {
            return grad;
        }

        auto current_grad = grad;
        // Backward pass through layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
        return current_grad;
    }

    // Actualiza todos los parámetros con learning rate lr
    void update_parameters(T lr) override {
        for (auto& layer : layers) {
            layer->update_parameters(lr);
        }
    }

    // Método de conveniencia para optimización
    void optimize(T lr) {
        update_parameters(lr);
    }

    /**
     * @brief Advanced training with early stopping and adaptive learning rate
     * @param X Input data tensor [batch_size, input_features]
     * @param Y Target data tensor [batch_size, output_features]
     * @param epochs Maximum number of training epochs
     * @param lr Initial learning rate
     * @param patience Early stopping patience (epochs without improvement)
     * @param min_delta Minimum change to qualify as improvement
     * @return Training metrics including final loss and convergence info
     */
    struct TrainingMetrics {
        T final_loss;
        size_t epochs_trained;
        bool converged;
        std::vector<T> loss_history;
        T best_loss;
    };

    TrainingMetrics train_advanced(const utec::algebra::Tensor<T,2>& X,
                                 const utec::algebra::Tensor<T,2>& Y,
                                 size_t epochs, T lr,
                                 size_t patience = 10, T min_delta = T(1e-6)) {
        if (layers.empty()) {
            throw std::runtime_error("Cannot train empty network");
        }

        if (X.shape()[0] != Y.shape()[0]) {
            throw std::invalid_argument("Batch size mismatch between X and Y");
        }

        TrainingMetrics metrics{};
        metrics.loss_history.reserve(epochs);

        T best_loss = std::numeric_limits<T>::max();
        size_t epochs_without_improvement = 0;
        T current_lr = lr;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            try {
                // Forward pass with error checking
                auto predictions = forward(X);

                // Compute loss with numerical stability
                T loss = criterion.forward(predictions, Y);

                // Check for numerical issues
                if (std::isnan(loss) || std::isinf(loss)) {
                    throw std::runtime_error("Loss became NaN or Inf at epoch " + std::to_string(epoch));
                }

                metrics.loss_history.push_back(loss);

                // Early stopping logic
                if (loss < best_loss - min_delta) {
                    best_loss = loss;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement++;
                }

                // Adaptive learning rate (reduce on plateau)
                if (epochs_without_improvement > patience / 2) {
                    current_lr *= T(0.95); // Reduce learning rate by 5%
                }

                // Early stopping
                if (epochs_without_improvement >= patience) {
                    metrics.converged = true;
                    break;
                }

                // Backward pass
                auto loss_grad = criterion.backward();
                backward(loss_grad);

                // Update parameters with current learning rate
                optimize(current_lr);

                metrics.final_loss = loss;
                metrics.epochs_trained = epoch + 1;

                // Progress reporting every 10% of epochs
                if (epoch % (epochs / 10 + 1) == 0) {
                    std::cout << "Epoch " << epoch << "/" << epochs
                             << " - Loss: " << std::scientific << std::setprecision(4) << loss
                             << " - LR: " << current_lr << std::endl;
                }

            } catch (const std::exception& e) {
                throw std::runtime_error("Training failed at epoch " + std::to_string(epoch) + ": " + e.what());
            }
        }

        metrics.best_loss = best_loss;
        return metrics;
    }

    // Simplified training interface (backward compatibility)
    T train(const utec::algebra::Tensor<T,2>& X, const utec::algebra::Tensor<T,2>& Y, size_t epochs, T lr) {
        auto metrics = train_advanced(X, Y, epochs, lr);
        return metrics.final_loss;
    }

    /**
     * @brief Evaluate model performance on test data
     * @param X_test Test input data
     * @param Y_test Test target data
     * @return Evaluation metrics
     */
    struct EvaluationMetrics {
        T test_loss;
        T accuracy;  // For classification tasks
        T mean_absolute_error;
        size_t num_samples;
    };

    EvaluationMetrics evaluate(const utec::algebra::Tensor<T,2>& X_test,
                              const utec::algebra::Tensor<T,2>& Y_test) {
        if (X_test.shape()[0] != Y_test.shape()[0]) {
            throw std::invalid_argument("Test batch size mismatch");
        }

        EvaluationMetrics metrics{};
        metrics.num_samples = X_test.shape()[0];

        // Forward pass (no gradient computation needed)
        auto predictions = forward(X_test);

        // Compute test loss
        metrics.test_loss = criterion.forward(predictions, Y_test);

        // Compute additional metrics
        T total_abs_error = T(0);
        size_t correct_predictions = 0;

        for (size_t i = 0; i < metrics.num_samples; ++i) {
            for (size_t j = 0; j < Y_test.shape()[1]; ++j) {
                T pred_val = predictions(i, j);
                T true_val = Y_test(i, j);

                total_abs_error += std::abs(pred_val - true_val);

                // Simple accuracy for binary classification
                if (Y_test.shape()[1] == 1) {
                    bool pred_class = pred_val > T(0.5);
                    bool true_class = true_val > T(0.5);
                    if (pred_class == true_class) correct_predictions++;
                }
            }
        }

        metrics.mean_absolute_error = total_abs_error / (metrics.num_samples * Y_test.shape()[1]);
        metrics.accuracy = static_cast<T>(correct_predictions) / metrics.num_samples;

        return metrics;
    }

    /**
     * @brief Get model statistics for debugging and analysis
     */
    struct ModelStatistics {
        size_t total_parameters;
        size_t num_layers;
        std::vector<std::pair<std::string, size_t>> layer_info;
        T gradient_norm;
    };

    ModelStatistics get_model_stats() const {
        ModelStatistics stats{};
        stats.num_layers = layers.size();
        stats.total_parameters = 0;

        // This would require extending the ILayer interface to provide parameter counts
        // For now, we provide basic information
        for (size_t i = 0; i < layers.size(); ++i) {
            stats.layer_info.emplace_back("Layer_" + std::to_string(i), 0);
        }

        return stats;
    }

    // Get number of layers
    size_t size() const {
        return layers.size();
    }

    // Check if network is empty
    bool empty() const {
        return layers.empty();
    }
};

} // namespace utec::neural_network
