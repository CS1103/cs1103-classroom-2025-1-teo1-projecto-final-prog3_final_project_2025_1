#pragma once

#include "utec/algebra/tensor.h"

namespace utec::neural_network {

    template<typename T>
    class MSELoss {
    private:
        utec::algebra::Tensor<T,2> last_pred, last_target;

    public:
        MSELoss() : last_pred(1, 1), last_target(1, 1) {}

        // Devuelve la pérdida media
        T forward(const utec::algebra::Tensor<T,2>& pred, const utec::algebra::Tensor<T,2>& target) {
            auto pred_shape = pred.shape();
            auto target_shape = target.shape();

            if (pred_shape != target_shape) {
                throw std::invalid_argument("Prediction and target shapes must match");
            }

            // Cache for backward pass
            last_pred = pred;
            last_target = target;

            T total_loss = T(0);
            size_t total_elements = pred_shape[0] * pred_shape[1];

            for (size_t i = 0; i < pred_shape[0]; ++i) {
                for (size_t j = 0; j < pred_shape[1]; ++j) {
                    T diff = pred(i, j) - target(i, j);
                    total_loss += diff * diff;
                }
            }

            return total_loss / total_elements;
        }

        // Devuelve dL/dpred
        utec::algebra::Tensor<T,2> backward() {
            auto pred_shape = last_pred.shape();
            utec::algebra::Tensor<T,2> grad(pred_shape);

            T scale = T(2) / (pred_shape[0] * pred_shape[1]);

            for (size_t i = 0; i < pred_shape[0]; ++i) {
                for (size_t j = 0; j < pred_shape[1]; ++j) {
                    grad(i, j) = scale * (last_pred(i, j) - last_target(i, j));
                }
            }

            return grad;
        }
    };

} // namespace utec::neural_network
