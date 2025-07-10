#pragma once

#include "layer.h"
#include "utec/algebra/tensor.h"

namespace utec::neural_network {

    template<typename T>
    class ReLU : public ILayer<T> {
    private:
        utec::algebra::Tensor<T,2> mask; // para almacenar dónde x>0

    public:
        ReLU() : mask(1, 1) {} // Initialize with dummy size, will be resized in forward

        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
            auto x_shape = x.shape();

            // Resize mask if needed
            if (mask.shape() != x_shape) {
                mask = utec::algebra::Tensor<T,2>(x_shape);
            }

            utec::algebra::Tensor<T,2> result(x_shape);

            for (size_t i = 0; i < x_shape[0]; ++i) {
                for (size_t j = 0; j < x_shape[1]; ++j) {
                    if (x(i, j) > T(0)) {
                        result(i, j) = x(i, j);
                        mask(i, j) = T(1);
                    } else {
                        result(i, j) = T(0);
                        mask(i, j) = T(0);
                    }
                }
            }

            return result;
        }

        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
            auto grad_shape = grad.shape();
            utec::algebra::Tensor<T,2> result(grad_shape);

            for (size_t i = 0; i < grad_shape[0]; ++i) {
                for (size_t j = 0; j < grad_shape[1]; ++j) {
                    result(i, j) = grad(i, j) * mask(i, j);
                }
            }

            return result;
        }
    };

} // namespace utec::neural_network
