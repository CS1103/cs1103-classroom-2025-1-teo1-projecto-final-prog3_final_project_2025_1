#pragma once

#include "utec/algebra/tensor.h"

namespace utec::neural_network {

    template<typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;

        // Forward: recibe batch x features, devuelve batch x units
        virtual utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) = 0;

        // Backward: recibe gradiente de salida, devuelve gradiente de entrada
        virtual utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) = 0;

        // Update parameters with learning rate
        virtual void update_parameters(T learning_rate) {}
    };

} // namespace utec::neural_network
