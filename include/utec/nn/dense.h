#pragma once

#include "layer.h"
#include "utec/algebra/tensor.h"
#include <random>

namespace utec::neural_network {

template<typename T>
class Dense : public ILayer<T> {
private:
    utec::algebra::Tensor<T,2> W, dW;     // [in_feats, out_feats] y su gradiente
    utec::algebra::Tensor<T,1> b, db;     // [out_feats] y su gradiente
    utec::algebra::Tensor<T,2> last_x;    // cache de entrada para backward

    void initialize_weights(size_t in_feats, size_t out_feats) {
        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        T stddev = std::sqrt(T(2.0) / (in_feats + out_feats));
        std::normal_distribution<T> dist(T(0), stddev);

        for (size_t i = 0; i < in_feats; ++i) {
            for (size_t j = 0; j < out_feats; ++j) {
                W(i, j) = dist(gen);
            }
        }

        b.fill(T(0));
    }

public:
    Dense(size_t in_feats, size_t out_feats)
        : W(in_feats, out_feats), dW(in_feats, out_feats),
          b(out_feats), db(out_feats), last_x(1, 1) {
        initialize_weights(in_feats, out_feats);
        dW.fill(T(0));
        db.fill(T(0));
    }

    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        // Cache input for backward pass
        last_x = x;

        auto x_shape = x.shape();
        auto w_shape = W.shape();
        size_t batch_size = x_shape[0];
        size_t out_feats = w_shape[1];

        // Check dimension compatibility
        if (x_shape[1] != w_shape[0]) {
            throw std::invalid_argument("Input features don't match weight matrix dimensions");
        }

        // Result: batch_size x out_feats
        utec::algebra::Tensor<T,2> result(batch_size, out_feats);

        // Matrix multiplication: result = x @ W + b
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_feats; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < w_shape[0]; ++k) {
                    sum += x(i, k) * W(k, j);
                }
                result(i, j) = sum + b(j);
            }
        }

        return result;
    }

    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
        auto grad_shape = grad.shape();
        auto x_shape = last_x.shape();
        auto w_shape = W.shape();

        size_t batch_size = grad_shape[0];
        size_t in_feats = w_shape[0];
        size_t out_feats = w_shape[1];

        // Compute gradient w.r.t. weights: dW = x^T @ grad
        dW.fill(T(0));
        for (size_t i = 0; i < in_feats; ++i) {
            for (size_t j = 0; j < out_feats; ++j) {
                for (size_t b = 0; b < batch_size; ++b) {
                    dW(i, j) += last_x(b, i) * grad(b, j);
                }
            }
        }

        // Compute gradient w.r.t. bias: db = sum(grad, axis=0)
        db.fill(T(0));
        for (size_t j = 0; j < out_feats; ++j) {
            for (size_t b = 0; b < batch_size; ++b) {
                db(j) += grad(b, j);
            }
        }

        // Compute gradient w.r.t. input: dx = grad @ W^T
        utec::algebra::Tensor<T,2> dx(batch_size, in_feats);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_feats; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < out_feats; ++k) {
                    sum += grad(i, k) * W(j, k);
                }
                dx(i, j) = sum;
            }
        }

        return dx;
    }

    void update_parameters(T learning_rate) override {
        auto w_shape = W.shape();
        auto b_shape = b.shape();

        // Update weights: W -= lr * dW
        for (size_t i = 0; i < w_shape[0]; ++i) {
            for (size_t j = 0; j < w_shape[1]; ++j) {
                W(i, j) -= learning_rate * dW(i, j);
            }
        }

        // Update biases: b -= lr * db
        for (size_t i = 0; i < b_shape[0]; ++i) {
            b(i) -= learning_rate * db(i);
        }
    }

    // Getters for testing
    const utec::algebra::Tensor<T,2>& get_weights() const { return W; }
    const utec::algebra::Tensor<T,1>& get_biases() const { return b; }
};

} // namespace utec::neural_network
