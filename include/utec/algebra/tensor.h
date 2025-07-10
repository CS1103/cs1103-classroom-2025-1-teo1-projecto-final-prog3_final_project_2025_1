#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace utec::algebra {

template<typename T, size_t Rank>
class Tensor {
private:
    std::array<size_t, Rank> shape_;
    std::vector<T> data_;

    // Helper to calculate linear index from multi-dimensional indices
    template<typename... Idxs>
    size_t calculate_index(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == Rank, "Number of indices must match tensor rank");
        std::array<size_t, Rank> indices = {static_cast<size_t>(idxs)...};

        size_t linear_idx = 0;
        size_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            linear_idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return linear_idx;
    }

    size_t total_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
    }

public:
    // Constructor with shape array
    explicit Tensor(const std::array<size_t, Rank>& shape)
        : shape_(shape), data_(total_size()) {}

    // Variadic constructor
    template<typename... Dims>
    explicit Tensor(Dims... dims)
        : shape_{static_cast<size_t>(dims)...}, data_(total_size()) {
        static_assert(sizeof...(dims) == Rank, "Number of dimensions must match tensor rank");
    }

    // Variadic access operators
    template<typename... Idxs>
    T& operator()(Idxs... idxs) {
        return data_[calculate_index(idxs...)];
    }

    template<typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        return data_[calculate_index(idxs...)];
    }

    // Linear access for reshape test case
    T& operator[](size_t idx) {
        if (idx >= data_.size()) {
            throw std::out_of_range("Linear index out of bounds");
        }
        return data_[idx];
    }

    const T& operator[](size_t idx) const {
        if (idx >= data_.size()) {
            throw std::out_of_range("Linear index out of bounds");
        }
        return data_[idx];
    }

    // Shape information
    const std::array<size_t, Rank>& shape() const noexcept {
        return shape_;
    }

    // Reshape
    void reshape(const std::array<size_t, Rank>& new_shape) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
        if (new_size != data_.size()) {
            throw std::invalid_argument("Reshape cannot change total number of elements");
        }
        shape_ = new_shape;
    }

    // Variadic reshape
    template<typename... Dims>
    void reshape(Dims... dims) {
        static_assert(sizeof...(dims) == Rank, "Number of dimensions must match tensor rank");
        reshape(std::array<size_t, Rank>{static_cast<size_t>(dims)...});
    }

    // Fill with value
    void fill(const T& value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Arithmetic operations
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for addition");
        }
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for subtraction");
        }
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    // Element-wise multiplication with broadcasting support
    Tensor operator*(const Tensor& other) const {
        // Simple broadcasting: allow multiplication if shapes match or one dimension is 1
        bool can_broadcast = true;
        std::array<size_t, Rank> result_shape = shape_;

        for (size_t i = 0; i < Rank; ++i) {
            if (shape_[i] != other.shape_[i]) {
                if (shape_[i] == 1) {
                    result_shape[i] = other.shape_[i];
                } else if (other.shape_[i] == 1) {
                    result_shape[i] = shape_[i];
                } else {
                    can_broadcast = false;
                    break;
                }
            }
        }

        if (!can_broadcast) {
            throw std::invalid_argument("Incompatible shapes for multiplication");
        }

        Tensor result(result_shape);

        // For simplicity, implement basic broadcasting for 2D case
        if constexpr (Rank == 2) {
            for (size_t i = 0; i < result_shape[0]; ++i) {
                for (size_t j = 0; j < result_shape[1]; ++j) {
                    size_t this_i = (shape_[0] == 1) ? 0 : i;
                    size_t this_j = (shape_[1] == 1) ? 0 : j;
                    size_t other_i = (other.shape_[0] == 1) ? 0 : i;
                    size_t other_j = (other.shape_[1] == 1) ? 0 : j;

                    result(i, j) = (*this)(this_i, this_j) * other(other_i, other_j);
                }
            }
        } else {
            // For other ranks, just do element-wise if shapes match
            if (shape_ == other.shape_) {
                for (size_t i = 0; i < data_.size(); ++i) {
                    result.data_[i] = data_[i] * other.data_[i];
                }
            } else {
                throw std::invalid_argument("Broadcasting not implemented for this rank");
            }
        }

        return result;
    }

    // Scalar multiplication
    Tensor operator*(const T& scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // Transpose for 2D tensors only
    Tensor transpose_2d() const {
        static_assert(Rank == 2, "transpose_2d only works for 2D tensors");

        std::array<size_t, 2> new_shape = {shape_[1], shape_[0]};
        Tensor result(new_shape);

        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    // Access to raw data for neural network operations
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
};

} // namespace utec::algebra
