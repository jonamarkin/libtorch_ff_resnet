#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <torch/torch.h>
#include <cereal/archives/binary.hpp>
#include <sstream>
#include <vector>

struct TensorWrapper {
    std::vector<int64_t> sizes;
    std::vector<float> data;
    c10::ScalarType dtype;
    std::vector<TensorWrapper> tensor_list;  // **Supports multiple tensors (model weights)**
    int type;  // **New: Flag to identify if this is model weights (0) or data (1)**

    TensorWrapper() = default;

    // **Wrap single tensor (for data)**
    explicit TensorWrapper(torch::Tensor tensor, int type_flag) : type(type_flag) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type();
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    // **Wrap multiple tensors (for model weights)**
    explicit TensorWrapper(const std::vector<torch::Tensor>& tensors, int type_flag) : type(type_flag) {
        for (const auto& tensor : tensors) {
            tensor_list.emplace_back(TensorWrapper(tensor, type_flag));  // ✅ Now we pass the type flag correctly
        }
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive(sizes, data, dtype, tensor_list, type);
    }

    // **Convert back to a single tensor (for training data)**
    torch::Tensor toTensor() const {
        return torch::from_blob(const_cast<float*>(data.data()), sizes,
                                torch::TensorOptions().dtype(dtype)).clone();
    }

    // **Convert back to a list of tensors (for model weights)**
    std::vector<torch::Tensor> toTensorList() const {
        std::vector<torch::Tensor> tensors;
        for (const auto& wrapped : tensor_list) {
            tensors.push_back(wrapped.toTensor());
        }
        return tensors;
    }

    // **Get type of data (0 = model weights, 1 = training data)**
    int getType() const {
        return type;
    }
};

#endif // SERIALIZE_HPP
