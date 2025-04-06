// #ifndef SERIALIZE_HPP
// #define SERIALIZE_HPP

// #include <torch/torch.h>
// #include <cereal/archives/binary.hpp>
// #include <sstream>
// #include <vector>

// // Enum to indicate the type of data in TensorWrapper
// enum class TensorType { DATA, WEIGHTS };

// struct TensorWrapper {
//     std::vector<int64_t> sizes;
//     std::vector<float> data;
//     c10::ScalarType dtype;
//     TensorType type;  // NEW: Field to store the type of tensor

//     TensorWrapper() = default;

//     // Constructor for normal data (image batches)
//     explicit TensorWrapper(torch::Tensor tensor, TensorType t = TensorType::DATA) : type(t) {
//         sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
//         dtype = tensor.scalar_type();
//         data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
//     }

//     template<class Archive>
//     void serialize(Archive & archive) {
//         archive(sizes, data, dtype, type);
//     }

//     torch::Tensor toTensor() const {
//         return torch::from_blob(const_cast<float*>(data.data()), sizes,
//                                 torch::TensorOptions().dtype(dtype)).clone();
//     }
// };

// #endif // SERIALIZE_HPP


#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <torch/torch.h>
#include <cereal/archives/binary.hpp>
#include <vector>

// Enum to indicate type of tensor
enum class TensorType { DATA, WEIGHTS };

struct TensorWrapper {
    std::vector<int64_t> sizes;
    std::vector<float> data;
    c10::ScalarType dtype;
    TensorType type;
    std::vector<TensorWrapper> tensor_list;  // ✅ Allows multiple tensors (for model weights)

    TensorWrapper() = default;

    // **Constructor for normal data (image batches)**
    explicit TensorWrapper(torch::Tensor tensor, TensorType t = TensorType::DATA) : type(t) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type();
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    // **Constructor for multiple tensors (for model weights)**
    explicit TensorWrapper(const std::vector<torch::Tensor>& tensors, TensorType t = TensorType::WEIGHTS) : type(t) {
        for (const auto& tensor : tensors) {
            tensor_list.emplace_back(TensorWrapper(tensor, TensorType::WEIGHTS));
        }
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive(sizes, data, dtype, type, tensor_list);
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
};

#endif // SERIALIZE_HPP


