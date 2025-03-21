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
    
    bool is_model_update = false;  // Flag for model updates

    // Model parameters
    std::vector<std::vector<int64_t>> param_sizes;
    std::vector<std::vector<float>> param_data;
    std::vector<c10::ScalarType> param_dtypes;

    TensorWrapper() = default;

    // 📌 Constructor for dataset batch
    explicit TensorWrapper(torch::Tensor tensor) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type();
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
        is_model_update = false;
    }

    // 📌 Constructor for sending both model and batch together
    explicit TensorWrapper(const std::vector<torch::Tensor>& params, torch::Tensor tensor) {
        is_model_update = true;
        storeTensor(tensor);
        storeModel(params);
    }

    // 📌 Constructor for sending only model parameters
    explicit TensorWrapper(const std::vector<torch::Tensor>& params) {
        is_model_update = true;
        storeModel(params);
    }

    template<class Archive>
    void serialize(Archive& archive) {
        archive(sizes, data, dtype, is_model_update, param_sizes, param_data, param_dtypes);
    }

    // Convert back to dataset batch
    torch::Tensor toTensor() const {
        return torch::from_blob(const_cast<float*>(data.data()), sizes,
                                torch::TensorOptions().dtype(dtype)).clone();
    }

    // Convert back to model parameters
    std::vector<torch::Tensor> toModel() const {
        std::vector<torch::Tensor> params;
        for (size_t i = 0; i < param_sizes.size(); i++) {
            torch::Tensor param = torch::from_blob(const_cast<float*>(param_data[i].data()), param_sizes[i],
                                                   torch::TensorOptions().dtype(param_dtypes[i])).clone();
            params.push_back(param);
        }
        return params;
    }

private:
    void storeTensor(const torch::Tensor& tensor) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type();
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    void storeModel(const std::vector<torch::Tensor>& params) {
        for (const auto& param : params) {
            param_sizes.push_back(std::vector<int64_t>(param.sizes().begin(), param.sizes().end()));
            param_dtypes.push_back(param.scalar_type());
            param_data.emplace_back(param.data_ptr<float>(), param.data_ptr<float>() + param.numel());
        }
    }
};

// 📌 Serialization Functions
inline std::string serializeTensor(const torch::Tensor& tensor) {
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive archive(ss);
        TensorWrapper wrapper(tensor);
        archive(wrapper);
    }
    return ss.str();
}

inline torch::Tensor deserializeTensor(const std::string& serialized) {
    std::stringstream ss(serialized);
    cereal::BinaryInputArchive archive(ss);
    TensorWrapper wrapper;
    archive(wrapper);
    return wrapper.toTensor(); 
}

#endif // SERIALIZE_HPP
