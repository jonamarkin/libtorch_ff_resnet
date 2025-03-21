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

    TensorWrapper() = default;
    explicit TensorWrapper(torch::Tensor tensor) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type(); // Store dtype
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive(sizes, data, dtype);  // Serialize dtype
    }

    torch::Tensor toTensor() const {
        return torch::from_blob(const_cast<float*>(data.data()), sizes,
                                torch::TensorOptions().dtype(dtype)).clone();  // Restore dtype
    }
};


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
