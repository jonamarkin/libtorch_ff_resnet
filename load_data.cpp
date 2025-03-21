#include <torch/torch.h>
#include <cstddef>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

// CIFAR100 custom dataset
class CIFAR100 : public torch::data::Dataset<CIFAR100> {
private:
    torch::Tensor images_;
    torch::Tensor labels_;
    
    void read_data(const std::string& root, bool is_train) {
        std::string filename = root + (is_train ? "/train.bin" : "/test.bin");
        std::ifstream file(filename, std::ios::binary);
        
        if (!file) {
            throw std::runtime_error("Cannot open dataset file: " + filename);
        }

        size_t num_images = is_train ? 50000 : 10000;
        const int image_size = 32 * 32 * 3;
        
        // Prepare tensors
        images_ = torch::empty({static_cast<long>(num_images), 3, 32, 32}, torch::kFloat32);
        labels_ = torch::empty(num_images, torch::kInt64);
        
        std::vector<uint8_t> buffer(image_size + 2);  // +2 for coarse and fine labels
        
        for (size_t i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char*>(buffer.data()), image_size + 2);
            
            // Second byte is fine label (index 1)
            labels_[i] = static_cast<int64_t>(buffer[1]);
            
            // Convert image data (normalize to 0-1)
            auto image_data = images_[i];
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        image_data[c][h][w] = buffer[2 + c * 1024 + h * 32 + w] / 255.0f;
                    }
                }
            }
        }
    }

public:
    // Constructor
    explicit CIFAR100(const std::string& root, bool train = true) {
        read_data(root, train);
    }

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    // Return the total size of our dataset
    torch::optional<size_t> size() const override {
        return images_.size(0);
    }
};

int main() {
    try {
        // Create train dataset
        auto train_dataset = CIFAR100("./data/cifar-100-binary/cifar-100-binary", true)
            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());

        // Create loader
        auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(32).workers(2));

        // Iterate through the loader
        for (auto& batch : *train_loader) {
            auto data = batch.data;
            auto target = batch.target;

            std::cout << "Batch size: " << data.size(0) << '\n';
            std::cout << "Data shape: " << data.sizes() << '\n';
            std::cout << "Target shape: " << target.sizes() << '\n';
            break;  // Just show first batch
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return -1;
    }

    return 0;
}