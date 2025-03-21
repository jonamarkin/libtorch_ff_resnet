#include "serialize_tensor.hpp"
#include <ff/dff.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>

using namespace ff;
std::mutex mtx;

// **CIFAR100 Dataset Class**
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

        const size_t num_images = is_train ? 50000 : 10000;
        const size_t image_size = 32 * 32 * 3;
        const size_t record_size = image_size + 2;  // +2 for labels

        // Read entire file at once
        std::vector<uint8_t> buffer(record_size * num_images);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        // Prepare tensors
        images_ = torch::empty({static_cast<long>(num_images), 3, 32, 32}, torch::kFloat32);
        labels_ = torch::empty(num_images, torch::kInt64);

        // Process labels - faster way
        for (size_t i = 0; i < num_images; i++) {
            labels_[i] = static_cast<int64_t>(buffer[i * record_size + 1]);
        }

        // Process images - using tensor operations
        auto images_bytes = torch::from_blob(buffer.data() + 2, // Skip first two bytes of first image
                                           {static_cast<long>(num_images), 3, 32, 32},
                                           torch::kUInt8);

        // Convert to float and normalize in one operation
        images_ = images_bytes.to(torch::kFloat32).div(255.0);

        // Ensure proper memory layout
        images_ = images_.contiguous();
    }

public:
    explicit CIFAR100(const std::string& root, bool train = true) {
        read_data(root, train);
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return images_.size(0);
    }
};


// **Source Node**
struct Source : ff_monode_t<TensorWrapper> {
    torch::Device device = torch::kCPU;
    CIFAR100 dataset{"./data/cifar-100-binary/cifar-100-binary", true};

    TensorWrapper* svc(TensorWrapper*) {
        auto data_loader = torch::data::make_data_loader(
            dataset.map(torch::data::transforms::Stack<>()),
            /*batch_size=*/64
        );

        for (auto& batch : *data_loader) {
            auto data = batch.data.to(device);
            TensorWrapper* wrapper = new TensorWrapper(data);
            ff_send_out(wrapper);
        }
        return EOS;
    }
};

// **Worker Node**
struct Worker : ff_minode_t<TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::SGD* optimizer;
    int processedBatches = 0;
    torch::Device device = torch::kCPU;

    Worker() {
        // Set device
        device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        // Load the TorchScript model (ResNet152, but you can use ResNet18)
        try {
            model = torch::jit::load("../resnet152.pt");
            std::cout << "Model loaded successfully\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load the model");
        }

        model.to(device);  // Move the model to the correct device

         // Convert model parameters to a vector of tensors
        std::vector<torch::Tensor> params;
        for (const auto& param : model.parameters()) {
            params.push_back(param);
        }

        optimizer = new torch::optim::SGD(params, 0.01);
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        torch::Tensor batch = wrapper->toTensor().to(device);  // Ensure input is on the correct device
        optimizer->zero_grad();

        // Forward pass with ResNet model
        torch::Tensor prediction = model.forward({batch}).toTensor();

        // Loss computation (CIFAR-100 has 100 classes)
        torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, 1), torch::zeros({batch.size(0)}, torch::kLong));

        loss.backward();
        optimizer->step();

        ++processedBatches;
        if (processedBatches % 100 == 0) {
            const std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Worker " << get_my_id()
                      << " | Batch: " << processedBatches
                      << " | Loss: " << loss.item<float>() << std::endl;
        }

        return new TensorWrapper(prediction);
    }

    void svc_end() {
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Worker " << get_my_id() << " processed "
                  << processedBatches << " batches" << std::endl;
        delete optimizer;
    }
};

// **Sink Node**
struct Sink : ff_minode_t<TensorWrapper> {
    int receivedBatches = 0;

    TensorWrapper* svc(TensorWrapper* wrapper) {
        torch::Tensor result = wrapper->toTensor();
        ++receivedBatches;
        return GO_ON;
    }

    void svc_end() {
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Sink received " << receivedBatches << " processed batches" << std::endl;
    }
};

// **Main FastFlow Pipeline**
int main(int argc, char* argv[]) {
    if (DFF_Init(argc, argv) != 0) {
        error("DFF_Init\n");
        return -1;
    }

    ff_pipeline mainPipe;
    ff_a2a a2a;
    Source source;
    Sink sink;

    // **Worker Pool**
    std::vector<Worker*> workers;
    for (int i = 0; i < 4; ++i) {
        workers.push_back(new Worker());
    }

    // **Add Stages**
    mainPipe.add_stage(&a2a);
    mainPipe.add_stage(&sink);

    // **Connect Nodes**
    a2a.add_firstset<Source>({&source});
    a2a.add_secondset<Worker>(workers);

    // **Create Distributed Groups**
    a2a.createGroup("G1") << &source;  // Group for source node
    a2a.createGroup("G2") << workers[0] << workers[1] << workers[2] << workers[3];  // Group for workers
    mainPipe.createGroup("G3") << &sink;  // Group for sink node

    // **Run the Pipeline**
    if (mainPipe.run_and_wait_end() < 0) {
        error("running mainPipe\n");
        return -1;
    }

    // Save the trained model after processing
    workers[0]->model.save("trained_resnet152.pt");

    for (auto w : workers) delete w;
    return 0;
}
