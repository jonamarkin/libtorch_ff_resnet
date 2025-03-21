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

// **CIFAR100 Dataset**
class CIFAR100 : public torch::data::Dataset<CIFAR100> {
private:
    torch::Tensor images_, labels_;

    void read_data(const std::string& root, bool is_train) {
        std::string filename = root + (is_train ? "/train.bin" : "/test.bin");
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open dataset file: " + filename);

        size_t num_images = is_train ? 50000 : 10000;
        size_t record_size = 32 * 32 * 3 + 2;

        std::vector<uint8_t> buffer(record_size * num_images);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        images_ = torch::from_blob(buffer.data() + 2, {static_cast<long>(num_images), 3, 32, 32}, torch::kUInt8)
                     .to(torch::kFloat32) / 255.0;
        labels_ = torch::from_blob(buffer.data(), {static_cast<long>(num_images)}, torch::kInt64);
    }

public:
    explicit CIFAR100(const std::string& root, bool train = true) { read_data(root, train); }

    torch::data::Example<> get(size_t index) override { return {images_[index], labels_[index]}; }
    torch::optional<size_t> size() const override { return images_.size(0); }
};

// **Source Node**
struct Source : ff_monode_t<TensorWrapper, TensorWrapper> {
    torch::jit::script::Module model;
    CIFAR100 dataset{"./data/cifar-100-binary/cifar-100-binary", true};
    int max_epochs = 10;

    Source() { model = torch::jit::load("../resnet152.pt"); }

    TensorWrapper* svc(TensorWrapper* feedback) {
        auto data_loader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), 64);

        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            for (auto& batch : *data_loader) {
                if (feedback) {  // **Wait for updated model from Sink**
                    std::vector<torch::Tensor> updated_params = feedback->toModel();
                    auto it = updated_params.begin();
                    for (auto param : model.parameters()) param.copy_(*it++);
                }

                // Send updated model & batch
                std::vector<torch::Tensor> params;
                for (const auto& param : model.parameters()) params.push_back(param.clone());
                ff_send_out(new TensorWrapper(params, batch.data));
            }
        }
        return EOS;
    }
};

// **Worker Node**
struct Worker : ff_minode_t<TensorWrapper, TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::SGD* optimizer;
    torch::Device device = torch::kCPU;
    int processedBatches = 0;

    Worker() {
        if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (!wrapper) return nullptr;

        std::vector<torch::Tensor> received_params = wrapper->toModel();
        torch::Tensor batch = wrapper->toTensor().to(device);

        if (processedBatches == 0) {  // **Initialize model**
            model = torch::jit::load("../resnet152.pt");
            model.to(device);
            auto it = received_params.begin();
            for (auto param : model.parameters()) param.copy_(*it++);

            std::vector<torch::Tensor> params_;
            for (const auto& param : model.parameters()) {
                params_.push_back(param);
            }
            optimizer = new torch::optim::SGD(params_, 0.01);
        }

        optimizer->zero_grad();
        torch::Tensor prediction = model.forward({batch}).toTensor();
        torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, 1), torch::zeros({batch.size(0)}, torch::kLong));
        loss.backward();
        optimizer->step();

        ++processedBatches;
        if (processedBatches % 100 == 0) {
            const std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Worker " << get_my_id() << " | Loss: " << loss.item<float>() << std::endl;
        }

        // **Send updated model to Sink**
        std::vector<torch::Tensor> updated_params;
        for (const auto& param : model.parameters()) updated_params.push_back(param.clone());
        return new TensorWrapper(updated_params);
    }
};

// **Sink Node - Synchronization**
struct Sink : ff_minode_t<TensorWrapper, TensorWrapper> {
    int num_workers;
    int updates_received = 0;
    std::vector<torch::Tensor> aggregated_params;

    explicit Sink(int num_workers) : num_workers(num_workers) {}

    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (!wrapper) return nullptr;

        std::vector<torch::Tensor> received_params = wrapper->toModel();
        if (updates_received == 0) aggregated_params = received_params;
        else for (size_t i = 0; i < received_params.size(); i++) aggregated_params[i] += received_params[i];

        updates_received++;

        if (updates_received == num_workers) {
            for (auto& param : aggregated_params) param /= num_workers;
            ff_send_out(new TensorWrapper(aggregated_params));  // **Send to Source**
            updates_received = 0;
        }
        return nullptr;
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
    Sink sink(4);

    // **Worker Pool**
    std::vector<Worker*> workers;
    for (int i = 0; i < 4; ++i) {
        workers.push_back(new Worker());
    }

    // **Add Stages**
    mainPipe.add_stage(&a2a);
    mainPipe.add_stage(&sink);

    // **Connect Nodes**
    // Connect Source to Worker pool via 'a2a'
    a2a.add_firstset<Source>({&source});
    a2a.add_secondset<Worker>(workers);

    // **Create Distributed Groups**
    // Create groups for proper management of stages
    a2a.createGroup("G1") << &source;  // Group for source node
    a2a.createGroup("G2") << workers[0] << workers[1] << workers[2] << workers[3];  // Group for workers
    mainPipe.createGroup("G3") << &sink;  // Group for sink node
    mainPipe.wrap_around();

    // **Run the Pipeline**
    if (mainPipe.run_and_wait_end() < 0) {
        error("running mainPipe\n");
        return -1;
    }

    // After all the batches have been processed, save the trained model from one of the workers
    workers[0]->model.save("trained_resnet152.pt");

    // Clean up the worker pool
    for (auto w : workers) {
        delete w;
    }

    return 0;
}

