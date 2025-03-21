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

const int NUM_WORKERS = 4;
const int NUM_EPOCHS = 10;
const int BATCH_SIZE = 64;

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
        const size_t record_size = image_size + 2;

        std::vector<uint8_t> buffer(record_size * num_images);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        images_ = torch::empty({static_cast<long>(num_images), 3, 32, 32}, torch::kFloat32);
        labels_ = torch::empty(num_images, torch::kInt64);

        for (size_t i = 0; i < num_images; i++) {
            labels_[i] = static_cast<int64_t>(buffer[i * record_size + 1]);
        }

        auto images_bytes = torch::from_blob(buffer.data() + 2, {static_cast<long>(num_images), 3, 32, 32}, torch::kUInt8);
        images_ = images_bytes.to(torch::kFloat32).div(255.0).contiguous();
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
    torch::jit::script::Module model;
    CIFAR100 dataset{"./data/cifar-100-binary/cifar-100-binary", true};
    int num_epochs = NUM_EPOCHS;
    int epoch = 0;

    Source() {
        try {
            model = torch::jit::load("../resnet152.pt");
            std::cout << "[Source] Initial model loaded\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load model");
        }
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (epoch >= num_epochs) return EOS;

        if (wrapper) {
            // **Received updated model from Sink**
            std::vector<torch::Tensor> new_weights = wrapper->toTensorList();
            int i = 0;
            for (const auto& param : model.parameters()) {
                param.set_data(new_weights[i++]);
            }
            std::cout << "[Source] Model updated for epoch " << (epoch + 1) << "\n";
        }

        // **Send model to all workers**
        std::vector<torch::Tensor> model_weights;
        for (const auto& param : model.parameters()) {
            model_weights.push_back(param.clone());
        }
        ff_send_out(new TensorWrapper(model_weights, 0));

        // **Send training data to workers**
        auto data_loader = torch::data::make_data_loader(
            dataset.map(torch::data::transforms::Stack<>()),
            BATCH_SIZE
        );

        for (auto& batch : *data_loader) {
            auto data = batch.data.to(torch::kCPU);
            ff_send_out(new TensorWrapper(data, 1));  // Training data
        }

        epoch++;
        return GO_ON;
    }
};

// **Worker Node**
struct Worker : ff_minode_t<TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::SGD* optimizer;
    bool model_initialized = false;

    Worker() { optimizer = nullptr; }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (wrapper->getType() == 0) {
            update_model(wrapper->toTensorList());
            return GO_ON;
        } else if (wrapper->getType() == 1) {
            if (!model_initialized) {
                std::cerr << "[Worker] No model received yet!\n";
                return GO_ON;
            }

            torch::Tensor batch = wrapper->toTensor();
            optimizer->zero_grad();
            torch::Tensor prediction = model.forward({batch}).toTensor();
            torch::Tensor loss = torch::nn::functional::cross_entropy(prediction, torch::zeros({batch.size(0)}, torch::kLong));

            loss.backward();
            optimizer->step();

            std::vector<torch::Tensor> model_weights;
            for (const auto& param : model.parameters()) {
                model_weights.push_back(param.clone());
            }

            ff_send_out(new TensorWrapper(model_weights, 0));
            return GO_ON;
        }

        return GO_ON;
    }

    void update_model(std::vector<torch::Tensor> new_weights) {
        if (!model_initialized) {
            model = torch::jit::script::Module();
            model_initialized = true;
        }
        int i = 0;
        for (const auto& param : model.parameters()) {
            param.set_data(new_weights[i++]);
        }

        std::vector<torch::Tensor> params;
        for (const auto& param : model.parameters()) {
            params.push_back(param);
        }

        optimizer = new torch::optim::SGD(params, 0.01);
        std::cout << "[Worker] Model successfully updated.\n";
    }
};

// **Sink Node**
struct Sink : ff_minode_t<TensorWrapper> {
    std::vector<std::vector<torch::Tensor>> weight_accumulator;
    int weights_received = 0;

    Sink() { weight_accumulator.resize(NUM_WORKERS); }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        std::vector<torch::Tensor> worker_weights = wrapper->toTensorList();
        for (size_t i = 0; i < worker_weights.size(); i++) {
            weight_accumulator[i].push_back(worker_weights[i]);
        }

        weights_received++;

        if (weights_received == NUM_WORKERS) {
            std::vector<torch::Tensor> averaged_weights;
            for (size_t i = 0; i < weight_accumulator.size(); i++) {
                averaged_weights.push_back(torch::stack(weight_accumulator[i]).mean(0));
            }

            ff_send_out(new TensorWrapper(averaged_weights, 0));
            weights_received = 0;
            weight_accumulator.clear();
        }

        return GO_ON;
    }
};




int main(int argc, char* argv[]) {
    if (DFF_Init(argc, argv) != 0) {
        error("DFF_Init\n");
        return -1;
    }

    ff_pipeline mainPipe;  // Main FastFlow pipeline
    ff_a2a a2a;            // All-to-All communication module

    Source source;
    Sink sink;

    // **Worker Pool**
    std::vector<Worker*> workers;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workers.push_back(new Worker());
    }

    // **Build the Pipeline**
    // mainPipe.add_stage(&source);  // Source sends model + data
    mainPipe.add_stage(&a2a);     // A2A handles workers
    mainPipe.add_stage(&sink);    // Sink collects updates

    // **Connect A2A Stages**
    a2a.add_firstset<Source>({&source});  // Source sends model + data
    a2a.add_secondset<Worker>(workers);  // Workers process training

    // **Create Distributed Groups**
    a2a.createGroup("G1") << &source;  
    a2a.createGroup("G2") << workers[0] << workers[1] << workers[2] << workers[3];  
    mainPipe.createGroup("G3") << &sink;

    // **Enable Wrap-around (Feedback from Sink to Source)**
    //mainPipe.wrap_around();

    // **Run the Pipeline**
    if (mainPipe.run_and_wait_end() < 0) {
        error("running mainPipe\n");
        return -1;
    }

    // **Save the final trained model**
    source.model.save("trained_resnet152.pt");

    // **Cleanup**
    for (auto w : workers) delete w;
    
    return 0;
}
