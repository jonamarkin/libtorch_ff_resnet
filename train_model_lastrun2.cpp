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
    int num_epochs = 10;

    TensorWrapper* svc(TensorWrapper*) {
        static int epoch =0;

        if (epoch >= num_epochs) {
            return EOS;
        }
        
        std::cout<< "Starting epoch " << (epoch + 1) << std::endl;
        auto data_loader = torch::data::make_data_loader(
            dataset.map(torch::data::transforms::Stack<>()),
            /*batch_size=*/64
        );

        for (auto& batch : *data_loader) {
            auto data = batch.data.to(device);
            //TensorWrapper* wrapper = new TensorWrapper(data);
            ff_send_out(new TensorWrapper(data));
        }
        epoch++;
        return GO_ON;
        
        
        //return EOS;
        
    }
};

// **Worker Node**
struct Worker : ff_monode_t<TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::SGD* optimizer;
    torch::Device device = torch::kCPU;

    Worker() {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "CUDA available! Training on GPU.\n";
        }

        try {
            model = torch::jit::load("../resnet152.pt");
            model.to(device);
            std::cout << "Model loaded successfully\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load the model");
        }

        // Convert model parameters to a vector of tensors
        std::vector<torch::Tensor> params;
        for (const auto& param : model.parameters()) {
            params.push_back(param);
        }

        optimizer = new torch::optim::SGD(params, 0.01);
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        // **1. If receiving updated weights, apply them**
        if (!wrapper->toTensorList().empty()) {
            update_model(wrapper->toTensorList());
            return GO_ON;
        }

        // **2. Otherwise, process a training batch**
        torch::Tensor batch = wrapper->toTensor().to(device);
        optimizer->zero_grad();

        torch::Tensor prediction = model.forward({batch}).toTensor();
        torch::Tensor loss = torch::nn::functional::cross_entropy(prediction, torch::zeros({batch.size(0)}, torch::kLong));

        loss.backward();
        optimizer->step();

        // **3. Send updated weights to Sink**
        std::vector<torch::Tensor> model_weights;
        for (const auto& param : model.parameters()) {
            model_weights.push_back(param.clone());
        }

        ff_send_out(new TensorWrapper(model_weights));

        return GO_ON;
    }

    //Update model weights from sink
    void update_model(std::vector<torch::Tensor> new_weights) {
        int i=0;
        for (const auto& param : model.parameters()) {
            param.set_data(new_weights[i++]);
        }
        std::cout << "Worker " << get_my_id() << " updated model weights" << std::endl;
    }
};



// **Sink Node**
// struct Sink : ff_monode_t<TensorWrapper> {
//     std::vector<std::vector<torch::Tensor>> weight_accumulator;
//     int weights_received = 0;

//     Sink() {
//         weight_accumulator.resize(NUM_WORKERS);
//     }

//     TensorWrapper* svc(TensorWrapper* wrapper) {
//         std::vector<torch::Tensor> worker_weights = wrapper->toTensorList();

//         if (weight_accumulator.empty()) {
//             weight_accumulator.resize(worker_weights.size());
//         }

//         for (size_t i = 0; i < worker_weights.size(); i++) {
//             weight_accumulator[i].push_back(worker_weights[i]);
//         }

//         weights_received++;

//         if (weights_received == NUM_WORKERS) {
//             std::vector<torch::Tensor> averaged_weights;
//             for (size_t i = 0; i < weight_accumulator.size(); i++) {
//                 averaged_weights.push_back(torch::stack(weight_accumulator[i]).mean(0));
//             }

//             for (int i = 0; i < NUM_WORKERS; ++i) {
//                 ff_send_out_to(new TensorWrapper(averaged_weights), i);
//             }

//             weights_received = 0;
//             weight_accumulator.clear();
//         }

//         return GO_ON;
//     }
// };

struct Sink : ff_minode_t<TensorWrapper> {
    std::vector<std::vector<torch::Tensor>> weight_accumulator;
    int weights_received = 0;

    Sink() {
        weight_accumulator.resize(NUM_WORKERS);
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        std::vector<torch::Tensor> worker_weights = wrapper->toTensorList();

        if (weight_accumulator.empty()) {
            weight_accumulator.resize(worker_weights.size());
        }

        for (size_t i = 0; i < worker_weights.size(); i++) {
            weight_accumulator[i].push_back(worker_weights[i]);
        }

        weights_received++;

        if (weights_received == NUM_WORKERS) {
            std::vector<torch::Tensor> averaged_weights;
            for (size_t i = 0; i < weight_accumulator.size(); i++) {
                averaged_weights.push_back(torch::stack(weight_accumulator[i]).mean(0));
            }

            // **Send updated weights to ALL workers**
            ff_send_out(new TensorWrapper(averaged_weights));  

            weights_received = 0;
            weight_accumulator.clear();
        }

        return GO_ON;
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
    std::vector<Sink*> sinks;
    for (int i = 0; i < 4; ++i) {
        workers.push_back(new Worker());
        sinks.push_back(new Sink());
    }

    // **Add Stages**
    mainPipe.add_stage(&source);
    mainPipe.add_stage(&a2a);
    //mainPipe.add_stage(&sink);

    // **Connect Nodes**
    a2a.add_firstset<Worker>(workers);
    a2a.add_secondset<Sink>(sinks);

    // **Create Distributed Groups**
    mainPipe.createGroup("G1") << &source;  // Group for source node
    a2a.createGroup("G2") << workers[0] << workers[1] << workers[2] << workers[3];  // Group for workers
    //a2a.createGroup("G3") << &sink;  // Group for sink node
    a2a.createGroup("G3") << sinks[0] << sinks[1] << sinks[2] << sinks[3]; 

    a2a.wrap_around();

    // mainPipe.wrap_around();

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
