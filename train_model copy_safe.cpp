#include "serialize_tensor.hpp"
#include <ff/dff.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <queue>


using namespace ff;
std::mutex mtx;

const int NUM_WORKERS = 4;
const int NUM_EPOCHS = 5;
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
    torch::jit::script::Module model;
    CIFAR100 dataset{"./data/cifar-100-binary/cifar-100-binary", true};
    bool first = true;
    int epoch = 0;

    Source () {
        device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        try {
            model = torch::jit::load("../resnet18_cifar100.pt");
            std::cout << "Model loaded successfully\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load the model");
        }

        model.to(device);
    }

    TensorWrapper* svc(TensorWrapper* w) {
        if (epoch >= NUM_EPOCHS) {
            std::cout << "Training complete after " << NUM_EPOCHS << " epochs!\n";
            return EOS;  // ✅ Stop execution after required epochs
        }

        if (first) {
            first = false;

             // **Send initial model weights to workers**
            // std::vector<torch::Tensor> model_weights;
            // for (const auto& param : model.parameters()) model_weights.push_back(param.clone());

            // std::cout << "Sending initial model weights to workers\n";
            // ff_send_out(new TensorWrapper(model_weights, TensorType::WEIGHTS)); 



            // for(int i=0; i< NUM_EPOCHS; i++){
            //     std::cout << "Source: Starting epoch " << (epoch + 1) << "\n";
            //     auto data_loader = torch::data::make_data_loader(
            //         dataset.map(torch::data::transforms::Stack<>()),
            //         /*batch_size=*/64
            //     );

            //     for (auto& batch : *data_loader) {
            //         auto data = batch.data.to(device);
            //         //std::cout << "Source: Sending data to workers\n";
            //         ff_send_out(new TensorWrapper(data, TensorType::DATA));
            //     }
            // }

            std::cout << "Source node started, sending data to workers\n";
            auto data_loader = torch::data::make_data_loader(
                dataset.map(torch::data::transforms::Stack<>()),
                /*batch_size=*/64
            );

            for (auto& batch : *data_loader) {
                // auto data = batch.data.to(device);
                // //std::cout << "Source: Sending data to workers\n";
                // ff_send_out(new TensorWrapper(data, TensorType::DATA));

                // Send different batches to different workers in round-robin fashion
                int target_worker = batch_counter % NUM_WORKERS;
                batch_counter++;
                
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                
                // Pack both data and labels together
                std::vector<torch::Tensor> batch_pair = {data, target};
                ff_send_out_to(new TensorWrapper(batch_pair, TensorType::DATA), target_worker);
            }


            return GO_ON;
        } else {
            
            // When receiving weights from Sink via Feedback
            try {
                std::vector<torch::Tensor> new_weights = w->toTensorList();
                
                // Check if we received the correct type of data
                if (new_weights.empty()) {
                    std::cerr << "Error: Received empty weights list in Source" << std::endl;
                    return GO_ON;
                }
                
                int i = 0;
                for (const auto& param : model.parameters()) {
                    if (i < new_weights.size()) {
                        param.set_data(new_weights[i++].to(device));
                    } else {
                        std::cerr << "Error: Not enough weights received in Source" << std::endl;
                        break;
                    }
                }
                std::cout << "Source: Model updated for epoch " << (epoch + 1) << "\n";
                epoch++;
            } catch (const std::exception& e) {
                std::cerr << "Error updating model in Source: " << e.what() << std::endl;
            }
        
            return GO_ON;
        }
    }
};



struct Worker : ff_minode_t<TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::SGD* optimizer;
    int processedBatches = 0;
    torch::Device device = torch::kCPU;
    bool model_initialized = true;

    std::queue<torch::Tensor> data_queue;  // Queue to store incoming data

    Worker() {
        device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        try {
            model = torch::jit::load("../resnet18_cifar100.pt");
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
        //optimizer = nullptr;
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        if(wrapper->type == TensorType::DATA){
            // **Received data from Source**
            // Store incoming data in the queue
            data_queue.push(wrapper->toTensor().to(device));
            
            while (!data_queue.empty()) {
                torch::Tensor batch = data_queue.front();
                data_queue.pop();
    
                // Train on the batch and ensure it completes
                TensorWrapper* result = train(batch);
                
                // Only send out updated model parameters after training completes
                if (result != nullptr) {
                    // Delete the result to avoid memory leak
                    delete result;
                    
                    // Now that training is complete, send the updated model weights to Sink
                    std::vector<torch::Tensor> model_params;
                    for (const auto& param : model.parameters()) {
                        model_params.push_back(param.clone());
                    }
    
                    ff_send_out(new TensorWrapper(model_params, TensorType::WEIGHTS));
                }
            }
        }
    
        if(wrapper->type == TensorType::WEIGHTS){
            // **Received updated model from Sink**
            std::cout << "Worker " << get_my_id() << " received model update!\n";
            
            // Apply new weights to model
            update_model(*wrapper);
        }
        
        return EOS_NOFREEZE;
    }
    
    TensorWrapper* train(const torch::Tensor& batch) {
        try {
            optimizer->zero_grad();
            torch::Tensor prediction = model.forward({batch}).toTensor();
            torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, 1),
                                                torch::zeros({batch.size(0)}, torch::kLong));
    
            loss.backward();
            optimizer->step();
            ++processedBatches;
    
            if (processedBatches % 100 == 0) {
                std::cout << "Worker " << get_my_id() << " | Batch: " << processedBatches
                        << " | Loss: " << loss.item<float>() << std::endl;
            }
    
            return new TensorWrapper(prediction);
        }
        catch(const std::exception& e) {
            std::cerr << "Error during training: " << e.what() << '\n';
            return nullptr;  // Return null if training fails
        }
    }

    void update_model(const TensorWrapper& wrapper) {
        // Get the list of tensors (separate parameters)
        std::vector<torch::Tensor> new_weights = wrapper.toTensorList();
        
        // Update each parameter with the corresponding weight tensor
        int i = 0;
        for (const auto& param : model.parameters()) {
            param.set_data(new_weights[i++]);
        }
        
        // Reinitialize optimizer with updated parameters
        std::vector<torch::Tensor> params;
        for (const auto& param : model.parameters()) {
            params.push_back(param);
        }
        delete optimizer;  // Delete old optimizer first to avoid memory leak
        optimizer = new torch::optim::SGD(params, 0.01);
        
        std::cout << "Worker " << get_my_id() << " updated model weights.\n";
    }

    void svc_end() {
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Worker " << get_my_id() << " processed "
                  << processedBatches << " batches" << std::endl;
        delete optimizer;
    }
};


struct Sink : ff_minode_t<TensorWrapper> {
    int receivedBatches = 0;
    int workersReported = 0;
    std::vector<torch::Tensor> accumulatedParameters;
    bool parametersInitialized = false;
    torch::Device device = torch::kCPU;
    
    Sink() {
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        std::cout << "Sink initialized" << std::endl;
    }

    TensorWrapper* svc(TensorWrapper* wrapper) {
        // Handle prediction results (DATA type)
        if (wrapper->type == TensorType::DATA) {
            ++receivedBatches;
            if (receivedBatches % 100 == 0) {
                std::cout << "Sink: Received " << receivedBatches << " prediction batches" << std::endl;
            }
            return GO_ON;
        }
        
        // Handle weights updates (WEIGHTS type)
        if (wrapper->type == TensorType::WEIGHTS) {
            std::vector<torch::Tensor> worker_params = wrapper->toTensorList();
            
            // If this is the first set of parameters we've seen in this epoch
            if (!parametersInitialized) {
                accumulatedParameters.clear();
                for (const auto& param : worker_params) {
                    accumulatedParameters.push_back(param.clone());
                }
                parametersInitialized = true;
            } else {
                // Add these parameters to our running total
                for (size_t i = 0; i < worker_params.size(); i++) {
                    accumulatedParameters[i] += worker_params[i];
                }
            }
            
            workersReported++;
            std::cout << "Sink: Worker " << workersReported << "/" << NUM_WORKERS 
                     << " reported weights." << std::endl;
            
            // If all workers have reported, average the parameters and send back
            if (workersReported == NUM_WORKERS) {
                std::cout << "Sink: All workers reported. Computing averaged model." << std::endl;
                
                // Divide accumulated parameters by number of workers to get average
                for (auto& param : accumulatedParameters) {
                    param.div_(static_cast<float>(NUM_WORKERS));
                }
                
                // Reset for next epoch
                workersReported = 0;
                receivedBatches = 0;
                parametersInitialized = false;
                
                std::cout << "Sink: Sending averaged model back to Source" << std::endl;
                return new TensorWrapper(accumulatedParameters, TensorType::WEIGHTS);
            }
        }
        
        return GO_ON;
    }
};


// **Feedback Node**
struct FeedBack : ff_monode_t<TensorWrapper> {
    TensorWrapper* svc(TensorWrapper* wrapper) {
        std::cout << "Feedback: Sending updated weights to Source\n";
        ff_send_out_to(wrapper, 0);
        return GO_ON;
    }
};


// **Main FastFlow Pipeline**
int main(int argc, char* argv[]) {

     // **Start timing**
    auto start_time = std::chrono::high_resolution_clock::now();
    if (DFF_Init(argc, argv) != 0) {
        error("DFF_Init\n");
        return -1;
    }
    

    ff_pipeline mainPipe;
    ff_a2a a2a;
    Source source;
    //Start source1;
    Sink sink;
    FeedBack feedback;

    // **Worker Pool**
    std::vector<Worker*> workers;
    for (int i = 0; i < 4; ++i) {
        workers.push_back(new Worker());
    }

    // **Add Stages**
    //mainPipe.add_stage(&source1);
    mainPipe.add_stage(&a2a);
    mainPipe.add_stage(&sink);
    mainPipe.add_stage(&feedback);

    // **Connect Nodes**
    a2a.add_firstset<Source>({&source});
    a2a.add_secondset<Worker>(workers);

    // **Create Distributed Groups**
    //source1.createGroup("G1") << &source1;  // Group for source node
    a2a.createGroup("G2") << &source; // Group for source node
    a2a.createGroup("G3") << workers[0] << workers[1] << workers[2] << workers[3];  // Group for workers
    sink.createGroup("G4") << &sink;  // Group for sink node
    feedback.createGroup("G5") << &feedback;  // Group for feedback node

    mainPipe.wrap_around();  // Wrap the pipeline

    // **Run the Pipeline**
    if (mainPipe.run_and_wait_end() < 0) {
        error("running mainPipe\n");
        return -1;
    }

     // **End timing**
     auto end_time = std::chrono::high_resolution_clock::now();
     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

     // **Print elapsed time**
     std::cout << "Training completed in " << duration.count() << " seconds.\n";

    // Save the trained model after processing
    workers[0]->model.save("trained_resnet152.pt");

    for (auto w : workers) delete w;
    return 0;
}
