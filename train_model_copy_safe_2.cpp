#include <torch/torch.h>
#include <torch/script.h>
#include <ff/dff.hpp>
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>

using namespace ff;
std::mutex mtx;

// Configuration constants
const int NUM_WORKERS = 4;
const int NUM_EPOCHS = 10;
const int BATCH_SIZE = 64;
const float LEARNING_RATE = 0.001;
const std::string MODEL_PATH = "resnet152.pt";
const std::string DATA_PATH = "./data/cifar-100-binary/cifar-100-binary";
const std::string OUTPUT_MODEL_PATH = "resnet152_trained_distributed.pt";

// Enum to indicate type of tensor message
enum class TensorType { 
    DATA_BATCH,    // Contains features and labels
    WEIGHTS,       // Model weights
    GRADIENTS,     // Gradients from workers
    EPOCH_START,   // Marker for epoch start
    EPOCH_END,     // Marker for epoch end
    VALIDATION,    // Validation results
    STOP           // Signal to stop processing
};

// Tensor wrapper for serialization and message passing
struct TensorWrapper {
    std::vector<int64_t> sizes;
    std::vector<float> data;
    TensorType type;
    int metadata;  // Additional info (epoch number, worker ID, etc.)
    std::vector<TensorWrapper> tensor_list;  // For multiple tensors

    TensorWrapper() = default;

    // Constructor for marker messages (no data)
    explicit TensorWrapper(TensorType t) : type(t), metadata(0) {}

    // Constructor for marker messages with metadata
    explicit TensorWrapper(TensorType t, int meta) : type(t), metadata(meta) {}

    // Constructor for single tensor
    explicit TensorWrapper(torch::Tensor tensor, TensorType t = TensorType::DATA_BATCH) : type(t), metadata(0) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    // Constructor for tensor pairs (features + labels)
    explicit TensorWrapper(const torch::Tensor& features, const torch::Tensor& labels, TensorType t = TensorType::DATA_BATCH) 
        : type(t), metadata(0) {
        // Store features in this wrapper
        sizes.assign(features.sizes().begin(), features.sizes().end());
        data.assign(features.data_ptr<float>(), features.data_ptr<float>() + features.numel());
        
        // Store labels in tensor_list
        TensorWrapper label_wrapper;
        label_wrapper.sizes.assign(labels.sizes().begin(), labels.sizes().end());
        
        // Handle int64 labels by converting to float
        std::vector<float> label_data(labels.numel());
        for (int64_t i = 0; i < labels.numel(); i++) {
            label_data[i] = static_cast<float>(labels[i].item<int64_t>());
        }
        label_wrapper.data = std::move(label_data);
        tensor_list.push_back(label_wrapper);
    }

    // Constructor for multiple tensors (model weights)
    explicit TensorWrapper(const std::vector<torch::Tensor>& tensors, TensorType t = TensorType::WEIGHTS) : type(t), metadata(0) {
        for (const auto& tensor : tensors) {
            TensorWrapper wrapped;
            wrapped.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
            wrapped.data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
            tensor_list.push_back(wrapped);
        }
    }

    // Convert back to a single tensor
    torch::Tensor toTensor() const {
        return torch::from_blob(const_cast<float*>(data.data()), sizes, 
                                torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }

    // Get labels tensor from tensor_list
    torch::Tensor getLabels() const {
        if (tensor_list.empty()) {
            throw std::runtime_error("No labels found in tensor_list");
        }
        
        const auto& label_wrapper = tensor_list[0];
        std::vector<int64_t> int_labels(label_wrapper.data.size());
        for (size_t i = 0; i < label_wrapper.data.size(); i++) {
            int_labels[i] = static_cast<int64_t>(label_wrapper.data[i]);
        }
        
        return torch::from_blob(int_labels.data(), label_wrapper.sizes, 
                               torch::TensorOptions().dtype(torch::kInt64)).clone();
    }

    // Convert back to a list of tensors (for model weights)
    std::vector<torch::Tensor> toTensorList() const {
        std::vector<torch::Tensor> tensors;
        for (const auto& wrapped : tensor_list) {
            tensors.push_back(torch::from_blob(
                const_cast<float*>(wrapped.data.data()), 
                wrapped.sizes, 
                torch::TensorOptions().dtype(torch::kFloat32)).clone());
        }
        return tensors;
    }

    // Serialization for FastFlow
    template<class Archive>
    void serialize(Archive & archive) {
        archive(sizes, data, type, metadata, tensor_list);
    }
};

// CIFAR100 dataset loader
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

        // Process labels
        for (size_t i = 0; i < num_images; i++) {
            labels_[i] = static_cast<int64_t>(buffer[i * record_size + 1]);
        }

        // Process images
        auto images_bytes = torch::from_blob(buffer.data() + 2, // Skip first two bytes
                                           {static_cast<long>(num_images), 3, 32, 32},
                                           torch::kUInt8);

        // Convert to float and normalize
        images_ = images_bytes.to(torch::kFloat32).div(255.0);
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

// =====================================================================
// Source Node: Manages model, distributes data, and controls training flow
// =====================================================================
struct Source : ff_monode_t<TensorWrapper> {
    torch::Device device;
    torch::jit::script::Module model;
    CIFAR100 train_dataset;
    CIFAR100 test_dataset;
    int epoch;
    float best_accuracy;
    
    Source() 
        : device(torch::kCPU),  // Explicitly initialize device
        train_dataset(DATA_PATH, true), 
        test_dataset(DATA_PATH, false),
        epoch(0), 
        best_accuracy(0.0f) {
            
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        try {
            model = torch::jit::load(MODEL_PATH);
            std::cout << "Model loaded successfully\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load the model");
        }

        model.to(device);
    }


    TensorWrapper* svc(TensorWrapper* w) {
        // First run or after receiving weights from previous epoch
        if (w == nullptr || w->type == TensorType::WEIGHTS) {
            // Update model if we received weights from previous epoch
            if (w != nullptr && w->type == TensorType::WEIGHTS) {
                std::vector<torch::Tensor> new_weights = w->toTensorList();
                int param_idx = 0;
                for (const auto& param : model.parameters()) {
                    param.data() = new_weights[param_idx++].to(device);
                }
                
                // Validate model
                float accuracy = validate();
                std::cout << "Epoch " << epoch << " completed. Validation accuracy: " 
                          << accuracy << "%" << std::endl;
                
                // Save best model
                if (accuracy > best_accuracy) {
                    best_accuracy = accuracy;
                    model.save(OUTPUT_MODEL_PATH);
                    std::cout << "New best model saved with accuracy: " << best_accuracy << "%" << std::endl;
                }
                
                // Increment epoch and check if we're done
                epoch++;
                if (epoch >= NUM_EPOCHS) {
                    std::cout << "Training completed after " << NUM_EPOCHS << " epochs!" << std::endl;
                    for (int i = 0; i < NUM_WORKERS; i++) {
                        ff_send_out_to(new TensorWrapper(TensorType::STOP), i);
                    }
                    return EOS;
                }
            }
            
            // Send initial model weights to all workers
            std::vector<torch::Tensor> model_weights;
            for (const auto& param : model.parameters()) {
                model_weights.push_back(param.clone());
            }
            
            for (int i = 0; i < NUM_WORKERS; i++) {
                ff_send_out_to(new TensorWrapper(model_weights, TensorType::WEIGHTS), i);
            }
            
            // Send epoch start marker with current epoch number
            for (int i = 0; i < NUM_WORKERS; i++) {
                ff_send_out_to(new TensorWrapper(TensorType::EPOCH_START, epoch), i);
            }
            
            // Distribute data batches to workers in round-robin fashion
            auto data_loader = torch::data::make_data_loader(
                train_dataset.map(torch::data::transforms::Stack<>()),
                torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
            );
            
            int batch_idx = 0;
            for (auto& batch : *data_loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                
                // Round-robin distribution
                int target_worker = batch_idx % NUM_WORKERS;
                batch_idx++;
                
                ff_send_out_to(new TensorWrapper(data, target, TensorType::DATA_BATCH), target_worker);
            }
            
            // Send epoch end marker
            for (int i = 0; i < NUM_WORKERS; i++) {
                ff_send_out_to(new TensorWrapper(TensorType::EPOCH_END, epoch), i);
            }
            
            return GO_ON;
        }
        
        return GO_ON;
    }
    
    float validate() {
        model.eval();
        torch::NoGradGuard no_grad;
        
        size_t correct = 0;
        size_t total = 0;
        
        auto test_loader = torch::data::make_data_loader(
            test_dataset.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
        );
        
        for (const auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(data);
            auto output = model.forward(inputs).toTensor();
            
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().item<int64_t>();
            total += targets.size(0);
        }
        
        return static_cast<float>(correct) / total * 100.0f;
    }
};

// =====================================================================
// Worker Node: Performs the actual training
// =====================================================================
struct Worker : ff_minode_t<TensorWrapper> {
    torch::jit::script::Module model;
    torch::optim::Adam* optimizer;
    torch::Device device;
    int worker_id;
    int current_epoch;
    int batches_processed;
    float epoch_loss;
    bool model_initialized;
    
    Worker() : device(torch::kCPU), 
                current_epoch(0), 
               batches_processed(0), 
               epoch_loss(0.0f),
               model_initialized(false) {
        worker_id = get_my_id();
        
        device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
        
        try {
            model = torch::jit::load(MODEL_PATH);
            std::cout << "Worker " << worker_id << ": Model loaded successfully\n";
        } catch (const c10::Error& e) {
            std::cerr << "Worker " << worker_id << ": Error loading model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load the model");
        }
        
        model.to(device);
        
        // Initialize optimizer
        std::vector<torch::Tensor> params;
        for (const auto& param : model.parameters()) {
            params.push_back(param);
        }
        optimizer = new torch::optim::Adam(params, LEARNING_RATE);
    }
    
    ~Worker() {
        delete optimizer;
    }
    
    TensorWrapper* svc(TensorWrapper* wrapper) {
        // Stop signal received
        if (wrapper->type == TensorType::STOP) {
            std::cout << "Worker " << worker_id << " stopping" << std::endl;
            return EOS;
        }
        
        // Model weights update
        if (wrapper->type == TensorType::WEIGHTS) {
            std::vector<torch::Tensor> new_weights = wrapper->toTensorList();
            
            // Update model parameters
            int param_idx = 0;
            for (const auto& param : model.parameters()) {
                param.data() = new_weights[param_idx++].to(device);
            }
            
            // Reinitialize optimizer with updated parameters
            delete optimizer;
            std::vector<torch::Tensor> params;
            for (const auto& param : model.parameters()) {
                params.push_back(param);
            }
            optimizer = new torch::optim::Adam(params, LEARNING_RATE);
            
            model_initialized = true;
            std::cout << "Worker " << worker_id << ": Model updated with new weights" << std::endl;
            return GO_ON;
        }
        
        // Epoch start
        if (wrapper->type == TensorType::EPOCH_START) {
            current_epoch = wrapper->metadata;
            batches_processed = 0;
            epoch_loss = 0.0f;
            std::cout << "Worker " << worker_id << ": Starting epoch " << current_epoch << std::endl;
            return GO_ON;
        }
        
        // Process data batch
        if (wrapper->type == TensorType::DATA_BATCH && model_initialized) {
            auto data = wrapper->toTensor().to(device);
            auto target = wrapper->getLabels().to(device);
            
            // Train on batch
            model.train();
            optimizer->zero_grad();
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(data);
            auto output = model.forward(inputs).toTensor();
            
            auto loss = torch::nn::functional::cross_entropy(output, target);
            
            loss.backward();
            optimizer->step();
            
            float loss_val = loss.item<float>();
            epoch_loss += loss_val;
            batches_processed++;
            
            if (batches_processed % 50 == 0) {
                std::cout << "Worker " << worker_id << " | Epoch: " << current_epoch 
                          << " | Batch: " << batches_processed 
                          << " | Loss: " << loss_val 
                          << " | Avg Loss: " << (epoch_loss / batches_processed) << std::endl;
            }
            
            return GO_ON;
        }
        
        // Epoch end
        if (wrapper->type == TensorType::EPOCH_END) {
            std::cout << "Worker " << worker_id << ": Completed epoch " << current_epoch 
                      << " with " << batches_processed << " batches"
                      << " | Avg Loss: " << (epoch_loss / (batches_processed > 0 ? batches_processed : 1)) 
                      << std::endl;
            
            // Send model parameters to Sink
            std::vector<torch::Tensor> model_params;
            for (const auto& param : model.parameters()) {
                model_params.push_back(param.clone().detach().to(torch::kCPU));
            }
            
            return new TensorWrapper(model_params, TensorType::WEIGHTS);
        }
        
        return GO_ON;
    }
};

// =====================================================================
// Sink Node: Aggregates model parameters from workers
// =====================================================================
struct Sink : ff_minode_t<TensorWrapper> {
    int workers_reported;
    std::vector<torch::Tensor> accumulated_params;
    bool parameters_initialized;
    
    Sink() : workers_reported(0), parameters_initialized(false) {
        std::cout << "Sink node initialized" << std::endl;
    }
    
    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (wrapper->type == TensorType::WEIGHTS) {
            std::vector<torch::Tensor> worker_params = wrapper->toTensorList();
            
            // Initialize accumulated parameters if this is the first worker report
            if (!parameters_initialized) {
                accumulated_params.clear();
                for (const auto& param : worker_params) {
                    accumulated_params.push_back(param.clone());
                }
                parameters_initialized = true;
            } else {
                // Add these parameters to our running total
                for (size_t i = 0; i < worker_params.size(); i++) {
                    accumulated_params[i] += worker_params[i];
                }
            }
            
            workers_reported++;
            std::cout << "Sink: " << workers_reported << "/" << NUM_WORKERS 
                      << " workers reported weights" << std::endl;
            
            // If all workers have reported, average the parameters and send back
            if (workers_reported == NUM_WORKERS) {
                std::cout << "Sink: All workers reported. Computing averaged model." << std::endl;
                
                // Average the accumulated parameters
                for (auto& param : accumulated_params) {
                    param.div_(static_cast<float>(NUM_WORKERS));
                }
                
                // Reset for next epoch
                workers_reported = 0;
                parameters_initialized = false;
                
                // Return averaged weights to be sent to Source
                return new TensorWrapper(accumulated_params, TensorType::WEIGHTS);
            }
        }
        
        return GO_ON;
    }
};

// =====================================================================
// Feedback Node: Closes the training loop
// =====================================================================
struct Feedback : ff_monode_t<TensorWrapper> {
    TensorWrapper* svc(TensorWrapper* wrapper) {
        if (wrapper->type == TensorType::WEIGHTS) {
            std::cout << "Feedback: Sending updated weights to Source for next epoch" << std::endl;
            ff_send_out_to(wrapper, 0);  // Send to Source node
        }
        return GO_ON;
    }
};

// =====================================================================
// Main function
// =====================================================================
int main(int argc, char* argv[]) {
    // Initialize FastFlow
    if (DFF_Init(argc, argv) != 0) {
        std::cerr << "Error initializing FastFlow" << std::endl;
        return -1;
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create nodes
        ff_pipeline main_pipeline;
        ff_a2a a2a;
        Source source, source2;
        Sink sink;
        Feedback feedback;
        
        // Create worker pool
        std::vector<Worker*> workers;
        for (int i = 0; i < 4; ++i) {
            workers.push_back(new Worker());
        }

        
        // Set up the pipeline: Source -> Workers -> Sink -> Feedback -> Source
        main_pipeline.add_stage(&a2a);
        main_pipeline.add_stage(&sink);
        main_pipeline.add_stage(&feedback);
        
        // Set up the all-to-all pattern
        a2a.add_firstset<Source>({&source});
        a2a.add_secondset<Worker>(workers);
        
        // Create groups for nodes
        a2a.createGroup("G1") << &source << workers[0] << workers[1];
        a2a.createGroup("G2") << &source2 << workers[2] << workers[3];
        //a2a.createGroup("G3") << workers[2] << workers[3];
        sink.createGroup("G4") << &sink;
        feedback.createGroup("G5") << &feedback;
        
        // Enable wrap-around for feedback
        main_pipeline.wrap_around();
        
        std::cout << "Starting distributed training with " << NUM_WORKERS << " workers for " 
                  << NUM_EPOCHS << " epochs" << std::endl;
        
                  
        if (main_pipeline.run_and_wait_end() < 0) {
            std::cerr << "Error running pipeline" << std::endl;
            return -1;
        }
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        // Print results
        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
        std::cout << "Final model saved to " << OUTPUT_MODEL_PATH << std::endl;
        
        // Clean up workers
        for (auto* w : workers) {
            delete w;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}