/* 
 * FastFlow concurrent network:
 *
 * 
 *             -----------------------------
 *            |            |--> MiNode1 --> | 
 *            |  MoNode1-->|                | 
 *  Source -->|            |--> MiNode2 --> | ---->  Sink
 *            |  MoNode2-->|                |
 *            |            |--> MiNode3 --> |
 *             -----------------------------                            
 *            |<---------- A2A ------- ---->| 
 *  |<-------------------  pipe ----------------------->|
 *
 *
 *  distributed version:
 *
 *     G1                        G2
 *   --------          -----------------------
 *  |        |        |           |-> MiNode1 |
 *  | Source | ---->  | MoNode1 ->|           | -->|     ------
 *  |        |  |     |           |-> MiNode2 |    |    |      |
 *   --------   |      -----------------------     |--> | Sink |
 *              |               |  ^               |    |      |
 *              |               |  |               |     ------
 *              |               v  |               |       G4   
 *              |      -----------------------     | 
 *               ---> |                       | -->|  
 *                    | MoNode2 ->|-> MiNode3 |
 *                     -----------------------
 *                               G3
 */

 #include <ff/dff.hpp>
 #include <iostream>
 #include <mutex>
 #include <torch/torch.h>
#include <torch/script.h>
 
 #define ITEMS 100
 std::mutex mtx;  // used only for pretty printing
 
 using namespace ff;


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


struct TensorWrapper {
    std::vector<int64_t> sizes;
    std::vector<float> data;
    c10::ScalarType dtype;
    std::vector<TensorWrapper> tensor_list;  // **Supports multiple tensors (model weights)**
    int type;  // **New: Flag to identify if this is model weights (0) or data (1)**

    TensorWrapper() = default;

    // **Wrap single tensor (for data)**
    explicit TensorWrapper(torch::Tensor tensor, int type_flag) : type(type_flag) {
        sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
        dtype = tensor.scalar_type();
        data.assign(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    // **Wrap multiple tensors (for model weights)**
    explicit TensorWrapper(const std::vector<torch::Tensor>& tensors, int type_flag) : type(type_flag) {
        for (const auto& tensor : tensors) {
            tensor_list.emplace_back(TensorWrapper(tensor, type_flag));  // ✅ Now we pass the type flag correctly
        }
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive(sizes, data, dtype, tensor_list, type);
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

    // **Get type of data (0 = model weights, 1 = training data)**
    int getType() const {
        return type;
    }
};
 
 struct Source : ff_monode_t<int>{

    int MAX_ITERATIONS = 5;
    int curr_iter = 0;
    bool isFirstCall = true;

    std::string data_path = "./data/cifar-100-binary/cifar-100-binary";  // Path to the dataset
    int batch_size = 32;               // Batch size for the DataLoader

    


    int* svc(int* i){

        if(isFirstCall){
            //auto dataset = torch::data::datasets::CIFAR100(data_path).map(torch::data::transforms::Stack<>());
            CIFAR100 dataset{"./data/cifar-100-binary/cifar-100-binary", true};
        
        // Create the DataLoader
            auto data_loader = torch::data::make_data_loader(
                dataset.map(torch::data::transforms::Stack<>()),
                batch_size
            );

            // for(int i=0; i< ITEMS; i++){
            for (auto& batch : *data_loader){
                
                // auto data = batch.data.to(torch::kCPU);
                // auto labels = batch.target.to(torch::kCPU);

                ff_send_out_to(new int(1), 0);
                ff_send_out_to(new int(1), 1);
                ff_send_out_to(new int(1), 2);
                ff_send_out_to(new int(1), 3);
                // std::vector<torch::Tensor> batch_pair = {batch.data.clone(), batch.target.clone()};

                // TensorWrapper* wrapper = new TensorWrapper(batch_pair, 1);  // 1 for data
                // ff_send_out(wrapper);
            }

            isFirstCall = false;
            //return GO_ON;

        }else{
            ff::cout << "Source called again!\n";
            
            curr_iter++;

            if(curr_iter >= MAX_ITERATIONS){
                return EOS;
            }else{
                ff::cout << "Sending another batch: called again!\n";
                for(int i=0; i< 50; i++){
                    ff_send_out_to(new int(i), 0);
                    ff_send_out_to(new int(i), 1);
                    ff_send_out_to(new int(i), 2);
                    ff_send_out_to(new int(i), 3);
                }
            }
           
        }

        // }
         
         //
        return EOS_NOFREEZE;
            
     }
 
     void svc_end(){
         ff::cout << "Source ended!\n"; 
     }
 };
 
 struct MoNode : ff_monode_t<int>{
     int processedItems = 0;
     int* svc(int* i){
         ++processedItems;
 
         for(volatile long i=0;i<10000000; ++i);
        
         
         return i;
     }
 
     void svc_end(){
         const std::lock_guard<std::mutex> lock(mtx);
         ff::cout << "[SxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
     }
 };
 
 struct MiNode : ff_minode_t<int>{
     int processedItems = 0;
     int* svc(int* i){
         ++processedItems;
 
         for(volatile long i=0;i<50000000; ++i);
         
         return i;
     }
 
     void svc_end(){
         const std::lock_guard<std::mutex> lock(mtx);
         ff::cout << "[DxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
     }
 };
 
 struct Sink : ff_minode_t<int>{
     int sum = 0;
     int* svc(int* i){
         sum += *i;
        
        //  delete i;
        //  return GO_ON;
        //ff_send_out(i);
         //return 0;
        //Check if sum received is equal to the expected sum before sending it
        //ff::cout << "Current sum: " << sum << std::endl;
        if(sum % ITEMS == 0){
            const std::lock_guard<std::mutex> lock(mtx);
            ff::cout << "Heyyyy: " << sum << " (Expected: " << (ITEMS*(ITEMS-1))/2 << ")" << std::endl;
            //ff_send_out(new int(*i));

            return new int(*i);
        }else{
            //ff::cout << "Sum not up to" << sum << std::endl;
            return GO_ON;
        }
 
         //ff_send_out(i);
         //delete i;
         //return 0;
         //ff_send_out(i);
         //delete i;
        //return new int(*i);

        //return GO_ON;
     }
 
     void svc_end() {
         int local_sum = 0;
         for(int i = 0; i < ITEMS; i++) local_sum += i;
         const std::lock_guard<std::mutex> lock(mtx);
         ff::cout << "Sum: " << sum << " (Expected: " << local_sum << ")" << std::endl;
     }
 };
 
 
 int main(int argc, char*argv[]){
 
     if (DFF_Init(argc, argv) != 0) {
         error("DFF_Init\n");
         return -1;
     }
 
     // defining the concurrent network
     ff_pipeline mainPipe;
     Source source;
     ff_a2a a2a;
     Sink sink;
     mainPipe.add_stage(&source);
     mainPipe.add_stage(&a2a);
     mainPipe.add_stage(&sink);
 
     MoNode sx1, sx2, sx3, sx4;
     MiNode dx1, dx2, dx3, dx4, dx5;
 
     a2a.add_firstset<MoNode>({&sx1, &sx2});
     a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});
 
     //----- defining the distributed groups ------
     source.createGroup("G1");
     a2a.createGroup("G2") << &sx1 << &dx1 << &dx2;
     a2a.createGroup("G3") << &sx2 << &dx3;
     //a2a.createGroup("G4") << &sx3 << &dx4;
     //a2a.createGroup("G5") << &sx4 << &dx5;
     sink.createGroup("G4");

     mainPipe.wrap_around();
 
     // -------------------------------------------
 
     // running the distributed groups
     if (mainPipe.run_and_wait_end()<0) {
         error("running mainPipe\n");
         return -1;
     }
     
     return 0;
 }
 