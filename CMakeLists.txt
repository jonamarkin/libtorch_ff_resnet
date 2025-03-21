cmake_minimum_required(VERSION 3.10)
project(train_resnet)

# Optionally link to torchvision
option(USE_TORCHVISION "Whether to link to torchvision" OFF)

# Find necessary packages
find_package(Torch REQUIRED)

# Find MPI package if available
find_package(MPI REQUIRED)

if(USE_TORCHVISION)
  find_package(TorchVision REQUIRED)
endif()

# Add your executable
add_executable(train_resnet train_model.cpp)

# Link necessary libraries
target_link_libraries(train_resnet "${TORCH_LIBRARIES}")

# Link MPI
target_link_libraries(train_resnet MPI::MPI_CXX)

if(USE_TORCHVISION)
  target_link_libraries(train_resnet TorchVision::TorchVision)
endif()

# Set C++ standard
set_property(TARGET train_resnet PROPERTY CXX_STANDARD 17)
