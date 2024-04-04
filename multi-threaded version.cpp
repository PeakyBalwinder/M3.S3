#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>

// Function to initialize a vector with random values
void initializeVector(std::vector<int>& vec, int size) {
    vec.resize(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = rand() % 100; // Assigning random values (0 to 99)
    }
}

int main() {
    srand(time(0)); // Seed the random number generator with current time

    int size = 10000000; // Default size of vectors
    std::vector<int> A, B;

    // Initialize vectors A and B
    initializeVector(A, size);
    initializeVector(B, size);

    // Create result vector
    std::vector<int> result(size);

    // Get the number of threads from the user
    int numThreads;
    std::cout << "Enter the number of threads: ";
    std::cin >> numThreads;

    // Get available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Select the first platform
    cl::Platform platform = platforms.front();

    // Get available OpenCL devices on the selected platform
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // Select the first device
    cl::Device device = devices.front();

    // Create OpenCL context
    cl::Context context(device);

    // Create OpenCL command queue
    cl::CommandQueue queue(context, device);

    // Load kernel source code
    std::string kernelSource =
        "__kernel void vectorAddition(__global const int* A, __global const int* B, __global int* result, const int size) {\n"
        "    int i = get_global_id(0);\n"
        "    if (i < size) {\n"
        "        result[i] = A[i] + B[i];\n"
        "    }\n"
        "}\n";

    // Create OpenCL program from kernel source
    cl::Program program(context, kernelSource);

    // Build the OpenCL program
    program.build({device});

    // Create OpenCL kernel
    cl::Kernel kernel(program, "vectorAddition");

    // Allocate memory on the device for input and output vectors
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, B.data());
    cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, sizeof(int) * size);

    // Set kernel arguments
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, size);

    // Start measuring execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the OpenCL kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NDRange(numThreads));

    // Wait for kernel to finish
    queue.finish();

    // Stop measuring execution time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = stop - start;

    // Print the number of threads used
    std::cout << "Number of Threads Used: " << numThreads << std::endl;

    // Print execution time
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

    return 0;
}
