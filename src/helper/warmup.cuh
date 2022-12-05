
__global__ void warm_up_gpu_kernel(){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float va, vb, vc;
    va = 0.1f;
    vb = 0.2f;

    for(int i = 0; i < 10; i++)
        vc += ((float) tid + va * vb);
}

void warm_up_gpu() {
    warm_up_gpu_kernel<<<4096, 256>>>();
    cudaDeviceSynchronize();
}