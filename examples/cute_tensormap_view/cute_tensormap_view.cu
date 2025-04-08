#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>

// CUDA Driver API
#include <cuda.h>
#include <cuda_runtime.h>

// 用于获取 Driver EntryPoint（cuTensorMapEncodeTiled）
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled_v12000, CUtensorMap

// ------------------ Utilities for printing layout-like info ------------------ //

// 模拟 CUTE 的 print_layout 功能：给定 shape+stride，打印出 2D 矩阵布局
// 只适用于二维布局
struct SimpleLayout2D {
    int dim_m;   // shape M
    int dim_n;   // shape N
    int stride_m;
    int stride_n;
};

// 打印形状、stride 和矩阵内容（如果传入 buffer）
template <typename T>
void print_layout_2d(const SimpleLayout2D& layout, const std::vector<T>& buffer, const char* title) {
    std::cout << "----- " << title << " Layout -----" << std::endl;
    std::cout << "(" << layout.dim_m << "," << layout.dim_n << "):("
              << layout.stride_m << "," << layout.stride_n << ")\n";

    // 这里简单打印矩阵内容，按照 (m,n) 的逻辑坐标读取
    // 注意实际地址偏移 = m*stride_m + n*stride_n
    for (int m = 0; m < layout.dim_m; ++m) {
        for (int n = 0; n < layout.dim_n; ++n) {
            int offset = m * layout.stride_m + n * layout.stride_n;
            std::cout << std::setw(4) << buffer[offset] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// --------------- Get the Driver API function pointer for cuTensorMapEncodeTiled --------------- //
// 定义 cuTensorMapEncodeTiled 的函数指针类型
using PFN_cuTensorMapEncodeTiled_v12000 = CUresult (*)(
    CUtensorMap* tensorMap,
    CUtensorMapDataType dataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill
);

// 使用 dlsym 从 libcuda.so 中获取 cuTensorMapEncodeTiled 的地址
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // 打开 libcuda.so 动态库
    void* handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << std::endl;
        exit(1);
    }
    // 使用 dlsym 获取符号地址
    void* func_ptr = dlsym(handle, "cuTensorMapEncodeTiled");
    if (!func_ptr) {
        std::cerr << "dlsym failed to find cuTensorMapEncodeTiled: " << dlerror() << std::endl;
        exit(1);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

int main()
{
    // 初始化 CUDA 驱动和上下文
    CUresult initRes = cuInit(0);
    if (initRes != CUDA_SUCCESS) {
        std::cerr << "cuInit(0) failed with error code: " << initRes << std::endl;
        return 1;
    }
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed with error code: " << res << std::endl;
        return 1;
    }
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxCreate failed with error code: " << res << std::endl;
        return 1;
    }

    // ===================================================================
    // 1) 定义 global memory 的 “逻辑” shape = (16,16), row-major
    // 我们先在 host 上创建数据用于打印展示
    // ===================================================================
    SimpleLayout2D global_layout{16, 16, /*stride_m=*/16, /*stride_n=*/1};
    std::vector<int> host_global(global_layout.dim_m * global_layout.dim_n);

    // 给 host global buffer 填充一些值
    for (int i = 0; i < (int)host_global.size(); ++i) {
        host_global[i] = i;  // row-major: 0..255
    }
    print_layout_2d(global_layout, host_global, "Global Memory (Host copy)");

    // 现在分配 GPU global memory 并将数据拷贝上去
    int* d_global = nullptr;
    size_t global_bytes = global_layout.dim_m * global_layout.dim_n * sizeof(int);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_global), global_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMemcpy(d_global, host_global.data(), global_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (Host->Device) failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // ===================================================================
    // 2) 用 CUDA driver API 创建一个 cuTensorMap (2D row-major)
    //    让 TMA 知道这块 global memory 的布局
    // ===================================================================
    constexpr uint32_t rank = 2;
    uint64_t size_arr[rank] = {16, 16};    // fastest-moving dimension放在 size_arr[0]
    // 全局 stride: row-major => 每行字节数 = 16 * sizeof(int)
    uint64_t stride_arr[rank - 1] = {16 * sizeof(int)};

    // box_size：这代表 shared memory 要接收 tile 的大小（单位：元素数）
    uint32_t box_size[rank] = {16, 16};
    // element stride：这里都 =1
    uint32_t elem_stride[rank] = {1, 1};

    CUtensorMap tensor_map{};
    auto encode_func = get_cuTensorMapEncodeTiled();

    // 注意：这里 base_ptr 要传 GPU global memory 的指针，而非 host 内存指针
    void* base_ptr = reinterpret_cast<void*>(d_global);

    CUresult rc = encode_func(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,  // 数据类型 int32
        rank,
        base_ptr,       // 使用设备内存指针
        size_arr,       // globalDim
        stride_arr,     // globalStrides
        box_size,       // boxDim
        elem_stride,    // elementStrides
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (rc != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled failed with error code: " << rc << std::endl;
        return 1;
    }

    std::cout << "Created CUtensorMap with rank=2, size=(" << size_arr[0] << "," << size_arr[1]
              << "), stride=" << stride_arr[0] << " bytes, box_size=(" << box_size[0] << "," << box_size[1]
              << "), element_stride=(" << elem_stride[0] << "," << elem_stride[1] << ")\n\n";

    // ===================================================================
    // 3) 定义 shared memory 的“逻辑” shape = (16,16), 但故意用列主序 (stride=(1,16))
    //    然后模拟 copy: global -> shared
    // ===================================================================
    SimpleLayout2D shared_layout{16, 16, /*stride_m=*/1, /*stride_n=*/16};
    std::vector<int> shared_buffer(shared_layout.dim_m * shared_layout.dim_n, 0);

    // 模拟数据搬运：实际 GPU 程序中在 kernel 内会使用 cp.async.bulk.tensor 指令，
    // 这里我们手动 for-loop 模拟将 global memory 内容搬到 shared memory（两者逻辑矩阵相同，但物理地址由 stride 决定）
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            int global_off = m * global_layout.stride_m + n * global_layout.stride_n;
            int shared_off = m * shared_layout.stride_m + n * shared_layout.stride_n;
            shared_buffer[shared_off] = host_global[global_off]; // 这里直接用 host_global 数据作演示
        }
    }

    print_layout_2d(shared_layout, shared_buffer, "Shared Memory (After Copy)");

    // ===================================================================
    // 4) 小结
    // - global_layout, shared_layout 分别用 (shape, stride) 定义
    // - cuTensorMapEncodeTiled(...) 则用于描述 global memory 的布局，
    //   其中 size/stride/box_size 对应 global shape、global stride、以及 tile 尺寸。
    // - 真实 GPU 上，你会在 kernel 中发起 cp.async.bulk.tensor.2d.global.shared，
    //   并用 mbarrier/async-group 机制保证数据可见。
    // - 这里仅做概念演示。
    // ===================================================================

    // 释放设备内存和 CUDA 上下文
    cudaFree(d_global);
    cuCtxDestroy(cuContext);

    return 0;
}
