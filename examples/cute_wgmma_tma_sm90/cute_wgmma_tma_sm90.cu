/***************************************************************************************************
** Minimal TMA Copy Example on SM90:
**   Copies a single 2D tile (M=128, K=64) from global memory to shared memory,
**   then one thread moves it back to another global buffer, and host prints it.
***************************************************************************************************/

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/util/helper_cuda.hpp>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "my_utils.hpp"

using namespace cute;

template <class ElementA, class SmemLayoutA> struct SharedStorage
{
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA, class TmaA, class TD>
__global__ static void tma_copy_once(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A,
    CUTLASS_GRID_CONSTANT TmaA const
        tma_a, // must use CUTLASS_GRID_CONSTANT! otherwise cudaErrorIllegalAddress will occur.
    TD *D_out)
{
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    // Represent the full tensors
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // shape (BLK_M, BLK_K, K/BLK_K)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, SmemLayoutA>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});

    // ------------- TMA partition -------------
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));

    auto gA_grp = group_modes<0, 2>(gA);
    auto sA_grp = group_modes<0, 2>(sA);
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 &&
        threadIdx.z == 0)
    {
        custom_print_layout("SmemLayoutA: ", SmemLayoutA{});

        custom_print_layout("Global tile mA shape: ", layout(mA));
        custom_print_layout("Global tile gA shape: ", layout(gA));
        custom_print_layout("Global tile sA shape: ", layout(sA));

        custom_print_layout("After group_modes(gA) shape: ", layout(gA_grp));
        custom_print_layout("After group_modes(sA) shape: ", layout(sA_grp));

        custom_print_layout("tAgA shape: ", layout(tAgA));
        custom_print_layout("tAsA shape: ", layout(tAsA));
    }

    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    // Current tile index in gmem to read from
    int k_tile = 0;

    // Initialize Barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t *producer_mbar = smem.tma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
        }
    }
    // Ensure barrier init is complete on all CTAs
    cluster_sync();

    // Start async loads for all pipes
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
    //   that flips each cycle through K_PIPE_MAX.
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>(); // MMA  reads

    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // ------------- 将共享内存数据写回到全局 D_out -------------
    // 只用一个线程做拷贝
    if (threadIdx.x == 0)
    {
        // 共享内存中 A 的总元素个数，通过 cosize_v<SmemLayoutA> 可知（编译期常量）
        constexpr int total_elems = cosize_v<SmemLayoutA>;

        TA *shared_ptr = reinterpret_cast<TA *>(smem.A.begin());

        for (int i = 0; i < total_elems; ++i)
        {
            D_out[i] = shared_ptr[i];
            // D_out[i] = *((half_t *)(smem.A.begin()) + i);
        }
    }
}

// 3) Host 端主函数
int main(int argc, char **argv)
{
    int M = 64;
    int N = 64;
    int K = 64;

    using TA = cute::half_t;

    thrust::host_vector<TA> h_A(M * K);
    thrust::host_vector<TA> h_out(M * K);
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = TA(i);
        h_out[i] = TA(0);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TA> d_Out = h_out;

    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, M);

    {
        // print some info
        Tensor host_mA = make_tensor(h_A.data(), make_shape(M, K), dA);
        custom_print_func("HOST: host_mA\n", print_data<decltype(layout(host_mA)), TA>,
                          layout(host_mA), h_A.data());
    }

    auto bM = Int<64>{};
    auto bN = Int<16>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<1>{};                      // Pipeline

    std::cout << "My test: M,N,K, bM,bN,bK, bP = " << M << "," << N << "," << K << ", " << bM << "," << bN << "," << bK
              << ", " << bP << std::endl;

    // Define the smem layouts (static)

    // tile_to_shape -- Perform a product of a layout so that the result matches a target shape.
    // This is similar to blocked_product, but specifies the result shape instead of the
    //   product shape, which is more convenient in certain circumstances.
    // @param block The layout to repeat
    // @param trg_shape The target shape of the result
    // @param ord_shape The order of the modes of @a trg_shape to tile @a layout with.
    //                  Defaults to GenColMajor, so @a layout will repeat
    //                    across the first mode first, the second mode second, etc
    //                  E.g. Step<_2,_1,_3> will cause @a layout to repeat
    //                    across the second mode first, the first mode second, and the third mode last.
    // @pre rank(@a block) <= rank(@a trg_shape)
    // @post compatible(@a trg_shape, shape(@a result))

    // auto sA = tile_to_shape(GMMA::Layout_MN_INTER_Atom<cute::half_t>{}, make_shape(bM, bK, bP));
    // auto sA = tile_to_shape(GMMA::Layout_MN_SW32_Atom<cute::half_t>{}, make_shape(bM, bK, bP));
    // auto sA = tile_to_shape(GMMA::Layout_MN_SW64_Atom<cute::half_t>{}, make_shape(bM, bK, bP));
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cute::half_t>{}, make_shape(bM, bK, bP));
    printf("HOST: sA = ");
    print(layout(sA));
    print("\n");
    // custom_print_func("HOST: sA\n", print_layout<decltype(layout(sA))>, layout(sA(_, _, 0)));

    // Define the TMAs
    // Create Global memory tensors for TMA inspection
    Tensor mA = make_tensor(d_A.data().get(), make_shape(M, K), dA);
    // mA cannot be printed, since its data is on device

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));

    // ------------- Kernel 启动 -------------
    int smem_size = int(sizeof(SharedStorage<TA, decltype(sA)>));
    dim3 dimBlock(128, 1, 1);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(1, 1, 1);
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    std::cout << "smem_size = " << smem_size << std::endl;
    std::cout << "dimBlock = (" << dimBlock.x << ", " << dimBlock.y << ", " << dimBlock.z << ")" << std::endl;
    std::cout << "dimCluster = (" << dimCluster.x << ", " << dimCluster.y << ", " << dimCluster.z << ")" << std::endl;
    std::cout << "dimGrid = (" << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << ")" << std::endl;

    void const *kernel_ptr = reinterpret_cast<void const *>(
        &tma_copy_once<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA), TA>);

    // cudaFuncAttributeMaxDynamicSharedMemorySize 支持更大 shmem size
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr, prob_shape, cta_tiler,
                                                               d_A.data().get(), tmaA, d_Out.data().get(), M);

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    // ------------- 拷回并打印结果 -------------
    // thrust::copy(d_Out.begin(), d_Out.end(), h_out.begin());
    thrust::host_vector<TA> cute_result = d_Out;

    // shared memory 中地址线性增加，为了查看 shared memory 中数据，将其视为行优先的 tensor
    // tensor 的一行（列数）匹配物理 bank，保证输出的一列对应物理带宽
    int out_col = 128 / sizeof(TA);    // eg. TA = FP16, 64 * FP16 = 128 Bytes
    Tensor tensor_out = make_tensor(cute_result.data(), make_shape(bM * bK * bP / out_col, out_col), make_stride(out_col, Int<1>{}));
    printf("HOST: tensor_out = \n");
    print_data(layout(tensor_out), tensor_out.data());

    // // 打印前 10 个元素
    // std::cout << "Check first 10 elements from out:\n";
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << float(h_out[i]) << " ";
    // }
    // std::cout << "\n";

    // // 也可以检查 correctness
    // bool ok = true;
    // for (int i = 0; i < M * K; ++i)
    // {
    //     if (float(h_out[i]) != float(h_A[i]))
    //     {
    //         std::cout << "Mismatch at i=" << i << " got=" << float(h_out[i]) << " expected=" << float(h_A[i]) <<
    //         "\n"; ok = false; break;
    //     }
    // }
    // if (ok)
    // {
    //     std::cout << "TMA Copy Test SUCCESS!\n";
    // }

    return 0;
}
