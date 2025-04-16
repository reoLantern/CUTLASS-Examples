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

template <class ElementA, class ElementB,
          class SmemLayoutA, // (M,K,P)
          class SmemLayoutB> // (N,K,P)
struct SharedStorage
{
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA, class TmaA, class TB, class SmemLayoutB,
          class TmaB, class TC, class CStride, class TiledMma, class TD>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A,
    CUTLASS_GRID_CONSTANT TmaA const tma_a, // must use CUTLASS_GRID_CONSTANT! otherwise
                                            // cudaErrorIllegalAddress will occur.
    TB const *B, CUTLASS_GRID_CONSTANT TmaB const tma_b, TC *C, CStride dC, TiledMma mma,
    TD *D_out // for verify TMA {global A -> shmem -> global out} only
)
{
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler)); // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));              // (M,K) TMA Tensor
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));              // (N,K) TMA Tensor
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // shape (BLK_M, BLK_K, K/BLK_K)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // ------------- TMA partition -------------
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA)); // (TMA,k) and (TMA,PIPE)
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB)); // (TMA,k) and (TMA,PIPE)

    {
        // print some info
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
    }

    // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
    constexpr int tma_transaction_bytes =
        sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    //
    // PREFETCH
    //
    auto K_PIPE_MAX = size<1>(tAsA);

    // Total count of tiles
    int k_tile_count = size<1>(tAgA);
    // Current tile index in gmem to read from
    int k_tile = 0;

    // Initialize Barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t *producer_mbar = smem.tma_barrier;
    uint64_t *consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
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
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate accumulators and clear them
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // Allocate "fragments"
    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    //
    // PIPELINED MAIN LOOP
    //

    // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
    //   that flips each cycle through K_PIPE_MAX.
    auto write_state = cutlass::PipelineState<K_PIPE_MAX>(); // TMA writes
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();  // MMA  reads

    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 &&
            threadIdx.z == 0)
            printf("----------------- Start MMA -----------------\n");
    }

    {
        // Wait for Producer to complete
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // MMAs to cover 1 K_TILE
        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC); // (V,M) x (V,N) => (V,M,N)
        warpgroup_commit_batch();

        // Wait for all MMAs in a K_TILE to complete
        warpgroup_wait<0>();

        // Notify that consumption is done
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx == 0) && lane_predicate)
        {
            int pipe = write_state.index();
            // Wait for Consumer to complete consumption
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
        }
    }

    // CUTE_NO_UNROLL
    // while (k_tile_count > -K_PIPE_MAX)
    // {
    //   // Wait for Producer to complete
    //   int read_pipe = read_state.index();
    //   ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
  
    //   // MMAs to cover 1 K_TILE
    //   warpgroup_arrive();
    //   gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    //   warpgroup_commit_batch();
  
    //   // Wait for all MMAs in a K_TILE to complete
    //   warpgroup_wait<0>();
  
    //   // Notify that consumption is done
    //   ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    //   ++read_state;
  
    //   if ((warp_idx == 0) && lane_predicate)
    //   {
    //     int pipe = write_state.index();
    //     // Wait for Consumer to complete consumption
    //     ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
    //     // Set expected Tx Bytes after each reset / init
    //     ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
    //     copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
    //     copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    //     ++write_state;
    //   }
    //   --k_tile_count;
    //   ++k_tile;
    // }

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
    int M = 128;
    int N = 128;
    int K = 128;
    for (int i = 1; i < argc; i++) {
        if (std::strncmp(argv[i], "--M=", 4) == 0) {
            M = std::atoi(argv[i] + 4);  // 从第四个字符开始转换
        } else if (std::strncmp(argv[i], "--N=", 4) == 0) {
            N = std::atoi(argv[i] + 4);
        } else if (std::strncmp(argv[i], "--K=", 4) == 0) {
            K = std::atoi(argv[i] + 4);
        }
    }
    int m = M, n = N, k = K;

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;

    thrust::host_vector<TA> h_A(M * K);
    thrust::host_vector<TB> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N);

    thrust::host_vector<TA> h_out(M * K);
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = TA(i);
        h_out[i] = TA(0);
    }
    for (int j = 0; j < n * k; ++j)
        h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m * n; ++j)
        h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TA> d_Out = h_out;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    int ldA = m, ldB = n, ldC = m;

    auto prob_shape = make_shape(M, N, K);

    // Define TN strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

    {
        // print some info
        Tensor host_mA = make_tensor(h_A.data(), make_shape(M, K), dA);
        custom_print_func("HOST: host_mA\n", print_data<decltype(layout(host_mA)), TA>, layout(host_mA), h_A.data());
    }

    // Define CTA tile sizes (static)
    auto bM = Int<64>{};
    auto bN = Int<64>{};
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
    auto sA = tile_to_shape(GMMA::Layout_MN_INTER_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_INTER_Atom<TB>{}, make_shape(bN, bK, bP));

    printf("HOST: sA = ");
    print(layout(sA));
    print("\n");
    // custom_print_func("HOST: sA\n", print_layout<decltype(layout(sA))>, layout(sA(_, _, 0)));

    // Define the MMA
    // TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x32x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    // Define the TMAs
    // Create Global memory tensors for TMA inspection
    Tensor mA = make_tensor(d_A.data().get(), make_shape(M, K), dA);
    Tensor mB = make_tensor(d_B.data().get(), make_shape(N, K), dB);
    // mA cannot be printed, since its data is on device

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    // ------------- Kernel 启动 -------------
    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x), round_up(size(ceil_div(n, bN)), dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    std::cout << "smem_size = " << smem_size << std::endl;
    std::cout << "dimBlock = (" << dimBlock.x << ", " << dimBlock.y << ", " << dimBlock.z << ")" << std::endl;
    std::cout << "dimCluster = (" << dimCluster.x << ", " << dimCluster.y << ", " << dimCluster.z << ")" << std::endl;
    std::cout << "dimGrid = (" << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << ")" << std::endl;

    void const *kernel_ptr = reinterpret_cast<void const *>(
        &gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA), TB, decltype(sB),
                     decltype(tmaB), TC, decltype(dC), decltype(tiled_mma), TA>);

    // cudaFuncAttributeMaxDynamicSharedMemorySize 支持更大 shmem size
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Kernel Launch
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params, kernel_ptr, prob_shape, cta_tiler, d_A.data().get(), tmaA,
                                          d_B.data().get(), tmaB, d_C.data().get(), dC, tiled_mma, d_Out.data().get());

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    // ------------- 拷回并打印结果 -------------
    // thrust::copy(d_Out.begin(), d_Out.end(), h_out.begin());
    thrust::host_vector<TA> cute_result = d_Out;

    // shared memory 中地址线性增加，为了查看 shared memory 中数据，将其视为行优先的 tensor
    // tensor 的一行（列数）匹配物理 bank，保证输出的一列对应物理带宽
    int out_col = 128 / sizeof(TA); // eg. TA = FP16, 64 * FP16 = 128 Bytes
    Tensor tensor_out =
        make_tensor(cute_result.data(), make_shape(bM * bK * bP / out_col, out_col), make_stride(out_col, Int<1>{}));
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
