#include "./ear_qdq.cu"

// Setup node's mesh connections
template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constChannelsAtoBs[7];
__constant__ DeviceHandle<mscclpp::PortChannel> constChannelsBtoAs[7];

// DeviceSyncer for the current node
__device__ mscclpp::DeviceSyncer deviceSyncer;


__device__ void crossReduceOneRound(
    const fp8x2* __restrict__ outgoingBuffer,
    const float2* __restrict__ outgoingQParams,
    fp8x2* __restrict__ incomingBuffer,
    float2* __restrict__ incomingQParams,
    bool outgoingBufferIsA,
    const int localRank,   // Rank of the current node, 0 <= localRank < worldSize
    const int partnerRank, // Rank of the partner node, 0 <= partnerRank < worldSize
    const size_t bytesToSend
) {
    // Figure out thread's position in the grid
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;

    // Figure out peer channels for comms
    int partnerId = (partnerRank < localRank) ? partnerRank : partnerRank - 1;

    DeviceHandle<mscclpp::PortChannel>& outgoingChannel = (
        outgoingBufferIsA ? constChannelsAtoBs[partnerId] : constChannelsBtoAs[partnerId]
    );
    DeviceHandle<mscclpp::PortChannel>& incomingChannel = (
        outgoingBufferIsA ? constChannelsAtoBs[partnerId] : constChannelsBtoAs[partnerId]
    );

    if (globalThreadId == 0) {
        // Send data to the partner node, without waiting for it to be received
        printf("Rank %d: sending data to rank %d\n", localRank, partnerRank);
        outgoingChannel.put(0, bytesToSend);
        outgoingChannel.signal();
        outgoingChannel.flush();
        // QUESTION: should I flush here?
        // Wait for the partner data to be received
        incomingChannel.wait();
    }
    deviceSyncer.sync(gridDim.x, -1);
    __syncthreads();

    // Dequantize, accumulate, and quantize the data
    warpwiseInplaceDQAQ(incomingBuffer, outgoingBuffer, incomingQParams, outgoingQParams);

    // Wait for the partner node to have received the data
    // if (globalThreadId == 0) {
    //     printf("Rank %d: waiting for signal flush to rank %d\n", localRank, partnerRank);
        // outgoingChannel.flush();
    //     printf("Rank %d: flushing done\n", localRank);
    // }
    deviceSyncer.sync(gridDim.x, -1);
    __syncthreads();
}


__global__ void encodedCrossReduce(
    uint8_t* __restrict__ commBufferA,
    uint8_t* __restrict__ commBufferB,
    half2* __restrict__ inputOutputBuffer, // the input and output buffer
    const int localRank,
    const size_t numElements
) {
    assert(numElements % ELEMS_PER_BLOCK == 0);
    const int numScales = numElements / (WARPSIZE * ELEMS_PER_THREAD);
    const int bytesToSend = numElements + numScales * sizeof(float2);
    const int scalesOffset = numElements / 8; // elements are fp8 (1b) and qparams are float2 (8b)

    // Relocate thread in tensors
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalWarpId = globalThreadId / WARPSIZE;

    inputOutputBuffer += globalThreadId * PK_ELEMS_PER_THREAD;
    fp8x2* aQuantized = reinterpret_cast<fp8x2*>(commBufferA) + globalThreadId * PK_ELEMS_PER_THREAD;
    float2* aQParams = reinterpret_cast<float2*>(commBufferA) + scalesOffset + globalWarpId;
    fp8x2* bQuantized = reinterpret_cast<fp8x2*>(commBufferB) + globalThreadId * PK_ELEMS_PER_THREAD;
    float2* bQParams = reinterpret_cast<float2*>(commBufferB) + scalesOffset + globalWarpId;

    // Before communication, quantize the input buffer to commBufferA
    warpwiseQuantize(inputOutputBuffer, aQuantized, aQParams);
    // if (globalThreadId == 0) {
    //     printf("Rank %d, before 1st round: aQuantized[0]: %f\n", localRank, dequantizeFp8x2(aQuantized[0]).x);
    // }

    // 1st round
    // Comms: 0-1, 2-3, 4-5, 6-7
    // Before: 0  1  2  3  4  5  6  7
    // After:  01 01 23 23 45 45 67 67
    int partnerRank = (localRank % 2 >= 1) ? localRank - 1 : localRank + 1;
    crossReduceOneRound(aQuantized, aQParams, bQuantized, bQParams, true, localRank, partnerRank, bytesToSend);
    // if (globalThreadId == 0) {
    //     printf("Rank %d, after 1st round: aQuantized[0]: %f\n", localRank, dequantizeFp8x2(aQuantized[0]).x);
    // }

    // After 1st round, commBufferB contains the information for the next round

    // 2nd round
    // Comms: 0-2, 1-3, 4-6, 5-7
    // Before: 01   01   23   23   45   45   67   67
    // After:  0123 0123 0123 0123 4567 4567 4567 4567
    partnerRank = (localRank % 4 >= 2) ? localRank - 2 : localRank + 2;
    crossReduceOneRound(bQuantized, bQParams, aQuantized, aQParams, false, localRank, partnerRank, bytesToSend);
    // if (globalThreadId == 0) {
    //     printf("Rank %d, after 2nd round: aQuantized[0]: %f\n", localRank, dequantizeFp8x2(aQuantized[0]).x);
    // }

    // After 2nd round, commBufferA contains the information for the next round

    // 3rd and final round
    // Comms: 0-4, 1-5, 2-6, 3-7
    // Before: 0123     0123     0123     0123     4567     4567     4567     4567
    // After:  01234567 01234567 01234567 01234567 01234567 01234567 01234567 01234567
    partnerRank = (localRank + WORLD_SIZE / 2) % WORLD_SIZE;
    crossReduceOneRound(aQuantized, aQParams, bQuantized, bQParams, true, localRank, partnerRank, bytesToSend);
    // if (globalThreadId == 0) {
    //     printf("Rank %d, after 3rd round: aQuantized[0]: %f\n", localRank, dequantizeFp8x2(aQuantized[0]).x);
    // }

    // After 3rd round, commBufferB contains the final result

    // Dequantize the final result to the output buffer
    warpwiseDequantize(bQuantized, bQParams, inputOutputBuffer);

    // Debug: print the current scales (can be switched between A and B)
    // if (globalThreadId == 0) {
    //     float* scales = reinterpret_cast<float*>(commBufferB) + scalesOffset;
    //     printf("Rank %d, scales: ", localRank);
    //     for (int i = 0; i < numScales; i++) {
    //         printf("%f ", scales[i]);
    //     }
    //     printf("\n");
    // }
}
