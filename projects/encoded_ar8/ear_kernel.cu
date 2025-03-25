#include "./ear_qdq.cu"
#include <mscclpp/concurrency_device.hpp>


// Setup node's mesh connections
template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constChannelsAtoBs[7];
__constant__ DeviceHandle<mscclpp::PortChannel> constChannelsBtoAs[7];

// DeviceSyncer for the current node
__device__ mscclpp::DeviceSyncer deviceSyncer;


__device__ void crossReduceOneRound(
    const uint8_t* __restrict__ outgoingBuffer,
    uint8_t* __restrict__ incomingBuffer,
    bool outgoingBufferIsA,
    const int localRank,   // Rank of the current node, 0 <= localRank < worldSize
    const int partnerRank, // Rank of the partner node, 0 <= partnerRank < worldSize
    const size_t numElements,
    const size_t bytesToSend
) {
    // Figure out thread's position in the grid
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int totalThreadCount = gridDim.x * blockDim.x;

    // Figure out peer channels for comms
    int partnerId = (partnerRank < localRank) ? partnerRank : partnerRank - 1;

    DeviceHandle<mscclpp::PortChannel>& outgoingChannel;
    DeviceHandle<mscclpp::PortChannel>& incomingChannel;
    if (outgoingBufferIsA) {
        outgoingChannel = constChannelsAtoBs[partnerId];
        incomingChannel = constChannelsAtoBs[partnerId];
    } else {
        outgoingChannel = constChannelsBtoAs[partnerId];
        incomingChannel = constChannelsBtoAs[partnerId];
    }

    if (globalThreadId == 0) {
        // Send data to the partner node, without waiting for it to be received
        printf("Rank %d: sending data to rank %d\n", localRank, partnerRank);
        outgoingChannel.put(0, bytesToSend);
        outgoingChannel.signal();
        // QUESTION: should I flush here?
        // Wait for the partner data to be received
        incomingChannel.wait();
    }
    deviceSyncer.sync(gridDim.x, -1);

    // Parse buffer into quantized data and scales: x is modified in place while y is constant
    fp8x2* xQuantized = reinterpret_cast<fp8x2*>(incomingBuffer);
    float* xScales = reinterpret_cast<float*>(incomingBuffer + numElements);
    const fp8x2* yQuantized = reinterpret_cast<const fp8x2*>(outgoingBuffer);
    const float* yScales = reinterpret_cast<const float*>(outgoingBuffer + numElements);

    // Dequantize, accumulate, and quantize the data
    warpwiseInplaceDQAQ(xQuantized, yQuantized, xScales, yScales);

    // Wait for the partner node to have received the data
    if (globalThreadId == 0) {
        printf("Rank %d: waiting for signal flush to rank %d\n", localRank, partnerRank);
        outgoingChannel.flush();
        printf("Rank %d: flushing done\n", localRank);
    }
    deviceSyncer.sync(gridDim.x, -1);
}


__device__ void encodedCrossReduce(
    uint8_t* __restrict__ commBufferA,
    uint8_t* __restrict__ commBufferB,
    half2* __restrict__ inputOutputBuffer, // the input and output buffer
    const int localRank,
    const size_t numElements
) {
    assert(numElements % ELEMS_PER_BLOCK == 0);
    const int numScales = numElements / (WARPSIZE * ELEMS_PER_THREAD);
    const int bytesToSend = numElements + numScales * sizeof(float);

    // Before communication, quantize the input buffer to commBufferA
    warpwiseQuantize(
        inputOutputBuffer,
        reinterpret_cast<fp8x2*>(commBufferA),
        reinterpret_cast<float*>(commBufferA + numElements)
    );

    // 1st round
    // Comms: 0-1, 2-3, 4-5, 6-7
    // Before: 0  1  2  3  4  5  6  7
    // After:  01 01 23 23 45 45 67 67
    int partnerRank = (localRank % 2 >= 1) ? localRank - 1 : localRank + 1;
    crossReduceOneRound(commBufferA, commBufferB, true, localRank, partnerRank, numElements, bytesToSend);

    // After 1st round, commBufferB contains the information for the next round

    // 2nd round
    // Comms: 0-2, 1-3, 4-6, 5-7
    // Before: 01   01   23   23   45   45   67   67
    // After:  0123 0123 0123 0123 4567 4567 4567 4567
    partnerRank = (localRank % 4 >= 2) ? localRank - 2 : localRank + 2;
    crossReduceOneRound(commBufferB, commBufferA, false, localRank, partnerRank, numElements, bytesToSend);

    // After 2nd round, commBufferA contains the information for the next round

    // 3rd and final round
    // Comms: 0-4, 1-5, 2-6, 3-7
    // Before: 0123     0123     0123     0123     4567     4567     4567     4567
    // After:  01234567 01234567 01234567 01234567 01234567 01234567 01234567 01234567
    partnerRank = (localRank + WORLD_SIZE / 2) % WORLD_SIZE;
    crossReduceOneRound(commBufferA, commBufferB, true, localRank, partnerRank, numElements, bytesToSend);

    // After 3rd round, commBufferB contains the final result

    // Dequantize the final result to the output buffer
    warpwiseDequantize(
        reinterpret_cast<const fp8x2*>(commBufferB),
        reinterpret_cast<const float*>(commBufferB + numElements),
        inputOutputBuffer
    );
}
