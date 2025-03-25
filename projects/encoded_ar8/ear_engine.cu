#include "./ear_kernel.cu"

class EarEngine {
public:
    EarEngine(int rank) : rank_(rank) {
        bootstrap();
    }

    ~EarEngine() {}

    torch::Tensor launchEncodedCrossReduce(torch::Tensor& x) {

        // Checks on the input tensor
        TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
        TORCH_CHECK(x.scalar_type() == torch::ScalarType::Half, "Input tensor must be float16 (Half) type");

        // Infer number of blocks needed
        const size_t numElements = x.numel();
        assert(numElements % ELEMS_PER_BLOCK == 0);

        const size_t numBlocks = numElements / ELEMS_PER_BLOCK;
        dim3 grid(numBlocks);
        dim3 block(THREADS_PER_BLOCK);

        // Allocate two buffers of the same size as the input tensor
        // TODO: reduce the size of this buffer, it's larger than needed
        allocateCommsBuffers(numElements);
        printf("R%d: Allocated input buffers.\n", rank_);

        // Setup mesh connections (A)
        setupMeshConnections(channelsAtoBs_, commBufferA_.get(), commBufferB_.get(), commBufferSize_);
        CUDATHROW(cudaMemcpyToSymbol(constChannelsAtoBs, channelsAtoBs_.data(),
            sizeof(DeviceHandle<mscclpp::PortChannel>) * channelsAtoBs_.size()));
        printf("R%d: Copied channels 'A to Bs' to device.\n", rank_);

        // Setup mesh connections (B)
        setupMeshConnections(channelsBtoAs_, commBufferB_.get(), commBufferA_.get(), commBufferSize_);
        CUDATHROW(cudaMemcpyToSymbol(constChannelsBtoAs, channelsBtoAs_.data(),
            sizeof(DeviceHandle<mscclpp::PortChannel>) * channelsBtoAs_.size()));
        printf("R%d: Copied channels 'B to As' to device.\n", rank_);

        // Start proxy
        startProxy();

        CUDATHROW(cudaDeviceSynchronize());
        encodedCrossReduce<<<grid, block>>>(
            commBufferA_.get(),
            commBufferB_.get(),
            (half2*)x.data_ptr(),
            rank_,
            numElements
        );
        CUDATHROW(cudaDeviceSynchronize());
        return x;
    }

private:
    void bootstrap() {
        // Use longer timeout for initialization
        printf("R%d: Bootstrapping...\n", rank_);

        // Create TCP bootstrap
        std::string ip_port = "localhost:12000";
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, WORLD_SIZE);

        // Initialize with options
        bootstrap->initialize("127.0.0.1:50000");
        bootstrap->barrier();
        printf("R%d: Bootstrapping done.\n", rank_);

        // Create communicator and wait for all processes
        communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
        chanService_ = std::make_shared<mscclpp::ProxyService>();
    }

    void allocateCommsBuffers(size_t numElements) {
        const int numScales = numElements / (WARPSIZE * ELEMS_PER_THREAD);
        const int bytesToSend = numElements + numScales * sizeof(float);
        commBufferA_ = mscclpp::GpuBuffer<uint8_t>(bytesToSend).memory();
        commBufferB_ = mscclpp::GpuBuffer<uint8_t>(bytesToSend).memory();
        commBufferSize_ = bytesToSend;
    }

    void setupMeshConnections(std::vector<DeviceHandle<mscclpp::PortChannel>>& portChannels, void* send_buff,
                              void* recv_buff, size_t buff_size) {

        printf("R%d: Setting up mesh connections...\n", rank_);
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;

        // Register memory
        mscclpp::RegisteredMemory recvBufRegMem = communicator_->registerMemory(recv_buff, buff_size, transport);
        mscclpp::RegisteredMemory sendBufRegMem = communicator_->registerMemory(send_buff, buff_size, transport);
        //mscclpp::RegisteredMemory bufRegMem = communicator_->registerMemory(buff, buff_size, transport);
        printf("R%d: Registered memory.\n", rank_);

        // Connect with all other ranks
        for (int r = 0; r < WORLD_SIZE; ++r) {
            if (r == rank_) continue;
            connectionFutures.push_back(communicator_->connectOnSetup(r, 0, transport));
            communicator_->sendMemoryOnSetup(recvBufRegMem, r, 0);
            remoteRegMemories.push_back(communicator_->recvMemoryOnSetup(r, 0));
        }
        printf("R%d: Connected with all other ranks.\n", rank_);

        // Setup communicator
        communicator_->setup();
        printf("R%d: Setup communicator.\n", rank_);

        // Get connections
        std::transform(
            connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
            [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
        printf("R%d: Got connections.\n", rank_);

        auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
        for (size_t i = 0; i < connections.size(); ++i) {
            portChannels.push_back(mscclpp::deviceHandle(
                service->portChannel(service->buildAndAddSemaphore(*communicator_, connections[i]),
                                     service->addMemory(remoteRegMemories[i].get()), service->addMemory(sendBufRegMem))));
        }
        printf("R%d: Created %zu channels.\n", rank_, portChannels.size());

        // Setup communicator
        communicator_->setup();
        printf("R%d: Setup mesh connections all done.\n", rank_);
    }

    void startProxy() {
        this->chanService_->startProxy();
        communicator_->bootstrap()->barrier();
        printf("Started proxy\n");
    }

    // Private members in order of definition:
    // ...at initialization
    int rank_;
    // ...in bootstrap
    std::shared_ptr<mscclpp::Communicator> communicator_;
    std::shared_ptr<mscclpp::BaseProxyService> chanService_;
    // ...in allocateCommsBuffers
    std::shared_ptr<uint8_t> commBufferA_;
    std::shared_ptr<uint8_t> commBufferB_;
    size_t commBufferSize_;
    // ...in setupMeshConnections
    std::vector<DeviceHandle<mscclpp::PortChannel>> channelsAtoBs_;
    std::vector<DeviceHandle<mscclpp::PortChannel>> channelsBtoAs_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<EarEngine>(m, "EarEngine")
        .def(py::init<int>())
        .def("launchEncodedCrossReduce", &EarEngine::launchEncodedCrossReduce);
}
