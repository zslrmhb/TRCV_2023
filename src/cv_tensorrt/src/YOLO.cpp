//
// Created by hongbin on 3/6/23.
//

#include "../include/auto_aim/YOLO.hpp"

YOLO::YOLO()
{

};

bool YOLO::parse_args(int argc, char **argv,  std::string &engine) {
    if (argc != 2) return false;
    else engine = std::string(argv[1]);
    return true;
}

void YOLO::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void YOLO::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void YOLO::serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

void YOLO::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void YOLO::declearAndLoadParameter() {
    nh_->declare_parameter("DEBUG", rclcpp::ParameterValue(true));
    nh_->declare_parameter("yaw_offset");
    nh_->declare_parameter("pitch_offset");
    nh_->declare_parameter("small_armor.width");
    nh_->declare_parameter("small_armor.height");
    nh_->declare_parameter("big_armor.width");
    nh_->declare_parameter("big_armor.height");

    debug_ = nh_->get_parameter("DEBUG").as_bool();
    OFFSET_YAW = nh_->get_parameter("yaw_offset").as_int();
    OFFSET_PITCH = nh_->get_parameter("pitch_offset").as_int();
}

void YOLO::load_armor_data()
{
    float x, y, z = 0;

    double small_width = 140.0;
    double small_height = 60.0;
    double big_width = 230.0;
    double big_height = 60.0;
//    double small_width = nh_->get_parameter("small_armor.width").as_double(),
//            small_height = nh_->get_parameter("small_armor.height").as_double(),
//            big_width = nh_->get_parameter("big_armor.width").as_double(),
//            big_height = nh_->get_parameter("big_armor.height").as_double();
    if (debug_)
    {
        RCLCPP_INFO(nh_->get_logger(), "small marmor:\n width: %f, height: %f\n big armor:\n width: %f, height: %f\n ",
                    small_width, small_height, big_width, big_height);
    }
    x = -small_width / 2;
    y = small_height / 2;
    small_real_armor_points.emplace_back(x, y, z);
    x = small_width / 2;
    y = small_height / 2;
    small_real_armor_points.emplace_back(x, y, z);
    x = small_width / 2;
    y = -small_height / 2;
    small_real_armor_points.emplace_back(x, y, z);
    x = -small_width / 2;
    y = -small_height / 2;
    small_real_armor_points.emplace_back(x, y, z);

    //**********************************************************************//
    x = -big_width / 2;
    y = big_height / 2;
    big_real_armor_points.emplace_back(x, y, z);
    x = big_width / 2;
    y = big_height / 2;

    big_real_armor_points.emplace_back(x, y, z);
    x = big_width / 2;
    y = -big_height / 2;

    big_real_armor_points.emplace_back(x, y, z);
    x = -big_width / 2;
    y = -big_height / 2;

    big_real_armor_points.emplace_back(x, y, z);

}

cv::Point3f YOLO::getPose()
{
    cv::Mat rvec;
    cv::Mat tvec;

    cv::solvePnP(big_real_armor_points, final_armor_2Dpoints, cameraMatrix, distCoeffs, rvec, tvec, false,
             cv::SOLVEPNP_ITERATIVE);

    return cv::Point3f(tvec);

}


void YOLO::publishData(int16_t pitch, int16_t yaw)
{
    int32_t message = ((int32_t)pitch);
    message = message << 16;
    message = message | (yaw & 0xffff);
    std_msgs::msg::Int32 gimbel_message;
    gimbel_message.data = message;

    if (debug_)
    {
        RCLCPP_INFO(nh_->get_logger(), "%x %d\n", message, message);
    }
    publisher_->publish(std::move(gimbel_message));
}

YOLO::~YOLO()
{

};
