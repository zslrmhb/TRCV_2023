#include "../include/auto_aim/YOLO.hpp"
#include "../include/auto_aim/Camera.h"

static const int COLOR = 1; // BLUE : 1, RED: 0

inline float euclid_distance(float x1, float y1, float x2, float y2){
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

inline cv::Point2f get_center(float width, float height) {return cv::Point2f(width / 2, height / 2); }

bool while_flag = true;
void signal_handle(int flag)
{
    while_flag = false;
}


int main(int argc, char** argv) {


    cudaSetDevice(kGpuId);
    std::string engine_name = "";

    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr nh_ = rclcpp::Node::make_shared("auto_aim");
    YOLO detector(nh_);
    if(!detector.parse_args(argc, argv, engine_name)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./exe enginename" << std::endl;
        return -1;
    }
    detector.load_armor_data();

    Camera camera;
    camera.init();

    

 // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    detector.deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    detector.prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    std::string window_name = "Yolov5 USB Camera";







    signal(SIGINT, signal_handle);

    while (while_flag) {


        auto start = std::chrono::system_clock::now();


        RCLCPP_INFO_ONCE(nh_->get_logger(), "Start to process frame...");
        cv::Mat img;
        std::vector<cv::Mat> img_batch;
        //cap >> img;
        camera.getImage(img);

//        }
        if (img.empty()) continue;

        img_batch.push_back(img);
        cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

        // Run inference
        detector.infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);


        // NMS
         std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

        Detection best_det = {{0.0f}, -1,-1,{0.0f} };
        float smallest_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < res_batch.size(); i++) {
            for (size_t j = 0; j < res_batch[i].size(); j++) {
                if (res_batch[i][j].class_id == COLOR) {
                    if (res_batch[i][j].conf > 0.75) {
                        cv::Point2f target_center = get_center(res_batch[i][j].bbox[0], res_batch[i][j].bbox[1]);
                        cv::Point2f frame_center = get_center(640, 640);
                        float dist = euclid_distance(target_center.x, target_center.y, frame_center.x, frame_center.y);
                        if (dist < smallest_dist) {
                            best_det = res_batch[i][j];
                            //std::cout << best_det.conf << std::endl;
                        }
                    }
                }
            }
        }

                    if (best_det.class_id != -1) {
//            OFFSET_YAW = (OFFSET_INT_YAW - 1800);
//            OFFSET_PITCH = (OFFSET_INT_PITCH - 1800);
                        // get 4 corner for solvepnp
                        detector.final_armor_2Dpoints.clear();
                        cv::Point2f topLeft = {best_det.bbox[0] - best_det.bbox[2] / 2,
                                               best_det.bbox[1] - best_det.bbox[3] / 2};
                        cv::Point2f topRight = {best_det.bbox[0] + best_det.bbox[2] / 2,
                                                best_det.bbox[1] - best_det.bbox[3] / 2};
                        cv::Point2f bottomLeft = {best_det.bbox[0] - best_det.bbox[2] / 2,
                                                  best_det.bbox[1] + best_det.bbox[3] / 2};
                        cv::Point2f bottomRight = {best_det.bbox[0] + best_det.bbox[2] / 2,
                                                   best_det.bbox[1] + best_det.bbox[3] / 2};
                        detector.final_armor_2Dpoints = {topLeft, topRight, bottomLeft, bottomRight};


                        cv::Point3f target_3d = {0, 0, 0};
                        target_3d = detector.getPose();
                        int pitch = atan2(target_3d.y, target_3d.z) * 180 / CV_PI;
                        int yaw = atan2(target_3d.x, target_3d.z) * 180 / CV_PI;
                        RCLCPP_INFO(nh_->get_logger(), "topLeft.x:%f topLeft.y:%f \n", topLeft.x, topLeft.y);
                        RCLCPP_INFO(nh_->get_logger(), "x:%f y:%f z:%f\n", target_3d.x, target_3d.y, target_3d.z);
                        RCLCPP_INFO(nh_->get_logger(), "pitch: %d, yaw: %d\n", pitch, yaw);
                        detector.publishData(pitch, yaw);
                    }

                    auto end = std::chrono::system_clock::now();
//                    std::cout << "inference time: "
//                              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
//                              << std::endl;
                    double t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    draw_bbox(img, best_det);

                    std::string label = cv::format("Inference time : % fps", 1 / (t / 1000));
                    cv::putText(img, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));
                    cv::imshow(window_name, img);


                    if (cv::waitKey(1) == 113) {
                        std::cout << "\nEsc key is pressed by user. Stopping the video\n" << std::endl;
                        rclcpp::shutdown();
                        break;
                    }
                }


                // Release stream and buffers
                cudaStreamDestroy(stream);
                CUDA_CHECK(cudaFree(gpu_buffers[0]));
                CUDA_CHECK(cudaFree(gpu_buffers[1]));
                delete[] cpu_output_buffer;
                cuda_preprocess_destroy();

                // Destroy the engine
                context->destroy();
                engine->destroy();
                runtime->destroy();
                std::cout << "Memory Freed";

                return 0;
        }