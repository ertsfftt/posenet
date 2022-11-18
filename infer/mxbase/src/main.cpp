#include "MxBase/Log/Log.h"
#include "Posenet.h"

namespace {
    const uint32_t DEVICE_ID = 0;
    const std::string RESULT_PATH = "./result";
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 3) {
        LogWarn << "Please input image path, such as './ [om_file_path] [img_path] [dataset_name]'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.checkTensor = true;
    initParam.modelPath = argv[1];
    auto inferPosenet = std::make_shared<Posenet>();
    APP_ERROR ret = inferPosenet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Posenet init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[2];
    std::string dataset_name = argv[3];
    ret = inferPosenet->Process(imgPath, RESULT_PATH, dataset_name);
    if (ret != APP_ERR_OK) {
        LogError << "Posenet process failed, ret=" << ret << ".";
        inferPosenet->DeInit();
        return ret;
    }
    inferPosenet->DeInit();
    return APP_ERR_OK;
}
