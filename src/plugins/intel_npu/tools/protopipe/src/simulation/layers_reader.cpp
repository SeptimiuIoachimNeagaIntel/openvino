//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/layers_reader.hpp"
#include "scenario/inference.hpp"
#include "utils/error.hpp"
#include "utils/logger.hpp"

OpenVINOLayersReader& getOVReader() {
    static OpenVINOLayersReader reader;
    return reader;
}

static std::string getModelFileName(const InferenceParams& params) {
    if (std::holds_alternative<OpenVINOParams>(params)) {
        const auto& ov_params = std::get<OpenVINOParams>(params);
        if (std::holds_alternative<OpenVINOParams::ModelPath>(ov_params.path)) {
            return std::get<OpenVINOParams::ModelPath>(ov_params.path).model;
        } else {
            ASSERT(std::holds_alternative<OpenVINOParams::BlobPath>(ov_params.path));
            return std::get<OpenVINOParams::BlobPath>(ov_params.path).blob;
        }
    } else if (std::holds_alternative<ONNXRTParams>(params)) {
        return std::get<ONNXRTParams>(params).model_path;
    } else {
        THROW_ERROR("Unsupported model parameters type!");
    }
    // NB: Unreachable
    ASSERT(false);
}

static void adjustDynamicDims(LayersInfo& layers) {
    auto dims_to_string = [](const std::vector<int>& dims) {
        std::stringstream ss;
        std::copy(dims.begin(), dims.end(), std::ostream_iterator<int>(ss, " "));
        return ss.str();
    };
    for (auto& layer : layers) {
        auto& dims = layer.dims;
        if (std::find(dims.begin(), dims.end(), -1) == dims.end()) continue;
        std::stringstream info_msg;
        info_msg
            << "Dynamic shape: [" << dims_to_string(dims) << "] has been detected for layer "
            << layer.name
            << ". Data with shape [";
        std::for_each(dims.begin(), dims.end(), [](int &v) { v = v == -1 ? -v : v; });
        info_msg 
            << dims_to_string(dims)
            << "] will be provided as input for this layer.";
        LOG_INFO() << info_msg.str() << std::endl;
    }
}

InOutLayers LayersReader::readLayers(const InferenceParams& params) {
    LOG_INFO() << "Reading model " << getModelFileName(params) << std::endl;
    if (std::holds_alternative<OpenVINOParams>(params)) {
        const auto& ov = std::get<OpenVINOParams>(params);
        return getOVReader().readLayers(ov);
    }
    ASSERT(std::holds_alternative<ONNXRTParams>(params));
    const auto& ort = std::get<ONNXRTParams>(params);
    // NB: Using OpenVINO to read the i/o layers information for *.onnx model
    OpenVINOParams ov;
    ov.path = OpenVINOParams::ModelPath{ort.model_path, ""};
    auto inOutLayers = getOVReader().readLayers(ov, true /* use_results_names */);
    adjustDynamicDims(inOutLayers.in_layers);
    adjustDynamicDims(inOutLayers.out_layers);

    return inOutLayers;
}
