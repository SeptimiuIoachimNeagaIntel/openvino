// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class PriorBoxClustered : public Node {
public:
    PriorBoxClustered(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;

    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    std::vector<float> widths;
    std::vector<float> heights;
    std::vector<float> variances;
    bool clip;
    float step;
    float step_heights;
    float step_widths;
    float offset;

    int number_of_priors;
};

}  // namespace ov::intel_cpu::node
