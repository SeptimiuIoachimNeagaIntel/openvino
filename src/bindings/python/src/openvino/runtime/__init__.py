# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

import warnings
warnings.filterwarnings("once", category=DeprecationWarning, module="openvino.runtime")
warnings.warn(
    "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. "
    "Please replace `openvino.runtime` with `openvino`.",
    DeprecationWarning,
    stacklevel=1
)


from openvino._pyopenvino import get_version

__version__ = get_version()

# Openvino pybind bindings and python extended classes
from openvino._pyopenvino import Symbol
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import Input
from openvino._pyopenvino import Output
from openvino._pyopenvino import Node
from openvino._pyopenvino import Type
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Strides
from openvino._pyopenvino import CoordinateDiff
from openvino._pyopenvino import DiscreteTypeInfo
from openvino._pyopenvino import AxisSet
from openvino._pyopenvino import AxisVector
from openvino._pyopenvino import Coordinate
from openvino._pyopenvino import Layout
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import layout_helpers
from openvino._pyopenvino import OVAny
from openvino._pyopenvino import RTMap
from openvino.runtime.ie_api import Core
from openvino.runtime.ie_api import CompiledModel
from openvino.runtime.ie_api import InferRequest
from openvino.runtime.ie_api import Model
from openvino.runtime.ie_api import AsyncInferQueue
from openvino._pyopenvino import Version
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Extension
from openvino._pyopenvino import ProfilingInfo
from openvino._pyopenvino import get_batch
from openvino._pyopenvino import set_batch
from openvino._pyopenvino import serialize
from openvino._pyopenvino import save_model
from openvino._pyopenvino import shutdown

# Import opsets
from openvino.runtime import op
from openvino.runtime import opset1
from openvino.runtime import opset2
from openvino.runtime import opset3
from openvino.runtime import opset4
from openvino.runtime import opset5
from openvino.runtime import opset6
from openvino.runtime import opset7
from openvino.runtime import opset8
from openvino.runtime import opset9
from openvino.runtime import opset10
from openvino.runtime import opset11
from openvino.runtime import opset12
from openvino.runtime import opset13
from openvino.runtime import opset14
from openvino.runtime import opset15
from openvino.runtime import opset16

# Import runtime proxy modules for backward compatibility
from openvino.runtime import utils
from openvino.runtime import opset_utils
from openvino.runtime import exceptions

# Import properties API
from openvino.runtime import properties

# Helper functions for openvino module
from openvino.runtime.ie_api import tensor_from_file
from openvino.runtime.ie_api import compile_model
