// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_furthest_point_sampling(const at::Tensor& point_xyz, const at::Tensor& nearset_temp, int32_t num_points)
{
    at::Tensor output = at::empty({static_cast<int64_t>(point_xyz.sizes()[0]), static_cast<int64_t>(num_points)},
        nearset_temp.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnFurthestPointSampling, point_xyz, nearset_temp, num_points, output);
    return output;
}
