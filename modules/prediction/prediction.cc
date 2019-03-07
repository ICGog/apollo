/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/prediction/prediction.h"

#include <cmath>

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/time/time.h"
#include "modules/common/util/file.h"
#include "modules/prediction/common/prediction_gflags.h"
#include "modules/prediction/container/container_manager.h"
#include "modules/prediction/container/obstacles/obstacles_container.h"
#include "modules/prediction/container/pose/pose_container.h"
#include "modules/prediction/evaluator/evaluator_manager.h"
#include "modules/prediction/predictor/predictor_manager.h"
#include "modules/prediction/proto/prediction_obstacle.pb.h"

namespace apollo {
namespace prediction {

using ::apollo::common::ErrorCode;
using ::apollo::common::Status;
using ::apollo::common::TrajectoryPoint;
using ::apollo::common::adapter::AdapterConfig;
using ::apollo::common::adapter::AdapterManager;
using ::apollo::common::time::Clock;
using ::apollo::localization::LocalizationEstimate;
using ::apollo::perception::PerceptionObstacle;
using ::apollo::perception::PerceptionObstacles;

std::string Prediction::Name() const {
  return FLAGS_prediction_module_name;
}

Status Prediction::Init() {
  // Load prediction conf
  prediction_conf_.Clear();
  if (!common::util::GetProtoFromFile(FLAGS_prediction_conf_file,
                                      &prediction_conf_)) {
    return OnError("Unable to load prediction conf file: " +
                   FLAGS_prediction_conf_file);
  } else {
    ADEBUG << "Prediction config file is loaded into: "
           << prediction_conf_.ShortDebugString();
  }

  adapter_conf_.Clear();
  if (!common::util::GetProtoFromFile(FLAGS_prediction_adapter_config_filename,
                                      &adapter_conf_)) {
    return OnError("Unable to load adapter conf file: " +
                   FLAGS_prediction_adapter_config_filename);
  } else {
    ADEBUG << "Adapter config file is loaded into: "
           << adapter_conf_.ShortDebugString();
  }

  last_localization_seq_num_ = 0;
  last_perception_seq_num_ = 0;
  // Initialization of all managers
  AdapterManager::Init(adapter_conf_);
  ContainerManager::instance()->Init(adapter_conf_);
  EvaluatorManager::instance()->Init(prediction_conf_);
  PredictorManager::instance()->Init(prediction_conf_);

  CHECK(AdapterManager::GetLocalization()) << "Localization is not ready.";
  CHECK(AdapterManager::GetPerceptionObstacles()) << "Perception is not ready.";

  // Set perception obstacle callback function
  AdapterManager::AddPerceptionObstaclesCallback(&Prediction::RunOnce, this);
  // Set localization callback function
  AdapterManager::AddLocalizationCallback(&Prediction::OnLocalization, this);
  // Set planning callback function
  AdapterManager::AddPlanningCallback(&Prediction::OnPlanning, this);

  return Status::OK();
}

Status Prediction::Start() {
  return Status::OK();
}

void Prediction::Stop() {}

void Prediction::OnLocalization(const LocalizationEstimate& localization) {
  uint32_t seq_num = localization.header().sequence_num();
  if (last_localization_seq_num_ != 0 && seq_num - 1 != last_localization_seq_num_) {
    AINFO << "Last sequence num: " << last_localization_seq_num_ << " sequence num: " << seq_num;
  }
  last_localization_seq_num_ = seq_num;

  ObstaclesContainer* obstacles_container = dynamic_cast<ObstaclesContainer*>(
      ContainerManager::instance()->GetContainer(
          AdapterConfig::PERCEPTION_OBSTACLES));
  CHECK_NOTNULL(obstacles_container);

  PoseContainer* pose_container = dynamic_cast<PoseContainer*>(
      ContainerManager::instance()->GetContainer(AdapterConfig::LOCALIZATION));
  CHECK_NOTNULL(pose_container);

  pose_container->Insert(localization);
  PerceptionObstacle* pose_ptr = pose_container->ToPerceptionObstacle();
  if (pose_ptr != nullptr) {
    obstacles_container->InsertPerceptionObstacle(
        *(pose_ptr), pose_container->GetTimestamp());
  } else {
    ADEBUG << "Invalid pose found.";
  }

  ADEBUG << "Received a localization message ["
         << localization.ShortDebugString() << "].";
}

void Prediction::OnPlanning(const planning::ADCTrajectory& adc_trajectory) {
  ADCTrajectoryContainer* adc_trajectory_container =
      dynamic_cast<ADCTrajectoryContainer*>(
          ContainerManager::instance()->GetContainer(
              AdapterConfig::PLANNING_TRAJECTORY));
  CHECK_NOTNULL(adc_trajectory_container);
  adc_trajectory_container->Insert(adc_trajectory);
}

void Prediction::RunOnce(const PerceptionObstacles& perception_obstacles) {
  ADEBUG << "Received a perception message ["
         << perception_obstacles.ShortDebugString() << "].";
  uint32_t seq_num = perception_obstacles.header().sequence_num();
  if (last_perception_seq_num_ != 0 && seq_num - 1 != last_perception_seq_num_) {
    AINFO << "Last sequence num: " << last_perception_seq_num_ << " sequence num: " << seq_num;
  }
  last_perception_seq_num_ = seq_num;

  double start_timestamp = Clock::NowInSeconds();

  ObstaclesContainer* obstacles_container = dynamic_cast<ObstaclesContainer*>(
      ContainerManager::instance()->GetContainer(
          AdapterConfig::PERCEPTION_OBSTACLES));
  // TODO(ionel): Check why it takes a bit of time.
  CHECK_NOTNULL(obstacles_container);
  obstacles_container->Insert(perception_obstacles);
  double evaluation_start_timestamp = Clock::NowInSeconds();
  EvaluatorManager::instance()->Run(perception_obstacles);
  double predictor_start_timestamp = Clock::NowInSeconds();
  // TODO(ionel): Check why it takes a bit of time.
  PredictorManager::instance()->Run(perception_obstacles);
  double predictor_end_timestamp = Clock::NowInSeconds();

  auto prediction_obstacles =
      PredictorManager::instance()->prediction_obstacles();
  prediction_obstacles.set_start_timestamp(start_timestamp);
  prediction_obstacles.set_end_timestamp(Clock::NowInSeconds());

  for (auto const& prediction_obstacle :
       prediction_obstacles.prediction_obstacle()) {
    for (auto const& trajectory : prediction_obstacle.trajectory()) {
      for (auto const& trajectory_point : trajectory.trajectory_point()) {
        if (!IsValidTrajectoryPoint(trajectory_point)) {
          AERROR << "Invalid trajectory point ["
                 << trajectory_point.ShortDebugString() << "]";
          return;
        }
      }
    }
  }

  double prediction_end_time = Clock::NowInSeconds();
  double prediction_runtime = (prediction_end_time - start_timestamp) * 1000.0;

  AINFO << "Pre evaluator runtime (ms): " << std::fixed << std::setprecision(9)
	<< (evaluation_start_timestamp - start_timestamp) * 1000.0;
  AINFO << "Evaluator runtime (ms): " << std::fixed << std::setprecision(9)
	<< (predictor_start_timestamp - evaluation_start_timestamp) * 1000.0;
  AINFO << "Predictor runtime (ms): " << std::fixed << std::setprecision(9)
	<< (predictor_end_timestamp - predictor_start_timestamp) * 1000.0;
  AINFO << "Post predictor runtime (ms): " << std::fixed << std::setprecision(9)
	<< (prediction_end_time - predictor_end_timestamp) * 1000.0;
  AINFO << "Prediction runtime (ms): " << prediction_runtime
        << " for obstacles sent at (sec): "
	<< std::fixed << std::setprecision(9) << perception_obstacles.header().timestamp_sec();

  Publish(&prediction_obstacles);

  ADEBUG << "Received a perception message ["
         << perception_obstacles.ShortDebugString() << "].";
  ADEBUG << "Published a prediction message ["
         << prediction_obstacles.ShortDebugString() << "].";
}

Status Prediction::OnError(const std::string& error_msg) {
  return Status(ErrorCode::PREDICTION_ERROR, error_msg);
}

bool Prediction::IsValidTrajectoryPoint(
    const TrajectoryPoint& trajectory_point) {
  return trajectory_point.has_path_point() &&
         (!std::isnan(trajectory_point.path_point().x())) &&
         (!std::isnan(trajectory_point.path_point().y())) &&
         (!std::isnan(trajectory_point.path_point().theta())) &&
         (!std::isnan(trajectory_point.v())) &&
         (!std::isnan(trajectory_point.a())) &&
         (!std::isnan(trajectory_point.relative_time()));
}

}  // namespace prediction
}  // namespace apollo
