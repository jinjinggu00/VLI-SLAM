#include "g2o_optimization/g2o_optimization.h"

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "read_configs.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_line3d.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/edge_imu.h"
#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/edge_project_line.h"
#include "g2o_optimization/edge_relative_pose.h"


#include <g2o/solvers/csparse/linear_solver_csparse.h>
void GlobalBA131(MapPtr _map, const OptimizationConfig& cfg, bool point_outlier_rejection, 
    bool line_outlier_rejection, int first_iterations, int second_iterations){
  std::map<int, MappointPtr>& mappoints = _map->GetAllMappoints();
  std::map<int, MaplinePtr>& maplines = _map->GetAllMaplines();
  std::map<int, FramePtr>& keyframes = _map->GetAllKeyframes();
  CameraPtr camera = _map->GetCameraPtr();

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  int max_frame_id = 0;
  Eigen::Matrix4d Tcb = camera->BodyToCamera();
  Eigen::Matrix3d Rcb = Tcb.block<3, 3>(0, 0);
  Eigen::Vector3d tcb = Tcb.block<3, 1>(0, 3);
  Eigen::Matrix3d Rwg = _map->GetRwg();
  for(auto& kv : keyframes){
    bool fix_this_frame = (kv.first == keyframes.begin()->first);

    Eigen::Matrix4d Twc = kv.second->GetPose();
    Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
    Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);

    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;
    VIPose vi_pose(Rcw, tcw, Rcb, tcb);
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(fix_this_frame);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  } 
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(mpt->GetPosition());
    int point_id = kv.first + max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(false);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 4. line vertex
  int max_line_id = max_point_id;
  for(auto& kv : maplines){
    MaplinePtr mpl = kv.second;
    if(!mpl || !mpl->IsValid()) continue;

    g2o::VertexLine3D* line_vertex = new g2o::VertexLine3D();
    line_vertex->setEstimateData(mpl->GetLine3D());
    int line_id = kv.first + max_point_id;
    max_line_id = std::max(max_line_id, line_id);
    line_vertex->setId(line_id);
    line_vertex->setMarginalized(true);
    line_vertex->setFixed(false);
    optimizer.addVertex(line_vertex);
  }
  max_line_id++;

  int max_bias_id = -1;
  int gravity_direction_id = -1;
  if(_map->IMUInit()){
    max_bias_id = max_line_id;
    for(auto& kv : keyframes){
      bool fix_this_frame = (kv.first == keyframes.begin()->first);
      FramePtr frame = kv.second;

      // 5. velocity vertex
      assert(frame->VelocityIsInitialized());
      VertexVelocity* velocity_vertex = new VertexVelocity(frame->GetVelocity());
      int velocity_id = kv.first * 3 + max_line_id;
      velocity_vertex->setId(velocity_id);
      velocity_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(velocity_vertex);

      // 6. bias vertex
      Eigen::Vector3d gyr_bias, acc_bias;
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(fix_this_frame){
        gyr_bias = preinteration->bg + preinteration->dbg;
        acc_bias = preinteration->ba + preinteration->dba;
      }else{
        gyr_bias = preinteration->bg;
        acc_bias = preinteration->ba;
      }
      VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(gyr_bias);
      int gyr_bias_id = kv.first * 3 + 1 + max_line_id;
      gyr_bias_vertex->setId(gyr_bias_id);
      gyr_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(gyr_bias_vertex);

      VertexAccBias* acc_bias_vertex = new VertexAccBias(acc_bias);
      int acc_bias_id = kv.first * 3 + 2 + max_line_id;
      max_bias_id = std::max(max_bias_id, acc_bias_id);
      acc_bias_vertex->setId(acc_bias_id);
      acc_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(acc_bias_vertex);
    }

    // 7. gravity direction vertex, will be fixed in this function.
    gravity_direction_id = max_bias_id + 1;
    VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
    gravity_direction_vertex->setId(gravity_direction_id);
    gravity_direction_vertex->setFixed(true);
    optimizer.addVertex(gravity_direction_vertex);
  }

  // edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges; 
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  std::vector<EdgeSE3ProjectLine*> mono_line_edges; 
  std::vector<EdgeStereoSE3ProjectLine*> stereo_line_edges;
  const double thHuberMonoLine = sqrt(cfg.mono_line);
  const double thHuberStereoLine = sqrt(cfg.stereo_line);

  std::vector<EdgeIMU*> imu_edges; 
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  double fx = camera->Fx();
  double fy = camera->Fy();
  double cx = camera->Cx();
  double cy = camera->Cy();
  double bf = camera->BF();
  Eigen::Vector3d Kv;
  Kv << -fy * cx, -fx * cy, fx * fy;

  FramePtr last_frame = std::shared_ptr<Frame>(nullptr);
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;
    
    int frame_vertex_id = frame_id;

    // 8. point edges
    std::vector<MappointPtr>& frame_mpts = frame->GetAllMappoints();
    for(size_t i = 0; i < frame_mpts.size(); i++){
      MappointPtr mpt = frame_mpts[i];
      if(!mpt || !mpt->IsValid()) continue;

      Eigen::Vector3d keypoint;
      if(!frame->GetKeypointPosition(i, keypoint)) continue;
  
      int mpt_vertex_id = max_frame_id + mpt->GetId();
      if(keypoint(2) < 0){
        EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint.head<2>());
        e->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMonoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        optimizer.addEdge(e);
        mono_edges.push_back(e);
      }else{
        EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint);
        e->setInformation(Eigen::Matrix3d::Identity());
        g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->bf = bf;
        optimizer.addEdge(e);
        stereo_edges.push_back(e);
      }
    }

    // 9. line edges
    std::vector<MaplinePtr>& frame_mpls = frame->GetAllMaplines();
    for(size_t i = 0; i < frame_mpls.size(); i++){
      MaplinePtr mpl = frame_mpls[i];
      if(!mpl || !mpl->IsValid()) continue;

      Eigen::Vector4d line_left, line_right;
      if(!frame->GetLine(i, line_left)) continue;

      double cov = mpl->ObverserNum() > 3 ? 0.1 : 0.001;
      int mpl_vertex_id = max_point_id + mpl->GetId();
      if(!frame->GetLineRight(i, line_right)){
        EdgeSE3ProjectLine* e = new EdgeSE3ProjectLine();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpl_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(line_left);
        e->setInformation(Eigen::Matrix2d::Identity() * cov);
        g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMonoLine);

        e->fx = fx;
        e->fy = fy;
        e->Kv = Kv; 
        optimizer.addEdge(e);
        mono_line_edges.push_back(e);
      }else{
        EdgeStereoSE3ProjectLine* e = new EdgeStereoSE3ProjectLine();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpl_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        Vector8d line_2d;
        line_2d << line_left, line_right;
        e->setMeasurement(line_2d);
        e->setInformation(Eigen::Matrix4d::Identity() * cov);
        g2o::RobustKernelCauchy* rk = new g2o::RobustKernelCauchy;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereoLine);

        e->fx = fx;
        e->fy = fy;
        e->b = bf / fx;
        e->Kv = Kv;
        optimizer.addEdge(e);
        stereo_line_edges.push_back(e);
      }
    }

    // 10. imu edges
    if(_map->IMUInit() && last_frame){
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(!preinteration->Valid()) continue;

      EdgeIMU* e_imu = new EdgeIMU(preinteration);

      int last_frame_vertex_id = last_frame->GetFrameId();
      g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(last_frame_vertex_id);
      g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(last_frame_vertex_id * 3 + max_line_id);
      g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(last_frame_vertex_id * 3 + 1 + max_line_id);
      g2o::HyperGraph::Vertex *va1 = optimizer.vertex(last_frame_vertex_id * 3 + 2 + max_line_id);
      g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(frame_vertex_id);
      g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(frame_vertex_id * 3 + max_line_id);
      g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(frame_vertex_id * 3 + 1 + max_line_id);
      g2o::HyperGraph::Vertex *va2 = optimizer.vertex(frame_vertex_id * 3 + 2 + max_line_id);
      g2o::HyperGraph::Vertex *vG = optimizer.vertex(gravity_direction_id);
      if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

      e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
      e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
      e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
      e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
      e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
      e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

      if(last_frame_vertex_id == keyframes.begin()->first){
        g2o::RobustKernelCauchy *rki = new g2o::RobustKernelCauchy;
        e_imu->setRobustKernel(rki);
        e_imu->setInformation(e_imu->information() * 1e-2);
        rki->setDelta(sqrt(16.92));
      }
      optimizer.addEdge(e_imu);
      imu_edges.push_back(e_imu);


      // bias edges
      EdgeGyr* e_gyr = new EdgeGyr();
      EdgeAcc* e_acc = new EdgeAcc();

      e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
      e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
      e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

      Eigen::Matrix3d info_g = preinteration->Cov.block<3,3>(9,9).inverse();
      Eigen::Matrix3d info_a = preinteration->Cov.block<3,3>(12,12).inverse();
      e_gyr->setInformation(info_g);
      e_acc->setInformation(info_a);

      optimizer.addEdge(e_gyr);
      optimizer.addEdge(e_acc);
      gyr_edges.push_back(e_gyr);
      acc_edges.push_back(e_acc);
    }

    last_frame = frame;
  }


  std::cout << "GBA131 GRAPH vertices=" << optimizer.vertices().size() << " edges=" << optimizer.edges().size() << " imu=" << imu_edges.size() << std::endl;
  optimizer.setVerbose(true);
  // solve 
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  std::cout << "GBA131 BEFORE chi2=" << optimizer.activeChi2() << " robust=" << optimizer.activeRobustChi2() << std::endl;
  optimizer.optimize(first_iterations);
  optimizer.computeActiveErrors();
  std::cout << "GBA131 FIRST chi2=" << optimizer.activeChi2() << " robust=" << optimizer.activeRobustChi2() << std::endl;

  if(point_outlier_rejection)
  {
    // check inlier observations
    for(size_t i=0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
        e->setLevel(1);
      }
      /* Retain original g2o Cauchy kernel after original hard outlier rejection. */ 
    }

    for(size_t i=0; i < stereo_edges.size(); i++){    
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
      if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
          e->setLevel(1);
      }
      /* Retain original g2o Cauchy kernel after original hard outlier rejection. */
    }
  }

  if(line_outlier_rejection){
    for(size_t i=0; i < mono_line_edges.size(); i++){
      EdgeSE3ProjectLine* e = mono_line_edges[i];
      if(e->chi2() > cfg.mono_line){
        e->setLevel(1);
      }
      /* Retain original g2o Cauchy kernel after original hard outlier rejection. */
    }

    for(size_t i=0; i < stereo_line_edges.size(); i++){    
      EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
      if(e->chi2() > cfg.stereo_line){
          e->setLevel(1);
      }
      /* Retain original g2o Cauchy kernel after original hard outlier rejection. */
    }
  }

  if(second_iterations > 0){
    // optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(second_iterations);

    // check outlier observations   
    std::vector<std::pair<FramePtr, MappointPtr>> mappoint_outliers;
    std::vector<std::pair<FramePtr, MaplinePtr>> mapline_outliers;  
    for(size_t i = 0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      e->computeError();
      if(e->chi2() <= cfg.mono_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    for(size_t i = 0; i < stereo_edges.size(); i++){    
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];

      e->computeError();
      if(e->chi2() <= cfg.stereo_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    for(size_t i = 0; i < mono_line_edges.size(); i++){
      EdgeSE3ProjectLine* e = mono_line_edges[i];
      e->computeError();
      if(e->chi2() <= cfg.mono_line) continue;

      int mpl_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpl_id = mpl_vertex_id - max_point_id;
      int frame_id = frame_vertex_id;
      mapline_outliers.emplace_back(keyframes[frame_id], maplines[mpl_id]);
    }

    for(size_t i = 0; i < stereo_line_edges.size(); i++){    
      EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];

      e->computeError();
      if(e->chi2() <= cfg.stereo_line) continue;

      int mpl_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpl_id = mpl_vertex_id - max_point_id;
      int frame_id = frame_vertex_id;
      mapline_outliers.emplace_back(keyframes[frame_id], maplines[mpl_id]);
    }

    std::cout << "mappoint_outliers = " << mappoint_outliers.size() << std::endl;
    std::cout << "mapline_outliers = " << mapline_outliers.size() << std::endl;
    _map->RemoveOutliers(mappoint_outliers);
    _map->RemoveLineOutliers(mapline_outliers);
  }

  optimizer.computeActiveErrors();
  std::cout << "GBA131 AFTER chi2=" << optimizer.activeChi2() << " robust=" << optimizer.activeRobustChi2() << std::endl;
  // recover optimized data
  // keyframes
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;

    // pose
    int frame_vertex_id = frame_id;
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(frame_vertex_id));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.block<3, 3>(0, 0) = SE3quat.rotation().toRotationMatrix();
    Twc.block<3, 1>(0, 3) = SE3quat.translation();
    frame->SetPose(Twc);

    if(_map->IMUInit()){
      // velocity
      int velocity_vertex_id = frame_vertex_id * 3 + max_line_id;
      VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(velocity_vertex_id));
      Eigen::Vector3d velocity = velocity_vertex->estimate();
      frame->SetVelocaity(velocity);

      // biases
      int gyr_vertex_id = frame_vertex_id * 3 + 1 + max_line_id;
      int acc_vertex_id = frame_vertex_id * 3 + 2 + max_line_id;
      VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(gyr_vertex_id));
      VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(acc_vertex_id));
      Eigen::Vector3d gyr_bias = gyr_bias_vertex->estimate();
      Eigen::Vector3d acc_bias = acc_bias_vertex->estimate();
      frame->SetBias(gyr_bias, acc_bias);
    }
  } 

  // 3. points 
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    int point_id = kv.first + max_frame_id;
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_id));
    if(point_vertex){
      Eigen::Vector3d pw = point_vertex->estimate();
      mpt->SetPosition(pw);
    }
  }

  // 4. line vertex
  for(auto& kv : maplines){
    MaplinePtr mpl = kv.second;
    if(!mpl || !mpl->IsValid()) continue;

    int line_id = kv.first+max_point_id;
    g2o::VertexLine3D* line_vertex = static_cast<g2o::VertexLine3D*>(optimizer.vertex(line_id));
    if(line_vertex){
      g2o::Line3D line_3d = line_vertex->estimate();
      mpl->SetLine3D(line_3d);
      mpl->SetEndpointsValidStatus(_map->UppdateMapline(mpl));
    }
  }

}