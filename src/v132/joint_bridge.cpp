// Input/output glue around upstream g2o VertexSE3, EdgeSE3 and LM.
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/edge_se3_pointxyz_depth.h>
#include <g2o/types/slam3d/parameter_camera.h>
#include <set>
#include <algorithm>
#include "../v126_author_hdl_pointlio_gravity_20261008/upstream/include/g2o/edge_se3_priorvec.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <queue>

static Eigen::Isometry3d read_pose(std::istream& in){
  double tx,ty,tz,qx,qy,qz,qw;
  if(!(in>>tx>>ty>>tz>>qx>>qy>>qz>>qw))throw std::runtime_error("Invalid pose record");
  Eigen::Quaterniond q(qw,qx,qy,qz);
  Eigen::Vector3d t(tx,ty,tz);
  if(!t.allFinite()||!q.coeffs().allFinite()||std::abs(q.norm()-1)>1e-7)throw std::runtime_error("Nonfinite or nonunit pose");
  Eigen::Isometry3d T=Eigen::Isometry3d::Identity();T.linear()=q.normalized().toRotationMatrix();T.translation()=t;return T;
}

int main(int argc,char** argv){
  try{
    if(argc!=8)throw std::runtime_error("Usage: pose_graph_robust graph.txt poses.txt stats.json policy");
    const std::string policy(argv[4]);const std::string auxiliary(argv[5]);const double auxiliary_width=std::stod(argv[6]);if((auxiliary!="GemanMcClure"&&auxiliary!="Cauchy")||!std::isfinite(auxiliary_width)||auxiliary_width<=0)throw std::runtime_error("Invalid original auxiliary robust kernel");
    if(policy!="shared_huber"&&policy!="rotation_cauchy05"&&policy!="rotation_cauchy1")
      throw std::runtime_error("Unknown robust policy");
    std::ifstream input(argv[1]);int n,m,ng;
    if(!(input>>n>>m>>ng)||n<2||n>6000||m<1||m>40000||ng<0||ng>6000)throw std::runtime_error("Invalid bounded graph size");
    std::ifstream observations(argv[7]);int nc,np,no;
    if(!(observations>>nc>>np>>no)||nc<0||nc>2000||np<0||np>100000||no<0||no>200000||(np==0&&no!=0))throw std::runtime_error("Invalid bounded feature/depth size");
    g2o::SparseOptimizer optimizer;
    if(np){using S=g2o::BlockSolver<g2o::BlockSolverTraits<6,3>>;auto l=std::make_unique<g2o::LinearSolverEigen<S::PoseMatrixType>>();optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::make_unique<S>(std::move(l))));}else{
    using Solver=g2o::BlockSolver<g2o::BlockSolverTraits<6,6>>;
    auto linear=std::make_unique<g2o::LinearSolverEigen<Solver::PoseMatrixType>>();
    auto block=std::make_unique<Solver>(std::move(linear));
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block)));}
    std::vector<int> kinds;std::vector<g2o::VertexSE3*> vertices;std::vector<std::vector<int>> neighbors(n);
    for(int i=0;i<n;i++){
      auto* v=new g2o::VertexSE3;v->setId(i);v->setEstimate(read_pose(input));v->setFixed(i==0);
      if(!optimizer.addVertex(v))throw std::runtime_error("Cannot add pose vertex");vertices.push_back(v);
    }
    for(int k=0;k<m;k++){
      int i,j,kind;if(!(input>>i>>j>>kind)||i<0||j<0||i>=n||j>=n||i==j||(kind!=0&&kind!=1&&kind!=2))throw std::runtime_error("Invalid edge index/kind");
      auto T=read_pose(input);Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Zero();
      for(int d=0;d<6;d++){double value;if(!(input>>value)||!std::isfinite(value)||value<0)throw std::runtime_error("Invalid edge information");information(d,d)=value;}
      if((kind==0&&(information.diagonal().array()<=0).any())||information.diagonal().maxCoeff()<=0)throw std::runtime_error("Invalid nonnegative input information");
      double delta;if(!(input>>delta)||!std::isfinite(delta)||delta<0)throw std::runtime_error("Invalid robust threshold");
      // Only the adapter changes. Native g2o owns the residual, kernels and LM.
      auto attach = [&](const Eigen::Matrix<double,6,6>& info,int id,bool cauchy,double width) {
        auto* edge=new g2o::EdgeSE3;edge->setId(id);edge->setVertex(0,vertices[i]);edge->setVertex(1,vertices[j]);
        edge->setMeasurement(T);edge->setInformation(info);
        if(width>0){
          g2o::RobustKernel* kernel=kind==2 ? (auxiliary=="GemanMcClure"?static_cast<g2o::RobustKernel*>(new g2o::RobustKernelGemanMcClure):static_cast<g2o::RobustKernel*>(new g2o::RobustKernelCauchy)) : cauchy ? static_cast<g2o::RobustKernel*>(new g2o::RobustKernelCauchy)
                                          : static_cast<g2o::RobustKernel*>(new g2o::RobustKernelHuber);
          kernel->setDelta(kind==2?auxiliary_width:width);edge->setRobustKernel(kernel);
        }
        if(!optimizer.addEdge(edge))throw std::runtime_error("Cannot add edge");
      };
      if(kind==0&&policy!="shared_huber"){
        auto trans=information;trans.bottomRightCorner<3,3>().setZero();
        auto rot=information;rot.topLeftCorner<3,3>().setZero();
        attach(trans,k*2,false,delta);
        attach(rot,k*2+1,true,policy=="rotation_cauchy05"?.5:1.);
      }else attach(information,k*2,false,delta);
      kinds.push_back(kind);
      neighbors[i].push_back(j);neighbors[j].push_back(i);
    }
    std::vector<g2o::EdgeSE3PriorVec*> gravity_edges;
    for(int k=0;k<ng;k++) {
      int i;Eigen::Matrix<double,6,1> v;double weight;
      if(!(input>>i)||i<0||i>=n)throw std::runtime_error("Invalid gravity identity");
      for(int z=0;z<6;z++)if(!(input>>v[z]))throw std::runtime_error("Invalid vector observation");
      if(!(input>>weight)||!std::isfinite(weight)||weight<=0||!v.allFinite()||v.head<3>().norm()<1e-9||v.tail<3>().norm()<1e-9)throw std::runtime_error("Invalid gravity observation");
      auto* edge=new g2o::EdgeSE3PriorVec;edge->setId(2*m+k);edge->setVertex(0,vertices[i]);edge->setMeasurement(v);edge->setInformation(Eigen::Matrix3d::Identity()*weight);
      if(!optimizer.addEdge(edge))throw std::runtime_error("Cannot add original gravity factor");gravity_edges.push_back(edge);
    }
    
    std::vector<g2o::ParameterCamera*> cameras;std::vector<int> camera_nodes;
    for(int k=0;k<nc;k++){
      int id,node;if(!(observations>>id>>node)||id!=k||node<0||node>=n)throw std::runtime_error("Invalid camera identity");
      auto offset=read_pose(observations);double fx,fy,cx,cy;
      if(!(observations>>fx>>fy>>cx>>cy)||!std::isfinite(fx+fy+cx+cy)||fx<=0||fy<=0)throw std::runtime_error("Invalid camera intrinsics");
      auto* c=new g2o::ParameterCamera;c->setId(id);c->setKcam(fx,fy,cx,cy);c->setOffset(offset);
      if(!optimizer.addParameter(c))throw std::runtime_error("Cannot add native camera parameter");cameras.push_back(c);camera_nodes.push_back(node);
    }
    std::vector<g2o::VertexPointXYZ*> landmarks;std::vector<int> support(np,0);
    for(int k=0;k<np;k++){
      int id;Eigen::Vector3d X;if(!(observations>>id>>X.x()>>X.y()>>X.z())||id!=k||!X.allFinite())throw std::runtime_error("Invalid landmark IO");
      auto* v=new g2o::VertexPointXYZ;v->setId(n+k);v->setEstimate(X);v->setMarginalized(true);
      if(!optimizer.addVertex(v))throw std::runtime_error("Cannot add native shared landmark");landmarks.push_back(v);
    }
    std::vector<g2o::EdgeSE3PointXYZDepth*> depth_edges;std::vector<int> depth_camera,depth_landmark;
    std::set<std::pair<int,int>> observation_ids;
    for(int k=0;k<no;k++){
      int c,p;Eigen::Vector3d z,info;double delta;
      if(!(observations>>c>>p>>z.x()>>z.y()>>z.z()>>info.x()>>info.y()>>info.z()>>delta)||c<0||c>=nc||p<0||p>=np||!z.allFinite()||z.z()<=0||!info.allFinite()||(info.array()<=0).any()||!std::isfinite(delta)||delta<=0||!observation_ids.insert({c,p}).second)throw std::runtime_error("Invalid true feature/depth observation");
      auto* e=new g2o::EdgeSE3PointXYZDepth;e->setId(2*m+ng+k);e->setVertex(0,vertices.at(camera_nodes.at(c)));e->setVertex(1,landmarks.at(p));e->setParameterId(0,c);e->setMeasurement(z);e->setInformation(info.asDiagonal());
      auto* kernel=new g2o::RobustKernelHuber;kernel->setDelta(delta);e->setRobustKernel(kernel);
      if(!optimizer.addEdge(e))throw std::runtime_error("Cannot add original feature/depth factor");
      depth_edges.push_back(e);depth_camera.push_back(c);depth_landmark.push_back(p);support[p]++;
    }
    std::string unexpected;if(observations>>unexpected)throw std::runtime_error("Extra observation input");
    if(np&&std::any_of(support.begin(),support.end(),[](int k){return k<2;}))throw std::runtime_error("Every landmark needs separate real views");

    std::string extra;if(input>>extra)throw std::runtime_error("Unexpected graph data");
    std::vector<bool> visited(n,false);std::queue<int> queue;queue.push(0);visited[0]=true;int count=0;
    while(!queue.empty()){int i=queue.front();queue.pop();count++;for(int j:neighbors[i])if(!visited[j]){visited[j]=true;queue.push(j);}}
    if(count!=n)throw std::runtime_error("Disconnected pose graph");
    if(!optimizer.initializeOptimization())throw std::runtime_error("g2o initialization failed");
    optimizer.computeActiveErrors();double before=optimizer.activeRobustChi2();
    int iterations=optimizer.optimize(40);optimizer.computeActiveErrors();double after=optimizer.activeRobustChi2();
    if(iterations<0||!std::isfinite(before)||!std::isfinite(after)||after>before+1e-7*std::max(1.,before))throw std::runtime_error("g2o objective invalid or increased");
    std::ofstream poses(argv[2]);poses<<std::setprecision(17);
    for(auto* v:vertices){
      const auto& T=v->estimate();if(!T.matrix().allFinite())throw std::runtime_error("Nonfinite optimized pose");
      Eigen::Quaterniond q(T.linear());q.normalize();poses<<T.translation().transpose()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
    }
    if(!poses)throw std::runtime_error("Cannot write poses");
    std::ofstream stats(argv[3]);stats<<std::setprecision(17)<<"{\"initial_cost\":"<<before<<",\"final_cost\":"<<after<<",\"iterations\":"<<iterations<<",\"vertices\":"<<n<<",\"input_factors\":"<<m<<",\"native_factors\":"<<optimizer.edges().size()<<",\"policy\":\""<<policy<<"\",\"factor_audit\":[";
    bool first=true;
    for(auto* abstract:optimizer.edges()){
      auto* e=dynamic_cast<g2o::EdgeSE3*>(abstract);if(!e){if(dynamic_cast<g2o::EdgeSE3PriorVec*>(abstract)||dynamic_cast<g2o::EdgeSE3PointXYZDepth*>(abstract))continue;throw std::runtime_error("Unexpected native edge type");}
      int k=e->id()/2,kind=kinds.at(k);std::string component=kind==0&&policy!="shared_huber"?(e->id()%2?"rotation":"translation"):"shared";
      std::string kernel=component=="rotation"?"Cauchy":kind==2?auxiliary:"Huber";
      double chi=e->chi2();Eigen::Vector3d rho;
      if(e->robustKernel())e->robustKernel()->robustify(chi,rho);else rho<<chi,1.,0.;
      if(!rho.allFinite()||rho[1]<0)throw std::runtime_error("Invalid native robust factor");
      if(!first)stats<<",";first=false;
      stats<<"{\"original_edge_index\":"<<k<<",\"kind\":"<<kind<<",\"component\":\""<<component<<"\",\"kernel\":\""<<kernel<<"\",\"delta\":"<<(e->robustKernel()?e->robustKernel()->delta():0.)<<",\"chi2\":"<<chi<<",\"native_cost\":"<<rho[0]<<",\"native_weight\":"<<rho[1]<<",\"i\":"<<e->vertex(0)->id()<<",\"j\":"<<e->vertex(1)->id()<<",\"information\":[";
      for(int z=0;z<6;z++){if(z)stats<<",";stats<<e->information()(z,z);}stats<<"],\"measurement\":[";
      for(int z=0;z<16;z++){if(z)stats<<",";stats<<e->measurement().matrix()(z/4,z%4);}stats<<"]}";
    }
    stats<<"],\"gravity_audit\":[";
    first=true;for(auto* e:gravity_edges){if(!first)stats<<",";first=false;stats<<"{\"i\":"<<e->vertex(0)->id()<<",\"chi2\":"<<e->chi2()<<",\"information\":"<<e->information()(0,0)<<",\"measurement\":[";for(int z=0;z<6;z++){if(z)stats<<",";stats<<e->measurement()[z];}stats<<"]}";}stats<<"]";
    stats<<",\"feature_depth_audit\":[";first=true;
    for(size_t k=0;k<depth_edges.size();k++){
      auto* e=depth_edges[k];Eigen::Vector3d rho;e->robustKernel()->robustify(e->chi2(),rho);
      if(!std::isfinite(e->chi2())||!rho.allFinite()||rho[1]<0)throw std::runtime_error("Invalid native projection statistic");
      if(!first)stats<<",";first=false;
      stats<<"{\"camera\":"<<depth_camera[k]<<",\"node\":"<<camera_nodes[depth_camera[k]]<<",\"landmark\":"<<depth_landmark[k]<<",\"chi2\":"<<e->chi2()<<",\"cost\":"<<rho[0]<<",\"weight\":"<<rho[1]<<",\"measurement\":["<<e->measurement().x()<<","<<e->measurement().y()<<","<<e->measurement().z()<<"],\"information\":["<<e->information()(0,0)<<","<<e->information()(1,1)<<","<<e->information()(2,2)<<"]}";
    }
    stats<<"],\"shared_landmarks\":"<<np<<",\"camera_parameters\":"<<nc;
    std::ofstream pts(std::string(argv[2])+".landmarks.txt");pts<<std::setprecision(17);
    for(auto* v:landmarks){if(!v->estimate().allFinite())throw std::runtime_error("Nonfinite native landmark");pts<<v->estimate().transpose()<<"\n";}
    if(!pts)throw std::runtime_error("Landmark IO failed");
stats<<"}\n";
    if(!stats)throw std::runtime_error("Cannot write statistics");
    return 0;
  }catch(const std::exception& e){std::cerr<<"GRAPH FAILED: "<<e.what()<<std::endl;return 2;}
}
