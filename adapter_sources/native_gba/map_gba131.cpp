
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <set>
#include <sys/resource.h>
#include "map.h"
#include "read_configs.h"
#include "g2o_optimization/g2o_optimization.h"
void GlobalBA131(MapPtr,const OptimizationConfig&,bool,bool,int,int);
void export_frames(MapPtr map,const std::string& path){
 std::ofstream f(path);f<<std::setprecision(17);
 for(auto & kv:map->GetAllKeyframes()){
  auto fr=kv.second;auto T=fr->GetPose();f<<kv.first<<" "<<fr->GetTimestamp();
  for(int r=0;r<4;r++)for(int c=0;c<4;c++)f<<" "<<T(r,c);f<<"\n";
 }
}
int main(int argc,char**argv){
 if(argc!=7 && argc!=8){std::cerr<<"map config output mode iterations1 iterations2\n";return 2;}
 std::cout<<std::setprecision(17);
 MapPtr map;std::ifstream f(argv[1],std::ios::binary);if(!f)return 3;
 boost::archive::binary_iarchive ar(f);ar>>map;f.close();
 std::cout<<"MAP keyframes="<<map->GetAllKeyframes().size()<<" points="<<map->GetAllMappoints().size()<<" lines="<<map->GetAllMaplines().size()<<" imu="<<map->IMUInit()<<std::endl;
 // Exclude fixed held-image observation identities only in this in-memory copy.
 std::set<int> excluded;
 if(argc==8){std::ifstream ids(argv[7]);int id;while(ids>>id)excluded.insert(id);}
 size_t removed_points=0,removed_lines=0;
 for(int id:excluded){
  auto it=map->GetAllKeyframes().find(id);if(it==map->GetAllKeyframes().end())continue;
  auto fr=it->second;
  auto & pts=fr->GetAllMappoints();
  for(size_t k=0;k<pts.size();k++)if(pts[k]){pts[k]->RemoveObverser(id);fr->RemoveMappoint(int(k));removed_points++;}
  auto & lines=fr->GetAllMaplines();
  for(size_t k=0;k<lines.size();k++)if(lines[k]){lines[k]->RemoveObverser(id);fr->RemoveMapline(int(k));removed_lines++;}
 }
 size_t valid_points=0,valid_lines=0,velocities=0;
 for(auto & kv:map->GetAllMappoints())if(kv.second){if(kv.second->ObverserNum()==0)kv.second->SetBad();if(kv.second->IsValid())valid_points++;}
 for(auto & kv:map->GetAllMaplines())if(kv.second){if(kv.second->ObverserNum()==0)kv.second->SetBad();if(kv.second->IsValid())valid_lines++;}
 for(auto & kv:map->GetAllKeyframes())if(kv.second->VelocityIsInitialized())velocities++;
 std::cout<<"ADMISSION excluded_images="<<excluded.size()<<" removed_point_observations="<<removed_points<<" removed_line_observations="<<removed_lines<<" valid_points="<<valid_points<<" valid_lines="<<valid_lines<<" initialized_velocities="<<velocities<<std::endl;
 OptimizationConfig cfg;auto yaml=YAML::LoadFile(argv[2]);cfg.Load(yaml["optimization"]);
 export_frames(map,std::string(argv[3])+".before");
 std::string mode=argv[4];
 if(mode=="original")GlobalBA(map,cfg,true,true,std::stoi(argv[5]),std::stoi(argv[6]));
 else if(mode=="robust")GlobalBA131(map,cfg,true,true,std::stoi(argv[5]),std::stoi(argv[6]));
 else if(mode!="inspect")return 4;
 export_frames(map,argv[3]);
 rusage ru;getrusage(RUSAGE_SELF,&ru);std::cout<<"COMPLETE peak_rss_mib="<<ru.ru_maxrss/1024.0<<std::endl;
 std::exit(0);
}
