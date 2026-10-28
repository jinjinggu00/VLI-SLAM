# Third-party notices

This project integrates existing implementations. Existing copyright headers and licenses remain applicable.

- AirSLAM: https://github.com/sair-lab/AirSLAM — GPL-3.0, see LICENSE and AirSLAM_LICENSE.md.
- g2o: https://github.com/RainerKuemmerle/g2o — original linked solver and factors; no solver reimplementation.
- LightGlue/ALIKED: https://github.com/cvg/LightGlue — see upstream/lightglue/LICENSE and licenses/LightGlue.txt.
- LVI-SAM DepthRegister: https://github.com/TixiaoShan/LVI-SAM — see licenses/LVI-SAM.txt.
- Point-LIO: https://github.com/hku-mars/Point-LIO — external LiDAR-inertial auxiliary implementation, not vendored here.
- BoQ: https://github.com/amaralibey/Bag-of-Queries — external place retrieval implementation, not vendored here.
- hdl_graph_slam factor: https://github.com/koide3/hdl_graph_slam — see licenses/hdl_graph_slam.txt and the retained source header.

The publication runtime manifest checks the distributed code and required frozen inputs. The native solver, output comparison, original model code and measured-input hashes remain unchanged. Original archive provenance is recorded in SOURCE_MANIFEST.json.
