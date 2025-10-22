#ifndef DEFINES_H_
#define DEFINES_H_

namespace Refactor {

#define PATH_OPT_PARAM "../../../Parameters/OptParam.json"
#define PATH_SPEC_PARAM "../../../Parameters/SpecParam.json"
#define PATH_TRAFFIC "../../../Parameters/traffic.txt"
#define PATH_PREPARE "../../../Parameters/prepare.txt"
#define PATH_MAPPING "../../../Parameters/mapping.txt"
#define PATH_MESHCONNECT "../../../Parameters/meshconnect.txt"
#define PATH_MESHLAYER "../../../Parameters/Meshlayer.txt"
#define PATH_DUPLICATION "../../../Parameters/duplication.txt"
#define PATH_TILENUM "../../../Parameters/tileNum.txt" // "tiles.json"
#define PATH_LAYER "../../../Parameters/layer.txt"     // "layer.json"
#define PATH_PLACING "../../../Parameters/placing.txt"

#define PATH_3D_DIR "../../../../3D-Multi_Level_Opt"
#define PATH_HSIM_OUT "../../../../3D-Multi_Level_Opt/results/PPA.csv"
#define PATH_LAYER_OUT "../../../../3D-Multi_Level_Opt/results/layer_performance.csv"

#define INF 1e9

// Refer to the paper "COMN"
enum Mapping_method {
  COLUMN_TRANS = 0,
  MATRIX_TRANS = 1,
  ROW_TRANS = 2,
  LW_MATMUL_TRANS = 3,
  RW_MATMUL_TRANS = 4
};

enum Pipeline_method {
  EXTREME_PIPELINE = 1,
  HYBRID_PIPELINE = 2,
  DEFAULT_PIPELINE = 3
};

enum Activation { RELU = 0, MAXPOOL = 1, SIGMOID = 2 };

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

} // namespace Refactor

#endif // !DEFINES_H_