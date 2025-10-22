 /*
 * @file Mapping.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <math.h>
#include <set>

#include "Perf_Evaluator.h"
#include "Mapping.h"
#include "Mesh_Placing.h"
#include "defines.h"

namespace Refactor {

Mapping::Mapping(PyParam *_pyParam, PyWeight *_pyWeight, PyInput *_pyInput, PyActInput *_pyActInput) : pyParam(_pyParam), pyWeight(_pyWeight), pyInput(_pyInput), pyActInput(_pyActInput) {
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  mapping_optimized = optParam["mapping_optimized"];
  transform_method = optParam["mapping_method"];
  
  prepare_mode = optParam["prepare_mode"];
  stride = pyParam->stride.first;
  k1 = (pyWeight->shape.size() == 4) ? pyParam->kernel_size.first : 1;
  k2 = (pyWeight->shape.size() == 4) ? pyParam->kernel_size.second : 1;
  
  inChannels = pyWeight->shape[1];
  outChannels = pyWeight->shape[0];

  NVM_states = 2;
}

void Mapping::mapping_modules() {
  if (prepare_mode) {
    analysis();
    // cout << "Analysis Done!!!" << endl;
  } else {
    auto_mapping();
    // cout << "auto_mapping Done!!!" << endl;
  }
}

/**
 * @brief Compute array_size(int or double?) and compute_cycle, and write into
 * Parameters/prepare.txt in "a+" format.
 *
 */
void Mapping::analysis() {
  int compute_cycle = 1, weight_features = 1;
  for (auto i : pyWeight->shape) {
    weight_features *= i;
  }
  double array_size = weight_features * 2 * pyParam->w_precision / NVM_states;
  if (transform_method == LW_MATMUL_TRANS) {
    // Matrix mul, Weight on the left;
    for (int i = 1; i < pyInput->shape.size(); i++) {
      compute_cycle *= pyInput->shape[i];
    }
  } else if (transform_method == RW_MATMUL_TRANS) {
    // Matrix mul, Weight on the right;
    for (int i = 0; i < pyInput->shape.size() - 1; i++) {
      compute_cycle *= pyInput->shape[i];
    }
  } else if (pyWeight->shape.size() == 4) {
    // Conv layers
    compute_cycle = ((pyInput->shape[2] + 2 * pyParam->padding.first - pyParam->kernel_size.first) / pyParam->stride.first + 1) * ((pyInput->shape[3] + 2 * pyParam->padding.second - pyParam->kernel_size.second) / pyParam->stride.second + 1);
  } else if (pyWeight->shape.size() == 2) {
    // Linear layers
    compute_cycle = 1;
  } else {
    throw runtime_error("In analysis(), unidentified layer type found!!!");
  }
  // Attach current layer's info to the file.
  ofstream prepare(PATH_PREPARE, ios::app);
  prepare << setprecision(16);
  prepare << "layer: " << pyParam->layer << " array_size: " << array_size << " compute_cycle: " << compute_cycle << endl;
  prepare.close();
}

/**
 * @brief Map the designated weight onto Tiles, and calculate performance
 * information including area/latency/energy.
 * Following the order of : weight_transform -> weight_partition -> Htree_NoC ->
 * (CIM_num) -> Calculate_Buffer
 *
 */
void Mapping::auto_mapping() {
  // Read json files and other data files and prepare for mapping
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  vector<int> duplication; // duplication for each layer
  int val;
  ifstream f(PATH_DUPLICATION);
  while (f >> val) {
    duplication.push_back(val);
  }
  int curDup = duplication[pyParam->layer - 1];
  arraySize = specParam["Subarray"][0]; // side length of a subarray.
  bufferSizeTile = specParam["buffersizeTile"];
  bufferSizeTile *= 1024 * 8; // KB -> b
  Tile.push_back(specParam["Tile"][0]);
  Tile.push_back(specParam["Tile"][1]);
  Subarray.push_back(specParam["Subarray"][0]);
  Subarray.push_back(specParam["Subarray"][1]);
  // Adjust the Mapping method according to Weight Size
  if (mapping_optimized) {
    if (k1 * k2 * inChannels < arraySize) {
      transform_method = COLUMN_TRANS; // [k1*k2*inChannels,outChannels]
    } else if (inChannels > 2 * outChannels) {
      transform_method = ROW_TRANS; // [inChannels,k1*k2*outChannels]
    } else {
      transform_method = MATRIX_TRANS; // [k1*inChannels,k2*outChannels]
    }
  }

  vector<int> weight_unfold;         // 2 nums, in column and row dimension
  int buffer_demand;                 // Buffer for NN inputs of each layer
  if (pyWeight->shape.size() == 2) { // Matmul && Linear
    weight_unfold.push_back(inChannels);
    weight_unfold.push_back(outChannels);
    buffer_demand = inChannels * pyParam->a_precision;
  } else {
    weight_transform(weight_unfold, buffer_demand, pyInput->shape[3]);
  }
  PartitionInfo partitionInfo;
  weight_partition(partitionInfo, weight_unfold, buffer_demand, curDup);
  HtreeNoCInfo htreeNoCInfo;
  Htree_NoC(htreeNoCInfo, partitionInfo);
  // modify pyInput shape in terms of linear layer or matmul layer
  if (pyWeight->shape.size() == 2) {
    pyInput->shape.resize(4, 1); // (a,b) -> (a,b,1,1)
  }

  int input_vec_num = 1; // num of input vecs to Subarrays for Matmul
  if (transform_method == LW_MATMUL_TRANS) {
    for (int i = 1; i < pyInput->shape.size(); i++) {
      input_vec_num *= pyInput->shape[i];
    }
  } else if (transform_method == RW_MATMUL_TRANS) {
    for (int i = 0; i < pyInput->shape.size() - 1; i++) {
      input_vec_num *= pyInput->shape[i];
    }
  } else {
    input_vec_num = pyInput->shape[2] * pyInput->shape[3] / (stride * stride);
  }

  // Num of CIM arrays used when processing a layer. Note that arrays are not
  // fully utilized, so the values of Split_array[] are important.
  CIM_num = partitionInfo.Split_array[0] * partitionInfo.Split_array[1] * partitionInfo.Inter_tile * partitionInfo.Intra_tile * input_vec_num / curDup * pyParam->a_precision;

  auto Buffer_result = Calculate_Buffer(partitionInfo.Split_tile, curDup, input_vec_num);
  Buffer_read = Buffer_result.first, Buffer_write = Buffer_result.second;
  TileNoC_num = Calculate_TileNoC(partitionInfo, htreeNoCInfo.Tile_NoC, curDup, input_vec_num);
}

/**
 * @brief Use info from prepare.txt to calculate duplication of each layer.
 * array_size and compute_cycle are recorded in prepare.txt.
 */
void Mapping::pipeline_optimized() {
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  int pipeline_method;
  if (optParam["latency"] == false && optParam["area"] == false) {
    pipeline_method = DEFAULT_PIPELINE;
  } else if (optParam["latency"] == true && optParam["area"] == false) {
    pipeline_method = EXTREME_PIPELINE;
  } else {
    pipeline_method = HYBRID_PIPELINE;
  }
  f.close();
  optParam["pipeline_method"] = pipeline_method;
  ofstream of(PATH_OPT_PARAM);
  of << std::setw(2) << optParam << std::endl;
  of.close();
  // cout << "Saving pipeline_method!!" << endl;
  // Reading layer.txt
  ifstream f_layer(PATH_LAYER);
  // Attention: conv_layer here means "not Linear layer" after introducing matmul operator.
  int conv_layer, total_layer;
  f_layer >> conv_layer >> total_layer;

  ifstream f_prepare(PATH_PREPARE);
  string line;
  vector<int> duplication(total_layer, 1), compute_cycle(conv_layer);
  vector<double> array_size(conv_layer);
  vector<string> tokens;

  // cout << "Start dealing with pipeline!!" << endl;
  if (pipeline_method == DEFAULT_PIPELINE) {
    cout << "Default Pipeline is used." << endl;
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        duplication[cur_layer - 1] = 1;
      }
    }
  }
  else if (pipeline_method == EXTREME_PIPELINE) {
    cout << "Extreme Pipeline is used." << endl;
    int min_compute_cycles = INF;
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        compute_cycle[cur_layer - 1] = stoi(tokens[5]);
        min_compute_cycles = min(min_compute_cycles, compute_cycle[cur_layer - 1]);
      }
    }
    for (int l = 0; l < conv_layer; l++) {
      duplication[l] = compute_cycle[l] / min_compute_cycles;
    }
  }
  else if (pipeline_method == HYBRID_PIPELINE) {
    cout << "Hybrid Pipeline is used." << endl;
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        array_size[cur_layer - 1] = stod(tokens[3]);
        compute_cycle[cur_layer - 1] = stoi(tokens[5]);
      }
    }
    vector<int> tmp_duplication(total_layer, 1), tmp_compute_cycle = compute_cycle;
    vector<double> tmp_array_size = array_size;
    // Test the most suitable layer to calculate duplication nums
    double lowest_score = 1e20;
    for (int i = conv_layer - 1; i >= 0; i--) {
      array_size = tmp_array_size;
      compute_cycle = tmp_compute_cycle;
      fill(tmp_duplication.begin(), tmp_duplication.end(), 1);
      for (int j = 0; j < conv_layer; j++) {
        if (compute_cycle[j] >= compute_cycle[i]) {
          tmp_duplication[j] = compute_cycle[j] / compute_cycle[i];
          compute_cycle[j] = compute_cycle[i];
          array_size[j] = array_size[j] * tmp_duplication[j];
        }
      }
      double score = accumulate(array_size.begin(), array_size.end(), 0.0) * compute_cycle[i];
      if (score < lowest_score) {
        lowest_score = score;
        duplication = tmp_duplication;
      }
    }
  }
  ofstream of_dup(PATH_DUPLICATION);
  for (auto dup : duplication) {
    of_dup << dup << " ";
  }
  of_dup.close();
}

/**
 * @brief Get the total_energy,total_latency,total_area
 *
 */
void Mapping::Mesh_operation() {
  // cout << "Into Mesh_operation!!!" << endl;
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  int pipeline_method = optParam["pipeline_method"];

  ifstream f_traffic(PATH_TRAFFIC);
  // vector<vector<int>> traffic;
  map<pair<int, int>, int> traffic_mp; // The data transfer between x and y.
  int src, dst, packet;
  while (f_traffic >> src >> dst >> packet) {
    traffic_mp[{src, dst}] += packet;
  }

  double meshEnergy, meshArea, meshLatency = NOC_Perf(5, 5, 4, 1e9, 32);

  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  int MeshNoC = specParam["MeshNoC_flitband"];
  MeshNoC *= 8;

  ifstream f_tile(PATH_TILENUM);
  int tileNum;
  f_tile >> tileNum;
  int conv_layer, total_layer;
  ifstream f_layer(PATH_LAYER);
  f_layer >> conv_layer >> total_layer; // conv_layer not used here.
  Mesh_Placing *placing = new Mesh_Placing(MeshNoC, MeshNoC, tileNum, meshLatency);

  MeshInfo meshInfo;
  meshInfo = placing->Mesh_mapping_energy_pipeline(traffic_mp);

  double totalEnergy = 0;
  double totalLatency = 0;
  double totalArea = 0;
  string line;
  ifstream f_results(PATH_HSIM_OUT);
  auto trim = [](string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == string::npos) s.clear(); else s = s.substr(b, e - b + 1);
  };
  auto split_csv = [&](const string &s) {
    vector<string> out; out.reserve(32); string cur;
    for (char c : s) {
      if (c == ',') { out.push_back(cur); cur.clear(); }
      else { cur.push_back(c); }
    }
    out.push_back(cur);
    for (auto &t : out) trim(t);
    return out;
  };

  string header;
  if (getline(f_results, header)) {
    auto headers = split_csv(header);
    // Build a name->index map once
    unordered_map<string, size_t> idx;
    for (size_t i = 0; i < headers.size(); ++i) idx[headers[i]] = i;

    const char* kNoCEnergy = "Total NoC Energy (pJ)";
    const char* kNoCArea = "Total NoC Area (mm^2)";
    const char* kNoCLat = "Total NoC Latency (ns)";
    const char* kDynEnergy = "Total Dynamic Energy (pJ)";
    const char* kLeakEnergy = "Total Leakage Energy (pJ)";
    const char* k3DStackArea = "Total 3d stack Area (mm^2)";
    const char* kCompLat = "Total Computing Latency (ns)";

    auto has = [&](const char* key) { return idx.find(key) != idx.end(); };

    string row;
    while (getline(f_results, row)) {
      auto values = split_csv(row);
      auto safe_atod = [&](const char* key) -> double {
        auto it = idx.find(key);
        if (it == idx.end()) return 0.0;
        size_t i = it->second;
        if (i >= values.size()) return 0.0;
        try { return stod(values[i]); } catch (...) { return 0.0; }
      };

      if (has(kNoCEnergy)) totalEnergy += safe_atod(kNoCEnergy);
      if (has(kNoCArea)) totalArea += safe_atod(kNoCArea);
      if (has(kNoCLat)) totalLatency += safe_atod(kNoCLat);
      if (has(kDynEnergy)) totalEnergy += safe_atod(kDynEnergy);
      if (has(kLeakEnergy)) totalEnergy += safe_atod(kLeakEnergy);
      if (has(k3DStackArea)) totalArea += safe_atod(k3DStackArea);
      if (has(kCompLat)) totalLatency += safe_atod(kCompLat);
    }
  }

  free(placing);

  string mapOutPath = "../../../generate_data/mapping_out/mapping_out.txt";
  writeInfo(totalArea, totalLatency*1e-9, totalEnergy*1e-12, mapOutPath, total_layer);
}

/**
 * @brief Calculate the width and height of weight unfolded in Memcells
 * @param features : input.shape[3], the width of input feature
 *
 */
void Mapping::weight_transform(vector<int> &weight_unfold, int &buffer_demand, int features) {
  if (transform_method == COLUMN_TRANS) {
    // buffer_demand for COLUMN_TRANS = ((Nx × (Ky − 1)) + Kx) × Nif.
    // Nx = number of rows in the input feature map;
    // Ky and Kx = the number of columns and rows in the kernel;
    // Nif = number of input feature maps involved in the convolution step
    weight_unfold.push_back(k1 * k2 * inChannels);
    // double weights are required to represent signed values.
    weight_unfold.push_back(outChannels * pyParam->w_precision / NVM_states * 2);

    // Kx = Ky = (k - 1) / stride + 1;
    buffer_demand = (features * (k1 - 1) / stride + (k1 - 1) / stride + 1) * inChannels * pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS) {
    weight_unfold.push_back(k1 * inChannels);
    weight_unfold.push_back(k2 * outChannels * pyParam->w_precision / NVM_states * 2);
    buffer_demand = (features * (k1 - 1) / stride + 1) * inChannels * pyParam->a_precision;
  } else if (transform_method == ROW_TRANS) {
    weight_unfold.push_back(inChannels);
    weight_unfold.push_back(k1 * k2 * outChannels * pyParam->w_precision / NVM_states * 2);
    buffer_demand = (features * (k1 - 1) / stride + (k1 - 1) / stride + 1) * outChannels * pyParam->a_precision;
  }
}

/**
 * @brief Partition the weights into tiles and subarrays.
 *
 */
void Mapping::weight_partition(PartitionInfo &info, vector<int> weight_unfold, int buffer_demand, int duplication) {
  // We consider folding weights in partition. More specifically, when the
  // width/height of "weight_unfold" is greater than the size of a tile, try to
  // fold it into fewer tiles to increase utilization rate.

  // Caution: After folding, the HTree routing conditions will change, so in
  // HTree_NoC, the cost of routing has to be discussed in different scenarios.

  double Split_heightArray = (double)weight_unfold[0] / Subarray[0];
  double Split_widthArray = (double)weight_unfold[1] / Subarray[1];

  int intraTile, fold_copies;
  fold_copies = adjust_split_array(Split_heightArray, Split_widthArray);
  // Duplication has to be considered. The placement of duplication also
  // matters. When we place duplications in rows or columns, the buffer
  // read/write energy may change a lot.

  // Here, we assume that all duplications are placed in a column across tiles.
  // The partition may optimize after proposing a detailed "target function".

  // Duplication num in Row/Column
  int maxRowDup = MAX(MIN(Tile[0], floor(Tile[0] / Split_heightArray)), 1);
  int maxColDup = MAX(MIN(Tile[1], floor(Tile[1] / Split_widthArray)), 1);
  intraTile = maxRowDup * maxColDup; // Max duplications in a Tile
  // BufferSize also needs to be taken into consideration. Duplication in a tile
  // cannot exceed the buffer size.
  intraTile = MIN(MAX(1, bufferSizeTile / buffer_demand), intraTile);
  intraTile = MIN(intraTile, duplication); // cannot exceed "duplication".
  maxRowDup = ceil(sqrt((double)intraTile));
  while (intraTile % maxRowDup != 0) {
    maxRowDup--;
  }
  info.intraRowDup = maxRowDup;
  info.intraColDup = intraTile / maxRowDup;

  // Duplication num across tiles (apart from duplication in a tile).
  int TileDups = ceil((double)duplication / intraTile);

  // Duplications are stacked in rows.
  double Split_heightTile = ceil(MAX(Split_heightArray, Tile[0]) * TileDups / Tile[0]);
  double Split_widthTile = ceil(Split_widthArray / Tile[1]);
  Split_heightArray = Split_heightArray * TileDups / Split_heightTile; // resize to (0,Tile[0]]
  Split_widthArray /= Split_widthTile;                 // resize to (0,Tile[1]]
  int interTile = Split_heightTile * Split_widthTile;

  double demand_tiles = ceil((double)buffer_demand / bufferSizeTile);
  if (demand_tiles > interTile) {
    // The tiles already split do not have enough buffer:
    Split_heightTile = ceil(sqrt(demand_tiles));
    Split_widthTile = ceil(demand_tiles / Split_heightTile);
    // TODO: some problems with Split array???
    Split_heightArray = (double)weight_unfold[0] / Subarray[0] / Split_heightTile;
    Split_widthArray = (double)weight_unfold[1] / Subarray[1] / Split_widthTile;
    fold_copies = adjust_split_array(Split_heightArray, Split_widthArray);
    interTile = Split_heightTile * Split_widthTile;
  }

  // Split_tile is exactly the split of tiles in width/height considering
  // duplication; Split_array is the split of subarray in width/height without
  // considering duplication in tile!!! (but the duplication between tiles are
  // considered)
  info.Split_array.resize(2);
  info.Split_tile.resize(2);
  info.Split_array[0] = Split_heightArray;
  info.Split_array[1] = Split_widthArray;
  // Duplications are considered in Split_tile.
  info.Split_tile[0] = Split_heightTile;
  info.Split_tile[1] = Split_widthTile;

  info.Intra_tile = intraTile;
  info.Inter_tile = interTile;
  info.fold_copies = fold_copies;

  ofstream of(PATH_MAPPING, ios::app); // adding
  of << setprecision(16);
  of << "layer: " << pyParam->layer
      << " Split_tile[0]: " << info.Split_tile[0]
      << " Split_tile[1]: " << info.Split_tile[1]
      << " Split_array[0]: " << info.Split_array[0]
      << " Split_array[1]: " << info.Split_array[1]
      << " intraTile: " << intraTile << " interTile: " << interTile
      << " duplication: " << duplication << endl;
  of.close();
}

/**
 * @brief Adjust the value of Split_heightArray/Split_widthArray by folding
 * weight into many halves. Currently, we fold the weight only when one side of
 * weight <= 0.5 * Tile Size and another side > Tile Size.
 * @return Return "fold_copies" for "PartitionInfo". Refer to the definition of
 * fold_copies in "struct PartitionInfo" in Mapping.h.
 */
int Mapping::adjust_split_array(double &Split_heightArray, double &Split_widthArray) {
  int fold_copies = 1;
  bool horizontal_fold = false;
  // Vertical Folding. At most one branch of "while" can be executed.
  while (Split_heightArray * 2 < Tile[0] && Split_widthArray > Tile[1]) {
    Split_heightArray *= 2;
    Split_widthArray /= 2;
    fold_copies *= 2;
    if (fold_copies >= Tile[0]) {
      break;
    } // Fold times limited by Tile Size
  }

  // Horizontal Folding.
  while (Split_widthArray * 2 < Tile[1] && Split_heightArray > Tile[0]) {
    horizontal_fold = true;
    Split_widthArray *= 2;
    Split_heightArray /= 2;
    fold_copies *= 2;
    if (fold_copies >= Tile[1]) {
      break;
    } // Fold times limited by Tile Size
  }

  // change to negative value when performing horizontal folding.
  fold_copies = horizontal_fold ? -fold_copies : fold_copies;
  return fold_copies;
}

/**
 * @brief Similar to weight_partition, but multiple Tile Size can be chosen to
 * hold the weights. The Subarray Size is fixed here, however.
 * The problem becomes an optimization problem, and we'll first define the
 * search space and then go through all possible params to find the best one.
 */
void Mapping::weight_mixed_split(PartitionInfo &info, vector<int> weight_unfold, int buffer_demand, int duplication) {
  // 1. Define Search Space for Tile Size.
  int MAX_TILE_SIZES = 4; // How many types of Tile Size are allowed;
  vector<vector<int>> tile_sizes = {{2, 2}, {2, 4}, {4, 4}, {8, 8}};
}

/**
 * @brief establish Hierarchical Tree Network-on-Chip with routers.
 */
void Mapping::Htree_NoC(HtreeNoCInfo &info, PartitionInfo part_info) {
  // We need to Calculate how many input hops and output hops are needed to
  // transfer IFMs and OFMs into/out of subarrays. And in this function, the
  // usage of all the subarrays in all the tiles of the current
  // layer is assumed to be the SAME. So we only need to investigate the input
  // hops and output hops of ONE TILE here. The modeling procedure of Htree hops
  // calculation is in "Graphs.pptx".

  // NOTE: A hop means transferring a packet (with data amount to ONE SUBARRAY)
  // through a HTree router. So macroRows/macroCols really matter.

  // At present, Tile Size and Subarray Size are all assumed to be the power of
  // 2, and the width and height are the same.
  auto Split_array = part_info.Split_array;
  auto intra_dup = part_info.Intra_tile; // Duplications inside a Tile.
  auto fold_copies = part_info.fold_copies;
  int macroRows, macroCols;
  if (fold_copies > 0) { // fold vertically
    macroRows = ceil(Split_array[0] / fold_copies) * fold_copies;
    macroCols = ceil(Split_array[1]);
  } else { // fold horizontally
    macroRows = ceil(Split_array[0]);
    macroCols = ceil(Split_array[1] / -fold_copies) * -fold_copies;
  }

  macroRows = ceil((double)macroRows / part_info.intraRowDup);
  macroCols = ceil((double)macroCols / part_info.intraColDup);

  // Hops for a packet to be transferred into/out of a subarray
  int routing_hops = log2(Tile[0]); // Tile[0] should be the power of 2;
  int input_hops, output_hops;
  int curRows = Tile[0], curCols = Tile[1];
  if (fold_copies == 1) {
    // Condition 1: Intra-Tile duplication and folding are not considered in
    // this condition. The most simple condition.
    int sub_routers = 0; // The num of routers concerned
    for (int i = 1; i <= routing_hops; i++) {
      sub_routers += ceil((double)macroCols / curCols);
      curCols /= 2;
    }
    input_hops = macroRows * sub_routers;

    sub_routers = 0;
    for (int i = 1; i <= routing_hops; i++) {
      sub_routers += ceil((double)macroRows / curRows);
      curRows /= 2;
    }
    output_hops = macroCols * sub_routers;
  } else if (fold_copies > 1) {
    // Condition 2: Intra-Tile duplication not considered in this condition.
    // Folding weight vertically.
    int curDup = 1;
    // After folding weights, we divide the weights into several groups
    // row-wise to calculate input_hops.
    int router_sum = 0; // total sum of routers for one "group" of input.
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += curDup * ceil((double)macroCols / curCols);
      curCols /= 2;
      if (curDup < fold_copies) {
        curDup *= 2;
      }
    }
    input_hops = macroRows / fold_copies * router_sum;

    router_sum = 0;
    int group_rows = macroRows / fold_copies;
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += ceil((double)group_rows / curRows);
      curRows /= 2;
    }
    output_hops = macroCols * fold_copies * router_sum;
  } else {
    // Condition 3: Intra-Tile duplication not considered in this condition.
    // Folding weight horizontally.
    fold_copies = -fold_copies; // The only place where fold_copies changed!!!
    int router_sum = 0;
    int group_cols = macroCols / fold_copies;
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += ceil((double)group_cols / curCols);
      curCols /= 2;
    }
    input_hops = macroRows * fold_copies * router_sum;

    int used_routers; // The num of routers used counting from the column
    int divide_num = 1;
    int concerned_columns = macroCols;
    router_sum = 0;
    curCols = Tile[1];
    for (int i = routing_hops; i >= 1; i--) {
      // Search the use of routers from low levels to high levels.
      // Correspond to "yellow->blue->black" router in "Graphs.pptx".
      divide_num *= 2;
      used_routers = ceil((double)macroRows / divide_num);
      if ((1 << i) <= fold_copies) {
        concerned_columns /= 2;
      }
      router_sum += used_routers * concerned_columns;
      curCols /= 2;
    }
    output_hops = router_sum;
  }

  // Duplication are considered here. Just duplicate input_hops/output_hops.
  input_hops *= intra_dup;
  output_hops *= intra_dup;
  info.Tile_NoC.resize(2);
  info.Tile_NoC[0] = input_hops;
  info.Tile_NoC[1] = output_hops;
  info.Htree_level = routing_hops;
}

/**
 * @brief Calculate Buffer_read and Buffer_write. Buffer_read: the num of data
 * read from buffer; Buffer_write: the num of data write to buffer
 * 
 * @param input_vec_num: num of input vecs sent to Macros.
 * 
 * @return pair<int, int> first: Buffer_read; second: Buffer_write
 */
pair<int, int> Mapping::Calculate_Buffer(vector<double> Split_tile, int duplication, int input_vec_num) {
  int Buffer_read, Buffer_write;
  if (transform_method == COLUMN_TRANS) {
    Buffer_read = k1 * k2 * input_vec_num * pyInput->shape[1] * pyParam->a_precision * Split_tile[1];
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] * pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS) {
    Buffer_read = k1 * input_vec_num * pyInput->shape[1] * pyParam->a_precision * Split_tile[1];
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] * pyParam->a_precision;
  } else if (transform_method == ROW_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape[1] * pyParam->a_precision * Split_tile[1];
    Buffer_write = 2 * k2 * input_vec_num * outChannels * Split_tile[0] * pyParam->a_precision;
  } else if (transform_method == LW_MATMUL_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape[0] * pyParam->a_precision * Split_tile[1]; // pyInput->shape[0] = input vec dimension
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] * pyParam->a_precision;
  } else if (transform_method == RW_MATMUL_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape.back() * pyParam->a_precision * Split_tile[1]; // pyInput->shape.back() = input vec dimension
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] * pyParam->a_precision;
  } else {
    throw runtime_error("In Calculate_Buffer(), unidentified transform_method found!!!");
  }
  return pair<int, int>(Buffer_read, Buffer_write);
}

int Mapping::Calculate_TileNoC(PartitionInfo info, vector<int> Tile_NoC, int duplication, int input_vec_num) {
  int TileNoC_num = Tile_NoC[0] * info.Intra_tile * info.Inter_tile * info.Split_array[0] * Subarray[0] * input_vec_num / duplication * pyParam->a_precision;
  if (transform_method == COLUMN_TRANS || transform_method == LW_MATMUL_TRANS || transform_method == RW_MATMUL_TRANS) {
    TileNoC_num += Tile_NoC[1] * info.Intra_tile * info.Inter_tile * info.Split_array[1] * Subarray[1] * input_vec_num / duplication * pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS || transform_method == ROW_TRANS) {
    TileNoC_num += Tile_NoC[1] * info.Intra_tile * info.Inter_tile * info.Split_array[1] * Subarray[1] * input_vec_num / duplication / k1 * pyParam->a_precision;
  }
  return TileNoC_num;
}

/**
 * @brief Calculate ip_index. The format of each unit is: [src, dst, packet]
 *
 * @return vector<vector<int>>
 */
vector<vector<int>> Mapping::Mesh_NoC() {
  vector<vector<int>> traffic;
  // When Mesh_NoC is called, "mapping.txt" is fully calculated;
  // "meshconnect.txt" also implies the correlation between layers. We need to
  // replace "prelayer_tile" and "nextlayer_tile" in "meshconnect.txt", which
  // are set to (0,0) currently.

  /*** 1: Parse the Info in mapping.txt in advance ***/
  ifstream f_map(PATH_MAPPING);
  // record the range of virtual tiles used in each layer. Each item in
  // mapping_vTiles follows the format: [tile_start, tile_end].
  vector<vector<int>> mapping_vTiles;
  vector<int> duplication;
  vector<int> tile_rows; // The rows occupied by each layer;
  vector<int> tile_cols; // The columns occupied by each layer;
  string line;
  int cur_tile = 0;

  while (getline(f_map, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    vector<int> Split_tile(2);
    Split_tile[0] = stoi(tokens[3]), Split_tile[1] = stoi(tokens[5]);
    int tile_start = cur_tile, tile_end = cur_tile + Split_tile[0] * Split_tile[1] - 1;
    mapping_vTiles.push_back({tile_start, tile_end});
    duplication.push_back(stoi(tokens[15]));
    tile_rows.push_back(Split_tile[0]);
    tile_cols.push_back(Split_tile[1]);
    cur_tile += Split_tile[0] * Split_tile[1];
  }

  /*** 2: Parse the Info in meshconnect.txt ***/
  ifstream f_mesh(PATH_MESHCONNECT);
  string content;
  while (getline(f_mesh, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    int cur_layer = stoi(tokens[1]), next_layer = stoi(tokens[3]);
    int total_volumn = stoi(tokens[7]);
    int tile_start, tile_end;
    /*** 3: Write prelayer_tile and nextlayer_tile into file ***/
    if (line.find("prelayer_tile: 0 0") != string::npos) {
      tile_start = mapping_vTiles[cur_layer - 1][0];
      tile_end = mapping_vTiles[cur_layer - 1][1];
      line = line.replace(line.find("prelayer_tile: 0 0"), 18, "prelayer_tile: " + to_string(tile_start) + " " + to_string(tile_end));
    }
    if (line.find("nextlayer_tile: 0 0") != string::npos) {
      tile_start = mapping_vTiles[next_layer - 1][0];
      tile_end = mapping_vTiles[next_layer - 1][1];
      line = line.replace(line.find("nextlayer_tile: 0 0"), 19, "nextlayer_tile: " + to_string(tile_start) + " " + to_string(tile_end));
    }
    content += line + '\n';
    /*** 4: Deciding "traffic" for connected layers ***/
    // Partsum and concat operations can be done in routers. We decide that
    // these operations are performed in the "first" router (with the minimum
    // virtual tile id) of the layer to be passed packets. Other routers only
    // need to store and forward packets as normal routers do.
    tile_start = mapping_vTiles[cur_layer - 1][0];
    tile_end = mapping_vTiles[cur_layer - 1][1];
    int first_router_tile = mapping_vTiles[next_layer - 1][0];
    // (i) Packets from cur_layer are sent to the "first" router of next_layer
    for (int i = tile_start; i <= tile_end; i++) {
      // TODO: simplified packet calculation. What if A tile is not totally
      // occupied by unfolded weight?

      // outputs of tiles in a row are concatenated to form the whole output.
      int packet = total_volumn / tile_cols[cur_layer - 1] * pyParam->a_precision;
      traffic.push_back({i, first_router_tile, packet});
    }

    tile_start = mapping_vTiles[next_layer - 1][0];
    tile_end = mapping_vTiles[next_layer - 1][1];
    // (ii) Packets from cur_layer are sent to the "first" router of next_layer
    for (int i = tile_start + 1; i <= tile_end; i++) {
      int packet = total_volumn / tile_rows[next_layer - 1] * pyParam->a_precision;
      traffic.push_back({tile_start, i, packet});
    }
  }
  f_mesh.close();
  ofstream f_mesh_out(PATH_MESHCONNECT);
  f_mesh_out << content;
  f_mesh_out.close();
  ofstream of_traffic(PATH_TRAFFIC);
  for (auto vec : traffic) {
    for (auto id : vec) {
      of_traffic << id << " ";
    }
    of_traffic << endl;
  }
  of_traffic.close();

  // Pass the Tile num
  ofstream f_tile(PATH_TILENUM);
  f_tile << cur_tile;
  f_tile.close();
  return traffic;
}

/**
 * @brief Read MappingInfo from mapping.txt, to be validated.
 *
 */
void Mapping::readMappingInfo(ifstream &f, MappingInfo &mappingInfo, LayerInfo layerInfo) {
  string ll;
  while (getline(f, ll)) {
    istringstream iss(ll);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    int layer = stoi(tokens[1]);
    if (layerInfo.prelayer == layer) {
      mappingInfo.tile_rows[0] = stod(tokens[3]);
      mappingInfo.tile_cols[0] = stod(tokens[5]);
      mappingInfo.Intra_tile[0] = stod(tokens[11]);
      mappingInfo.tile_nums[0] = stod(tokens[13]);
      mappingInfo.duplication[0] = stod(tokens[15]);
    }
    if (layerInfo.nextlayer == layer) {
      mappingInfo.tile_rows[1] = stod(tokens[3]);
      mappingInfo.tile_cols[1] = stod(tokens[5]);
      mappingInfo.Intra_tile[1] = stod(tokens[11]);
      mappingInfo.tile_nums[1] = stod(tokens[13]);
      mappingInfo.duplication[1] = stod(tokens[15]);
    }
  }
}

/**
 * @brief Write performance and tile mapping info into files.
 *
 */
void Mapping::writeInfo(double area, double latency, double energy, string mapOutPath, int total_layer) {
  ofstream of(mapOutPath, ios::app);
  of << setprecision(8);
  ifstream f_mesh(PATH_MESHCONNECT);

  string line;
  of << " total energy (mJ): " << energy * 1000
     << " total latency (ms): " << latency * 1000
     << " total area (mm2): " << area << endl
     << endl;

  of << "##########################  mapping relationship between virtual tiles and physical tiles ##########################" << endl << endl;
  ifstream f_placing(PATH_PLACING);
  while (getline(f_placing, line)) {
    of << line << endl;
  }
  of.close();
}

} // namespace Refactor
