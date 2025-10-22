/**
 * @file Perf_Evaluator.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <ctime>
#include <chrono>

#include "Perf_Evaluator.h"
#include "defines.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

namespace Refactor {

void PPA_cost() {
  int techNode = 32;
  
  HISIM();

  // For NoC_mesh
  // NOC_Perf(5, 5, 4, 1e9, techNode);
  // PE_Perf();
}

void HISIM() {
  auto curDir = get_current_dir_name();
  chdir(PATH_3D_DIR);
  system("make sim_no_thermal");
  //system("make sim");
  chdir(curDir);
}

double PE_Perf() {
  ifstream f_perf(PATH_HSIM_OUT);

  double energy = 0, area = 0, latency = 0;
  int tier_num = 0,tiles_num = 0, pe_num = 0;
  string header, line;
  getline(f_perf, header); // Read the header line
  vector<string> headers;
  istringstream headerStream(header);
  string headerItem;
  while (getline(headerStream, headerItem, ',')) {
    headers.push_back(headerItem);
  }

  while (getline(f_perf, line)) {
    istringstream iss(line);
    string value;
    vector<string> values;
    while (getline(iss, value, ',')) {
      values.push_back(value);
    }
    for (size_t i = 0; i < headers.size(); ++i) {
      if (headers[i] == "Real Tiers") {
        tier_num = stoi(values[i]);
      } else if (headers[i] == "Tiles per Tier") {
        tiles_num = stoi(values[i]);
      } else if (headers[i] == "PEs per Tile") {
        pe_num = stoi(values[i]);
      } else if (headers[i] == "Total Computing Latency (ns)") {
        latency = stod(values[i])*1e-9/tier_num/tiles_num/pe_num;
      } else if (headers[i] == "Total Dynamic Energy (pJ)") {
        energy += stod(values[i])*1e-12/tier_num/tiles_num/pe_num;
      } else if (headers[i] == "Total 3d stack Area (mm^2)") {
        area = stod(values[i])/tiles_num/pe_num;
      }
    }
  }

  // json MacroPerf;
  // MacroPerf["energy"] = energy;
  // MacroPerf["area"] = area;
  // MacroPerf["latency"] = latency;
  // ofstream of_mesh(PATH_MACRO_PERF);
  // of_mesh << setw(2) << MacroPerf << endl;
  return energy, area, latency;
}

double NOC_Perf(int inPorts, int outPorts, int v_channels, double freq, int featureSize) {
  string line;
  auto curDir = get_current_dir_name();
  ifstream f_noc(PATH_HSIM_OUT);

  double energy = 0, area = 0, latency = 0;
  // read f_noc (csv file), find energy="Total NoC Power (W)" and area="Total NoC Area (mm^2)"
  string header;
  getline(f_noc, header); // Read the header line
  vector<string> headers;
  istringstream headerStream(header);
  string headerItem;
  while (getline(headerStream, headerItem, ',')) {
    headers.push_back(headerItem);
  }

  while (getline(f_noc, line)) {
    istringstream iss(line);
    string value;
    vector<string> values;
    while (getline(iss, value, ',')) {
      values.push_back(value);
    }
    for (size_t i = 0; i < headers.size(); ++i) {
      if (headers[i] == "Total NoC Power (W)") {
        energy = stod(values[i]) / 8 / freq / (inPorts + outPorts) * featureSize / 65;
      } else if (headers[i] == "Total NoC Area (mm^2)") {
        area = stod(values[i]) * featureSize / 65 * featureSize / 65;
      }
    }
  }

  latency = 1.0 / freq / v_channels * featureSize / 65 * featureSize / 65;
  // json MeshPerf;
  // MeshPerf["energy"] = energy;
  // MeshPerf["area"] = area;
  // MeshPerf["latency"] = 1 / freq / v_channels * featureSize / 65 * featureSize / 65;
  // ofstream of_mesh(PATH_MESH_PERF);
  // of_mesh << setw(2) << MeshPerf << endl;

  return energy, area, latency;
}
} // namespace Refactor
