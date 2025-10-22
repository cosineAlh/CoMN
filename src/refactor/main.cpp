/**
 * @file main.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <string>

#include "Mapping.h"
#include "Mesh_Placing.h"
#include "Perf_Evaluator.h"
#include "PyActInput.h"
#include "PyInput.h"
#include "PyParam.h"
#include "PyWeight.h"
#include "defines.h"
#include "json.hpp"

using namespace Refactor;
using namespace std;
using json = nlohmann::json;

bool do_mapping = false;
bool do_pipeline = false;
bool do_mesh = false;
bool do_PPA = false;

bool parse_arg(int argc, char **argv);

int main(int argc, char **argv) {
  if (!parse_arg(argc, argv)) {
    return 0;
  }

  PyParam *pyParam = new PyParam();
  PyWeight *pyWeight = new PyWeight();
  PyInput *pyInput = new PyInput();
  PyActInput *pyActInput = new PyActInput();

  Mapping *mapping = new Mapping(pyParam, pyWeight, pyInput, pyActInput);
  if (do_pipeline) {
    cout << "=============== Pipeline Opt ===============" << endl;
    mapping->pipeline_optimized();
  }
  if (do_PPA) {
    cout << "=============== PPA ===============" << endl;
    PPA_cost();
  }
  if (do_mesh) {
    cout << "=============== Mesh ===============" << endl;
    // Figure out traffic of NoC first.
    mapping->Mesh_NoC();
    mapping->Mesh_operation();
  }
  if (do_mapping) {
    // cout << "mapping_modules!!!" << endl;
    mapping->mapping_modules();
  }

  free(pyParam);
  free(pyWeight);
  free(pyInput);
  free(pyActInput);
  free(mapping);
  return 0;
}

bool parse_arg(int argc, char **argv) {
  if (argc == 2) {
    if (strcmp(argv[1], "--mapping_modules") == 0) {
      do_mapping = true;
    } else if (strcmp(argv[1], "--pipeline_optimized") == 0) {
      do_pipeline = true;
    } else if (strcmp(argv[1], "--mesh_operation") == 0) {
      do_mesh = true;
    } else if (strcmp(argv[1], "--PPA_cost") == 0) {
      do_PPA = true;
    } else {
      cout << "[CoMN_refactor] Method not supported here!!!" << endl;
      return false;
    }
    return true;
  } else {
    cout << "Arg format is not correct!!!" << endl;
    return false;
  }
}
