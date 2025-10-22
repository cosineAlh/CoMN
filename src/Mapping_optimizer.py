import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import random

from Parameters import *
from utils import *


def mapping(mapping_out_path="../generate_data/mapping_out/mapping_out.txt", onnx_model_path=None):
    os.makedirs(os.path.dirname(mapping_out_path), exist_ok=True)
    if not onnx_model_path or not os.path.isfile(onnx_model_path):
        with open(mapping_out_path, "w", encoding="utf-8") as f:
            f.write(f"ONNX model not found: {onnx_model_path}\n")
        return 0,0,0
    else:
        with open(mapping_out_path,"w") as f:
            f.write("#######################################################\n\n")
            f.write("####################### Mapping #######################\n\n")
            f.write("#######################################################\n\n")

    # ------ 1. PPA ------
    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --PPA_cost")
    os.chdir(curDir)

    updateParam("OptParam", "prepare_mode", True)

    # Clean previous files
    for fname in ["prepare.txt","placing.txt"]:
        os.system("rm " + os.path.join(app_path(), "Parameters/" + fname))

    batch = np.random.randn(1, 3, 224, 224).astype('float32')
    def _forward_once():
        extract_and_map(onnx_model_path, batch_shape=tuple(batch.shape))
    _forward_once()

    # ------ 2. Pipeline ------
    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --pipeline_optimized")
    os.chdir(curDir)

    updateParam("OptParam", "prepare_mode", False)
    for fname in ["Parameters/mapping.txt", "Parameters/traffic.txt"]:
        os.system("rm " + os.path.join(app_path(), fname))

    layerconnect_path = os.path.join(app_path(), "Parameters/layerconnect.txt")
    if os.path.isfile(layerconnect_path):
        os.system("mv " + layerconnect_path + " " + os.path.join(app_path(), "Parameters/meshconnect.txt"))
    with open(os.path.join(app_path(), "Parameters/tileNum.txt"),"w") as f:
        f.write(str(0))

    _forward_once()

    # ------ 3. Mesh ------
    os.chdir("./refactor/build")
    os.system("./main --mesh_operation")
    os.chdir(curDir)


if __name__ == "__main__":
    onnx_model_path = "/home/anlh/Workspace/3D-Multi_Level_Opt/onnx_model/resnet18.onnx"
    mapping_out_path = "../generate_data/mapping_out/mapping_out.txt"

    updateParam("OptParam", "mapping_finish", False)
    Specparameters = SpecParam()
    do_mapping = True
    if Specparameters["Subarray"][0] > 1024 or Specparameters["Subarray"][1] > 1024:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("The subarray size is too large, 128*128 subarray size is recommended\n")
        do_mapping = False
    if Specparameters["Subarray"][0] < 64 or Specparameters["Subarray"][1] < 64:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("The subarray size is too small, 128*128 subarray size is recommended\n")
        do_mapping = False
    if Specparameters["Tile"][0] > 16 or Specparameters["Tile"][1] > 16:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("There are too many macros per Tile\n")
        do_mapping = False
    if Specparameters["Tile"][0] < 2 or Specparameters["Tile"][1] < 2:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("Not less than 4 macros per Tile are required\n")
        do_mapping = False
    if Specparameters["buffersizeTile"] < 2:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("Not less than 2KB buffer are required per Tile\n")
        do_mapping = False
    if Specparameters["buswidthTile"] > 128 or Specparameters["buswidthTile"] < 8:
        with open(mapping_out_path, "a+", encoding="utf-8") as f:
            f.write("Appropriate buffer bandwidth are required\n")
        do_mapping = False

    if do_mapping:
        mapping(mapping_out_path, onnx_model_path)
    updateParam("OptParam", "mapping_finish", True)
