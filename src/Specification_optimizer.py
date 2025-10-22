####################################################################################
####### Scenario 2: Optimizing IMC chip specification for multiple DNN models ######
####################################################################################

import numpy as np
import math
import sys
import json
import re
import os

from Parameters import *
from utils import app_path
from Mapping_optimizer import *
from Bayesian.bayesian_optimization import BayesianOptimization
temp = sys.stdout


def specification(Subarray, Macronumbers, buswidthTile, buffersizeTile, ColumnMUX, Meshflitband):
    global Spec_mapping_out_path, Spec_onnx_path
    optparam = OptParam()
    arrayrow = int(math.pow(2, round(math.log2(Subarray))))
    arraycol = int(arrayrow)
    Subarray = [arrayrow, arraycol]

    buffersizeTile =  int(math.pow(2, round(math.log2(buffersizeTile))))
    buswidthTile =  int(math.pow(2, round(math.log2(buswidthTile))))
    ColumnMUX =  int(math.pow(2, round(math.log2(ColumnMUX))))
    Meshflitband =  int(math.pow(2, round(math.log2(Meshflitband))))

    Set_Htreenums = [1, 4, 8, 16, 32, 64, 128, 256]
    Htreenums = Set_Htreenums
    Set_Htreesize = [[1, 1], [2, 2], [2, 4], [4, 4], [4, 8], [8, 8], [8, 16], [16, 16]]
    Set_Routernum = [0, 1, 3, 5, 11, 21, 43, 85]
    Htreenums.append(Macronumbers)
    macronums = sorted(Htreenums)
    Macronums = macronums.index(Macronumbers)
    Macronumbers = Set_Htreenums[Macronums]
    RouternumperTile = Set_Routernum[Macronums]
    Tile = Set_Htreesize[Macronums]
    
    updateParam('SpecParam','Subarray',Subarray)
    updateParam('SpecParam','Tile',Tile)
    updateParam('SpecParam', 'buswidthTile', buswidthTile)
    updateParam('SpecParam', 'buffersizeTile', buffersizeTile)
    updateParam('SpecParam', 'ColumnMUX', ColumnMUX)
    updateParam('SpecParam', 'MeshNoC_flitband', Meshflitband)
    
    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --PPA_cost")
    os.chdir(curDir)

    mapping(Spec_mapping_out_path, Spec_onnx_path)

    # Parse energy (mJ), latency (ms), and area (mm2) from mapping_out.txt
    # Convert energy to J and latency to s to keep internal units consistent
    map_path = Spec_mapping_out_path
    # Accept either a directory that contains mapping_out.txt or a direct file path
    map_file = os.path.join(map_path, "mapping_out.txt") if os.path.isdir(map_path) else map_path

    if not os.path.isfile(map_file):
        raise FileNotFoundError(f"mapping_out.txt not found at: {map_file}")

    pattern = re.compile(r"total energy \(mJ\):\s*([0-9eE+\-.]+)\s+total latency \(ms\):\s*([0-9eE+\-.]+)\s+total area \(mm2\):\s*([0-9eE+\-.]+)")
    energy = latency = area = None
    with open(map_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                energy_mj = float(m.group(1))
                latency_ms = float(m.group(2))
                area_mm2 = float(m.group(3))
                energy = energy_mj / 1000.0  # J
                latency = latency_ms / 1000.0  # s
                area = area_mm2  # mm^2
                break

    if energy is None or latency is None or area is None:
        raise ValueError("Failed to parse energy/latency/area from mapping_out.txt")
    if optparam['specification_optimized'] == True:
        perf_root_path = "../generate_data/performance_out"
        perf_path = perf_root_path + "/performance_out.txt"
        with open(perf_path,"a+") as f:
            f.write("\n")
            f.write(f"total energy (mJ): \t{energy * 1000}\t total latency (ms): \t{latency * 1000}\t total area (mm2): \t{area}\n")
            f.write("\n")

    output = DefinedPerformance(energy, latency, area)
    return output


def Specification_optimizer(mapping_out_path, onnx_model_path):
    global Spec_mapping_out_path, Spec_onnx_path
    Spec_mapping_out_path = mapping_out_path
    Spec_onnx_path = onnx_model_path

    verbose = 2
    optparam = OptParam()

    perf_root_path = "../generate_data/performance_out"
    perf_path = perf_root_path + "/performance_out.txt"
    if optparam['specification_optimized'] == True:
        with open(perf_path, "w+") as f:
            f.write("#######################################################\n\n")
            f.write("##################### Optimizing ######################\n\n")
            f.write("#######################################################\n\n")

    verbose = 2
    Specboundaries = SpecboundParam()
    pbounds = {'Subarray': (Specboundaries['minSubarray'], Specboundaries['maxSubarray']),
               'Macronumbers': (Specboundaries['minMacronumbers'], Specboundaries['maxMacronumbers']),
               'buswidthTile': (Specboundaries['minbuswidthTile'], Specboundaries['maxbuswidthTile']),
               'buffersizeTile': (Specboundaries['minbuffersizeTile'], Specboundaries['maxbuffersizeTile']),
               'ColumnMUX': (Specboundaries['minColumnMUX'], Specboundaries['maxColumnMUX']),
               'Meshflitband': (Specboundaries['minmeshflitband'], Specboundaries['maxmeshflitband'])}

    optimizer = BayesianOptimization(f=specification, pbounds=pbounds, random_state=1, verbose=verbose, path=perf_path)
    optimizer.maximize(init_points=optparam['init_points'], n_iter=optparam['search_iters'])

    arrayrow = int(math.pow(2, round(math.log2(optimizer.max['params']['Subarray']))))
    arraycol = int(arrayrow)
    Subarray = [arrayrow, arraycol]
    buffersizeTile = int(math.pow(2, round(math.log2(optimizer.max['params']['buffersizeTile']))))
    buswidthTile = int(math.pow(2, round(math.log2(optimizer.max['params']['buswidthTile']))))
    ColumnMUX = int(math.pow(2, round(math.log2(optimizer.max['params']['ColumnMUX']))))
    Set_Htreenums = [1, 4, 8, 16, 32, 64, 128, 256]
    Set_Htreesize = [[1, 1], [2, 2], [2, 4], [4, 4], [4, 8], [8, 8], [8, 16], [16, 16]]
    Set_Htreenums.append(optimizer.max['params']['Macronumbers'])
    macronums = sorted(Set_Htreenums)
    Macronums = macronums.index(optimizer.max['params']['Macronumbers'])
    Macronumbers = Set_Htreenums[Macronums]
    Tile = Set_Htreesize[Macronums]
    meshflitband = int(math.pow(2,round(math.log2(optimizer.max['params']['Meshflitband']))))

    if optparam['specification_optimized'] == True:
        with open(perf_path, "a+") as f:
            f.write("IMC specifications are shown as follows\n\n")
            f.write(f"Subarray size: {Subarray}\n")
            f.write(f"Tile size: {Tile}\n")
            f.write(f"ADC numbers per Macro: {ColumnMUX}\n")
            f.write(f"Buswidth of Tile buffer (B): {buswidthTile}\n")
            f.write(f"Buffer size per Tile (KB): {buffersizeTile}\n")
            f.write(f"the sysytem performance: {optimizer.max['target']}\n\n")

    return optimizer.max['target']

