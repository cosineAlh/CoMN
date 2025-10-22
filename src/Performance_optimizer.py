import sys

from Parameters import *
from Specification_optimizer import *

temp = sys.stdout

def performance_optimizer(mapping_out_path, onnx_model_path):
    optparam = OptParam()
    Specparameters = SpecboundParam()
    if optparam['specification_optimized'] == True:
        perf_root_path = "../generate_data/performance_out"
        perf_path = perf_root_path + "/performance_out.txt"
        if Specparameters['minSubarray'] < 32 or Specparameters['minSubarray'] > 512:
            with open(perf_path, 'w+', encoding="utf-8") as f:
                f.write('Minimum subarray size error ')
                f.write("\n")
            exit()
        if Specparameters['maxSubarray'] < 64 or Specparameters['maxSubarray'] > 1024:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum subarray size error')
                f.write("\n")
            exit()
        if Specparameters['minColumnMUX'] < 1 or Specparameters['minColumnMUX'] > 32:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum ADC numbers error')
                f.write("\n")
            exit()
        if Specparameters['maxColumnMUX'] < 1 or Specparameters['maxColumnMUX'] > 64:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum ADC numbers error')
                f.write("\n")
            exit()
        if Specparameters['minMacronumbers'] < 2 or Specparameters['minMacronumbers'] > 16:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum Macro numbers per Tile error')
                f.write("\n")
            exit()
        if Specparameters['maxMacronumbers'] < 2 or Specparameters['maxMacronumbers'] > 16:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum Macro numbers per Tile error')
                f.write("\n")
            exit()
        if Specparameters['minbuffersizeTile'] < 2 or Specparameters['minbuffersizeTile'] > 128:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum buffer size per Tile error')
                f.write("\n")
            exit()
        if Specparameters['maxbuffersizeTile'] < 2 or Specparameters['maxbuffersizeTile'] > 128:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum buffer size per Tile error')
                f.write("\n")
            exit()
        if Specparameters['minbuswidthTile'] > 128 or Specparameters['minbuswidthTile'] < 8:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum buffer bandwidth error')
                f.write("\n")
            exit()
        if Specparameters['maxbuswidthTile'] > 128 or Specparameters['maxbuswidthTile'] < 8:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum buffer bandwidth error')
                f.write("\n")
            exit()
        Specification_optimizer(mapping_out_path, onnx_model_path)


if __name__ == '__main__':
    onnx_model_path = "/home/anlh/Workspace/3D-Multi_Level_Opt/onnx_model/resnet18.onnx"
    mapping_out_path = "../generate_data/mapping_out/mapping_out.txt"

    performance_optimizer(mapping_out_path, onnx_model_path)