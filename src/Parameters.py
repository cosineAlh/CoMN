import json
import os

from utils import app_path


def layer_connect(filename, prelayer, nextlayer, type, volumn, prelayer_tile=None, nextlayer_tile=None, Dir="Parameters"):
    opt = OptParam(Dir=Dir)
    if opt.get("evaluate_mode", False) is True:
        # Compute volumn if a sequence/array is provided; otherwise cast to int
        try:
            # Treat numpy arrays and lists/tuples uniformly
            if hasattr(volumn, "__iter__") and not isinstance(volumn, (str, bytes)):
                prod = 1
                # Support both 2D and 4D shapes; multiply all provided dims
                for v in volumn:
                    prod *= int(v)
                volumn_val = int(prod)
            else:
                volumn_val = int(volumn)
        except Exception:
            # Fallback: keep original value if unexpected
            volumn_val = volumn

        # Default tiles to (0,0) when not provided
        if prelayer_tile is None:
            prelayer_tile = (0, 0)
        if nextlayer_tile is None:
            nextlayer_tile = (0, 0)

        # Only log valid layer links (aligns with template starting from layer 1)
        if int(prelayer) > 0:
            out_dir = os.path.join(app_path(), Dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{filename}.txt")
            with open(out_path, "a+", encoding="utf-8") as f:
                f.write(
                    "prelayer: {prelayer}\t"
                    "nextlayer: {nextlayer}\t"
                    "type: {type}\t"
                    "volumn: {volumn}\t"
                    "prelayer_tile: {pt0} {pt1} \t"
                    "nextlayer_tile: {nt0} {nt1} \n".format(
                        prelayer=int(prelayer),
                        nextlayer=int(nextlayer),
                        type=str(type),
                        volumn=volumn_val,
                        pt0=int(prelayer_tile[0]),
                        pt1=int(prelayer_tile[1]),
                        nt0=int(nextlayer_tile[0]),
                        nt1=int(nextlayer_tile[1]),
                    )
                )


def loadParam(param, Dir="Parameters"):
    with open(os.path.join(app_path(), Dir,"{}.json").format(param), "r", encoding="utf-8") as f:
        data = json.loads(f.read())
        return data

def loadPerfParam(param):
    with open(os.path.join(app_path(), "Performance/{}.json").format(param), "r", encoding="utf-8") as f:
        data = json.loads(f.read())
        return data

def saveParam(param, udata):
    with open(os.path.join(app_path(), "Performance/{}.json").format(param), "w", encoding="utf-8") as f:
        f.write(json.dumps(udata))

def updateParam(param, paramname, udata, Dir="Parameters"):
    dict_data = loadParam(param,Dir=Dir)
    dict_data[paramname] = udata
    with open(os.path.join(app_path(), Dir,"{}.json").format(param), "w", encoding="utf-8") as f:
        f.write(json.dumps(dict_data))

def SpecParam(Dir="Parameters"):
    Specification = loadParam("SpecParam",Dir=Dir)
    return Specification

def SpecboundParam():
    Specboundaries = loadParam("SpecboundParam")
    return Specboundaries

def NeuronsynpaseParam():
    NeurSynap = loadParam("NeuronsynpaseParam")
    return NeurSynap

def OptParam(Dir="Parameters"):
    Opt = loadParam("OptParam",Dir=Dir)
    return Opt

def DefinedPerformance(energy, latency, area):
    # latency unit is second, energy unit is mJ, area unit is mm2
    Opt = OptParam()
    if Opt["energy"] == True:
        optimizedperformance = energy
    if Opt["latency"] == True:
        optimizedperformance = latency
    if Opt["area"] == True:
        optimizedperformance = area
    if Opt["latency"] == True and Opt["energy"] == True:
        optimizedperformance = energy / latency
    if Opt["area"] == True and Opt["energy"] == True:
        optimizedperformance = energy * area
    if Opt["area"] == True and Opt["latency"] == True:
        optimizedperformance = latency * area
    if Opt["energy"] == True and Opt["area"] == True and Opt["latency"] == True:
        optimizedperformance = latency * area * energy

    return 1 / optimizedperformance
