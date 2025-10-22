import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


# ---------------- In-memory I/O controls ----------------
# Toggle file I/O with env var CIM_NO_FILE_IO=1
_LAYER_COUNTS: Optional[Tuple[int, int]] = None

def app_path():
    """Returns the base application path."""
    curdir = os.getcwd()
    return os.path.dirname(curdir)

def set_layer_counts(conv_layers: int, total_layers: int) -> None:
    global _LAYER_COUNTS
    _LAYER_COUNTS = (int(conv_layers), int(total_layers))

def get_layer_counts():
    """Return cached (conv_layers, total_layers) or None if not set."""
    return _LAYER_COUNTS


# ---------------- ONNX extractor ----------------

def _ensure_dirs():
    base = os.path.join(app_path(), "src/data_transmiss")
    os.makedirs(base, exist_ok=True)
    return base

def _write_pyinput(shape: Tuple[int, ...], a_bits: int, input_sparsity: float = 0.0) -> None:
    base = _ensure_dirs()
    with open(os.path.join(base, "PyInput.txt"), "w", encoding="utf-8") as f:
        f.write("shape: ")
        f.write(",".join(str(int(s)) for s in shape))
        f.write("\n")
        f.write(f"input_sparsity: {float(input_sparsity)}\n")

def _write_pyparam(layer: int, stride: Tuple[int, int], kernel: Tuple[int, int], padding: Tuple[int, int], w_bits: int, a_bits: int) -> None:
    base = _ensure_dirs()
    with open(os.path.join(base, "PyParam.txt"), "w", encoding="utf-8") as f:
        f.write(f"layer: {layer}\n")
        f.write(f"stride: {stride[0]},{stride[1]}\n")
        f.write(f"kernel_size: {kernel[0]},{kernel[1]}\n")
        f.write(f"padding: {padding[0]},{padding[1]}\n")
        f.write(f"w_precision: {w_bits}\n")
        f.write(f"a_precision: {a_bits}\n")

def _write_pyweight(shape: Tuple[int, ...], weights: Optional[np.ndarray], w_bits: int) -> None:
    base = _ensure_dirs()
    with open(os.path.join(base, "PyWeight.txt"), "w", encoding="utf-8") as f:
        f.write("shape: ")
        f.write(",".join(str(int(s)) for s in shape))
        f.write("\n")
        # Approximate sparsity from provided weights if available, else 0.0
        if weights is not None and weights.size > 0:
            scale = float(2 ** w_bits - 1)
            # Match original: mean(abs(W))/scale/2
            weight_sparsity = float(np.mean(np.abs(weights)) / scale / 2.0)
        else:
            weight_sparsity = 0.0
        f.write(f"weight_sparsity: {weight_sparsity}\n")

def _write_pyactinput(shape: Tuple[int, ...], mode: int) -> None:
    base = _ensure_dirs()
    with open(os.path.join(base, "PyActInput.txt"), "w", encoding="utf-8") as f:
        f.write("shape: ")
        f.write(",".join(str(int(s)) for s in shape))
        f.write("\n")
        f.write(f"mode: {int(mode)}\n")

def _collect_shapes(m: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    g = m.graph
    # graph inputs/outputs
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            continue
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                # Unknown dim -> -1
                dims.append(-1)
        shapes[vi.name] = tuple(dims)
    return shapes

def _collect_initializers(m: onnx.ModelProto) -> Dict[str, np.ndarray]:
    inits: Dict[str, np.ndarray] = {}
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        inits[init.name] = arr
    return inits

def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
            if a.type == onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == onnx.AttributeProto.FLOATS:
                return [float(v) for v in a.floats]
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else str(a.s)
    return default

def _as_pair(x: Optional[List[int]], default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    if not x:
        return default
    if len(x) == 1:
        return (int(x[0]), int(x[0]))
    return (int(x[0]), int(x[1]))

def _pads_to_pair(pads: Optional[List[int]]) -> Tuple[int, int]:
    # ONNX pads for 2D: [pad_top, pad_left, pad_bottom, pad_right]
    if not pads or len(pads) < 2:
        return (0, 0)
    return (int(pads[0]), int(pads[1]))

def _run_mapper(cmd: str) -> None:
    cur = os.getcwd()
    try:
        os.chdir("./refactor/build")
        os.system(cmd)
    finally:
        os.chdir(cur)

def _node_label(node, fallback_idx: int) -> str:
    name = getattr(node, 'name', '') or ''
    op = getattr(node, 'op_type', 'Op')
    base = name if name else f"{op}_{fallback_idx}"
    return base

def extract_and_map(onnx_model_path: str, batch_shape: Optional[Tuple[int, int, int, int]] = None, override_a_bits: Optional[int] = None, override_w_bits: Optional[int] = None, call_mapper: bool = True) -> None:
    """
    Iterate ONNX nodes and emit PyInput.txt, PyParam.txt, PyWeight.txt for Conv/Gemm/MatMul.
    If batch_shape is provided (N, C, H, W), it's used for the model input shape when unknown.
    """
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError(onnx_model_path)

    from Parameters import OptParam
    a_bits = int(override_a_bits if override_a_bits is not None else 8)
    w_bits = int(override_w_bits if override_w_bits is not None else 8)

    # If in prepare mode (first pass), start with a clean layerconnect file
    try:
        opt_flags = OptParam()
        if opt_flags.get('prepare_mode', False):
            lc_path = os.path.join(app_path(), 'Parameters', 'layerconnect.txt')
            if os.path.isfile(lc_path):
                try:
                    os.remove(lc_path)
                except OSError:
                    pass
    except Exception:
        pass

    # Load and infer shapes
    model = onnx.load(onnx_model_path)
    print(f"[ONNX-Import] Loading: {onnx_model_path}")

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        # Continue without inferred shapes
        pass

    shapes = _collect_shapes(model)
    inits = _collect_initializers(model)

    # If model input has unknown dims, patch with provided batch_shape for reporting
    if model.graph.input:
        in0 = model.graph.input[0].name
        shp = list(shapes.get(in0, ()))
        if len(shp) == 0 and batch_shape is not None:
            shapes[in0] = batch_shape
        elif batch_shape is not None and len(shp) == 4:
            # only fill unknowns
            patched = []
            for i, v in enumerate(shp):
                patched.append(batch_shape[i] if (v is None or v < 0) else v)
            shapes[in0] = tuple(patched)

    layer_idx = 1
    conv_count = 0
    fc_count = 0

    # Track previous compute layer to create connection records
    prev_compute_idx: Optional[int] = None

    def _sanitize_dims(seq: Tuple[int, ...]) -> Tuple[int, ...]:
        dims: list[int] = []
        for s in seq:
            try:
                v = int(s)
            except Exception:
                continue
            if v > 0:
                dims.append(v)
        return tuple(dims) if dims else (0,)

    def _record_connection(next_idx: int, type_label: str, vol_dims: Tuple[int, ...]):
        # Lazily import to avoid circulars
        try:
            from Parameters import layer_connect as _lc
            pre_idx = prev_compute_idx if prev_compute_idx is not None else 0
            _lc(
                filename='layerconnect',
                prelayer=pre_idx,
                nextlayer=next_idx,
                type=type_label,
                volumn=vol_dims,
                Dir='Parameters',
            )
        except Exception:
            # Non-fatal: logging connection should not break mapping
            pass

    def emit_for_conv(node: onnx.NodeProto):
        nonlocal layer_idx, conv_count, prev_compute_idx
        x_name = node.input[0] if node.input else None
        w_name = node.input[1] if len(node.input) > 1 else None
        W = inits.get(w_name)
        if W is None:
            # No constant weights; still write shape-only
            w_shape = (0, 0, 0, 0)
        else:
            w_shape = tuple(int(s) for s in W.shape)

        strides = _as_pair(_get_attr(node, 'strides'), (1, 1))
        pads = _pads_to_pair(_get_attr(node, 'pads'))
        if len(w_shape) == 4:
            kH, kW = int(w_shape[2]), int(w_shape[3])
        else:
            kH, kW = 0, 0

        x_shape = shapes.get(x_name, (-1, -1, -1, -1))

        _write_pyinput(x_shape, a_bits, input_sparsity=0.0)
        _write_pyparam(layer_idx, strides, (kH, kW), pads, w_bits, a_bits)
        _write_pyweight(w_shape, W, w_bits)

        if call_mapper:
            _run_mapper("./main --mapping_modules")

        # Record layer-to-layer connection (use next layer's type and its input dims)
        curr_idx = layer_idx
        conv_label = f"conv{kH}*{kW}" if (kH > 0 and kW > 0) else "conv"
        _record_connection(curr_idx, conv_label, _sanitize_dims(x_shape))
        prev_compute_idx = curr_idx
        layer_idx += 1
        # Track counts like internal importer
        conv_count += 1

    def emit_for_linear_like(node: onnx.NodeProto, transposeA: int = 0, transposeB: int = 0):
        nonlocal layer_idx, fc_count, prev_compute_idx
        a_name = node.input[0] if node.input else None
        b_name = node.input[1] if len(node.input) > 1 else None
        A = inits.get(a_name)
        B = inits.get(b_name)

        # Determine which one is the weight matrix (constant)
        weight_arr = None
        input_name = a_name
        if B is not None:
            weight_arr = B
            if transposeB:
                weight_arr = weight_arr.T
        elif A is not None:
            weight_arr = A
            input_name = b_name
            if transposeA:
                weight_arr = weight_arr.T

        w_shape = tuple(int(s) for s in weight_arr.shape) if weight_arr is not None else (0, 0)
        x_shape = shapes.get(input_name, (-1, -1))

        # Linear-like params: kernel/padding set to 0, stride 1
        _write_pyinput(x_shape, a_bits, input_sparsity=0.0)
        _write_pyparam(layer_idx, (1, 1), (0, 0), (0, 0), w_bits, a_bits)
        _write_pyweight(w_shape, weight_arr, w_bits)

        if call_mapper:
            _run_mapper("./main --mapping_modules")
        # Record connection for fully-connected like layer
        curr_idx = layer_idx
        _record_connection(curr_idx, "fc", _sanitize_dims(x_shape))
        prev_compute_idx = curr_idx
        layer_idx += 1
        fc_count += 1

    def emit_for_activation(node: onnx.NodeProto, mode: int):
        # Activation/pooling path writes PyActInput and triggers activation mapper.
        x_name = node.input[0] if node.input else None
        x_shape = shapes.get(x_name, tuple())
        _write_pyactinput(x_shape, mode)

    # Walk nodes
    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        name = _node_label(node, idx)
        if 'downsample' in name:
            continue
        if op == 'Conv':
            emit_for_conv(node)
        elif op == 'Gemm':
            tA = int(_get_attr(node, 'transA', 0) or 0)
            tB = int(_get_attr(node, 'transB', 0) or 0)
            emit_for_linear_like(node, transposeA=tA, transposeB=tB)
        elif op == 'MatMul':
            emit_for_linear_like(node)
        elif op == 'Relu':
            emit_for_activation(node, mode=0)
        elif op == 'Sigmoid':
            emit_for_activation(node, mode=2)
        elif op == 'MaxPool':
            emit_for_activation(node, mode=1)
        elif op == 'AveragePool':
            # In project code, Avgpool_S uses mode 1 as well
            emit_for_activation(node, mode=1)
        else:
            # Skip non-mapping ops
            continue

    # Persist or cache layer counts for downstream tools
    total_layers = conv_count + fc_count
    try:
        set_layer_counts(conv_count, total_layers)
    except Exception:
        pass
    params_dir = os.path.join(app_path(), "Parameters")
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, "layer.txt"), "w", encoding="utf-8") as f:
        f.write(str(conv_count) + "\n")
        f.write(str(total_layers) + "\n")
