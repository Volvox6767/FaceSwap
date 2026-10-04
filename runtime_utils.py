"""Consistent ONNX Runtime settings, including DirectML requirements."""
from pathlib import Path
import onnxruntime as ort


def available_providers():
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'DmlExecutionProvider' in available:
        from gpu_devices import preferred_dml_device
        return [('DmlExecutionProvider', {'device_id': str(preferred_dml_device())}), 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def session_options(providers=None):
    providers = providers or available_providers()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    if 'DmlExecutionProvider' in names:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def create_session(path: Path, providers=None):
    providers = providers or available_providers()
    if not Path(path).is_file():
        raise FileNotFoundError(f'Model bulunamadı: {path}')
    return ort.InferenceSession(str(path), sess_options=session_options(providers), providers=providers)
