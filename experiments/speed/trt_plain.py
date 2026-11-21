# A float16 engine built and timed with no PyTorch in the process, to try a TensorRT made for another CUDA than the
# project's (trt_core.py made the ONNX file and the chunk: <work folder>/core_float16.onnx, x16.npy, ref.npy).
#   <a python with tensorrt, cuda-python and numpy> experiments/speed/trt_plain.py <work folder> [name of the export]
import sys, time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as rt


def ok(r):
    assert int(r[0]) == 0, r[0]
    return r[1] if len(r) == 2 else r[1:]


if __name__ == "__main__":
    work, name = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "float16")
    log = trt.Logger(trt.Logger.ERROR)
    cuda = [line.split()[-1] for line in open("/proc/self/maps") if "libcudart" in line or "libcublas.so" in line]
    plan = work / f"plain_{name}.plan"
    if not plan.exists():
        builder = trt.Builder(log)
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(net, log)
        assert parser.parse_from_file(str(work / f"core_{name}.onnx")), [str(parser.get_error(i)) for i in range(parser.num_errors)]
        cfg = builder.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)
        t0 = time.time(); plan.write_bytes(bytes(builder.build_serialized_network(net, cfg)))
        print(f"engine built in {time.time() - t0:.0f} s", flush=True)
    engine = trt.Runtime(log).deserialize_cuda_engine(plan.read_bytes()); ctx = engine.create_execution_context()
    cuda = sorted({Path(line.split()[-1]).name for line in open("/proc/self/maps") if "libcudart" in line or "libcublas.so" in line or "libnvinfer.so" in line})
    x = np.ascontiguousarray(np.load(work / "x16.npy")); ref = np.load(work / "ref.npy")
    out = np.empty(tuple(engine.get_tensor_shape("mask")), np.float16)
    dx, dy = ok(rt.cudaMalloc(x.nbytes)), ok(rt.cudaMalloc(out.nbytes))
    ok(rt.cudaMemcpy(dx, x.ctypes.data, x.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice))
    took = []
    for i in range(12):
        t0 = time.perf_counter(); assert ctx.execute_v2([int(dx), int(dy)]); ok(rt.cudaDeviceSynchronize()); took.append(time.perf_counter() - t0)
    ok(rt.cudaMemcpy(out.ctypes.data, dy, out.nbytes, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    y = out.astype(np.float32)
    print(f"TensorRT {trt.__version__} with {', '.join(cuda)}: {name} {1000 * np.median(took[2:]):.1f} ms a chunk "
          f"({1000 * min(took[2:]):.0f} to {1000 * max(took[2:]):.0f}) | from float32 {10 * np.log10(((y - ref) ** 2).mean() / (ref ** 2).mean()):.1f} dB")
