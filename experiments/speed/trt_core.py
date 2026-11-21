# A separation model's core (band split, transformer blocks, mask estimators: everything between the STFT and its
# inverse) exported to ONNX and built as a TensorRT engine, against the ways PyTorch runs it. One chunk of a real take.
#
#   uv run python experiments/speed/trt_core.py <work folder> torch  [model.ckpt]   the chunk, PyTorch's times, the reference masks
#   uv run python experiments/speed/trt_core.py <work folder> export [model.ckpt]   three ONNX files: float32, float16, autocast
#   uv run python experiments/speed/trt_core.py <work folder> trt    [model.ckpt]   engines built, timed, compared
#   uv run python experiments/speed/trt_core.py <work folder> ort    [model.ckpt]   the same files in ONNX Runtime on the GPU
#   uv run python experiments/speed/trt_core.py <work folder> versus [model.ckpt]   compiled PyTorch and the engines in turn, 1 and 2 chunks
#   uv run python experiments/speed/trt_core.py <work folder> whole  [model.ckpt]   30 s separated with the engine in the model's place
# After the model's name: which exports or engines (float16x2: two chunks at once; float16@5: the builder's level 5).
#
# TensorRT is not a dependency of the project: it is looked for in <work folder>/pkgs
# (uv pip install --python .venv/bin/python --target <work folder>/pkgs tensorrt-cu13 onnxscript), ONNX Runtime's GPU
# build in <work folder>/ort (... --target <work folder>/ort onnxruntime-gpu).
# Nothing is written outside the work folder; the take is only read.
import logging, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

TAKE = "new_tracks/drive-download-20261107T141948Z-1-001/261105_235855 Biancaneve"
SR = 44100


def db(a, ref):  # how far a is from ref, dB re ref
    a, ref = a.float(), ref.float()
    return round(float(10 * np.log10(float(((a - ref) ** 2).mean()) / (float((ref ** 2).mean()) + 1e-30) + 1e-30)), 1)


def separator(work, model, **flags):
    from audio_separator.separator import Separator
    from mooz import config, separate
    cfg = config.load(None)["stems"]
    (work / "out").mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=logging.ERROR, model_file_dir=str(separate.MODEL_DIR), output_dir=str(work / "out"), output_format="WAV",
                    use_soundfile=True, sample_rate=SR, normalization_threshold=1.0, amplification_threshold=0.0, **flags)
    sep.arch_specific_params["MDXC"].update(segment_size=256, override_model_segment_size=False, pitch_shift=0)
    separate._load(sep, model, cfg.get("custom_models", {}))
    return sep


def piece(work):
    from mooz import separate
    p = work / "piece.wav"
    if not p.exists():
        src = next(Path(TAKE).glob("*TrMic*"))
        x, fs = sf.read(src, dtype="float32", always_2d=True, start=100 * 48000, stop=130 * 48000)
        sf.write(p, separate._resample(x, fs, SR).astype(np.float32), SR, subtype="FLOAT")
    return p


def core_of(model):
    """The model between its STFT and its inverse STFT, as a module of real numbers: (b, t, f*c) -> (b, stems, t, f*c)."""
    import torch

    class Core(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            x = m.band_split(x)
            for block in m.layers:
                assert len(block) == 2, "a linear transformer in the block: not written here"
                time_transformer, freq_transformer = block
                b, t, f, d = x.shape
                x = x.transpose(1, 2).reshape(b * f, t, d)
                x = time_transformer(x)
                x = x.reshape(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
                x = freq_transformer(x)
                x = x.reshape(b, t, f, d)
            x = m.final_norm(x)
            return torch.stack([fn(x) for fn in m.mask_estimators], dim=1)

    return Core(model)


def timed(fn, n=8, warm=2):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    took = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); took.append(time.perf_counter() - t0)
    return float(np.median(took))


def torch_part(work, model):
    """The chunk the model's core gets, how much of a separation the core is, and PyTorch's ways of running it."""
    import torch
    sep = separator(work, model, use_autocast=True)
    net = sep.model_instance.model_run
    seen, clock = {}, {"core": 0.0, "n": 0}

    def before(mod, args):
        seen.setdefault("x", args[0].detach().float().clone())
        torch.cuda.synchronize(); clock["t"] = time.perf_counter()

    def after(mod, args, out):
        torch.cuda.synchronize(); clock["core"] += time.perf_counter() - clock["t"]; clock["n"] += 1

    hooks = [net.band_split.register_forward_pre_hook(before), net.mask_estimators[-1].register_forward_hook(after)]
    sep.separate(str(piece(work)))  # (once to warm up, then the one that counts)
    clock.update(core=0.0, n=0)
    torch.cuda.synchronize(); t0 = time.perf_counter(); sep.separate(str(piece(work))); torch.cuda.synchronize(); whole = time.perf_counter() - t0
    for h in hooks:
        h.remove()
    x = seen["x"]
    inst = sep.model_instance
    print(f"{model}: 30 s in {whole:.2f} s, of which the core {clock['core']:.2f} s in {clock['n']} chunks ({100 * clock['core'] / whole:.0f} %); "
          f"a chunk is {tuple(x.shape)}, overlap {inst.overlap}, {sum(p.numel() for p in net.parameters()) / 1e6:.0f} M weights, "
          f"{len(net.layers)} blocks, {len(net.mask_estimators)} stems", flush=True)
    core = core_of(net).eval()
    out = {}
    with torch.no_grad():
        ref = core(x); out["float32"] = (timed(lambda: core(x)), ref)
        with torch.autocast("cuda", dtype=torch.float16):
            out["autocast float16 (before 20 November)"] = (timed(lambda: core(x)), core(x).float())
            for b in (2, 4):
                xb = x.repeat(b, 1, 1)
                try:
                    out[f"autocast float16, {b} chunks at once"] = (timed(lambda: core(xb)) / b, core(xb)[:1].float())
                except torch.OutOfMemoryError:
                    print(f"   {b} chunks at once: out of GPU memory")
    torch.save({"x": x.cpu(), "ref": ref.cpu(), "auto": out["autocast float16 (before 20 November)"][1].cpu()}, work / "chunk.pt")
    del sep, net, core, inst; torch.cuda.empty_cache()
    sep = separator(work, model, use_autocast=True, use_torch_compile=True)
    core = core_of(sep.model_instance.model_run).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        t0 = time.perf_counter(); core(x); torch.cuda.synchronize(); first = time.perf_counter() - t0
        out["autocast float16, blocks compiled (now)"] = (timed(lambda: core(x)), core(x).float())
        for b in (2, 4):
            xb = x.repeat(b, 1, 1)
            try:
                out[f"compiled, {b} chunks at once"] = (timed(lambda: core(xb)) / b, core(xb)[:1].float())
            except Exception as e:
                print(f"   compiled, {b} chunks at once: {type(e).__name__}")
    for name, (t, y) in out.items():
        print(f"   {name:42} {1000 * t:7.1f} ms a chunk" + ("" if name == "float32" else f" | from float32 {db(y, ref)} dB"), flush=True)
    print(f"   (the first compiled call: {first:.1f} s)")


def export_part(work, model, *only):
    import torch
    d = torch.load(work / "chunk.pt", weights_only=True); x = d["x"].cuda()
    sep = separator(work, model, use_autocast=False)
    core = core_of(sep.model_instance.model_run).eval()

    class Autocast(torch.nn.Module):  # float16 where PyTorch's autocast puts it, float32 for the rest (the norms)
        def __init__(self, c):
            super().__init__(); self.c = c

        def forward(self, x):
            with torch.autocast("cuda", dtype=torch.float16):
                return self.c(x).float()

    import copy
    half = copy.deepcopy(core).half()
    ways = (("float32", core, x), ("autocast", Autocast(core), x), ("float16", half, x.half()), ("float16x2", half, x.half().repeat(2, 1, 1)),
            ("bfloat16", copy.deepcopy(core).bfloat16() if "bfloat16" in only else None, x.bfloat16()))
    for name, mod, inp in [w for w in ways if not only or w[0] in only]:
        f = work / f"core_{name}.onnx"
        for old in work.glob(f"core_{name}.onnx*"):
            old.unlink()
        for dynamo in (True, False):
            try:
                t0 = time.time()
                with torch.no_grad():
                    torch.onnx.export(mod, (inp,), str(f), input_names=["x"], output_names=["mask"], dynamo=dynamo,
                                      **({"external_data": True} if dynamo else {"opset_version": 18}))
                size = sum(p.stat().st_size for p in work.glob(f"core_{name}.onnx*")) / 2**20
                print(f"   {name}: exported ({'new' if dynamo else 'old'} exporter) in {time.time() - t0:.0f} s, {size:.0f} MB", flush=True)
                break
            except Exception as e:
                print(f"   {name}: the {'new' if dynamo else 'old'} exporter failed: {type(e).__name__}: {str(e)[:300]}", flush=True)


def engine_of(work, name, trt, log, level=None):
    plan = work / (f"core_{name}.plan" if level is None else f"core_{name}@{level}.plan")
    if not plan.exists():
        builder = trt.Builder(log)
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(net, log)
        if not parser.parse_from_file(str(work / f"core_{name}.onnx")):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))[:600])
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)
        if level is not None:
            cfg.builder_optimization_level = level  # 3 unless told; 5 looks longest for the fastest kernels
        t0 = time.time()
        made = builder.build_serialized_network(net, cfg)
        if made is None:
            raise RuntimeError("the engine was not built")
        plan.write_bytes(bytes(made))
        print(f"   {plan.stem}: engine built in {time.time() - t0:.0f} s, {plan.stat().st_size / 2**20:.0f} MB", flush=True)
    t0 = time.time()
    engine = trt.Runtime(log).deserialize_cuda_engine(plan.read_bytes())
    return engine, time.time() - t0


class Engine:
    """A built engine called with a PyTorch tensor on the GPU, answering with one."""

    def __init__(self, engine, trt):
        import torch
        self.engine, self.ctx = engine, engine.create_execution_context()
        kinds = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.BF16: torch.bfloat16}
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.kind = kinds[engine.get_tensor_dtype(self.inp)]
        self.out = torch.empty(tuple(engine.get_tensor_shape(self.outp)), dtype=kinds[engine.get_tensor_dtype(self.outp)], device="cuda")

    def __call__(self, x):
        import torch
        x = x.to(self.kind).contiguous()
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.outp, self.out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("the engine did not run")
        return self.out


def trt_part(work, model, *only):
    import torch
    sys.path.append(str(work / "pkgs"))
    import tensorrt as trt
    log = trt.Logger(trt.Logger.ERROR)
    d = torch.load(work / "chunk.pt", weights_only=True); x, ref, auto = d["x"].cuda(), d["ref"].cuda(), d["auto"].cuda()
    print(f"TensorRT {trt.__version__}; PyTorch's autocast float16 is {db(auto, ref)} dB from float32 on this chunk")
    for name in only or ("float32", "autocast", "float16"):
        name, _, level = name.partition("@")
        if not (work / f"core_{name}.onnx").exists():
            print(f"   {name}: no ONNX file"); continue
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            engine, load = engine_of(work, name, trt, log, int(level) if level else None)
            run = Engine(engine, trt)
            n = 2 if name.endswith("x2") else 1
            xs = x.repeat(n, 1, 1)
            t = timed(lambda: run(xs)) / n
            y = run(xs)[:1].float().clone(); torch.cuda.synchronize()
            name = name + ("@" + level if level else "")
            bad = int((~torch.isfinite(y)).sum())
            print(f"   {name:10} {1000 * t:7.1f} ms a chunk | from float32 {db(y, ref)} dB" + (f" ({bad} values not finite)" if bad else "")
                  + f" | engine's GPU memory {engine.device_memory_size_v2 / 2**30:.1f} GB | loaded in {load:.1f} s", flush=True)
            del run, engine
        except Exception as e:
            print(f"   {name}: {type(e).__name__}: {str(e)[:500]}", flush=True)


def versus_part(work, model, rounds=4):
    """PyTorch's compiled blocks and the float16 engines in turn in one process, one and two chunks at once:
    single timings on this laptop move by a tenth with the card's temperature."""
    import torch
    sys.path.append(str(work / "pkgs"))
    import tensorrt as trt
    x = torch.load(work / "chunk.pt", weights_only=True)["x"].cuda(); x2 = x.repeat(2, 1, 1)
    sep = separator(work, model, use_autocast=True, use_torch_compile=True)
    core = core_of(sep.model_instance.model_run).eval()
    log = trt.Logger(trt.Logger.ERROR)
    one, two = Engine(engine_of(work, "float16", trt, log)[0], trt), Engine(engine_of(work, "float16x2", trt, log)[0], trt)

    def compiled(v):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return core(v)

    ways = {"PyTorch, blocks compiled, a chunk at a time (now)": (lambda: compiled(x), 1), "PyTorch, blocks compiled, two chunks at once": (lambda: compiled(x2), 2),
            "TensorRT float16, a chunk at a time": (lambda: one(x), 1), "TensorRT float16, two chunks at once": (lambda: two(x2), 2)}
    for fn, _ in ways.values():
        fn(); fn()
    took = {k: [] for k in ways}
    for _ in range(int(rounds)):
        for k, (fn, n) in ways.items():
            took[k].append(timed(fn, n=5, warm=1) / n)
    base = float(np.median(took["PyTorch, blocks compiled, a chunk at a time (now)"]))
    for k, v in took.items():
        print(f"   {k:52} {1000 * np.median(v):6.1f} ms a chunk ({1000 * min(v):.0f} to {1000 * max(v):.0f}) | {np.median(v) / base:.2f} of now", flush=True)


def ort_part(work, model, *only):
    """The same ONNX files run by ONNX Runtime's CUDA provider (its GPU build in <work folder>/ort), tensors staying on the GPU."""
    import torch  # first: it brings the CUDA libraries the provider needs
    sys.path.insert(0, str(work / "ort"))
    import onnxruntime as ort
    d = torch.load(work / "chunk.pt", weights_only=True); x, ref = d["x"].cuda(), d["ref"].cuda()
    print(f"ONNX Runtime {ort.__version__}, providers {ort.get_available_providers()}")
    for name in only or ("float32", "autocast", "float16"):
        try:
            torch.cuda.empty_cache()
            opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; opts.log_severity_level = 3
            whole = work / f"core_{name}_one_file.onnx"  # (the exporter puts every constant in a side file; ONNX Runtime wants the small ones inside)
            if not whole.exists():
                import onnx
                onnx.save(onnx.load(str(work / f"core_{name}.onnx")), str(whole))
            t0 = time.time()
            sess = ort.InferenceSession(str(whole), opts, providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})])
            load = time.time() - t0
            if sess.get_providers()[0] != "CUDAExecutionProvider":
                print(f"   {name}: the GPU provider did not load ({sess.get_providers()})"); continue
            if "bfloat16" in sess.get_inputs()[0].type:
                raise NotImplementedError("the session was made, but its tensors are bfloat16 and the binding here is not written for it")
            half = sess.get_inputs()[0].type == "tensor(float16)"
            n = 2 if name.endswith("x2") else 1
            xs = (x.half() if half else x).repeat(n, 1, 1).contiguous()
            out_half = sess.get_outputs()[0].type == "tensor(float16)"
            y = torch.empty((n, *ref.shape[1:]), dtype=torch.float16 if out_half else torch.float32, device="cuda")
            bind = sess.io_binding()
            bind.bind_input("x", "cuda", 0, np.float16 if half else np.float32, tuple(xs.shape), xs.data_ptr())
            bind.bind_output("mask", "cuda", 0, np.float16 if out_half else np.float32, tuple(y.shape), y.data_ptr())
            t = timed(lambda: sess.run_with_iobinding(bind)) / n
            print(f"   {name:10} {1000 * t:7.1f} ms a chunk | from float32 {db(y[:1], ref)} dB | session made in {load:.0f} s", flush=True)
            del sess, bind
        except Exception as e:
            print(f"   {name}: {type(e).__name__}: {str(e)[:500]}", flush=True)


def whole_part(work, model, name="autocast"):
    """30 s separated by the library with the engine in the place of the model's core: the time and the stems."""
    import torch
    sys.path.append(str(work / "pkgs"))
    import tensorrt as trt
    from mooz import separate

    def stems(sep, runs=2):
        took = []
        for _ in range(runs):
            for old in (work / "out").glob("*"):
                old.unlink()
            torch.cuda.synchronize(); t0 = time.perf_counter(); files = sep.separate(str(piece(work))); torch.cuda.synchronize()
            took.append(time.perf_counter() - t0)
        outs = {}
        for f in files:
            f = Path(f) if Path(f).is_absolute() else work / "out" / f
            m = separate._STEM_IN_NAME.search(f.name)
            outs[(m.group(1) if m else f.stem).lower()] = torch.from_numpy(sf.read(f, dtype="float32", always_2d=True)[0])
        return took[-1], outs

    sep = separator(work, model, use_autocast=False)
    t32, ref = stems(sep, 1)
    del sep; torch.cuda.empty_cache()
    sep = separator(work, model, use_autocast=True, use_torch_compile=True)
    t_now, now = stems(sep)
    del sep; torch.cuda.empty_cache()
    sep = separator(work, model, use_autocast=True)
    net = sep.model_instance.model_run
    engine, _ = engine_of(work, name, trt, trt.Logger(trt.Logger.ERROR))
    run = Engine(engine, trt)

    class InPlace(torch.nn.Module):  # stands where the band split stood and gives the masks; the rest of the core is emptied
        def __init__(self):
            super().__init__(); self.kind = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x):
            return run(x.float()).float().clone()

    class Stem(torch.nn.Module):
        def __init__(self, i):
            super().__init__(); self.i = i

        def forward(self, masks):
            return masks[:, self.i]

    n = len(net.mask_estimators)
    net.band_split, net.layers, net.final_norm = InPlace().cuda(), torch.nn.ModuleList([]), torch.nn.Identity()
    net.mask_estimators = torch.nn.ModuleList([Stem(i) for i in range(n)])
    torch.cuda.empty_cache()
    t_trt, got = stems(sep)
    loud = [k for k in ref if float((ref[k] ** 2).mean()) > 1e-9]
    print(f"{model}, 30 s of the take, whole separations:")
    print(f"   float32                         {t32:5.2f} s")
    print(f"   compiled blocks (now)           {t_now:5.2f} s | from float32, dB re each stem: { {k: db(now[k], ref[k]) for k in loud} }")
    print(f"   TensorRT engine ({name:8})      {t_trt:5.2f} s | from float32, dB re each stem: { {k: db(got[k], ref[k]) for k in loud} }")
    print(f"   the engine against now's stems: { {k: db(got[k], now[k]) for k in loud} }")


if __name__ == "__main__":
    work, part = Path(sys.argv[1]), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "BS-Roformer-SW.ckpt"
    {"torch": torch_part, "export": export_part, "trt": trt_part, "ort": ort_part, "versus": versus_part, "whole": whole_part}[part](work, model, *sys.argv[4:])
