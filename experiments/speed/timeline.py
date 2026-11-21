# Where the main process spends a take's wall time when the chains are on workers: start, end and name of its stages.
# Run from the project root: MOOZ_MEMORY_MAX=off GLIBC_TUNABLES=glibc.malloc.hugetlb=1 uv run python experiments/speed/timeline.py <out.json> <scratch>/rec -c <settings> --preset default --no-chain-cache --again
# (the workers start afresh and load this file again: nothing may run at its top level)
def main():
    import sys, time, json
    from mooz import cli, pipeline, effects, cleanup, export, loudness, eq, stages, profile, multitrack, character, trim, dynamics, stereo
    global T0
    T0 = time.perf_counter(); log = []; depth = [0]
    def wrap(mod, name, label=None):
        f = getattr(mod, name)
        def g(*a, **k):
            t0 = time.perf_counter() - T0; depth[0] += 1
            try: return f(*a, **k)
            finally:
                depth[0] -= 1; log.append((round(t0, 2), round(time.perf_counter() - T0, 2), depth[0], label or f"{mod.__name__.split('.')[-1]}.{name}"))
        setattr(mod, name, g)
    for n in ("_tracks", "_drum_bus", "_balance", "_ride", "_finalize", "_consistency", "_characterise", "_condition", "_merged", "separate_take", "_bounds"):
        if hasattr(pipeline, n): wrap(pipeline, n)
    for m, n in ((effects, "hit_trigger"), (cleanup, "buzz_frames"), (cleanup, "buzz_fingerprint"), (cleanup, "detect_hum"), (cleanup, "remove_hum"), (export, "write_wav"), (export, "finish"),
                 (loudness, "summary"), (loudness, "measure"), (loudness, "true_peak_db"), (eq, "band_spectrum"), (eq, "apply_fir"), (dynamics, "multiband"), (dynamics, "compress"), (dynamics, "soft_clip"), (dynamics, "limit"),
                 (effects, "saturator"), (stereo, "mid_side"), (profile, "targets"), (trim, "fade"), (stages.StageWriter, "flush"), (stages.StageWriter, "write"), (stages.StageWriter, "adopt")):
        if hasattr(m, n): wrap(m, n)
    import concurrent.futures as cf
    res = cf.Future.result
    def result(self, *a, **k):
        t0 = time.perf_counter() - T0
        try: return res(self, *a, **k)
        finally:
            if time.perf_counter() - T0 - t0 > 0.05: log.append((round(t0, 2), round(time.perf_counter() - T0, 2), depth[0], "waiting for a worker"))
    cf.Future.result = result
    try: cli.main(sys.argv[2:])
    except SystemExit: pass
    try:
        from mooz import report
    except Exception: pass
    json.dump({"total": time.perf_counter() - T0, "log": sorted(log)}, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
