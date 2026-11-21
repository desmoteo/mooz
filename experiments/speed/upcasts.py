# With [audio] float = 32: which functions are given float32 audio and hand back float64 audio, or are given float64
# audio at all (something before them widened it). "Audio" is a 2-D array of samples x 1 or 2 channels. One chain
# after the other, so that everything runs in this process. Run from the project root:
#   MOOZ_MEMORY_MAX=off MOOZ_WORKERS=1 uv run python experiments/speed/upcasts.py <scratch>/rec -c <settings> --preset default --no-chain-cache --again
# (the workers start afresh and load this file again: nothing may run at its top level)
import functools, inspect, sys
from collections import Counter


def main():
    import numpy as np
    from mooz import augment, bed, cleanup, cli, dynamics, effects, eq, loudness, note_eq, pipeline, plugins, space, stereo, trim, export
    seen, given = Counter(), Counter()
    audio = lambda a: isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] <= 2 and a.shape[0] >= 1024  # noqa: E731
    def kinds(v):
        if audio(v): return {str(v.dtype)}
        if isinstance(v, tuple): return set().union(*(kinds(u) for u in v)) if v else set()  # (not lists: the drum samples)
        return set()
    def wrap(mod, name, f):
        @functools.wraps(f)
        def g(*a, **k):
            ins = set().union(*(kinds(v) for v in list(a) + list(k.values()))) if a or k else set()
            r = f(*a, **k)
            outs = kinds(r)
            where = f"{mod.__name__.split('.')[-1]}.{name}"
            if "float64" in ins: given[where] += 1
            if "float32" in ins and "float64" in outs: seen[where] += 1
            return r
        setattr(mod, name, g)
    for mod in (augment, bed, cleanup, dynamics, effects, eq, loudness, note_eq, pipeline, plugins, space, stereo, trim, export):
        for name, f in list(vars(mod).items()):
            if inspect.isfunction(f) and f.__module__ == mod.__name__ and not inspect.isgeneratorfunction(f) and name not in ("main",):
                wrap(mod, name, f)
    try: cli.main(sys.argv[1:])
    except SystemExit: pass
    print("\ngiven float32 audio, hands back float64 audio:", file=sys.stderr)
    for k, v in seen.most_common(): print(f"   {v:4d}x {k}", file=sys.stderr)
    print("given float64 audio (after the stems' stage begins, something widened it):", file=sys.stderr)
    for k, v in given.most_common(): print(f"   {v:4d}x {k}", file=sys.stderr)


if __name__ == "__main__":
    main()
