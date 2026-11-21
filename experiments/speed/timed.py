# One run of mooz with a timer on each stem's chain (clean-up, studio, mix, report detail), the hit triggers and the bed step.
# Run from the project root: MOOZ_MEMORY_MAX=off uv run python experiments/speed/timed.py <out.json> <scratch>/rec -c <settings> --preset default --no-chain-cache --again
# The chains come in the order kick, snare, toms, then the other stems by name.
# (the workers start afresh and load this file again: nothing may run at its top level)
def main():
    import sys, time, json
    from mooz import cli, effects, bed, cleanup, pipeline, report, export, stages
    log = []
    def wrap(mod, name):
        f = getattr(mod, name)
        def g(*a, **k):
            t0 = time.perf_counter(); r = f(*a, **k); log.append((f"{mod.__name__.split('.')[-1]}.{name}", round(time.perf_counter() - t0, 2))); return r
        g.__wrapped__ = f
        setattr(mod, name, g)
    for n in ("cleanup_track", "studio_track", "mix_track", "begin_detail", "end_detail", "hit_trigger"): wrap(effects, n)
    wrap(bed, "remove_bed"); wrap(cleanup, "remove_buzz")
    t0 = time.perf_counter()
    try: cli.main(sys.argv[2:])
    except SystemExit: pass
    json.dump({"total": time.perf_counter() - t0, "log": log}, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
