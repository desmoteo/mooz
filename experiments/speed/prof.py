# One run of mooz under cProfile. Run from the project root, on a scratch copy of a take (docs/SPEED.md, Reproduce):
#   MOOZ_MEMORY_MAX=off uv run python experiments/speed/prof.py <out.prof> <scratch>/rec -c <settings> --preset default --no-chain-cache --again
# (the workers start afresh and load this file again: nothing may run at its top level)
def main():
    import cProfile, pstats, sys
    from mooz import cli
    out = sys.argv[1]; args = sys.argv[2:]
    pr = cProfile.Profile(); pr.enable()
    try:
        cli.main(args)
    except SystemExit:
        pass
    pr.disable(); pr.dump_stats(out)


if __name__ == "__main__":
    main()
