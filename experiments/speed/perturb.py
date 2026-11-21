# Run from the project root: MOOZ_MEMORY_MAX=off uv run python experiments/speed/perturb.py <scratch>/rec -c <settings> --preset default --no-chain-cache --again
# The same run, with the envelope follower's result moved by one part in 1e15: how far does the master move?
# (the workers start afresh and load this file again: nothing may run at its top level)
def main():
    import sys
    from mooz import cli, dynamics
    plain = dynamics.release_hold
    dynamics.release_hold = lambda *a, **k: plain(*a, **k) * (1.0 + 1e-15)
    try: cli.main(sys.argv[1:])
    except SystemExit: pass


if __name__ == "__main__":
    main()
