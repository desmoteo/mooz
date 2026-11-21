# Speed: where the time goes and what would shorten it

Status, 20 November 2026: measured, and built (sections 5, 7 and 9): a take's processing went from
about 250 s to about 95 s on the same machine. The
stems and stage files are the same to the sample or within −135 dB, and the master is 0.005 dB
from the old one, which is how far any change in a stem's last bit moves it; with the audio in
float32 (the default since 20 November, `[audio] float`) the stems are the same down to the floor of
a 24-bit file. A take's first run, the separation, went from 1063 s to 659 s (section 10). Sections 1 to 4 are the assessment as it was made, before anything was
changed; [DESIGN.md](DESIGN.md) section 8 has the times and memory of the program as it is.

In short:

- A take's **first run** is mostly separation on the GPU. The models already run at their own
  overlap, 2 for most (the first assessment said 8 and that halving it would halve the time: wrong,
  corrected in section 10). The library's compiled model blocks and one separating process for the
  run take a first run from 1063 s to 659 s (section 10).
- **Processing** (stems already separated) took 266 s for a 5-minute take on 1.2 of the 16
  cores, with no single hot spot. Huge pages, three small rewrites, files written on a thread
  the stem chains in three processes and two changes to the filters bring it to about 130 s.
- **Float32** was first estimated as not worth it, then measured with a prototype (section 7) and
  built (section 9): the stems the same to the floor of a 24-bit file, a tenth and more off the time,
  a little off the memory. **Other FFT or BLAS libraries**, the GPU and IIR filters by FFT
  would gain little or nothing.

## 1. What was measured, and how

The take is Biancaneve (`261105_235855`, 298 s, a room pair and one close track), with the
band's `rehearsal.toml`, the default profile, the chain cache off (`--no-chain-cache`) and
`--again`, so every stage runs. The runs were made on a scratch copy of the take (links to the
recorder's files and to the separation caches), so the band's masters were not touched.

The machine:

| | |
|---|---|
| processor | Intel Core Ultra 9 285H: 16 cores, no hyper-threading (6 at 5.4 GHz, 8 at 4.5 GHz, 2 at 2.5 GHz), AVX2, no AVX-512 |
| memory | 30 GB, of which the desktop (browser, editor) held about 20 GB during the runs |
| GPU | NVIDIA RTX PRO 1000, 8 GB |
| disk | NVMe under dm-crypt |
| power | on mains, profile "balanced" |
| software | Python 3.12, numpy 2.5 (OpenBLAS, 16 threads), scipy 1.18, torch 2.14 with CUDA, numba 0.68 (installed, not used by `mooz`) |

The tools, all in [experiments/speed/](../experiments/speed/): the whole run under
`/usr/bin/time -v` and under cProfile (`prof.py`), the same run with a timer on each stem's
chain (`timed.py`), micro-benchmarks of the candidate changes on a real 5-minute stereo stem
(`bench.py`), chains in parallel processes (`scale.py`), and the separation at three overlaps
(`overlap.py`). The logs of the runs are next to them.

## 2. Where the time goes

### The first run: separation

From the separation of 7 November (Miramare, a 240 s take, room and close track):

| model | room | close track |
|---|---|---|
| BS-Roformer-SW on the recording | 53 s | 51 s |
| Revive v2 (vocals) | 75 s | 75 s |
| BS-Roformer ep 317 (vocals) | 137 s | 137 s |
| xlancer (bass) | 46 s | 46 s |
| BS-Roformer-SW on the instrumental | 48 s | 48 s |
| DrumSep, 5 stems | 16 s | |
| DrumSep, 6 stems | 25 s | |
| de-reverb (vocals) | 35 s | |
| all models | 435 s | 357 s |

The whole separation took 851 s: 792 s in the models and 59 s in loading them, combining and
files. It runs once per take and is cached. Mixed precision (autocast) is already on. The GPU
holds 1.6 of its 8 GB at the peak, but it is busy the whole time: memory is not what limits it.

### Processing

| | |
|---|---|
| wall time | 266 s |
| processor time | 249 s user + 68 s system: 118 % of one core |
| memory, highest | 3.4 GB |
| page faults | 7.4 million |

By phase (cProfile, 253 s in all):

| phase | time |
|---|---|
| the stems (`_tracks`): chains, triggers, bed, blend, balance, stage files | 186 s |
| of which the eight chains themselves | 126 s |
| of which the hit triggers | 9 s |
| of which the bed under the hi-hat | 8 s |
| loudness, clipper and limiter (`_finalize`) | 13 s |
| MP3 export (two files) | 8 s |
| character of the song | 6 s |
| the rest: reading, conditioning, bus, report | about 40 s |

By chain (`timed.py`):

| stem | time | of which clean-up | of which studio chain |
|---|---|---|---|
| bass | 38.8 s | 14.6 | 21.5 |
| guitar | 17.5 s | 1.8 | 10.8 |
| vocals | 14.7 s | 0.0 | 12.7 |
| snare | 14.5 s | 3.5 | 9.6 |
| cymbals | 12.1 s | 2.8 | 7.9 |
| hi-hat | 10.4 s | 2.8 | 6.1 |
| kick | 9.6 s | 0.0 | 8.2 |
| toms | 8.4 s | 2.9 | 3.9 |

By what the processor is doing, across all phases:

| primitive | time | calls |
|---|---|---|
| IIR filtering (`scipy.signal.sosfilt`, mostly zero-phase) | 35 s | 444 |
| reductions (`mean`, `max`, `sum`), mostly across the two channels | 26 s | 413 642 |
| envelope follower (`dynamics.release_hold`) | 22 s | 136 |
| polyphase resampling (4x oversampling for true peak, clipper, saturation) | 19 s | 2 414 |
| reading and writing audio files | 14 s | 291 |
| FFTs (STFT, FIR by overlap-add) | 13 s | about 95 000 |
| waiting for the MP3 encoder | 8 s | 2 |
| one-pole smoothing (`lfilter`) | 7 s | 121 |
| moving average (`uniform_filter1d`) | 5 s | 73 |

Three things follow. The time is spread thin: the largest primitive is 14 % of the run, so no
single optimisation changes the picture. The program runs on one core of 16. And a fifth of the
processor time is the system's: each temporary array of a whole stem (229 MB in float64) is
mapped, faulted in page by page and unmapped again.

## 3. Findings

### 3.1 Separation: the overlap is the lever

**Wrong, corrected on 20 November (section 10).** The program does not run the models at the library's
fallback of 8: each model is run at the overlap its own configuration names, 2 for BS-Roformer-SW, 4 for
ep 317. The table below is right as a measurement (the time is proportional to the overlap, and the stems at
4 and 2 are that far from those at 8), but "now" is its last column for BS-Roformer-SW and its middle one
for ep 317, not its first. There is no factor of two to be had here.

`audio-separator` runs its Roformer and MDX23C models over windows that overlap, by default 8
times (`overlap` in its `mdxc_params`), and `mooz` does not set it
([separate.py](../mooz/separate.py), where the segment size is set). Measured on 60 s of the
take:

| model | overlap 8 | overlap 4 | overlap 2 |
|---|---|---|---|
| BS-Roformer-SW, time | 42.0 s | 20.9 s | 11.2 s |
| BS-Roformer-SW, stems against overlap 8 | | −36 to −43 dB | −29 to −37 dB |
| BS-Roformer ep 317, time | 66.4 s | 33.6 s | 17.5 s |
| BS-Roformer ep 317, stems against overlap 8 | | −35 to −36 dB | −30 to −31 dB |

The time is proportional to the overlap. The stems at overlap 4 differ from those at 8 by a
signal 35 to 43 dB under the stem itself, which is far below what the separation itself gets
wrong (these models' own error is 10 to 13 dB under the stem on benchmarks). The two DrumSep
models and the de-reverb model were not measured; the library applies the same setting to them.

The library does not batch windows for Roformer models (its own comment: "negligible
performance improvements"), so the free GPU memory buys nothing.

### 3.2 Huge pages: a fifteenth of the run for one variable

With `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (glibc 2.35 or later, transparent huge pages set
to "madvise", as here) the allocator asks for 2 MB pages for large arrays:

| | as it is | with huge pages |
|---|---|---|
| wall time | 266 s | 249 s |
| system time | 68 s | 48 s |
| page faults | 7.4 million | 1.9 million |
| memory, highest | 3.41 GB | 3.44 GB |

The output is the same. The variable must be set before Python starts; the launcher already
starts the run a second time inside a memory-capped scope
([cli.py](../mooz/cli.py) `_run_in_memory_scope`), which is the place for it.

### 3.3 Three small rewrites

Each on a real stereo stem of 298 s:

| what | as it is | rewritten | result |
|---|---|---|---|
| mean of the two channels, `x.mean(axis=1)` | 0.166 s | `0.5 * (L + R)`: 0.028 s | identical |
| peak of the two channels, `np.abs(x).max(axis=1)` | 0.339 s | `np.maximum(abs(L), abs(R))`: 0.060 s | identical |
| mean square of the two channels | 0.192 s | 0.058 s | identical |
| envelope follower (log, running maximum, exp) | 0.198 s | one compiled loop (numba): 0.056 s | equal to −245 dB |
| true-peak envelope (4x oversampling in chunks) | 1.89 s | the same chunks on 8 threads: 0.32 s | identical |

numpy reduces along an axis of length 2 one pair at a time, which is why the first three are
six times slower than they need be; such reductions are in 12 modules. The envelope follower is
called 136 times a take. scipy's resampling and filtering release the interpreter lock, so
threads work without any change to the mathematics: the same oversampling loop is also in the
clipper, the saturator and the harmonics. Together these are about 45 s of the 266.

### 3.4 Workers: 2.5 times, not 16

`scale.py` runs k jobs at once, each like a stem's chain (12 zero-phase filters, 8 envelopes,
2 FIR filters, an STFT and back, on a whole stem):

| jobs at once | float64: each takes | work done | float32: each takes | work done |
|---|---|---|---|---|
| 1 | 11.3 s | 1.0x | 9.6 s | 1.0x |
| 2 | 14.3 s | 1.6x | 11.6 s | 1.6x |
| 4 | 18.7 s | 2.4x | 14.5 s | 2.6x |
| 6 | 26.9 s | 2.5x | 26.2 s | 2.1x |
| 8 | | | 37.7 s | 2.0x |

Four jobs at once do the work of 2.5; more than four do no more, and in float32 less. The six
fast cores cannot all run at their single-core speed, the other ten are slower, and every job
streams whole stems through memory. So on this machine the ceiling for work in parallel is about
2.5 times, with 4 workers. The power profile was "balanced"; "performance" may raise the
ceiling and was not tried, since it is a system setting.

Threads inside one chain give less: 4 threads on 8 single-channel filters are 2.7 times faster
than one, and two channels on two threads 1.5 times.

What it means for a take: the chains are 126 s one after the other and the longest (bass) is
39 s. Four at once would take about 55 s, since each runs about 1.5 times slower in company.
Each chain is already a unit with a name made of everything it reads
([chaincache.py](../mooz/chaincache.py)), which is what running them in separate processes
needs. What they share is small: the hit triggers (made before the chains), the raw snare stem
for the bed step, and two results to add up after (the kit gate's curve and what the bed step
took).

Memory is what limits the workers. A chain needs up to 4 GB on a take of 5:40, and the desktop
leaves about 10 GB: 2 chains at once as things are, 4 only with the stems in float32 or with
the browser closed. The run's memory cap already knows what the machine can spare.

### 3.5 Float32: for memory, not for speed

| operation | float64 | float32 | error of float32 |
|---|---|---|---|
| zero-phase band-pass 2–8 kHz, order 4 | 0.39 s | 0.31 s | −99 dB |
| zero-phase high-pass 30 Hz, order 4 | 0.37 s | 0.31 s | −68 dB |
| zero-phase peak 80 Hz, Q 4 | 0.36 s | 0.29 s | −65 dB |
| the same three, float32 samples and float64 arithmetic | | 0.32 to 0.35 s | −133 to −152 dB |
| FIR EQ, 8191 taps | 0.45 s | 0.22 s | −132 dB |
| STFT and back, one channel | 0.87 s | 0.78 s | |
| loudness (BS.1770) | 0.18 s | 0.17 s | the same to 0.001 LU |
| true-peak envelope | 1.89 s | 1.58 s | 1.6e-7 |
| gain times signal | 0.060 s | 0.034 s | |
| 10^(g/20) | 0.105 s | 0.053 s | |

IIR filters run one sample after the other whatever the precision, so they gain a fifth at
most, and with float32 arithmetic the low filters are right only to −65 dB: with dozens of
them in a chain that is not acceptable. Keeping the samples in float32 and the filter's
arithmetic in float64 is accurate (−133 dB or better) and gains a tenth. FIR filtering and
plain arithmetic are twice as fast in float32. Over a whole chain that is 10 to 15 %.

(Measured since with a prototype, section 7: the stems come out the same to the floor of 24-bit
files, a take is 12 % faster, and the memory does not go down until the arithmetic is float32 too.)

Its real worth is memory: half of it, which is what lets four chains run at once (3.4), and
fewer pages to fault in. Its cost: every array in the chains, the blend and the bus changes
type, the tests that compare to the sample loosen to about −130 dB, and the masters are no
longer the ones made today bit for bit.

### 3.6 Files

| | time | size |
|---|---|---|
| FLAC 24-bit, default level | 0.47 s | 47 MB |
| FLAC 24-bit, level 0 | 0.30 s | 49 MB |
| WAV float32 | 0.08 s | 114 MB |
| reading a WAV as float64, as float32 | 0.05 s, 0.02 s | |
| reading a FLAC | 0.19 s | |

A take writes 33 stage files (about 10 s) and two MP3s (8 s, in another process, which the
run waits for). Writing them from a thread while the next stage computes takes that time off
the run without changing a byte. The lowest FLAC level saves a third of the encoding for files
4 % larger.

### 3.7 What gains nothing

- **Another FFT library** (pyFFTW, MKL) or FFT workers: all the FFTs of a take are 13 s, and 8
  workers took the FIR EQ from 0.22 s to 0.18 s.
- **BLAS**: OpenBLAS is used by a few matrix products only.
- **Batches on the GPU**: not used by the library for these models.
- **Float32 arithmetic in the IIR filters**: see 3.5.

## 4. Steps

In the order I would take them. The gains are for a take of 5 minutes; those of steps 2 to 5
are estimates from the measurements above and are not simply additive. Steps 2 to 7 are built
(section 5 has what they gave); 1 and 8 are not.

| | step | gain | effort | risk |
|---|---|---|---|---|
| 1 | ~~Separation at overlap 4~~ (wrong: the models are at 2 and 4 already, section 10) | | | |
| 2 | Huge pages in the launcher | 17 s | one variable | none seen |
| 3 | The channel reductions written per channel | about 20 s | small, 12 modules | none: identical results |
| 4 | The envelope follower as a compiled loop | about 15 s | small | first call compiles (cached on disk) |
| 5 | The oversampling loops on threads | about 10 s | small | none: identical results |
| 6 | Stage files and MP3s written from a thread | about 10 s | small | the run must wait for them before it ends |
| 7 | The stem chains in 3 or 4 processes | about 60 s | medium | memory; progress events and errors from workers |
| 8 | Stems in float32 | 10 to 15 % of the chains; half the memory | large | masters change at −130 dB; tests loosen |

1. **Overlap.** Add `overlap` to `[stems]` (default 4) and pass it where `segment_size` is
   passed. It must be part of the separation cache's key, so that stems made at 8 and at 4 are
   not mixed, and it means separating the takes again to have them all alike. Before: separate
   one take at 4, run it, and compare by ear and by the report with the one at 8. Overlap 2 is
   there for a quick look at a new take.
2. **Huge pages.** Set `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` where the run is started again
   in its scope, unless the variable is already set. Check: the master is the same file.
3. **Reductions.** A helper for the mono sum and for the stereo-linked peak, used wherever
   `mean(axis=1)` and `abs(...).max(axis=1)` are on whole signals. Check: the masters do not
   change.
4. **Envelope follower.** `release_hold` as a numba loop over the samples, the 2-D case over
   its rows. numba is already installed (librosa needs it); it becomes a dependency of `mooz`
   itself. Check: the null tests, and the master against today's.
5. **Oversampling.** `oversampled_chunks` handing its chunks to a small thread pool, in order.
6. **Files.** One writer thread with a queue, joined before the report is written.
7. **Chains in parallel.** A pool of processes started once per run; a chain goes to a worker
   with what it reads and writes its result where the chain cache keeps it, and the main
   process adds up the kit gate and the bed step as it does now for kept chains. The number of
   workers from the memory the run may use, 4 at most. Check: the same master as with one
   worker, to the byte. Running several takes at once instead is simpler and meets the same
   ceiling, but each take needs its whole memory and the takes' tone targets depend on each
   other.
8. **Float32.** Only if step 7 is held back by memory. Samples in float32, filter arithmetic in
   float64.

Steps 2 to 6 together are about 70 s of 266 with no change to the sound; with step 7 a take
would take about 150 s. Every change to the chains' code renames the kept chains, so the first
run after each step makes them again.

Not tried: the power profile "performance" (3.4), `torch.compile` or TensorRT for the models (both tried since:
sections 10 and 11),
a pool that keeps models loaded across takes (loading and combining are 59 s of 851).

## 5. What was built (19 November 2026)

Steps 2 to 7, on the same take. Each was checked against the take as the old code made it: 27
audio files (the master, the mix, the pre-master parts and every stage file), compared sample
by sample.

| step | what | against the old code's files |
|---|---|---|
| 2 | huge pages: the launcher sets `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` unless the variable is set ([cli.py](../mooz/cli.py)) | 27 of 27 the same to the sample |
| 3 | `loudness.mono`, `peak` and `power` in place of the reductions across the channels, in 8 modules | the same to the sample |
| 5 | the oversampling loops read ahead on threads; the true-peak envelope on threads ([loudness.py](../mooz/loudness.py), `MOOZ_THREADS`) | the same to the sample |
| 6 | stage files encoded on a thread of their own ([stages.py](../mooz/stages.py)); the MP3 encoders started and waited for at the end ([export.py](../mooz/export.py)) | the same to the sample |
| 4 | the envelope follower as a compiled loop ([compiled.py](../mooz/compiled.py), numba) | stems and pre-master parts within −135 dB; mix and master at −72 dB (below) |
| 7 | the stem chains in processes of their own ([pipeline.py](../mooz/pipeline.py) `_chain_job`, `[chains] workers`, `MOOZ_WORKERS`) | the same to the sample as with one chain after the other, at 2, 3 and 4 workers |

**The −72 dB.** The compiled follower gives the same values as the old formula to one part in
10^12, and with it 16 of the 27 files are the same to the sample and the stems and the
pre-master parts within −135 to −185 dB. The mix and the master differ by −72 dB, evenly over
the whole take: a gain 0.005 dB apart. That is not the follower. Moving the follower's result by
one part in 10^15 moves the mix by the same −72 dB: the bus compressors set their thresholds at
a percentile of the level over the loud blocks, and whether a block counts as loud is a yes or
no, so any change in the last bit of a stem shifts those thresholds by a hair. A take made twice
by the same code is the same to the sample; a take made before and after any change to a chain's
arithmetic is 0.005 dB apart.

**Times.** One run each, old and new code in turn on the same machine (it was shared with other
work, and warm: single runs vary by a tenth):

| | old code | steps 2 to 6 | with the chains on workers |
|---|---|---|---|
| quiet pair | 245 s | 188 s | |
| busier pair | 269 s | 206 s | |
| the last pairs | 253 s, 249 s | | 145 s (3 workers), 165 s (2 workers) |

In the last pairs the number of workers was left to the program: three the first time, two the
second, when less memory was to spare.

By the number of workers, the same afternoon (all with steps 2 to 6):

| workers | take | the bass chain | a worker's memory at most |
|---|---|---|---|
| 1 | 210 s | 39 s | |
| 2 | 148 s | 39 s | 2.7 GB |
| 3 | 135 s | 48 s | 2.6 GB |
| 4 | 135 s | 56 s | 2.6 GB |

A fourth worker gains nothing, as section 3.4 had it: every chain slows down in company, and the
take waits for the bass. So the workers are three at most. Their number is set by the memory to
spare when the take starts: a worker needs about 0.6 GB and 0.4 GB per minute of take, the run
itself 1.5 GB and 0.25 per minute. With 12 GB to spare that is 3 workers for a take of 5
minutes and 2 for one of 8; with 8 GB, 2 and 1. A take under a minute is made in one process.
The memory limit of the run covers the workers too.

What the estimates of section 4 missed: the reductions across the channels and the threads on
the oversampling gave 16 s where 30 were expected (the consumers of the oversampled chunks still
work one chunk at a time), and the workers 60 to 75 s as expected.

**How it works, where it matters to someone changing the code.**

- A chain is one function, `_chain`, that reads its stem's file, the hit triggers and its
  settings. A worker makes it and keeps it as [chaincache.py](../mooz/chaincache.py) keeps
  chains: in the take's kept chains, or in the run's work folder where none are kept. The run
  takes it up as it takes up a kept one. The hit triggers are made once by the run and read by
  the workers from files in the work folder, without a copy.
- The workers are started afresh (spawn), so nothing a test puts in a function's place reaches
  them: the tests set `MOOZ_WORKERS=1` (`tests/conftest.py`) except the one that checks the
  workers.
- Whoever reads or links a stage file calls `stage.flush()` first.
- The first run after a change to `compiled.py` compiles it (a second); the result is kept in
  `mooz/__pycache__`.

Left as it was: the consumers of the oversampled chunks (clipper, saturator, harmonics) work one
chunk at a time; zero-phase filters run one channel after the other (two threads would be 1.5
times faster and need the memory of both at once).

## 6. Reproduce


A scratch copy of a take, so that nothing is written next to the recordings:

```sh
S=/tmp/mooz-speed; R="new_tracks/drive-download-20261107T141948Z-1-001"; T="261105_235855 Biancaneve"
mkdir -p "$S/rec/$T" "$S/rec/processed"
ln -s "$PWD/$R/$T"/* "$S/rec/$T/"
for d in stems_close stems_ensemble stems_ensemble_merged; do ln -s "$PWD/$R/processed/$d" "$S/rec/processed/$d"; done
cp rehearsal.toml "$S/"
export MOOZ_MEMORY_MAX=off   # stay in one process, so that the profiler sees it
```

Then, from the project's folder:

```sh
/usr/bin/time -v uv run mooz "$S/rec" -c "$S/rehearsal.toml" --preset default --no-chain-cache --again
uv run python experiments/speed/prof.py  "$S/run.prof"    "$S/rec" -c "$S/rehearsal.toml" --preset default --no-chain-cache --again
uv run python experiments/speed/timed.py "$S/chains.json" "$S/rec" -c "$S/rehearsal.toml" --preset default --no-chain-cache --again
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 /usr/bin/time -v uv run mooz "$S/rec" -c "$S/rehearsal.toml" --preset default --no-chain-cache --again
uv run python experiments/speed/bench.py "$S"
uv run python experiments/speed/scale.py
uv run python experiments/speed/overlap.py "$S"
```

One file is still written in the real cache through the links, with the same content every
time: the take's buzz frames (`stems_ensemble_merged/<id>/_buzz/<take>.npy`).

**Only for runs that read the stems.** A run that separates (`--separate-again`, a changed `[stems]` plan)
writes through the links into the real project: on 20 November a check of `--separate-again` on this set-up
replaced the real kept stems of Biancaneve. For such a run, make the three stems folders real and empty.

To compare two outputs of the take sample by sample, and to see how far a change in the last bit
moves the master (section 5):

```sh
cp -r "$S/rec/processed/full" "$S/before"     # then change the code, run again, and:
uv run python experiments/speed/same.py "$S/before" "$S/rec/processed/full"
uv run python experiments/speed/perturb.py "$S/rec" -c "$S/rehearsal.toml" --preset default --no-chain-cache --again
```

For times, run the old and the new code in turn (a copy of `mooz/` each, first on `PYTHONPATH`):
single runs on a machine doing other things vary by a tenth, and by more when it is hot.

## 7. After section 5: two more steps, and three things measured (19 and 20 November 2026)

### Built: the filters

- Where only the mono sum of a filtered signal is wanted (hit triggers, band envelopes, the hats' own hits,
  the character's band), the mono sum is filtered: half the filtering, the same to rounding.
- `loudness.filtfilt` filters the two channels on a thread each: the same samples, both channels'
  temporaries held at once.

Tested (`tests/test_speed.py`) and checked on the take: the stems within −135 dB of the old code's, the
master at the same −72 dB as in section 5. Timed in turn, 3 workers: 135 s and 140 s before, 131 s and
129 s after. About 7 s of a take, 5 %. The worker's and the run's memory at their highest are as they were
(2.6 GB and 2.7 GB).

With everything built, a take of 5 minutes is at about 130 s with three workers, from about 250 s.

### Measured and not built: the GPU, and IIR filters by FFT

([experiments/speed/](../experiments/speed/) `gpu.py`, `fdiir.py`)

- **The GPU** (float64 / float32, transfers included, a stem of 5 minutes): STFT and back 1.07 s on the CPU,
  0.14 / 0.06 s; true-peak envelope 0.7–1.0 s, 0.21 / 0.11 s; FIR EQ 0.47 s, 0.27 / 0.12 s; zero-phase IIR in
  the frequency domain 0.41 s, 0.25 / 0.15 s but −104 dB from today's filters; the envelope follower slower.
  About 10–15 s of a take, for a second code path, 1–3 s of start per process and 2.9 GB of the card per
  float64 operation: not worth it now.
- **Zero-phase IIR by FFT on the CPU**: at best as fast as the recursive filters (0.37–0.54 s in float32 blocks
  on threads against 0.37–0.43 s), slower otherwise, and −103 to −124 dB from them: no.

### Measured: float32

Section 3.5 estimated float32 from single operations and judged it worth its memory, not its speed. A
prototype says otherwise on both counts. It is kept in `experiments/speed/f32_prototype/` (`_f32.py` and
`prototype.diff`, applied to a copy of `mooz/`, never to the program): `MOOZ_F32=chains` keeps the audio in
float32 through the stem chains, `all` through the blend, the bus and the master too. Per-sample gains and
other control signals stay float64, and so does the arithmetic of the IIR filters (float64 coefficients,
the result stored in float32). The audio is cast where it enters and leaves every function of the effect
modules, so arithmetic written with float64 gains inside a function is still done in float64 and rounded on
the way out: the rounding is that of a float32 program, its speed and memory are not yet.

**Sound: nothing against it.** The take in float32 (`all`) against the same code in float64
(`f32_prototype/float32_all_against_float64.txt`, made with `delta.py`):

| | whole file | loudest 0.4 s of the difference | largest single sample |
|---|---|---|---|
| the eight stems (`2_stems`) | −136 to −155 dBFS | −124 to −143 dBFS | −98 to −126 dBFS |
| the drum stems after the clean-up | −160 to −173 dBFS | −149 to −160 dBFS | −126 dBFS |
| the drum bus, the pre-master parts | −138 to −144 dBFS | −116 to −140 dBFS | −93 to −126 dBFS |
| the mix and the master | −87 dBFS (−72 dB re the file) | −82 dBFS | −65 dBFS |

The stage files are 24-bit, whose own step is −138 dBFS: in the stems the difference is at the floor of the
files it is measured in. No decision came out differently: the reports name the same hits, samples placed,
gates and bed, the same loudness, peak and thresholds. The mix and the master are 0.002 dB apart, which is
what any change in a stem's last bit does to them (section 5). One field of the report differs, the rider's
`avg_up_db` (0.7, 1.3 or 1.4 dB over three renders with the same `max_up_db`, `max_down_db` and
`riding_up_pct`): it is a mean over every sample where the gain is above zero, and after the smoothing that
includes samples at 1e-17. It is a weakness of that figure, in float64 too, not a difference in the sound.

One take; a threshold can in principle fall the other way on another one, as with any change to a chain.

**Speed: 6 % with the chains in float32, 12 % with everything**, from a prototype that still does much of
its arithmetic in float64. In turn, 3 workers:

| | first | second |
|---|---|---|
| float64 | 131 s | 123 s |
| float32 in the chains | 118 s | 120 s |
| float32 everywhere | 112 s | 111 s |

The bass chain, which the take waits for, goes from 46–50 s to 38–42 s.

**Memory: none gained by the prototype.** A worker is at 2.5 GB in place of 2.6 GB, and with float32
everywhere at 2.8 GB; the run at 2.7 to 2.8 GB. The highest point is inside an effect, where a float64 gain
times the audio makes float64 temporaries, and the casts at the functions' edges add copies. To halve the
memory the arithmetic itself has to be float32.

**What a real float32 would take.** About 80 places in 12 modules: 35 where a gain curve multiplies the
audio (the linear gain cast to float32 first), 35 where an array is made without a type, a few filters
called on scipy directly. The control signals, the running sums and the IIR arithmetic stay float64
(section 3.5). The tests that compare to the sample would compare to about −120 dB. Expected then, not
measured: more than the 12 % (the multiplications and the FIR at twice the speed, half the page faults) and
a worker near 1.6 GB, which lets three workers fit where two do now.

So float32 is sound-wise as harmless as the compiled envelope follower, and worth about as much time as the
two filter steps and the huge pages together. It is not built: it was not among the steps asked for.

## 8. Reassessed after the steps (20 November 2026)

With everything of sections 5 and 7 built, a take of 5 minutes takes about 129 s with three workers. Where
that goes now (`timeline.py`: the main process's stages; `prof.py` with one chain after the other for the
functions):

| | wall time |
|---|---|
| before the chains: cache checks, measuring, conditioning, the song's character | 10 s |
| the eight chains on three workers (158 s of work between them) | 59 s |
| after them, still in the stems' stage: blend, group balance, drum bus, rider | 15 s |
| the bus: multiband 5 s, tape 12 s, the glue's threshold search 5.5 s | 23 s |
| the master: clipper 4 s, limiter 2 s, measuring 3.5 s | 10 s |
| files, MP3, report | 6 s |

More than half the take is now the main process working alone, and the chain phase is no longer waiting
for the bass (49 s) but for all the chains' work shared out among three (the hi-hat's is the last to end).
By function, one chain after the other (174 s under the profiler): IIR filtering 18 s, of which the loudness
meter's K-weighting 8 s in 52 calls; the per-stem detail for the tracks report 13 s (a spectrum and a
loudness of every stem before and after); FFTs 13 s; attack smoothing 7 s in 121 calls; the bass's pitch
tracker 9 s; reading and writing audio 8 s.

### Libraries (`libs.py`)

| what | now | the alternative | result |
|---|---|---|---|
| zero-phase IIR, stereo | scipy `sosfiltfilt`: 0.21 s | one compiled loop (numba): 0.25–0.37 s | the same to the last bit, and slower: scipy's is as good as it gets |
| K-weighting, one way | scipy `sosfilt`: 0.15 s | compiled: 0.21 s | the same, slower |
| attack smoothing (one pole) | scipy `lfilter`: 0.054 s | compiled: 0.018 s | the same to the last bit, 3 times faster |
| `10 ** (g / 20)`, `20 log10(x)` on a stem | numpy: 0.11 s | numexpr on threads: 0.02 s | the same to the last bit, 5 to 6 times faster |
| `tanh` on a stem | numpy: 0.22 s | numexpr: 0.034 s | −300 dB, 6.5 times faster |
| gain times a stereo stem | numpy: 0.058 s | numexpr: 0.033 s | the same, 1.7 times faster |
| 4x oversampling | scipy `resample_poly`: 1.14 s | libsoxr (HQ, VHQ): 0.55–0.60 s | another filter: −62 dB from scipy's, and another true peak |

So: nothing to gain on the filters; numexpr (a new dependency, small) for the element-wise mathematics of
the detectors and gain curves; the compiled one-pole for the smoothing; soxr no, it changes what the
clipper, the saturators and the true-peak meter see. Other FFT libraries were already ruled out (3.7).

### Stereo where mono would do (`stereo.py`, `mono_bass_prototype/`)

Side against mid on the takes with merged stems: bass −17 dB and vocals −19 dB (nine tenths of each is the
mono close track), the other stems −5 to −10 dB (the room: really stereo). The detection paths already
filter the mono sum (section 7). The settings make one stem mono, the bass (`width = 0`), and only at the end
of its chain: the whole chain works on two channels and then sums them.

The bass chain on its mono sum from the start (prototype, `MOOZ_MONO=1`), in turn with the same code in
stereo, 3 workers: the bass chain 48–49 s to 36–37 s, the take 133 and 127 s to 123 and 126 s: about 5 s.
The bass as it enters the mix differs by −40 dB re itself (0.1 dB in its level), the master by −46 dB: the
compressor, the saturator and the exciter answer the mid, not each channel, and what the side did to them
is gone. Small, and not rounding. The vocals are as near to mono, but their settings keep half the side
(`width = 0.5`), which holds the room: not a candidate. The spectrum and loudness of the report need both
channels.

### The tape plugin (`tape.py`)

The largest single item of the bus: 10 to 12 s for the take, one call, one thread. Its two channels do not
touch each other (the left with a silent right is the left of the stereo call, to the last bit), but a
plugin with one silent channel takes as long as with two, so a thread per channel gains nothing. In six
pieces of time on six threads it takes 2 s, and comes out −82 dB from the one call (largest sample
−75 dBFS), the same with 0.5 s or 10 s of run-in: pedalboard's own note in `plugins.py` says as much. About
8 s of a take for a difference of −82 dB on the bus.

### What is left, in order

(Steps a to f and h were built the same day: section 9. Left: g, i, j, k.)

No change to the sound (the same samples, or the same to rounding):

| | step | of a take |
|---|---|---|
| a | The glue's threshold search works the level out once, not eight times, and applies no gain until it has the threshold | about 3 s |
| b | numexpr for the detectors' and gain curves' element-wise mathematics (threads limited in the workers) | 4 to 6 s (estimate) |
| c | Attack smoothing as a compiled loop | about 2 s (4.5 s of the chains' work) |
| d | The hit triggers made once for the character and the chains | 1.6 s |
| e | One K-weighting for a summary's loudness and left/right balance, and its two channels on two threads | 1 to 3 s |
| f | The bass's pitch kept with the stem (it reads only the stem): nothing on the first run, 9 s of the bass chain on every later one, the second profile's included | later runs |
| g | The per-stem detail left out when no tracks report is written | 4 s, only without reports |

A small change to the sound:

| | step | of a take | the change |
|---|---|---|---|
| h | float32 (section 7) | 15 s or more | stems at the 24-bit floor, master 0.002 dB |
| i | The tape plugin in pieces of time on threads | about 8 s | −82 dB on the bus |
| j | The bass chain in mono | about 5 s | the bass −40 dB re itself |
| k | ~~The separation at overlap 4~~ (wrong, section 10); the compiled model blocks of section 10 | about a third of a take's first run | the stems as far from float32 as now |

Steps a to e together are about 10 to 15 s of the 129. Past them the take is bounded by the chains' work
shared among three workers and by a bus and master that work on one stereo signal in order: the next real
gain is float32, or a machine that keeps more than three workers fed.

## 9. Built: the six steps of section 8, and float32 (20 November 2026)

### The six steps

| | step | where | against the code before |
|---|---|---|---|
| a | the glue's threshold search on the level alone | [dynamics.py](../mooz/dynamics.py) `average_reduction`, `compress(level=)` | the same to the sample |
| b | element-wise mathematics on threads | [expr.py](../mooz/expr.py) (numexpr), used by the detectors and gain curves of `dynamics.py` and `effects.py` | the same to the sample |
| c | attack smoothing compiled | [compiled.py](../mooz/compiled.py) `smooth` | the same to the sample |
| d | the hit triggers made once a take | [pipeline.py](../mooz/pipeline.py) `_hit_trigger` | the same to the sample |
| e | one K-weighting for a summary, its channels on threads | [loudness.py](../mooz/loudness.py) `_k_weighted`, `summary` | the same to the sample |
| f | the bass's pitch kept with the take's chains | [augment.py](../mooz/augment.py) `pyin`, `<chains>/<take>/_pitch/` | the same to the sample |

All six together: 27 of the take's 27 audio files the same to the sample as before them. The pitch is kept
under the samples the tracker reads and its settings, so a change to the bass's chain or a second profile
finds it (9 s of the bass's chain); with `--no-chain-cache` it is not kept.

### Float32

`[audio] float = 32` (the default; 64 for everything in 64 bits as before). The samples are float32 from
where the stems' stage begins to the master. What measures and decides stays in 64 bits: the recording's
loudness, the trim, the hum, the song's character, the hit triggers and buzz frames (read from the stems in
64 bits), every gain curve and detector, the loudness meter, and the arithmetic of the IIR filters (float64
coefficients; the result is stored in float32). A gain curve is worked out in 64 bits and cast once, where it
multiplies the audio (`expr.gained`, `expr.times`).

With `float = 64` the take is the same to the sample as before the change (27 of 27 files). With 32, against
64:

| | re the file itself |
|---|---|
| the stems (`2_stems`) | −110 to −121 dB |
| the drum stems after the clean-up, the drum bus, the pre-master parts | −117 to −127 dB |
| the mix and the master | −72 dB (0.002 dB of gain: what any change in a stem's last bit does) |

as the prototype had it (section 7), and the same report but for the rider's `avg_up_db`. Two tests keep it
so: the test song in 32 and in 64 bits (stems within −90 dB, the same hits found and samples placed), and a
run in which no function of the effect modules may hand back 64-bit audio when given 32
(`experiments/speed/upcasts.py` is the same check on a real take, with names).

Memory, the take of 5 minutes: a worker 2.3 GB (2.6 in 64 bits), the run itself 2.4 GB (2.75). Less than
half is saved because the gain curves and detectors, one value per sample in 64 bits, are now as large as
the stereo audio they act on. The number of workers is worked out from these figures for each precision.

### Times

In turn, 3 workers, the code before the six steps, with them, and with float32 as well:

| | first | second |
|---|---|---|
| before (section 7) | 121 s | 119 s |
| the six steps | 112 s | 112 s |
| the six steps and float32 | 94 s | 95 s |

The six steps take 8 s off a take (the kept pitch not counted: these runs keep nothing), float32 another 17 s:
120 s to 95 s, and the bass chain from 45 s to 37 s. The machine was quieter than on the 19th, when the same
"before" took 129 to 135 s. Against the code of the morning of 19 November (245 s on its quietest run) a
take is 2.6 times faster.

## 10. The separation: precision, compilation, export (20 November 2026)

**A correction first.** Sections 3.1, 4 and 8 said the models run at an overlap of 8 and that 4 would halve
a take's first run. They do not: `audio-separator` takes each model's own `num_overlap` when none is given,
and falls back to 8 only where a model names none.

| model | overlap it runs at |
|---|---|
| BS-Roformer-SW, Revive v2, xlancer bass, the 5-stem DrumSep, the de-reverb | 2 |
| BS-Roformer ep 317, the 6-stem DrumSep | 4 |
| the karaoke model of step 5 (off) | 8 |

Only ep 317 has a step left to take (4 to 2: about 68 s of its 137 s per source, for stems of its own about
−31 dB from now's). The mistake was mine: I read the fallback in the library and did not ask the loaded model.

**What the library offers** (`sep_modes.py`: 30 s of the take, each stem against float32):

| | BS-Roformer-SW | ep 317 | from float32 | GPU memory |
|---|---|---|---|---|
| float32 | 10.4 s | 34.7 s | | 1.5 GB |
| autocast float16 (as now) | 5.7 s | 14.7 s | −62 to −72 dB | 1.6 GB |
| native float16 | 5.3 s | 14.1 s | −61 to −70 dB | 0.9 GB |
| autocast float16, blocks compiled | 2.9 s | 8.2 s | −63 to −71 dB | 1.4 GB |
| native float16, blocks compiled | 2.9 s | 8.1 s | −62 to −70 dB | 0.7 GB |
| autocast bfloat16 | 5.8 s | 14.7 s | −43 to −54 dB | 1.6 GB |

- **bfloat16**: as fast as float16 and 20 dB further from float32 (8 bits of mantissa against 11). No.
- **Native float16** (the weights themselves in half precision): 5 % faster than autocast, half the GPU
  memory. Not worth a change by itself.
- **Compiled blocks** (`use_torch_compile`, the library's "regional" `torch.compile` of the repeated
  transformer blocks, for BS- and Mel-Band Roformer): 1.8 to 2 times faster, and no further from float32
  than the program is now. The first run after a model is loaded compiles: 5 to 11 s. The program loads
  each model afresh for every source, 13 loads for a take with a close track.
- **int8 quantisation**: PyTorch has none for these models on a GPU; it comes with TensorRT or ONNX
  Runtime. A separator's output is a waveform at 60 dB and more under its input in places: the stems at
  float16 are already 65 dB from float32, and 8 bits would be heard.
- **TensorRT export**: not tried. Neither `tensorrt` nor an ONNX Runtime with a GPU is installed, and the
  model's forward pass holds the STFT and its inverse on complex numbers, band splitting through einops
  and rotary embeddings: the STFT would have to be taken out of the model and the rest exported in real
  numbers. The attention is already one fused kernel (`scaled_dot_product_attention` in float16). With the
  compiled blocks giving a factor of two for one flag, an export would be days of work for an unknown rest.
  (Wrong on the work: section 11 exports it in an afternoon. The rest turned out small.)

**What it would give a take's first run.** Of the 851 s of Miramare (240 s, a close track), 751 s are
Roformer models. Compiled: about 395 s, plus 13 compilations of about 8 s: 851 s to about 600 s. With each
model loaded once for all the takes of a run and not once a source, the compilations are paid once a run:
about 500 s a take. The stems would differ from the kept ones at about −65 dB re themselves (float16's own
noise, between one way of running it and another), so the kept stems stay valid: nothing has to be
separated again unless one wants every take made the same way.

### Built: compiled blocks, and one separating process for the run

- `[stems] compile = true` (the default, on a GPU): the separators are made with `use_torch_compile`.
- The separation was a process per source, each loading every model of its plan: 13 loads for a take with a
  close track, and a compilation at each would have eaten the gain. It is now one process for as long as there
  is something to separate ([separate.py](../mooz/separate.py) `_ask`, `_serve`): the run hands it one source
  after another, and its models stay loaded, each in a separator of its own, and compiled. All seven models of
  the plans together are 3.7 GB of weights and 4.6 GB at work on the 7.5 GB card; where a card has less than
  2 GB free for one more, the least recently used model goes first. The run ends the process before the stem
  chains (`separate.release()`), which want the memory: 4 GB of main memory were its own.
- **The compiled code is kept** in `~/.cache/mooz/compiled` (PyTorch's own cache of what it compiles,
  `TORCHINDUCTOR_CACHE_DIR`, 23 MB). Measured on all seven models in one process: 17 s of compiling with
  nothing kept, 8 s with the folder there. What stays is the tracing of the blocks, done at every start; the
  models of one architecture share it within a run.

Measured on Biancaneve (298 s, the room track and one close track), the first run with nothing kept:

| | before | now |
|---|---|---|
| the whole take | 1063 s | 659 s |
| BS-Roformer-SW on a source | 67 s | 35 s (47 s the first time: loading and compiling) |
| Revive v2 | 96 s | 57 to 62 s |
| ep 317 | 175 s | 100 s |
| xlancer bass | 59 s | 29 s |
| the de-reverb | 44 s | 33 s |
| the two DrumSep models (not Roformer: not compiled) | 56 s | 54 s |

The new stems against those kept from 7 November: −55 to −63 dB re each stem, the two that are next to
silent (the room's dry vocal, `other`) more against themselves. That is float16's noise between two ways of
running the same model, 40 dB and more under what a separation gets wrong, and the kept stems stay as
they are: the setting is not in the cache's name, so nothing is separated again.

A job that fails ends the process (its GPU may not be fit to go on) and the next source starts another.

### Kept for later

`changes_since_baseline.diff` is everything built on 19 November against the code as it was that morning
(with the uncommitted work of 16 November in both), for timing old against new (`ab.sh.txt`) and for
committing it apart. `changes_20_november.diff` is what was built on the 20th on top of it (the six steps and float32).
`timing_20_november.txt`, `timeline_20_november.json` and `sep_modes.log.txt` have the runs above. Nothing of this
is committed.

## 11. The separation exported: TensorRT, ONNX Runtime, a container (20 November 2026)

The question: would the separation models exported to TensorRT and served from a Docker container help, and
by how much. Measured, not built. In short: the export works, the engine is 13 % faster than the compiled
blocks on the part of a model it can replace, a take's first run would gain 5 to 10 %, the stems come out 6 to
8 dB less exact, and the container adds nothing to the speed. ONNX Runtime on the GPU is 2.4 times slower
than what the program does now.

**What can be exported** ([trt_core.py](../experiments/speed/trt_core.py)). A Roformer's forward pass is an STFT,
a core in real numbers (band split, 12 transformer blocks, mask estimators), and the inverse STFT on complex
numbers. The core is 91 % of a separation's time for BS-Roformer-SW and 97 % for ep 317; the STFT, its inverse
and the overlap-add stay in PyTorch whatever is done. Every chunk has the same shape, (1, 801, 4100), the last
one of a source too (the library moves it back to end with the source), so an engine of one fixed shape does.
PyTorch's new exporter (`torch.onnx.export(dynamo=True)`) wrote the core at the first attempt, in 35 to 80 s
a model, in float32, in float16 and in bfloat16. TensorRT 11.5 (CUDA 13 build, `pip install tensorrt-cu13`,
4.5 GB) ran in the same process as PyTorch 2.14 without a conflict. Its networks are typed by the ONNX file
(there is no "FP16" switch any more): the precision is chosen at the export.

**One chunk, the ways in turn in one process** (`versus`; ms a chunk, and against now):

| | BS-Roformer-SW | ep 317 | from float32 (the masks) |
|---|---|---|---|
| PyTorch float32 | 1618 | 2897 | |
| PyTorch autocast float16 (until 20 November) | 875 | 1214 | −61 to −63 dB |
| PyTorch, blocks compiled (now) | 461 | 678 | −62 to −64 dB |
| PyTorch, blocks compiled, two chunks at once | 364 (0.79) | 645 (0.95) | −62 to −64 dB |
| TensorRT float16 | 401 (0.87) | 590 (0.87) | −53 to −57 dB |
| TensorRT float16, two chunks at once | 404 (0.87) | 592 (0.87) | −56 dB |
| TensorRT float16, builder level 5 | 412 | | −56 dB |
| TensorRT bfloat16 | 322 (0.70) | 521 (0.77) | −41 to −44 dB |
| TensorRT float32 (TF32 inside) | 967 | | −70 dB |
| ONNX Runtime 1.32, CUDA provider, float16 | 1092 (2.4) | | −62 dB |
| ONNX Runtime 1.32, CUDA provider, bfloat16 | 1096 (2.4) | | −44 dB |

- **TensorRT float16: 0.87 of now on both models**, and the same with the CUDA 12 build of TensorRT
  ([trt_plain.py](../experiments/speed/trt_plain.py): 390 and 600 ms), which rules out the float16 attention
  regression its release notes name for CUDA 13.0 on Blackwell cards. An engine is built in 35 to 55 s, is
  0.35 GB on disk, loads in 0.2 to 0.5 s and takes 0.4 GB of GPU memory at work.
- **A whole separation of 30 s** with the engine in the place of the model's core (`whole`, BS-Roformer-SW):
  3.04 s as now, 2.90 s with the engine (5 %; the wiring copies the masks once more than a built version
  would). The stems: −55 to −60 dB from float32 where the compiled blocks are −63 to −69 dB, and −54 to −60 dB
  from the stems the program makes now.
- **A take's first run**, estimated from that: Biancaneve's 659 s to about 590 to 625 s. The engines would
  also save the compilation at a run's start (8 to 17 s).
- **bfloat16 in TensorRT** is faster (0.70 and 0.77), unlike in PyTorch (section 10), and as inexact as
  there: −41 to −44 dB, 20 dB further from float32 than now.
- **ONNX Runtime** (the same ONNX files, CUDA provider, every node on the GPU, all graph optimisations,
  tensors bound on the GPU): 2.4 times slower than now in float16 and in bfloat16. It runs the attention as
  separate matrix products and a softmax; PyTorch and TensorRT have one kernel for it. Its TensorRT provider
  would build the engine measured above.
- **int8, FP8**: not measured. TensorRT 11 takes quantisation only as Q/DQ nodes in the graph, from a
  quantisation tool and calibration data, and bfloat16 (8 bits of mantissa) shows where fewer bits lead.

**What a container would add.** No speed: it would run this engine on this GPU. Docker 29.1 with the NVIDIA
runtime is on the machine (not run for this). TensorRT's libraries are 4.5 GB and PyTorch with its CUDA
libraries about 7 GB (it is still needed for the STFT, unless that is rewritten), so an image is 10 GB and more
where the system disk has 29 GB free; installing two TensorRT builds side by side for these measurements
filled that disk once. Engines cannot be shipped in an image: one is made for a GPU model and a TensorRT
version, so they would be built on the first run (seven models, about 6 minutes, 2.4 GB) and kept on a
volume, and built again after a driver, TensorRT or library update. And the serving is there already: one
separating process for the run with its models loaded (`separate._serve`, section 10). TensorRT did not need
isolating either: it ran inside the project's own process.

**The cost of building it**: the export and engine cache for five Roformer models (two measured; the
Mel-Band de-reverb and the two DrumSep models, which are not Roformers, not tried), the library's forward pass
taken apart (pinned to `audio-separator` 0.47.0), a 4.5 GB optional dependency, and stems 6 to 8 dB less exact,
for 5 to 10 % of a take's first run.

**What the measurements found instead**, without a new dependency:

- **Two chunks at once** in the compiled blocks: 0.79 for BS-Roformer-SW (run twice a source: four times and about
  140 s a take with a close track, so about 30 s of it), 0.95 for ep 317 (about 10 s of its 200 s); four at once is no better. The library runs one chunk at a time ("batch_size
  is not utilized" for Roformers), so its loop over the chunks would be replaced. The other three
  Roformers were not measured. About the size of TensorRT's gain, at float16's own noise from now's stems.
- **ep 317 at an overlap of 2** in place of 4 (section 10): about 50 s a source, 100 s a take with a close track, more than
  TensorRT gives the whole take, for stems of its own about −31 dB from now's: a change of sound to decide by ear.

`trt_core.log.txt` has the runs. TensorRT and ONNX Runtime were installed in a scratch folder and removed
again; the project's environment and lock file were not touched.

