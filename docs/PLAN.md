# Plan: a simpler rehearsal-master

Status, 7 November 2026: step 1 (safety net) is done, apart from two deletions that wait
for a yes; step 2 (the settings-only trial) has run and waits for the listening. Nothing that changes the sound is implemented: the audio code and
`rehearsal.toml` are as they were. It replaces no other document: [DESIGN.md](DESIGN.md)
still describes the pipeline as it runs today.

## Aims

1. **Simpler.** Fewer stages and fewer settings, one chain for every song.
2. **No hi-hat or cymbal artifacts**, accepting that less of the snare's resonance (the
   wires buzzing along with bass, guitars and voice) is removed.
3. **Drum samples quieter** in the mix.
4. **The original recording mixed in**: the whole stereo recording, making up a quarter of
   the mix's volume as heard (measured loudness, not a fader position).
5. **The result:** good quality, homogeneous from song to song, good loudness and clarity.

## What the reports say today

From the five default-preset reports in `processed/reports/` (runs of early October 2026).

| measurement | Stanza OK | Riempio i vuoti | Miramare | Abitudine Strato | Vita |
|---|---|---|---|---|---|
| buzz removal, snare (dB in 1.2–12 kHz) | −0.6 | −0.6 | −0.2 | −0.2 | −0.2 |
| buzz removal, toms | −0.2 | −0.1 | −0.1 | −0.4 | −0.2 |
| buzz removal, hi-hat | −4.1 | −2.1 | −1.9 | −4.5 | −4.3 |
| buzz removal, cymbals | −3.8 | −3.4 | −3.3 | −3.9 | −4.1 |
| snare hit gate (average dB down) | 5.6 | 4.6 | 5.6 | 6.0 | 4.6 |
| samples add to the kick (dB) | +0.8 | +2.0 | +2.1 | +1.7 | +1.6 |
| samples add to the snare | +1.1 | +3.0 | +3.1 | +2.9 | +3.2 |
| samples add to the hi-hat | +0.7 | +0.7 | +1.1 | +0.6 | **+11.1** |
| samples add to toms / cymbals | +0.1 / +0.1 | +0.2 / +0.1 | +0.1 / +0.1 | +0.1 / +0.1 | +0.5 / +0.1 |
| loudness range, recording → master (LU) | 11.3 → 7.8 | 5.7 → 2.4 | 6.4 → 2.9 | 7.2 → 4.5 | 4.3 → 2.2 |
| vocals re the remix before balance (LU) | −1.6 | −11.4 | −7.3 | −12.5 | −12.2 |
| balance gain on the vocals (dB, limit ±6) | −5.6 | +2.6 | +0.6 | +5.7 | +5.0 |
| limiter active (% of the time) | 6.3 | 0.2 | 5.2 | 7.4 | 0.7 |
| song style applied | dynamic | wall | steady | dynamic | wall |

The buzz figures are the energy change of the whole stem in the buzz band, hits included,
so the reduction between snare hits is larger than the snare row shows.

What follows from them:

- The buzz fingerprint removal does most of its work on the hi-hat and cymbals (2–4.5 dB)
  and almost none on the snare and toms. The snare's resonance is mainly handled by its
  hit gate. The stage that is most likely to cause hat and cymbal artifacts is also the
  one the snare needs least.
- The studio hi-hat samples raised Vita's hi-hat stem by 11 dB. Nothing limits how much
  samples may add.
- The loudness range is roughly halved in three songs out of five.
- The vocals sit anywhere from −1.6 to −12.5 LU before the balance, and the balance then
  moves them by up to 5.7 dB against a 6 dB limit. The measurement compares whole-song
  loudness, so it mixes up level with arrangement (Stanza opens with voice and drums
  alone).
- Three different styles are applied to five songs, each changing bus compression, tape
  and width. That works against homogeneity.

## Decisions

### 1. Snare resonance: keep the gates, drop the fingerprint removal

- Remove the buzz fingerprint stage: the learning step, the library in `stems/_buzz/`,
  the factorisation and the `[buzz]` settings ([cleanup.py](../rehearsal_master/cleanup.py),
  the buzz part of `_tracks` in [pipeline.py](../rehearsal_master/pipeline.py)).
- Keep the snare and tom hit gates, including the snare's extra cut of its wire band
  between hits (`gate.buzz_extra_db`), and the kit gate on hi-hat and cymbals while the
  drummer is not playing.
- Nothing is removed from the hi-hat and cymbal stems while the kit plays. The 2 dB dips
  on kick hits (`ducks`) are a candidate to go as well.

Cost: the buzz that the splitter files under hi-hat and cymbals stays in those stems.
Benefit: no masking artifacts on them, about 150 lines and one cross-song library less.

### 2. Samples: one quiet layer on kick, snare and toms; none on hats and cymbals

| stem | today | planned starting point |
|---|---|---|
| kick, snare, toms: the drummer's own samples | 0.3 (about −10 dB under each hit) | 0.15 (about −16 dB) |
| kick, snare, toms: studio samples | 0.4 (about −8 dB) | 0.2 (about −14 dB) |
| hi-hat: studio samples | 0.8 (about −2 dB) | off |
| cymbals: studio crash / ride samples | 0.5 / 0.35 | off |

- Add a limit: samples may not raise a stem's level by more than 1.5 dB; beyond that
  their mix is scaled down for that song and the report says so.
- With the hat and cymbal samples gone, the crash / ride / bell classification
  (`cymbal_augment`) and the brightness-matched library layer can be deleted.
- Whether to keep two layers on the shells or only the drummer's own is decided by
  listening (step 5).

The hat and cymbal samples were added because the separated stems are dull at the hits.
The original recording blended in (next decision) gives that top back without samples.

### 3. The original recording blended in, replacing the residual

Today the mix is the processed stems plus the residual (the recording minus the raw stems)
at `residual_gain_db`. Planned instead:

    mix before the bus = processed stems + g × original

with g set per song so that the original carries a share of 0.25 of the two parts'
measured loudness (`original_share = 0.25`).

- "Share" is of loudness energy, measured the way the rest of the pipeline measures level
  (integrated loudness, BS.1770): the original's loudness is put at
  10·log10(share / (1 − share)) LU against the processed part. For 0.25 that is 4.8 LU
  under it, the same on a loud song and a quiet one. This follows how sends are already
  set (`level_lu` against the dry track).
- The original is the trimmed recording after the rumble filter and hum notches.
- It is blended before the bus chain (stereo, tonal EQ, tape, glue), so the tape's phase
  shift applies to both alike.
- The residual add-back and `residual_gain_db` go: the original already holds everything
  the separation missed.

Consequences to know:

- Everything the clean-up removes comes back at that share: buzz, room, bleed. However
  deep a gate goes on a stem, that sound drops by at most about 9 dB in the mix. That is
  the accepted trade, and one more reason not to keep deep clean-up stages.
- Level moves shrink by about a third: a vocal raised 5 dB in the processed part ends up
  about 3.5 dB up in the mix. The balance therefore has to aim its targets at the blended
  mix, not at the processed part alone (step 3).
- The blend only works while the processed stems stay time-aligned with the recording.
  They are today (linear-phase EQ, gain-only dynamics, samples aligned to hits). The one
  exception is tape on a single track, which shifts phase: only the `indie` preset does
  that.
- It fixes a defect: today the vocal room that the de-reverb step removes re-enters
  through the residual (on Vita the residual is −17.6 dB re the mix, of which the removed
  vocal room is −18.1 dB; with no room reduction the residual is about −31 dB).

### 4. One chain for every song

- Song styles stop changing the mix (`character.adapt = false`, then the style tables are
  removed). The measurements stay in the report.
- Homogeneity rests on four things, all kept: the group balance, the per-stem tone match,
  the consistency EQ on the mix, and the loudness target.
- Their targets move out of side-effect libraries into one **band profile** file written
  by an explicit `learn` command and only read while processing. Processing a song twice
  then gives the same file, whatever was processed before.
- The balance measures each group only while it and the rest of the band play together,
  instead of comparing whole-song loudness.

### 5. What stays and what goes

A stage stays only if switching it off is audibly worse in a blind A/B on at least two of
the five songs, or moves one of the [targets](#targets) the wrong way.

| | stays | goes | decided by A/B |
|---|---|---|---|
| separation | the ensemble plan | cascade plan, demucs backend, mask refinement, backing-vocals step | vocal de-reverb step |
| drums | hit gates, kit gate, EQ, tone match, own samples, drum bus balance and glue | buzz fingerprint, cymbal and hi-hat samples | kick shaper vs transient shaper (one of them), studio samples on the shells, shared shell room |
| bass | expander, EQ, tone match, fundamental restore, compressor, kick duck | `note_eq` (off today, moved to `experiments/`) | exciter, bass enhancer, transient shaper, saturator, resonance suppressor, harmonic clean-up |
| guitars, vocals | EQ, tone match, compressor (vocals), vocal unmasking duck, plate and slapback | | resonance suppressor, saturator, exciter |
| mix bus | original blend, mono lows, tonal EQ, consistency EQ, glue, loudness, limiter | residual add-back, song styles | multiband compressor, clipper, tape |
| presets | default, albini | indie, modern, interpol: parked (below) | |

**Parked presets.** indie, modern and interpol move to `presets/parked/`: not listed by
`--preset`, not tested, not kept working while the chain is simplified. What only they use
goes with them: the pan split, the dual bass chain, the second compressor, sample
flattening, and the tape echo, delay and chorus sends. Grit stays (albini uses it), as do
the plate and slapback. Their existing masters in `processed/` are left as they are. They
come back, one at a time, once the simple chain sounds right.

## Review findings and where they are handled

Every weakness found in the review of 5 November 2026, with the step that deals with it.
All are still present.

### Verified defects

| defect | where | handled by |
|---|---|---|
| removed vocal room comes back through the residual | [pipeline.py:288](../rehearsal_master/pipeline.py#L288) | step 3 (decision 3) |
| song style overrides per-song `[files]` settings (documented the other way round) | [pipeline.py:421](../rehearsal_master/pipeline.py#L421) | step 4 (decision 4: styles removed) |
| per-song `[files."X".tracks.<track>]` does not reach the tracks based on it (`guitar_left` / `guitar_right`) | [config.py:92](../rehearsal_master/config.py#L92) | step 3 |
| presets write different content to the same files in the shared stems cache (`_vocals_recombined.wav`, `_proc_*.wav`, `_buzz/`) | [pipeline.py:401](../rehearsal_master/pipeline.py#L401), [pipeline.py:265](../rehearsal_master/pipeline.py#L265) | step 3; `_buzz/` goes in step 4 |
| `librosa` imported but not declared as a dependency | [augment.py:409](../rehearsal_master/augment.py#L409), [note_eq.py:133](../rehearsal_master/note_eq.py#L133) | done (step 1) |
| plain `uv run pytest` collects `experiments/` and fails | [pyproject.toml](../pyproject.toml) | done (step 1) |
| the docs gave wrong test counts (36 and 49 for 56) | README, DESIGN | done on 5 November 2026 |
| latent: if the loudness loop has not settled after 5 rounds, the reported gain is not the one applied | [pipeline.py:29](../rehearsal_master/pipeline.py#L29) | step 3 |
| latent: a kit piece the balance marks as dropped is not removed from the drum bus (cannot happen with today's groups) | [pipeline.py:127](../rehearsal_master/pipeline.py#L127) | step 3 |

### Design weaknesses

| weakness | evidence | handled by |
|---|---|---|
| results are not reproducible: tone targets, buzz fingerprint and mix-tone library change as a side effect of processing; stale entries are never purged; a new rehearsal folder starts with empty libraries; the band's balance targets are typed into the packaged `defaults.toml` | a song's master depends on which songs ran before it | step 4 (buzz library gone), step 6 |
| the balance mixes up arrangement with level | vocals from −1.6 to −12.5 LU before the balance; gains of −5.6 and +5.7 dB against a ±6 dB limit | step 7 |
| the loudness range is roughly halved, nothing guards it, and the stage that causes it is not known | 5.7 → 2.4, 6.4 → 2.9, 4.3 → 2.2 LU; albini, with no bus compression, 6.4 → 4.4 | step 5 (trace), step 7 (guard), step 8 (report) |
| mix decisions are not validated: only the separation has a benchmark; the rest is a 923-line settings file tuned by ear on five songs | | step 5 |
| too many overlapping stages | the bass alone runs ten processors plus EQ and a duck | decision 5, steps 4 and 5 |
| slow iteration: any tweak re-runs about 4 minutes per song; stage files are written but never reused | | step 9 |
| no end-to-end test; `process_file` and `_tracks` hold 365 lines of orchestration, file handling and caching | the 56 tests before step 1 exercised single functions on synthetic signals | test: done (step 1); split: step 9 |
| not a git repository; the code sits inside 170 GB of data, about 34 GB of it experiment virtualenvs, on a disk 81 % full | | git: done (step 1); disk: waiting for a yes |
| fragile separation glue: it patches audio-separator internals and the global `torch.load`, and the dependency is only bounded from below (`>=0.47`) | [separate.py:398-410](../rehearsal_master/separate.py#L398-L410) | done (step 1): pinned, with a test |
| batch handling: one failing file aborts the batch; `rehearsal.toml` and `--learn-balance` read from the current folder while outputs go next to the input | [cli.py:258](../rehearsal_master/cli.py#L258), [cli.py:266](../rehearsal_master/cli.py#L266) | step 3, step 6 |

### The review's seven points

| review point | step here |
|---|---|
| 1. Safety net | step 1 |
| 2. Correctness fixes | step 3 (the residual fix became the original blend) |
| 3. Explicit band profile | step 6 |
| 4. Stage functions and cache | step 9 |
| 5. Objective quality loop | step 5 |
| 6. Balance and dynamics rework | step 7 |
| 7. Prune | step 4 |

New with the aims: step 2 (listening with settings only) and step 8 (album report).

## Steps

Each step ends with the docs updated: README tuning table, DESIGN, `defaults.toml`
comments.

1. **Safety net.** Done on 6 November 2026, except the deletions.
   - Done: a git repository with two commits (the project as it stood, then this step);
     `.gitignore` for audio, `processed/`, virtualenvs, model caches and the third-party
     checkouts under `experiments/`.
   - Done: `testpaths = ["tests"]`; `librosa` declared.
   - Done: [tests/test_pipeline.py](../tests/test_pipeline.py) runs `process_file` on a
     12 s synthetic song with fixture stems, for default and albini: finite output, the
     loudness target within 0.1 LU, the true-peak ceiling, the reports and stage files.
   - Done: audio-separator pinned to 0.47.0, and
     [tests/test_separation_glue.py](../tests/test_separation_glue.py) checks the
     internals `separate.py` patches.
   - Done: each experiment virtualenv's package list is recorded next to its scripts
     (`experiments/*/requirements-*.txt`, with the third-party checkouts and their commits).
   - **Waiting for a yes:** deleting the four experiment virtualenvs (about 34 GB) and the
     stray `processed/processed/` (170 MB). The model caches under `experiments/`
     (`prompt_sep/hf_cache` 60 GB, `bass_sr/hf_home` 6 GB) can be downloaded again and are
     the larger saving.

   *Done when:* plain `uv run pytest` passes (it does: 61 checks, about a minute), the
   first commit exists (it does) and the disk space is back (not yet).
2. **Listen to the aims with settings only** (no code). Run on 7 November 2026 with
   [rehearsal-trial.toml](../rehearsal-trial.toml) (two passes, 45 minutes) into
   `processed/trial/`; the current masters in `processed/` are untouched, both at −14 LUFS,
   so the two can be compared directly. What the reports say, trial against current:
   - Hi-hat and cymbals: the buzz stage takes nothing from them (before: 1.9–4.5 dB); the
     snare's 0.2–0.6 dB is unchanged. Samples add half as much to kick and snare
     (0.4–1.6 dB, before 0.8–3.2); none on hats and cymbals (Vita's +11 dB is gone).
   - The master's top is lower: 6–16 kHz down 0.6–2.1 dB, most on Riempio i vuoti and Vita.
     Two causes: no studio hats and cymbal samples, and the kit balance now pulls the
     cymbals 2–4 dB further down, because the buzz left in that stem counts as cymbal
     level. If the hats and cymbals sound too far back, the first thing to try is
     `[drum_bus.balance] offsets = { cymbals = 4.0 }` (now 1.5), or the kit balance should
     measure the cymbals in a band the buzz does not share.
   - Without the song styles, the two "wall" songs (Riempio i vuoti, Vita) get 1.2 dB of
     glue instead of 1.8 and no parallel drum compression, so the limiter works harder on
     them (4.3 % and 7.4 % of the time, before 0.2 % and 0.7 %); their loudness range is
     up by 0.4 LU. Everything else (loudness, true peak, group balance, 60–120 Hz) is
     within 0.5 dB of before.

   *Done when:* the sample levels and the hat/cymbal trade are confirmed or adjusted by ear
   (waiting for the listening). Listen above all to Vita and Abitudine Strato (the largest
   hat and cymbal changes) and to the kick and snare punch on Riempio i vuoti.
3. **Original blend and correctness.**
   - Add the blend (decision 3) and remove the residual add-back; the balance aims at the
     blended mix.
   - Settings resolved in one function, in the documented order, with track inheritance
     applied last.
   - Work files in a per-run folder under the output folder, not in the stems cache.
   - A failing file is reported and the batch goes on; the exit code says so.
   - `rehearsal.toml` read from the input's folder (the current folder as a fallback).
   - The two latent defects in the table above.

   *Done when:* with every processor off the output equals the recording (null test), the
   original measures 4.8 LU under the processed part in every song, and two presets can
   run on one song at the same time.
4. **Cut (prune).**
   - Delete what the "goes" column lists; park the three presets.
   - Fix the sample levels from step 2; add the 1.5 dB sample limit.
   - Remove `process.sh` (the earlier ffmpeg/sox script); move `note_eq` to `experiments/`.
   - DESIGN describes what the pipeline does; its fix-by-fix history ("first version's
     artefacts", "tried and dropped") moves to the git history.

   *Done when:* the end-to-end test passes and the settings file and package are clearly
   smaller (aim: `defaults.toml` from 923 lines to about 500).
5. **Objective quality loop.**
   - Extend the room benchmark ([experiments/sep_bench](../experiments/sep_bench): 12
     simulated-room songs with their close-mic sources) past the separation: processed
     stems against the close-mic sources (tone, attack), balance against the known
     levels, loudness range in and out.
   - Trace the loudness-range loss stage by stage on the five songs, from the stage files.
   - A blind, loudness-matched A/B of two settings files (an extension of `--ab`).
   - Each stage in the "decided by A/B" column switched off in turn, on the benchmark and
     by ear; the ones that lose are deleted.

   *Done when:* every remaining stage has a recorded reason to exist, and the stages that
   cost the loudness range are named.
6. **Band profile.**
   - A `learn` command writes tone targets, balance targets (mix and kit) and mix tone to
     one file, built from a named set of songs, so nothing stale survives.
   - The file lives with the band's settings, not inside `processed/`, and serves any
     rehearsal folder. Processing only reads it.
   - The band's numbers leave `defaults.toml`, which keeps the published priors only.
     `--learn-balance` is replaced.

   *Done when:* two runs of the same song are bit-identical, the order of songs does not
   matter, and a new folder with a single song gets the full targets.
7. **Balance and dynamics rework** (driven by step 5).
   - Each group measured only while it and the rest of the band play together.
   - A warning when a balance gain exceeds 4 dB.
   - The stages named in step 5 reined in until the master keeps the loudness range in
     [Targets](#targets); the run warns when it does not.

   *Done when:* on the five songs no balance gain exceeds 4 dB without an understood
   reason, and the loudness-range target is met.
8. **Album report.** One table over all songs with the numbers under
   [Targets](#targets), printed after a batch, with a warning where one is missed.
   *Done when:* the five songs meet the targets or the misses are understood.
9. **Stage functions and cache.**
   - `process_file` and `_tracks` split into stages with explicit inputs and outputs.
   - Each stem's processed result cached under its resolved settings, so a balance, bus
     or master tweak re-runs only the mix.
   - `--jobs` across songs, inside the memory cap.

   *Done when:* the outputs null against step 8's, and a mix-level tweak re-runs in under
   30 s per song.

Order: step 1 protects everything after it. Step 2 costs nothing and may change the rest.
The cuts in step 4 come before the benchmark in step 5 (less to measure) and the profile
in step 6 (the buzz library disappears with them). Step 7 needs step 5's numbers. The
cache in step 9 comes last: it needs the determinism from step 6, and it is cheaper to
build on the shorter chain.

## Trying it today with settings only

Everything except the original blend can be heard now by adding this to `rehearsal.toml`
(checked: it loads with the current code).

```toml
[buzz]
stems = ["snare", "toms"]        # nothing taken from the hi-hat and cymbals

[character]
adapt = false                    # one chain for every song

[tracks.kick]
augment = { mix = 0.15, library = { mix = 0.2 } }

[tracks.snare]
augment = { mix = 0.15, library = { mix = 0.2 } }

[tracks.toms]
augment = { mix = 0.15, library = { mix = 0.2 } }

[tracks.hihat]
augment = { enabled = false }    # no studio hats

[tracks.cymbals]
cymbal_augment = { enabled = false }
```

## Targets

Proposed; to be confirmed after step 2.

| aim | number | today |
|---|---|---|
| loudness | −14.0 ± 0.1 LUFS, true peak ≤ −1.0 dBTP | met |
| loudness without strain | limiter active under 5 % of the time | 0.2–7.4 % |
| dynamics kept | master keeps at least 60 % of the recording's loudness range | 42–69 % |
| homogeneous levels | each group within ±1 dB of its target after the balance, with no gain at the limit | gains up to 5.7 of 6 dB |
| homogeneous tone | master's long-term spectrum within ±1.5 dB of the songs' median, 100 Hz–8 kHz | not reported today |
| hats and cymbals untouched | clean-up takes under 1 dB from either stem | 1.9–4.5 dB |
| samples stay under the kit | samples add at most 1.5 dB to any stem | up to 3.2 dB (snare), 11.1 dB (Vita's hi-hat) |
| clarity | voice-to-band ratio in 1–4 kHz reported per song, within ±1.5 dB across songs | not reported today |

The numbers do not replace listening; they catch a song that drifts from the others.

## Settled (5 November 2026)

1. "The original track mixed in" means the whole stereo recording.
2. 25 % is a share of the volume as heard, not an amplitude weight: a 0.25 share of
   measured loudness, which puts the original 4.8 LU under the processed part.
3. indie, modern and interpol are parked; default and albini stay.

## Open question

The share is defined on loudness energy, which is what the pipeline's meters measure. On
the psychoacoustic scale, where 10 dB sounds twice as loud, "a quarter as loud as the
rest" would instead put the original about 16 LU under the processed part, a much
subtler blend. `original_share` is one setting, so step 3 can audition both ends (0.25
and about 0.03) before fixing the default.
