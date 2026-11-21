# Plan: a simpler Mooz

Status, 11 November 2026: steps 1–4 and 10 are done, and so are two parts of the others (step 9's
kept stem chains, step 5's blind comparison); the rest of steps 5–9 is not started.
[DESIGN.md](DESIGN.md) describes the pipeline as it runs today.

- Steps 1–4: the safety net, the listening trial (judged better by ear), the original blend, and
  the trial's settings as the defaults. Decision of 7 November: the code is kept; nothing is
  pruned, and the stages that are off by default stay available as settings. The deletions step 1
  was waiting for have been made.
- Since then, outside the numbered steps: takes of several files (a Zoom's room and inputs), see
  "Close tracks"; the bed under the hi-hat and the original at a tenth, see "The bed under the
  hi-hat".
- Step 10, new on 10 November: a web app for projects, remastering, settings, progress and
  listening to every stage of a take ([GUI.md](GUI.md)).
- 10 November: in a merged stem the close track's stem now prevails, with a tenth of the room's,
  see "The close stem prevails". The masters of 9 November are not remade yet: they still have
  the old shares.
- 10 November: the mix bus and the master stage measured step by step, and a listening trial of
  five changes on three takes (`trial-base` against `trial-pro`), see "Mix and master,
  reassessed". Waiting to be judged by ear; the defaults are unchanged.
- 11 November: each stem's chain is kept from run to run, so a change to the mix or the master
  takes a third of the time (Miramare: 336 s, then 112 s, the same file byte for byte); and two
  profiles' masters can be compared blind in the app, the verdicts kept in `listening.json`. See
  "Kept chains and comparing blind". The trial of 10 November is still to be judged, now with this.
- 11 November: the merged bass was losing its lowest notes: lined up by one delay, the close
  stem and the room's were 137–158° apart at 63 Hz and took 4–6 dB from each other there. The
  close stem's lows are now turned to the room's phase (`[multitrack] phase_below_hz`). On three
  takes the master gains 3–5 dB at 63 Hz, see "The bass's lowest notes in the merge". To be
  heard: `trial-lows` against `trial-base`.
- 16 November: runs of the rehearsal of 12 November stopped half-way with nothing said: the 4 GB
  memory cap, reached by a take of 5:40. The cap now goes with the machine, a stopped run says why,
  and a take that is up to date is not made again. And the mix was liked better than the master by
  ear: the mix is now written next to it (`_mix`), and the two can be compared blind at the same
  loudness. See "Stopped runs, and the mix against the master".
- Where the 8 takes of 5 November stand against the targets, and what that says about the order
  of the remaining steps: "Where it stands" below. In short: the limiter now works 14–18 % of
  the time (the target is under 5 %), which makes step 5's trace the next thing to do.

## Where it stands (10 November 2026)

From the 16 reports of 9 November (`processed/full/` and `processed/simple_bass/` next to the
takes; the two profiles agree within 0.3 on every figure below, so one column serves both).

| target | on the 8 takes | |
|---|---|---|
| −14.0 ± 0.1 LUFS, true peak ≤ −1.0 dBTP | −14.0 (one −14.1), −1.0 | met |
| limiter active under 5 % of the time | 13.6–18.1 %, at most 2.0–2.3 dB | **missed on every take** |
| master keeps at least 60 % of the recording's loudness range | 40, 64, 58, 77, 66, 40, 55, 55 % | met on 3 of 8 |
| no balance gain at the limit (±6 dB) | over 4 dB on two takes: Riempio's guitars +4.9, and the short Stanza's drums +6.0 (the limit) and vocals −5.7 | met on the 7 full takes |
| each group within ±1 dB of its target after the balance | on the 7 full takes the four groups sit 0.2–1.8 dB above their targets, together to within 0.6 dB in each take; from take to take a group moves by up to 1.0 dB (vocals −6.1 to −5.1 LU) | met as a pattern: the balance sets the groups against each other, not their absolute level |
| buzz clean-up takes under 1 dB from hi-hat and cymbals | nothing: it is off on both. The bed step takes 5–13 dB between the hat's hits, 0.2–0.3 dB at them, by design | met |
| samples add at most 1.5 dB to any stem | kick +1.5 to +1.8, snare +0.1 to +0.9, toms 0 to +0.4 | kick over by up to 0.3 |
| tone within ±1.5 dB of the songs' median; voice-to-band ratio | not reported (step 8) | |

What follows:

- **The limiter.** (Looked at since: "Mix and master, reassessed". It is 0.1 dB on average, and the
  clipper can take most of it.) On the five September songs with the blend at a quarter it worked
  2.0–5.4 % of the time (step 3); on these takes it is three to four times that. Three things changed
  between the two sets and nothing here says which one did it: the recordings (another evening,
  with a close track), the close stems merged into voice and bass, and the original at a tenth
  instead of a quarter. One song on its own, mastered once per change, would tell. This is step
  5's trace, and it now comes first.
- **The loudness range** is still roughly halved where the recording had little to begin with
  (Miramare and Riempio i vuoti: 40 %). Same trace.
- **The kick's samples** are just over the limit that step 2 proposed and that was never built.
  Either the limit is built (a few lines in `augment.py`) or the target becomes 2 dB.
- **Reproducibility (step 6)** is unchanged: tone targets and the mix-tone library still depend
  on which songs were processed before. The web app makes this more visible, not worse: it runs
  the same command line.
- **Iteration time (step 9)**, as it was on 10 November: about 4 minutes per take and profile for
  any tweak. The web app's progress events show where the time goes: on Miramare (4 minutes, stems
  cached, 259 s) the stem chains take 65 % (the bass 35 s, the snare 30 s), the mix bus 10 %, the
  character 8 % (`progress.STAGES`). Since 11 November the stem chains and the character are kept
  from run to run: a mix or master tweak is a third of a full run ("Kept chains and comparing
  blind").

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
  the factorisation and the `[buzz]` settings ([cleanup.py](../mooz/cleanup.py),
  the buzz part of `_tracks` in [pipeline.py](../mooz/pipeline.py)).
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

### 5. What is on and what is off

Decision of 7 November 2026: **the code stays.** Nothing is deleted or parked; what the trial
switched off is off by default and can be switched back on in `rehearsal.toml`. The earlier
plan of pruning a third of the package is dropped; the simplification is in what runs, not
in what exists.

| | on by default | off by default (settings) | to decide by A/B (step 5) |
|---|---|---|---|
| separation | the ensemble plan | cascade plan, demucs backend, mask refinement, backing-vocals step | vocal de-reverb step |
| drums | hit gates, kit gate, EQ, tone match, own and studio samples at half level, drum bus balance and glue | buzz fingerprint on hats and cymbals, cymbal and hi-hat samples | kick shaper vs transient shaper, studio samples on the shells, shared shell room |
| bass | expander, EQ, tone match, fundamental restore, harmonic clean-up, compressor, transient, saturator, exciter, enhancer, kick duck | `note_eq` | exciter, bass enhancer, transient shaper, saturator, resonance suppressor, harmonic clean-up |
| guitars, vocals | EQ, tone match, compressor (vocals), vocal unmasking duck, plate and slapback | | resonance suppressor, saturator, exciter |
| mix bus | original blend, mono lows, tonal EQ, multiband, tape, glue, consistency EQ, clipper, loudness, limiter | song styles (`character.adapt`), residual add-back (gone) | multiband compressor, clipper, tape |
| presets | default, albini, indie, modern, interpol (all kept; the three built on the default inherit the new defaults) | | |

## Review findings and where they are handled

Every weakness found in the review of 5 November 2026, with the step that deals with it.
All are still present.

### Verified defects

| defect | where | handled by |
|---|---|---|
| removed vocal room comes back through the residual | [pipeline.py:288](../mooz/pipeline.py#L288) | fixed (step 3): the residual is no longer added; the original is blended |
| song style overrides per-song `[files]` settings (documented the other way round) | [pipeline.py:421](../mooz/pipeline.py#L421) | fixed (step 3): the style is now a layer under the song's own settings |
| per-song `[files."X".tracks.<track>]` does not reach the tracks based on it (`guitar_left` / `guitar_right`) | [config.py:92](../mooz/config.py#L92) | fixed (step 3): tracks expanded after every layer |
| presets write different content to the same files in the shared stems cache (`_vocals_recombined.wav`, `_proc_*.wav`, `_buzz/`) | [pipeline.py:401](../mooz/pipeline.py#L401), [pipeline.py:265](../mooz/pipeline.py#L265) | fixed (step 3): work files under `<out dir>/.work/<song>/`; `_buzz/` goes in step 4 |
| `librosa` imported but not declared as a dependency | [augment.py:409](../mooz/augment.py#L409), [note_eq.py:133](../mooz/note_eq.py#L133) | done (step 1) |
| plain `uv run pytest` collects `experiments/` and fails | [pyproject.toml](../pyproject.toml) | done (step 1) |
| the docs gave wrong test counts (36 and 49 for 56) | README, DESIGN | done on 5 November 2026 |
| latent: if the loudness loop has not settled after 5 rounds, the reported gain is not the one applied | [pipeline.py:29](../mooz/pipeline.py#L29) | fixed (step 3) |
| latent: a kit piece the balance marks as dropped is not removed from the drum bus (cannot happen with today's groups) | [pipeline.py:127](../mooz/pipeline.py#L127) | fixed (step 3) |

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
| not a git repository; the code sits inside 170 GB of data, about 34 GB of it experiment virtualenvs, on a disk 81 % full | | done (step 1): a repository; the virtualenvs and model caches are gone, the disk at 74 % |
| fragile separation glue: it patches audio-separator internals and the global `torch.load`, and the dependency is only bounded from below (`>=0.47`) | [separate.py:398-410](../mooz/separate.py#L398-L410) | done (step 1): pinned, with a test |
| batch handling: one failing file aborts the batch; `rehearsal.toml` and `--learn-balance` read from the current folder while outputs go next to the input | [cli.py:258](../mooz/cli.py#L258), [cli.py:266](../mooz/cli.py#L266) | fixed (step 3): per-file errors reported, the batch goes on; settings read from the recordings' folder. `--learn-balance` itself: step 6 |

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
   - Done (found so on 10 November 2026): the four experiment virtualenvs, the model caches
     under `experiments/` and the stray `processed/processed/` are gone; `experiments/` is
     0.5 GB and the disk is at 74 % (was 81 %).

   *Done when:* plain `uv run pytest` passes (it does: 61 checks then, 103 now), the
   first commit exists (it does) and the disk space is back (it is).
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
3. **Original blend and correctness.** Done on 7 November 2026.
   - Done: the blend (decision 3), `[mix] original_share = 0.25`; the residual add-back and
     `residual_gain_db` are gone. The balance measures each group in the blend and solves
     for the gain on the stems ([pipeline.py](../mooz/pipeline.py) `_balance`).
   - Done: settings resolved in one place ([config.py](../mooz/config.py)), in
     the documented order, tracks expanded after every layer; the command line is a layer.
   - Done: work files under `<out dir>/.work/<song>/`, removed when the run ends; nothing is
     written to the stems cache any more but the stems, `key.json` and `_profile.json`.
   - Done: a failing song is reported and the batch goes on (exit code 1 at the end);
     `rehearsal.toml` is read from the recordings' folder, then the current one.
   - Done: the two latent defects.
   - Run on the five songs into `processed/trial-blend/` (the trial settings plus the
     blend; A/B files there too). Against the trial: the original sits at −4.77 LU under the
     stems in every song (gain −3.2 to −4.1 dB); the vocals end within 0.6 LU of their
     target in all five (the stems' gain is larger than the change wanted, as designed:
     −9.4 dB on the stems for −5.1 wanted on Stanza, +8.0 for +6.0 on Abitudine); the
     limiter works less (2.0–5.4 % of the time, trial 4.3–7.4 %); loudness range up
     0.1–0.6 LU; 2–6 kHz up 0.8–1.4 dB and 6–16 kHz up 0.2–0.8 dB (the original brings
     back some of the top the trial lost); 4–5 minutes per song as before.

   *Done when:* with every processor off the output equals the recording (null test: yes,
   [tests/test_pipeline.py](../tests/test_pipeline.py), within −60 dB), the original
   measures 4.8 LU under the processed part in every song (yes), and two presets can run on
   one song at the same time (the shared cache is no longer written to; not tried live).
4. **Keep the configuration.** Done on 7 November 2026, in place of the pruning step.
   - Done: the trial settings are the defaults (`defaults.toml`): buzz removal on snare and
     toms only, samples at 0.15 / 0.2, no studio hats or cymbal samples, `character.adapt =
     false`; with the blend of step 3 at 0.25. `rehearsal-trial.toml` is gone (its content
     is the default; it stays in the git history).
   - Done: README, DESIGN and `defaults.toml` describe what runs by default and what is
     available, with the dates and reasons of the changes.
   - Not done, by decision: no code removed, no presets parked. The 1.5 dB sample limit is
     deferred to step 5 (the samples are now half as loud and the hi-hat samples are off).
   - `processed/` still holds the masters of the old chain; a plain `uv run mooz`
     now produces what `processed/trial-blend/` holds.

   *Done when:* a run with no settings file gives the approved sound (the defaults load and
   the 66 tests pass; a null comparison against `trial-blend` is still to be made).
5. **Objective quality loop.**
   - Extend the room benchmark ([experiments/sep_bench](../experiments/sep_bench): 12
     simulated-room songs with their close-mic sources) past the separation: processed
     stems against the close-mic sources (tone, attack), balance against the known
     levels, loudness range in and out.
   - Trace the loudness-range loss stage by stage on the five songs, from the stage files.
   - A blind, loudness-matched A/B of two settings files (an extension of `--ab`). Done on
     11 November, in the web app instead: two profiles' masters of a take as A and B
     ("Kept chains and comparing blind").
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
     or master tweak re-runs only the mix. Done on 11 November (`chaincache.py`), without the
     split into stage functions: the stem loop of `_tracks` asks the cache per stem.
   - `--jobs` across songs, inside the memory cap.

   *Done when:* the outputs null against step 8's, and a mix-level tweak re-runs in under
   30 s per song. (11 November: the outputs are the same file with and without the cache; a
   mix-level tweak takes 112 s on a 4-minute take, not 30. What the 112 s are is in "Kept chains
   and comparing blind".)

10. **Web app.** Done on 10 November 2026: [GUI.md](GUI.md).
    - A local page over the command line: a project from a folder of recordings, takes and
      profiles ticked and remastered, the settings in a form, progress by stage.
    - Added the same day: every profile can be edited (a built-in one as the project's version
      of it), and each take has a page that draws the pipeline, with the recording, the stems,
      every stage file and the master to play at the same moment of the song.
    - For it, the command line takes several `--preset`, a project can hold presets of its own
      (`presets/<name>.toml` next to its `rehearsal.toml`), and the pipeline reports its stages
      (`progress.py`).

    *Done when:* a take is remastered from the page, start to finish, and the tests pass (both: a
    4-minute and a 47-second take into a throw-away folder; 99 checks).

Order: step 1 protects everything after it. Step 2 costs nothing and may change the rest.
The cuts in step 4 come before the benchmark in step 5 (less to measure) and the profile
in step 6 (the buzz library disappears with them). Step 7 needs step 5's numbers. The
cache in step 9 comes last: it needs the determinism from step 6, and it is cheaper to
build on the shorter chain. (The cache was built first after all, on 11 November: it does not
need the determinism, because what changes from outside, a stem's tone target and the buzz
fingerprint, is part of each chain's key. A chain is made again when they change.)

## Snare buzz in the other stems: assessment (7 November 2026)

The question: the snare's wires buzz along with bass, guitars and voice; the drum splitter
spreads that buzz over the hi-hat, cymbal and tom stems; by default nothing removes it from
the hats and cymbals any more. What can still be done? Measured on the five songs with the
cached stems and the `trial-blend` stage files ([experiments/buzz/](../experiments/buzz/):
`buzz_where.py` and `buzz_lift.py`, run from the project root with `uv run python`).

**Where and when it is.** The buzz-only situation, kit idle (no kick, tom or snare for a
while) while the band plays, is 0.3–7.6 % of each song (Stanza 6.1 %, Abitudine 7.6 %,
Riempio 0.3 %). In those frames, of the drum stems, the cymbals stem holds the most
1.2–12 kHz energy in four songs out of five (e.g. Abitudine −56 dB against hi-hat −68 and
snare −84), then the hi-hat, then the snare, the toms nothing. Caveat: this measure cannot
tell buzz from a cymbal wash ringing into a break; the correlation of those frames' level
with the exciters' level (0.4–0.6 on Stanza and Abitudine, where the frames are many) says
that a good part of it does follow bass, guitars and voice, i.e. is buzz. The rest of the
time, between hits while the kit plays, the buzz sits under the hats' and cymbals' own
decay: the hi-hat stem between strokes is only 3–12 dB under its strokes, the cymbals
13–40 dB.

**What the chain does to it.** Nothing, in either direction: the hat and cymbal chains move
the between-stroke level and the stroke level together, within ±1 dB on every song (raw
stems against `2_stems`). The buzz is not lifted by processing; it is as audible as the
recording makes it, less the kit gate (−10 dB on hats and cymbals in idle passages) and
plus the original blend.

**The blend caps every stem-side fix.** Whatever a gate removes from a stem, the original's
share (0.25, amplitude 0.58 against the stems) brings back: a stem fully muted still leaves
that sound at −8.7 dB in the mix; the kit gate's −10 dB on the stems comes to about −5 dB.
Deepening the kit gate to 20 dB would gain about 2 dB more; the cap moves only with the
share (0.15 gives −10.6 dB) or by letting the original follow the same gate in the buzz
band, which is code.

**Tried and dropped, with numbers:**
- *Fingerprint removal (NMF) on hats and cymbals:* 2–4.5 dB out of the band, heard as
  artifacts; this is why it is off there (decision 1).
- *Predicting the buzz from its exciters* (earlier work): explained 12–60 %, mostly noise.
  Now: the correlation of the hat and cymbal stems' level with the exciters in buzz-only
  frames is 0.2–0.6, so a predictor would explain at most about a third.
- *Sustained/transient split (HPSS) on the hi-hat stem* (Vita and Stanza, margins 1–4):
  the buzz splits evenly between the two parts (−10.6 and −11.5 dB at margin 2), because
  wire rattle is itself a stream of micro-transients, not a steady noise; and the hat
  strokes are not cleanly percussive either (−4.8 against −9.8 dB). No separation there.

**What can be done, in order of effect:**
1. *In the room*, by far the largest: the strainer off (snares disengaged) in intros and
   breaks, a little damping on the snare wires or the resonant head, the bass amp away from
   the kit or turned down a little in the room, a high-pass on the bass amp. This removes
   the buzz from the recording, where no stage can reach it, and costs nothing. Worth a
   test at the next rehearsal: one song with the snares off in the intro.
2. *Kit gate deeper and quicker* on hats and cymbals (`kit_gate depth_db` 10 → 15,
   `hold_s` 2 → 1.5): settings only, acts only in idle passages (0.3–7.6 % of a song), about
   2 dB more in the mix because of the blend's cap; low risk, as the stem's own strong
   strokes still open it. To try by ear.
3. *A lower share or a gated original:* `original_share = 0.15` moves the cap from −8.7 to
   −10.6 dB and also brings less room and bleed back everywhere; or (code, step 5
   candidate) the original's 1.2–12 kHz band follows the kit gate in idle passages. Both
   change more than the buzz.
4. *A shallow hit gate on the hi-hat* (as the snare's, 4–6 dB, release 150 ms): the stem
   between strokes is only 3–12 dB under its strokes, so the gate would act on the hats'
   own decay as much as on the buzz. Likely to be heard as the artifacts we just removed;
   not recommended without a blind A/B.
5. *Better buzz removal than the fingerprint* is not in sight: the three model-based
   approaches tried (NMF, exciter prediction, HPSS) fail for the same reason, the buzz and
   the hats are the same kind of signal in the same band. A snare-off reference recording
   (the band playing a passage with the snares disengaged, then engaged) would give a true
   buzz template to test any future method against; without it, listening is the only
   judge.

**Done on 7 November 2026 (items 2 and 3):**
- Kit gate 15 dB (was 10), hold 1.5 s on the hi-hat and 2 s on the cymbals (were 2 and 3):
  defaults.
- The original follows the kit gate in the buzz band (`[mix] kit_gate_band_hz = [1200, 12000]`):
  while the drummer is not playing, that band of the original goes down by the gate's own
  curve, open as soon as either the hi-hat's or the cymbals' gate is. This removes the
  blend's cap in those passages. Cost: the band holds the guitars' and voice's top too, so
  intros and breaks are a little less bright in the blend.
- Measured on Stanza OK and Abitudine Strato (`processed/trial-gate/`, with A/B files),
  against `trial-blend`: in the idle passages (9 % and 11 % of the songs) the master's
  1.2–12 kHz against the loudness-matched recording went from +1.7 to −1.6 dB (Stanza) and
  from +1.5 to +0.5 dB (Abitudine); the full band in those passages, and everything while
  the kit plays, unchanged within 0.3 dB. The buzz itself sits 20–40 dB under the guitars
  in that band, so these figures are mostly the cost (less top in the breaks); what they
  bought in buzz is for the ear. Limiter and loudness range unchanged.
- To listen: `processed/trial-gate/` against `processed/trial-blend/` on those two songs,
  intros and breaks.

## Close tracks (7 November 2026)

Asked for: all stems from `*TrLR.wav`; just voice, bass and guitar (both guitars if sufficiently
audible) from each `Tr<n>.wav`; all of them used in the mix; the stems before the final master kept
for listening; the full and the simplified bass, stems and mixes.

**One deviation from the request: the room is `TrMic`, not `TrLR`.** On these recordings `TrLR` is
not the embedded mic. It is the recorder's own mix-down: exactly 1.413 `TrMic` + 1.000 `Tr2` on all 8
takes (residual −149 dB). Separating it would put the close track in every room stem and then in
the mix twice. `TrMic` is the stereo embedded mic. `[multitrack] room = "TrLR"` forces the literal
reading, and without a `TrMic` file `TrLR` is used anyway.

What the 8 takes hold, on the only input in use (`Tr2`, a mono mixer feed):

| stem | level (LU re the track) | used | delay to the room | polarity |
|---|---|---|---|---|
| voice | −2.5 to +0.7 | yes, every take | 9.6 to 11.5 ms | inverted, always |
| bass | −6.2 to −1.4 | yes, every take | 11 ms (16 ms on one Stanza) | as is, always |
| guitar | −26 to −20 | no: bleed on every take | no stable delay | |
| other | −7 and −2 on 2 takes, else silent | no: not asked for | 9.6 ms, the voice's path | |
| drums | −32 to −17 | no: not asked for | | |

- **Guitar.** It never reaches the close track as anything but bleed, so on these takes the guitar
  stem is the room's alone. The rule stays: guitar on an input is used when it is within 15 LU of
  the track's loudest stem, and two guitars on two inputs are both kept. A mono feed cannot be split
  into two guitars by position.
- **`other`.** An instrument on the voice's channel on Miramare and the full Stanza. By the earlier
  notes it may be the harmonica, not verified by ear. It stays in the room's `other` stem. Adding
  `"other"` to `[multitrack] stems` brings the close version in.
- **The merge is measured, not assumed.** In all bands from 50 Hz to 6.4 kHz the lined-up sum stays
  within ±2.2 dB of the parts' powers on every take, with the whole-take delay.
- **The short Stanza** (`261105_233934`, 47 s) is probably a false start. It is processed like the
  rest. It is one of 8 in the medians, which a single outlier does not move.
- **Result (8 November 2026).** Both versions of all 8 takes, with A/B files, are in
  `processed/full/` and `processed/simple_bass/` next to the takes, the stems before the master in
  each take's `stages/<take>/5_premix/` (they add up to the mix before the bus chain). Every master
  is at −14.0 LUFS and −1.0 dBTP. Loudness range is 2–7 LU, 14 on the short Stanza. Two passes were
  run so that every song's tone correction sees all 8 songs: it moves no band by more than 0.9 dB.
  The simple bass has the same lows (within ±1.5 dB below 500 Hz) and 3.3–8.1 dB less above
  500 Hz, where its harmonic stages were.
- **A defect found on the way:** a drum sample that starts abruptly stopped the song (a negative
  start cut an empty reference). Fixed with a test.

## The bed under the hi-hat (9 November 2026)

Heard on Biancaneve (`261105_235855`) from 54 s: snare-wire rattle all through the hi-hat stem,
and in the snare stem, whose hit gate already takes care of it (−18 dB between hits).

**What it is, measured.** A dark noise (peak near 3 kHz, −15 dB by 12 kHz) 15 dB under the hat's
hits, in every gap between them. It appears when the full band comes in, is 8.6 dB louder while the
singer sings, and has zero coherence with the close voice (the room vocal stem has 0.5–0.6): a
rattle the PA sets off, not a leak. It keeps its shape with the voice on or off, does not follow
the voice's level within a phrase, is not tied to kick hits and does not decay after cymbal hits.
Before the drum split it is 7 dB under the hat's hits; the 5-stem DrumSep puts it 18 dB under, the
6-stem one 10 dB, the 0.75/0.25 average 15 dB. The hat and the bed differ by 1.6 dB left to right:
no spatial cue. No gate reaches it: the kit gate never closes (the hats never stop) and the ducks
are keyed on hits it does not follow. The buzz fingerprint removal (`[buzz]`) on the hi-hat was
heard as artifacts earlier.

**What was done.** A stage 3 step for the hi-hat ([bed.py](../mooz/bed.py), `[bed]`,
details in [DESIGN.md](DESIGN.md)): hat, bed and the snare's burst as three shapes learnt from the
song, the stem keeping the hat's share per bin in 1.2–8 kHz, the first 25 ms of each hat hit and
everything above 8 kHz untouched, the burst keyed on the raw snare stem. On Biancaneve after 54 s
it takes 12.7 dB between hits and 0.3 dB at the hits (2–4 kHz). What it takes from the stem is
taken from the original too, as with the kit gate. The original is at a tenth of the loudness in
`rehearsal.toml` (9.5 LU under the stems; the default stays a quarter): the recording's full bed,
including the part the drum split drops into no stem, comes through it, and at a quarter it was
of the same order as the hat stem's own.

**Result on the 8 takes (9 November 2026, both versions, two passes; every master −14.0 LUFS,
−1.0 dBTP).** Taken between the hat's hits in 2–8 kHz, with the hits within 0.3 dB everywhere:

| take | between hits | hat / bed contrast | note |
|---|---|---|---|
| Miramare | −5.2 dB | 7.5 dB | its bed is half bright: a hat-like sound fills the gaps too |
| Vita (both) | −9.5, −10.6 dB | 17, 19 dB | |
| Stanza (both) | −12.8, −11.9 dB | 19, 18 dB | |
| Riempio i vuoti | nothing taken | 2.0 dB | what fills the gaps is as bright as the hat (an open hat or the tamburello): the step rightly stays out |
| Biancaneve (both) | −11.3, −11.0 dB | 15 dB | |

The tone correction between songs moves no band by more than 0.9 dB. The pre-master stems in
`5_premix/` add up to the mix before the bus chain as before.

**Not done.** The cymbals stem holds a bed 8 dB lower (its ride wash would be at risk; `[bed]
stems` can include it). Giving the hi-hat to the 5-stem model alone would gain 3 dB before any
clean-up, at the cost of the average's smoothing. The real fix is at the source: dampen the snare
or move the vocal monitor away from it at the next rehearsal.

## The close stem prevails (10 November 2026)

Asked for, after listening to Miramare's bass in the web app: the close track's stem is the
cleaner one, so it should prevail in the merged stem, with a little of the room's added,
in proportion to the two stems' quality if that can be measured.

**It could not be measured.** Six figures were tried on the voice and the bass of the 8 takes,
room stem against `Tr2` stem (scripts not kept: each is a few lines on the cached stems).

| figure | what it says | why it does not serve |
|---|---|---|
| the instrument's level against the rest of its track, in its band (voice 200 Hz–5 kHz, bass 50–700 Hz) | voice: room −13 to −5 dB, close +1 to +7. Bass: room −7 to 0, close −3.5 to +6 | as shares: voice 0.89–0.96, bass 0.62–0.86. Miramare's bass comes out at 0.62: less of the close stem than before, against what is heard. It measures the track the separation was given, not the stem it gave |
| the share of the stem's energy in time-frequency cells its track gives mostly to it | voice: 0.90–0.96 room, 0.92–0.96 close. Bass: 0.83–0.90 room, 0.90–0.98 close | the separation's output is near all-or-nothing per cell, so every stem looks alone; nothing for the voice |
| the stem's leftovers while its instrument pauses and the band plays | voice: 34–42 dB under the singing in the room's stem, 27–39 dB in the close one | the model shuts the room's voice in the pauses, the close track keeps its bleed and noise there: it ranks the room's voice cleaner. The bass pauses for under 3 s on 4 takes |
| coherence of the two stems, lined up | 64–90 % of the room's stem is predicted by the close one | the same number seen from either side: it says how much they share, not which one holds the rest |
| what the other track's stems explain in a stem beyond its own instrument | 1.6 % at most, in either | neither holds much of another instrument that a filter can find; what differs is elsewhere (the room, the amp, the separation's artefacts) |
| level against the rest of the track, weighted by the stem's own spectrum | voice: the close stem from 3.6 dB behind to 8 dB ahead | changes sign between takes |

Two recordings of one instrument, each with leftovers of its own, share one number (the
coherence); telling whose the leftovers are needs a third recording, or a judgement. So the
share stays a setting, and the judgement is the one asked for:

- **`[multitrack] close_share` is 0.9 for every instrument**: the room's stem 9.5 LU under the
  close one, a tenth of the loudness (the voice was at 0.8, bass and guitar at 0.7, `other` at 0.6).
  Which close stems count at all is decided as before, by level (`min_level_lu`).
- **The merged stem's loudness no longer follows the share**: `level_lu = 6.0` puts it 6 LU over
  the room's own stem. Until now the room's stem stayed as it was and the close one was added, so the
  sum grew with the share (5.2 LU over at 0.7, 7 LU at 0.8, 10 LU at 0.9) and the balance had to
  take it back. From the balance gains of 9 November:

  | the merged stem | voice gain, 8 takes | bass gain, 8 takes |
  |---|---|---|
  | as on 9 November (0.8 and 0.7) | −5.7 to +2.1 dB | −2.5 to 0 dB |
  | 0.9, the close stem raised | −8.7 to −0.9 | −7.1 to −4.6: over the 6 dB limit on 4 takes |
  | 0.9, at the room stem's loudness | +1.3 to +9.1: over on 3 takes | +2.7 to +5.2 |
  | 0.9, 6 LU over the room's stem (chosen) | 0.5 to 1.6 dB more than now | from 0.5 less to 0.3 more |

  The chains do not care (thresholds and drives follow the stem's own level).
- **Checked on the 8 takes** by running the merge into a scratch folder: in every merged voice and
  bass the room's part is 9.5–9.6 LU under the close part and the stem 6.0–6.1 LU over the room's
  own. The close stem's gain went from 13.5–17.1 dB to 14.9–18.2 dB (bass) and from 10.7–16.0 dB to
  10.0–15.2 dB (voice): `Tr2` is recorded 16 dB under the room track, and what it costs is its noise
  floor raised by that gain.
- **Not done:** the masters. The merged stems have a new cache folder (the merge's settings name
  it), and the tone targets are medians over that folder's songs, so all 8 takes need mastering
  again, twice for the targets to see all of them. Not judged by ear yet either.

## Mix and master, reassessed (10 November 2026)

Asked for: the steps of the mix and the master looked at critically, for a sound as good and as
professional as it can be. Measured on the 8 masters of 9 November, their stage files and
reports. Nothing here is judged by ear: that is what the trial at the end is for.

**What was found**, worst first:

| | measured | what it means |
|---|---|---|
| loudness range | lost in the mix bus, not in the master stage: on the 7 full takes 3.0–9.4 LU before the bus and 1.9–6.8 after it (Miramare 5.3 → 3.0, Biancaneve 7.2 → 4.6); the master stage takes 0.0–0.4 more | the bus levels the sections. By its steps (run one after the other on 4 takes): the bus compressor 0.7–1.3 LU, the multiband 0.3–1.0, the tape 0.1–0.2. The bus compressor is asked for 1.2 dB on average while the band plays; it gets there with 3.5–4.7 dB on the loud sections and none on the quiet ones |
| low end | the 40–80 Hz octave is 7–9 dB under the 80–160 Hz one in every master; at 50 and 63 Hz the masters are 10 and 7 dB under the bus EQ's own target | it is the band's sound, not a step's fault: the recording (−7 to −8 dB), the room's bass stem (−5 to −8), `Tr2`'s bass stem (−9 to −12) and the kick (−8 to −10) are all like this. The bus EQ's boost at 63 Hz stops at its 3 dB limit on all 8 takes. With the close stem at 0.9 the masters are 0.3–0.8 dB thinner there again |
| balance | after the balance the drums are 2.2 dB over the voice (drums −3.0 to −3.7 LU re the mix, voice −5.1 to −6.1) | the learnt targets keep the rehearsal room's balance; mix engineers put the voice about 4 dB over the drums on average (De Man 2017, the priors in `[balance]`, weighted 0.3) |
| width | left/right correlation 0.88–0.93 in the masters, 0.51–0.76 in the recordings; side 12–14.5 dB under mid | most of it is what a mix is: a loud voice, bass and kit in the middle (the parts' sum before the bus: 0.72–0.89). The bus's mono ramp up to 400 Hz adds 0.04–0.11 (on three takes) and takes 1–3 dB of side between 120 and 600 Hz; a ramp to 200 Hz would give about half of that back. Not in the trial |
| limiter | active 14–18 % of the time, 1.9–2.3 dB at most, but 0.08–0.10 dB on average (three takes) | it is the tails of drum hits, not a mix held down: the target of 5 % overstated the problem. The mix's peaks stand 16.3–17.6 dB over its loudness and the master's 13, so 3.3–4.6 dB of each hit has to go, and it is the limiter that takes the last 2 |
| tone above 100 Hz | within ±3 dB of the EQ's target in third octaves; from take to take within 1.5 dB per octave (the short Stanza aside) | lumps at 200, 400 and 630 Hz of +2.8 dB each (the room); nothing changed |
| bus tape | third harmonic −52 dB at the mix's level, −32 dB 10 dB over it; a 70 Hz tone puts sidebands on a 1 kHz one at −42 dB | clean where the mix sits, saturating on the hits (1.0–3.6 dB of their height). Left as it is |

**The trial.** Two profiles of the project (`presets/trial-base.toml`, `presets/trial-pro.toml`), so
that both show on a take's page in the web app and the same moment can be heard under each.
`trial-base` is the settings as they are, with the close stem at 0.9. `trial-pro` changes five
things, each for one row above:

| setting | from | to | for |
|---|---|---|---|
| `[glue] target_gr_db` | 1.2 | 0.6 | the loudness range |
| `[clipper] headroom_db` | 2.0 | 0.5 | the limiter: the hits' tops clipped (under 1 ms each) instead of the mix ducked after each |
| `[balance] prior_weight` | 0.3 | 0.5 | the voice against the drums |
| `[eq] max_boost_db` | 3.0 | 5.0 | the low end (only the 63 Hz band was at the limit) |
| `[tracks.bass] bass_fundamental.restore_db` | −3 | 0 | the low end: a note's fundamental as loud as its second harmonic, the median of raw studio bass tracks |

On Miramare, the first Vita and the first Biancaneve (10 November, all at −14.0 LUFS and −1.0 dBTP):

| | `trial-base` | `trial-pro` |
|---|---|---|
| loudness range | 2.9, 2.3, 4.3 LU (41, 64, 56 % of the recording's) | 3.4, 2.8, 5.0 LU (49, 78, 65 %) |
| bus compressor at most | 3.9, 3.9, 4.9 dB | 2.7, 2.6, 3.6 dB |
| limiter | 14–19 % of the time, 2.0–2.3 dB at most | 5.6–7.5 %, 0.6–1.0 dB |
| clipper | 0.07–0.09 % of the samples, 1.8–2.9 dB at most | 0.18–0.21 %, 3.0–4.1 dB |
| 40–80 Hz against 80–160 Hz | −9.7, −8.2, −9.2 dB | −7.8, −6.0, −6.5 dB |
| voice and drums re the mix | −5.9 and −3.6, −5.9 and −3.4, −5.1 and −3.0 LU | −4.8 and −4.4, −4.6 and −4.0, −4.0 and −3.7 LU |
| width | 0.88, 0.93, 0.92 | the same |

- The masters of 9 November (`full`) and `trial-base` differ by the close stem's share alone: the
  range within 0.1 LU, the limiter within 1 % of the time, the low end 0.3–0.8 dB thinner.
- The bus still takes 0.9–2.0 LU of the range in `trial-pro`. The multiband is the next step to
  look at (0.3–1.0 LU), then the mono ramp.
- What the September songs' limiter (2–5 % of the time) had that these takes do not is still not
  known; with the clipper at 0.5 dB it no longer matters much.

**To do:** listen (`trial-base` against `trial-pro`, and both against `full`). What is kept
becomes the defaults; then all 8 takes are mastered again.

Since 14 November `trial-pro`'s five settings are also a built-in preset, `pro`
([pro.toml](../mooz/presets/pro.toml)), so that a project in another folder can run them: a new
project's form in the web app now lists every built-in preset instead of a fixed six. Still not
judged by ear.

## Kept chains and comparing blind (11 November 2026)

Two things stood between a setting and a judgement of it: every tweak cost a full run, and the
judgement was made knowing which master was which. Both are built; neither changes the sound.

**Kept chains** ([chaincache.py](../mooz/chaincache.py), `[chains]` in the settings).

- What one stem's chain gives is kept in `processed/chains/<take>/<stem>/<key>/`: the processed
  stem as 32-bit float (the very samples the drum bus and the mix get), its part of the report,
  the curves of the stem figures, the kit gate's gain and what the bed clean-up took (the
  original follows both), and its stage files.
- The key is everything the chain reads: the code (every module but the command line, the
  figures, the file export and the progress events; the versions of numpy and scipy), the stem's
  file, the trim, the files the hit triggers come from, the stem's whole chain settings, its
  profile and tone target, the bed and buzz settings where they apply (with the buzz fingerprint
  itself), the sample libraries' index, the stage files' format. The same key: the chain is not
  run. So a bus, balance or master setting runs no chain, a stem's own setting runs that one,
  and two profiles that differ in one chain share the others.
- The song's character (stage 0) is kept the same way: it reads the recording and five raw
  stems, and no settings.
- Per stem the results of the last three settings stay; results of other code are removed.
  `--no-chain-cache` runs every chain and keeps none. A cache that cannot be written is said
  and the run goes on.
- Stage files and the processed stem are second names for the kept files where the disk allows
  it (hard links), so nothing is copied; a stage file is never written through such a name.

Measured on Miramare (4:00, a linked copy in a throw-away folder, stems separated before, A/B
files and figures on). The same run took 259 s on 10 November; the laptop was slower this
morning, with or without the cache:

| run | time | stem chains | |
|---|---|---|---|
| the cache off (`--no-chain-cache`) | 330 s | 9 made, 206 s | |
| nothing kept yet | 336 s | 9 made and kept, 209 s | keeping costs 6 s; the same file as with the cache off |
| the same again | 112 s | 9 kept, 1 s | the master is the same file, byte for byte |
| `simple_bass` after `default` | 136 s | 8 kept, the bass made (20 s) | the same file as a run of `simple_bass` on its own |

- The test suite holds the same on a synthetic song: a second run makes no chain and writes the
  same file; a bus setting changes none; a bass setting changes the bass only; and a run with
  the cache off gives the same file as one with it.
- On disk: 1.3 GB for the take, of which the nine processed stems are 0.8 GB (92 MB each).
- What the 112 s are: the mix bus 37 s (its compressor is run 8 times to find the threshold for
  its target), loudness and the limiter 19 s, blend and balance 15 s, writing 15 s (the two MP3
  files 9 s), learning the buzz and the hit triggers before the chains 10 s, reading and
  preparing 8 s, the drum bus 6 s, the figures 3 s.
- Not done: the learning of the buzz before the chains could be kept as well (10 s). Under 30 s,
  step 9's aim, needs the bus compressor's search and the limiter's loop made cheaper, which is
  work on the sound's own code and wants a null test of its own.
- Not tried: the cache on a disk without hard links (the code falls back to copies), two runs
  on one take at once, and a first run from the web app (the job's line shows "bass, kept";
  seen on the command line's events only).

**Comparing blind** ([GUI.md](GUI.md), job 6; [listening.py](../mooz/gui/listening.py)).

- On a take's page, two profiles' masters as A and B, drawn by lot, both decoded and started on
  one sample, each turned down to the quieter one's loudness; a passage can loop; the choice (A,
  B or no difference, with a note) is written to `listening.json` next to `rehearsal.toml`, and
  only then are the sides named.
- The takes' page adds the verdicts up per pair of profiles. A verdict stops counting when one
  of its two masters is made again.
- Checked in the test suite (the verdicts' file) and in headless Chrome on the linked copy of
  Miramare, `default` against `simple_bass` (the view, the keys, the loop, the saved verdict).
  Not checked: by a person, with ears. The first real use is the trial of 10 November:
  `trial-base` against `trial-pro` on Miramare, the first Vita and the first Biancaneve.

## The bass's lowest notes in the merge (11 November 2026)

Asked: are the bass's lows not decreased too much in the mix and the master? Measured on Miramare,
the first Vita and the first Biancaneve (`trial-base`), in third octaves, stage by stage.

**Against the recording, no.** Against its own whole level the master has more low end than the
recording: 40–80 Hz by 0.6 to 4.6 dB, 80–160 Hz by 1.7 to 5.6 dB. The bus adds about 2 dB to the
lower octave and 1–2 dB to the upper (its EQ gives +2.8 dB at 63 Hz); the master stage takes 0.2 dB. The balance turns the bass's
stem down by 2.3–3.3 dB, to −5.9 to −6.7 LU under the mix, which is still above what engineers
do (−8 to −9).

**Inside the bass stem, yes, at one point: the merge.**

| at 63 Hz, re each file's 160–1250 Hz | Miramare | Vita | Biancaneve |
|---|---|---|---|
| the room's bass stem | −10.1 | −9.0 | −15.5 |
| `Tr2`'s bass stem | −14.2 | −13.6 | −15.2 |
| the two merged | −18.3 | −16.7 | −18.3 |

- The merged stem had less at 63 Hz than either of its parts. Inside it, the room's part and the
  close part were 158°, 153° and 137° apart at 63 Hz (coherence 0.93–0.95) and about 90–120° at
  80 Hz: the sum was 6.3, 5.7 and 4.4 dB under what the two give in phase at 63 Hz, 2.6–4.0 dB
  at 80 Hz, and 1.9–3.3 dB under the close part alone at 63 Hz.
- Why: one delay (11 ms) lines the two up where the bass's body is (50–700 Hz as a whole). At the
  bottom the phase between them runs away, about 20 ms' worth between 50 and 80 Hz: what a low
  cut on the mixer's feed and the room do to the lowest notes. The earlier check ("within
  ±2.2 dB in every band") was in whole octaves against the summed powers, and did not show it.
- The same happened once more in the mix: the original recording under the stems holds the bass
  in the room's phase, 116–146° from the bass stem at 63 Hz. Adding it took another 0.5–1.6 dB
  there.
- The voice does not have this: its phase against the close stem differs from band to band all
  the way up (the PA in the room), with nothing lost in particular at the bottom.

**The fix** (`multitrack.phase_to_room`, `[multitrack] phase_below_hz = { bass = 250.0 }`, the
other instruments 0): below 250 Hz the lined-up close stem's phase is turned to the room's,
band by band, from the two stems' cross-spectrum over the take; its spectrum is untouched (an
all-pass, within 0.05 dB; 0.2 dB at one band). It is followed only where the two are clearly the
same sound. The room is the reference because the other stems and the original are in its phase.

| `trial-lows` over `trial-base` | Miramare | Vita | Biancaneve |
|---|---|---|---|
| merged bass, 63 / 80 Hz | +6.2 / +2.4 dB | +5.6 / +3.8 | +4.4 / +3.3 |
| master, 63 / 80 Hz | +5.2 / +0.9 dB | +3.5 / +2.4 | +3.2 / +1.9 |
| master, 100–160 Hz | −1.0 to +0.1 | −1.0 to −0.6 | −0.7 to −0.3 |
| master, 40–80 re 80–160 Hz | −9.7 → −6.3 dB | −8.2 → −3.7 | −9.2 → −6.0 |
| bass stem to the original, phase at 63 Hz | −146° → 4° | −131° → 8° | −116° → 1° |
| loudness range | 2.9 → 2.7 LU | 2.3 → 2.4 | 4.3 → 4.3 |
| limiter at work | 18.9 → 18.6 % | 16.8 → 16.8 | 14.3 → 14.2 |

- The recordings themselves have −7.9, −6.8 and −8.2 dB for 40–80 re 80–160 Hz: the masters now
  have a little more of the bottom octave than the room had, where they had less.
- What it costs: the close bass's lowest notes now sit where the room's do, up to about 20 ms
  later against their own harmonics than on `Tr2`. Not judged by ear.
- `trial-pro` got 2 dB at the same place another way (the bus EQ's limit raised, the bass's
  fundamentals restored). Whether those two are too much on top of the merge put right was
  measured the same day: `trial-pro-lows` is `trial-pro`'s settings with the fix (the masters in
  `processed/trial-pro` are left as they were). Third octaves of the masters against the bus EQ's
  own target (−2 dB per octave, fitted from 100 Hz up), Miramare / Vita / Biancaneve:

  | | 50 Hz | 63 Hz | 80 Hz | 40–80 re 80–160 Hz |
  |---|---|---|---|---|
  | the recording | −20 / −21 / −23 | −10.9 / −11.4 / −12.4 | −1.1 / −0.8 / −3.3 | −8.2 / −7.1 / −8.3 |
  | `trial-base` | −11 / −11 / −9 | −8.3 / −7.1 / −7.0 | +2.2 / +0.4 / −0.4 | −9.7 / −8.2 / −9.2 |
  | `trial-lows` | −12 / −12 / −9 | −2.9 / −3.3 / −3.6 | +3.3 / +3.1 / +1.7 | −6.3 / −3.7 / −6.0 |
  | `trial-pro` | −9 / −8 / −6 | −5.9 / −4.8 / −4.3 | +2.3 / +0.4 / −0.2 | −7.8 / −6.0 / −6.5 |
  | `trial-pro-lows` | −9 / −9 / −7 | −1.7 / −2.1 / −2.0 | +3.5 / +3.2 / +2.1 | −5.0 / −2.7 / −4.2 |

  By that target the two are not too much: with both, 63 Hz is still 2 dB under it and 50 Hz 7–9.
  What stands over it is 80 Hz, by 2–3.5 dB, and that comes from the fix, not from `trial-pro`:
  the thing to listen for is a boom on the notes around 80 Hz (E2, F2). The restoring of the
  fundamentals does less by itself once they are there (on Vita 5.5 → 3.4 dB on average).
  Elsewhere `trial-pro-lows` is `trial-pro`: range 3.1 / 2.9 / 5.0 LU, limiter 5–7 % of the time.
- Both sources are thin under 80 Hz (the recording falls 9 dB per third octave below it, `Tr2`'s
  bass stem 5–8 dB below 100 Hz). Whether that is a low cut on the recorder and on the mixer's channel, or
  the amp and the room, the audio does not say; the switches are worth a look before the next
  rehearsal.
- Not done: the 8 takes under `full` and `simple_bass` (they still have the old merge); the fix on
  other instruments (`phase_below_hz` is 0 for them: not measured as needed).

## Stopped runs, and the mix against the master (16 November 2026)

Reported after mastering the rehearsal of 12 November (7 takes, `default` and `pro`) from the web
app: a take sometimes stops with no error shown, and starting again makes every finished take
again; and on both profiles the mix sounds better than the master, punchier and louder.

**Why the runs stopped.** The system's log has it: on 14 November at 20:43 and on 16 November at
10:16 the run was killed inside its own memory scope ("Memory cgroup out of memory", 4.14 and
4.15 GB in use, the scope "Failed with result 'oom-kill'"). The cap was a fixed 4 GB.

| Lisa dice, 5:40, the longest take | highest memory in use |
|---|---|
| reading, preparing, stems from the cache | 0.9–1.1 GB |
| the song's character | 2.7 GB |
| the stem chains | 4.1 GB (the hi-hat's; the bass's 4.0, the snare's 3.9) |
| drum bus, blend, mix bus, master, files, figures | 2.4–3.3 GB |
| the same with every chain kept but the bass's | 3.7 GB (the bass's chain; else 3.1) |

- About 1.3 GB plus 0.5 GB per minute of take: a take over five and a half minutes is at the old
  cap, and whether it goes through was a matter of a few megabytes.
- A kill leaves the run no word to say. The app showed "the run stopped before this take was
  finished"; a terminal, "Killed".

What changed (none of it in the sound):

- **The cap goes with the machine**: four fifths of the memory free when the run starts, at most
  half of all there is, at least 4G (here 15G). `MOOZ_MEMORY_MAX` sets it; the old name works.
- **A stopped run says why.** The run is now a child of the process that starts it, which waits,
  asks systemd how the scope ended, and says "the system stopped the run: it needed more memory
  than the 15G it may use", to the terminal and to the app (the job's message and the take it
  stopped at).
- **What is up to date is not made again.** A take's report carries the name of what made it
  (program, the take's settings, the recorder's files); with that name and its files in place the
  take is left as it is, in the app ("up to date") and on the command line. `--again`, or "Also
  what is up to date" in the app, makes it again. What the other takes bring (tone targets) is
  not in the name.
- A run prints the most memory it has used after each take.
- Checked on a linked copy of Lisa dice: with the limit at 2G the run ends with the message and
  exit code 137; with the machine's 15G it goes through (4.0 GB at most, 315 s; `pro` after it
  156 s with 7 chains kept); the same command again makes nothing and says "up to date" twice.
- Not done: using less memory. The stem loop holds the recording twice, four trigger curves and
  each chain's own copies in 64-bit; halving that is work in `_tracks`, which costs every kept
  chain once.

**The mix against the master.** The master step is three things after the mix (`5_mix`): one
gain to −14 LUFS, the clipper, the limiter, with the ceiling at −1 dBTP. Measured on the 7 takes:

| | `default` | `pro` |
|---|---|---|
| the mix's own loudness | −10.5 to −13.6 LUFS | −9.8 to −13.2 LUFS |
| its highest peak | +3.4 to +5.7 dBTP | +4.2 to +6.5 dBTP |
| its peaks over its loudness (PLR) | 15.5–17.4 dB | 15.7–17.5 dB |
| the master's | 13.0 dB on every take | 13.0 dB on every take |
| so the master step takes off the peaks | 2.5–4.4 dB | 2.7–4.5 dB |
| and plays the take quieter than the mix by | 0.4–3.5 dB | 0.8–4.2 dB |
| clipper, at most / limiter at work | 0.8–2.6 dB / 9–15 % of the time | 2.2–4.0 dB / 2–7 % |

- Both things heard are there. **Louder**: on the take's page everything plays 12 dB under its
  real level, so the mix, whose peaks stand 3–6 dB over full scale, plays as it is, and it is up
  to 4 dB louder than the master. **Punchier**: the master has 2.5–4.5 dB less of every drum
  hit's height. That is not a setting gone wrong: −14 LUFS under a −1 dBTP ceiling leaves
  exactly 13 dB for the peaks, whatever the mix.
- A file cannot hold the mix as the page plays it. What it can hold is the mix with its highest
  peak at the ceiling: −16.5 to −18.5 LUFS, 2.5–4.5 dB quieter than the master, all peaks whole.
  That is now written next to every master: `<take>_mix.wav` and `.mp3` (`[output] mix`), the
  mix with the master's fades and that one gain. On Lisa dice: −17.4 LUFS (`default`) and −17.5
  (`pro`), PLR 16.4 and 16.5, against the masters' −14.0 and 13.0.
- Which of the two was liked for its level and which for its peaks, the ear can now say:
  Compare blind takes a master against its own mix and turns the louder down to the other.
- If the mix still wins at the same loudness, the remedy is a lower target, not another limiter:
  at −16 LUFS the master step takes 0.5–2.5 dB off the peaks, at −17 at most 1.5 (from the PLRs
  above; not run). The other way, to keep −14, is to bring the peaks down inside the mix, hit by
  hit on the drum stems, where 3 dB is heard less than on the whole mix. Neither is decided.

To do: master the 7 takes again (the files of 16 November have no `_mix` and no name of what made
them, and the kept chains are of the program before this change: one full run); then the blind
comparison of masters and mixes.

## Trying it today with settings only

Historical: this was the step-2 trial block. Since 7 November 2026 these are the defaults
and the block is no longer needed; it is kept to show what the trial changed.

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

The "today" column is from the five October reports; the 8 takes of November are in "Where it
stands" at the top.

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
3. indie, modern and interpol are parked; default and albini stay. (Superseded on 7 November:
   all presets are kept, see decision 5.)

## Open question

The share is defined on loudness energy, which is what the pipeline's meters measure. On
the psychoacoustic scale, where 10 dB sounds twice as loud, "a quarter as loud as the
rest" would instead put the original about 16 LU under the processed part, a much
subtler blend. `original_share` is one setting, so step 3 can audition both ends (0.25
and about 0.03) before fixing the default.
