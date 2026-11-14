# Mooz

Mooz (Master Out Of Zoom) turns stereo rehearsal recordings (one recorder in the room, e.g. a
Zoom) into balanced, mastered tracks: stems separated on the GPU, cleaned and processed one by
one, the drum kit mixed as one instrument, the band remixed and mastered, every stage saved for
listening. Originals are never modified. How it works: [docs/DESIGN.md](docs/DESIGN.md). Where
it stands and what is next: [docs/PLAN.md](docs/PLAN.md). The web app: [docs/GUI.md](docs/GUI.md).

## Usage

In a browser: projects, takes and profiles to tick, a Remaster button, progress by stage, the
settings and every profile in a form. Each take has a page of its own that draws the pipeline,
with the recording, the stems, every stage and the master to play at the same moment of the song.

```sh
uv run mooz-gui                   # http://127.0.0.1:8377, on this machine only
```

Or on the command line, which is what the web app runs:

```sh
uv run mooz                       # the recordings in rehearsal.toml's [input], with its [input] presets
uv run mooz "260928_224856_TrLR Miramare.WAV"   # one recording (or a whole take: any of its files)
uv run mooz ~/rehearsals/2026-10-05      # a whole folder, with its take folders
uv run mooz --preset simple_bass  # the shorter bass chain -> processed/simple_bass/
uv run mooz --preset default --preset albini   # several: every take once per preset ("default": none)
uv run mooz --separate-only       # only separate every track into the cache
uv run mooz --ab                  # also write loudness-matched A/B files
uv run mooz --preset albini       # Steve Albini-style mix -> processed/albini/
uv run mooz --preset indie        # gritty indie / garage rock -> processed/indie/
uv run mooz --preset modern       # modern pop-rock, radio-ready -> processed/modern/
uv run mooz --preset interpol     # dark post-punk, Interpol-like -> processed/interpol/
uv run mooz --target -16          # loudness target (LUFS)
uv run mooz --analyze             # measure only, write nothing
uv run mooz --no-chain-cache      # run every stem's chain again (they are kept from run to run otherwise)
uv run mooz --learn-balance       # print level targets learnt from all songs
```

The tape stage uses Chowdhury DSP's free CHOW Tape Model (VST3). Install it once; without
it a built-in saturator is used:

```sh
curl -LO https://github.com/jatinchowdhury18/AnalogTapeModel/releases/download/v2.11.4/ChowTapeModel-Linux-x64-2.11.4.deb
dpkg-deb -x ChowTapeModel-Linux-x64-2.11.4.deb chowtape && mkdir -p ~/.vst3
cp -r chowtape/usr/lib/vst3/CHOWTapeModel.vst3 ~/.vst3/
```

The drum augmentation also blends in studio drum samples (kick, snare and toms; hi-hat, crash and ride
samples exist too but are off). Build them once (about 15
minutes; they come from the Cambridge-MT raw multitracks and stay in your user cache):

```sh
uv run --with remotezip --with requests python tools/build_drum_samples.py
```

The first run of a song separates it (about 6–7 min on the GPU, cached afterwards); processing
takes about 4 min per song. The run is memory-capped (`REHEARSAL_MEMORY_MAX`, default 4G).

## Recordings

A take is every file from one moment. Zoom recorders write one per track,
`<YYMMDD>_<HHMMSS>_Tr<id>.WAV`, optionally in a folder per take named after the song
(`261105_225442 Miramare/`); two takes of a song keep their own names. A plain run works on
`[input] paths` in `rehearsal.toml`, now `new_tracks/drive-download-20261107T141948Z-1-001`.

| track | what it is | used for |
|---|---|---|
| `TrMic` | the built-in stereo microphone: the room | every stem, and the original under the mix |
| `Tr<n>` | input *n*: an external mic or a mixer: a close track | voice, bass and guitar, where it really holds them |
| `TrLR` | next to a `TrMic`, the recorder's own mix-down (measured: 1.413 `TrMic` + 1.000 `Tr2`, residual −149 dB); without one, the stereo pair itself | the A side of the A/B files; the room track only if there is no `TrMic` (or `[multitrack] room = "TrLR"`: the close track is then in it twice) |

Each close stem is lined up with the same instrument's room stem (delay and polarity, measured)
and added to it before its chain, so one chain and one fader serve each instrument. A stem more
than 15 LU under its track's loudest is bleed and stays out. A mono track holds one guitar stem:
two guitars on one mixer feed cannot be told apart, on two inputs both are used. Recordings with
a single file work as before.

Outputs and caches go in `processed/` next to the takes' folders.

## Outputs

In `processed/full/` and `processed/simple_bass/` (or the preset's folder), for every take:

| file | what |
|---|---|
| `<name>.wav` / `.mp3` | the master: 24-bit WAV and a 320 kbps MP3 tagged with title and date |
| `<name>_AB.wav` / `.mp3` | with `--ab`: original ↔ master every 10 s (original first: the recorder's mix-down if it wrote one), at the same loudness |
| `reports/<name>.png` | the mix stages: spectrum before/after, EQ, loudness over time, gain reduction |
| `reports/<name>_tracks.png` | every stem: level in the remix, spectrum, EQ, its dynamics over time |
| `reports/<name>.json` | every measurement and decision |
| `stages/<name>/` | the stages, to listen to (FLAC, all −12 dB so nothing clips and they line up in a DAW): |
| &nbsp;&nbsp;`character.json` | 0 · what kind of song it is (wall, ballad, build, dynamic, steady) and why, and every stem's profile |
| &nbsp;&nbsp;`1_raw/` | 1 · the separated stems, untouched (link to the cache) |
| &nbsp;&nbsp;`1_raw/models/` | 1 · every separation model's own outputs before they are averaged (`<step>_<input>/<model>/<stem>.flac`, 12 dB down, listed in `index.json`), and each average (`ensemble/`) |
| &nbsp;&nbsp;`1_raw_Tr<n>/` | 1 · a close track's separated stems (link to its cache) |
| &nbsp;&nbsp;`1_merged/` | 1 · what the chains start from: the room's stems with the close stems lined up and added (link) |
| &nbsp;&nbsp;`2_stems/` | 2 · every stem cleaned and studio-processed; `samples/`: the drum samples made from the song |
| &nbsp;&nbsp;`3_drum_cleanup/` | 3 · drum stems after the buzz/bleed clean-up, and `*_removed`: what it took out |
| &nbsp;&nbsp;`4_drum_bus` | 4 · the drum kit mixed as one instrument |
| &nbsp;&nbsp;`5_premix/` | 5 · every part as it goes into the mix, balanced and placed, the original too: they add up to the mix before the bus chain, to remix or listen to in a DAW |
| &nbsp;&nbsp;`5_mix` | 5 · the full mix before loudness and limiting |

## How it works, in short

1. **Prepare**: trim (2 s lead-in, fade-in), rumble filter, mains-hum notches.
2. **Separate** (stage 1): vocals (three models averaged; the de-reverb model's dry and room
   parts put back together, the room turned down smoothly: no glitches), guitars (optionally
   split in two by where their amps stood), bass, kick, snare, toms, hi-hat, cymbals (two
   DrumSep models averaged per piece), other. All from the whole recording, with weights chosen
   on rehearsal-room test songs ([experiments/sep_bench](experiments/sep_bench)). What the
   separation misses stays in the recording, which is blended under the stems (7). Every
   model's own outputs are kept for listening.
3. **Characterise** (stage 0): loudness curve, sections, contrast, build-up, density, tempo,
   drum and vocal activity, guitar distortion → a style (reported; it adjusts the mix stages
   only with `character.adapt = true`: by default every song gets the same chain). And
   every raw stem, before any clean-up: level, noise floor, tone, brightness, punch, compared
   with the same instrument's median over all songs.
4. **Clean the drums** (stage 3): the snare buzz that bass, guitars and voice set off is learnt
   and removed from the snare and tom stems (the hi-hat and cymbals share its band and are left
   alone: taking it from them was audible). The bed under the hi-hat, the snare's wires rattling
   along with the PA, which the drum split files under the hat in every gap between its hits, is
   taken out by its shape: each moment is read as bright hat + dark bed + the snare's burst, and
   the stem keeps the hat's share, its hits untouched. Bleed keyed to kick and snare hits is
   ducked; gates between hits and while the drummer is not playing.
5. **Studio stems** (stage 2): kick, snare and toms are reinforced with samples made from the
   drummer's own cleanest hits, the bass's fundamental is pitch-tracked and raised, the voice
   gets its chest resonance back when it goes thin. Each stem is matched to its instrument's
   usual tone and attack (optionally blended with raw studio tracks of it), then gets the
   EQ, dynamics and harmonics of a good close-miked track of that instrument (EQ choices: see
   *EQ references*). The expanders clean down to each stem's measured noise floor.
6. **Drum bus** (stage 4): kit pieces balanced against each other, glue and parallel compression.
7. **Mix and master** (stage 5): the original recording under the processed stems at a quarter
   of the loudness by default, a tenth in `rehearsal.toml` (`[mix] original_share`: what the
   separation took out is still there, at a known level; what the hi-hat's bed clean-up and the
   kit gate took is taken from it too), instruments placed and balanced in that blend (consistent from song to song,
   blended with how mix engineers balance a band: voice up, drums down a little), guitars
   making room for the voice's words, the low end widening gradually, the same overall tone
   in every song,
   bus EQ and compression, tape (CHOW Tape Model), loudness target, true-peak limiter.

## Presets

- **simple_bass**: the same mix with a much shorter bass chain, to compare with the full one:
  expander, EQ with tone match, compressor, fundamental restore and kick duck; no harmonic
  clean-up, resonance suppressor, transient shaper, saturator, exciter or enhancer.

- **albini**: Steve Albini's approach: the band in a room, as it played. The rehearsal room's
  defects are still fixed (hum, buzz, levels); the polish is dropped: no bus compression or
  clipper, no transient shaping, dry loud wide guitars, gritty bass, raw vocals compressed a
  few dB, the vocal room kept, a barely-glued kit with more room and cymbals, CHOW Tape hit
  a little hot, −18 LUFS.

- **indie**: gritty indie / garage rock, character over polish. A barely gated kit with a
  crushed, saturated room under it (the buzz fix stays). Tape echo on the vocals. A distorted
  copy of the bass grinding through the guitars. Guitar mids left forward, with tape taming
  the top. The mix on hot tape into a bus compressor that breathes with the kick, −14 LUFS.

- **modern**: modern pop-rock, built on the default, maximum impact and clarity:
  - *Vocals:* serial (fast + slow) compression, air with strict de-essing, a micro-pitch
    double for width instead of more reverb.
  - *Drums:* tuned studio drum samples at an even level on every hit, sharper and tighter.
  - *Low end:* the kick owning the sub, the bass levelled hard just above it.
  - *Guitars:* the two split and panned hard left and right, carved and ducked for the voice.
  - *Mix:* −14 LUFS.

- **interpol**: dark, spacious post-punk in the vein of Interpol's *Turn On the Bright Lights*:
  - *Bass:* a lead instrument, loud and picked, through a dual chain: clean, mono lows; a
    driven, chorused upper part.
  - *Guitars:* two interlocking, trebly guitars panned apart, with chorus, dotted-eighth
    delays at the song's tempo and a plate.
  - *Vocals:* the baritone kept full and set inside the band (long plate, slapback,
    quarter-note echo).
  - *Drums:* tight and driving, with reverb on the snare.
  - *Mix:* a darker tone, −14 LUFS.

Presets live in [mooz/presets/](mooz/presets/) and win over
`rehearsal.toml`; the song's style (only with `character.adapt = true`) and the per-song
`[files]` settings apply on top, in that order. A project can have presets of its own:
`presets/<name>.toml` next to its `rehearsal.toml`, used by name like the built-in ones (the web
app calls them profiles and can make one from a copy). Give each its own `[output] dir`, or it
overwrites the default's masters.

## Tuning

Put changes in `rehearsal.toml` (only the keys you change; every option is explained in
[defaults.toml](mooz/defaults.toml)). Per song: `[files."<part of name>"]`.

| it sounds… | try |
|---|---|
| the bass thin at the bottom on a take with a close track | it should not be: below 250 Hz the close bass is turned to the room's phase so the two add (`[multitrack] phase_below_hz`); `{ bass = 0.0 }` gives the earlier merge, to compare |
| voice / bass too dry and upfront, or too roomy | `[multitrack] close_share = { vocals = 0.8, bass = 0.7, guitar = 0.9, other = 0.9 }` (more of the room's stem; the default is 0.9 for all: the close stem with a tenth of the room's) |
| an instrument on the close track left out, or bleed let in | see `[multitrack]`: `min_level_lu` (default −15), `stems` (add `"other"`) |
| too much of the room recording / too processed | `[mix] original_share = 0.05` (drier, the stems in front) .. `0.35` (closer to the recording; 0 = stems only; the band's `rehearsal.toml` has 0.10, the default 0.25) |
| muddy / boxy | `eq.strength = 0.9` |
| too bright / not warm enough | `eq.tilt_db_per_oct = -2.3`; more guitar saturation `[tracks.guitar] saturator = { mix = 0.5 }` |
| vocals too roomy / too dry | `[stems.vocal_room] max_reduction_db = 9.0` (0 = as recorded; default 6) |
| more / less tape | `[saturation] chowtape = { drive = 0.7 }` (0..1; `bias = 0.5` grittier); `engine = "builtin"` for the old saturator |
| vocals too dry / too wet | `[tracks.vocals] sends = [...]`: the plate's `level_lu` (default −18; −14 wetter, −22 drier) |
| vocals lost in loud parts | `[rider] max_up_db = 4.0`, `strength = 0.8` (or `enabled = false`) |
| kit too separate / too roomy | `[drum_bus] room = { level_lu = -12.0 }` (−20 drier; `enabled = false`) |
| mix not glued / too squashed | `[glue] target_gr_db = 2.0` (or 0.6) |
| vocals too far forward / back (all songs) | `[balance] prior_weight = 0.0` (the band's own balance) .. `0.6` (more like a studio mix) |
| vocals too quiet / loud everywhere | `[balance] offsets = { vocals = 2.0 }` (or −2.0); same for `guitars`, `bass`, `drums` |
| kick or snare too quiet in the kit | `[drum_bus.balance] offsets = { kick = 2.0 }` (`snare`, `toms`, `cymbals`) |
| kick not punchy | `[tracks.kick] kick_shaper = { punch_db = 5.0 }`, or more samples: `augment = { mix = 0.5 }` |
| drums sound "sampled" | `[tracks.snare] augment = { library = { mix = 0.1 } }` (studio samples, default 0.2; `enabled = false` for none), `augment = { mix = 0.1 }` (the drummer's own, default 0.15) |
| kick / snare not punchy enough | `augment = { mix = 0.3, library = { mix = 0.4 } }` (the levels before November 2026); `flatten = 0.5` evens the hits out |
| hi-hat dull / low | `[tracks.hihat] gain_db = 5.0` (default 4); studio hats on its hits: `augment = { enabled = true }` (off by default, at 0.8 under each hit) |
| crashes / ride dull | `[tracks.cymbals] cymbal_augment = { enabled = true }` (studio cymbal hits by kind, off by default; `mix = { crash = 0.5, ride = 0.35, bell = 0.0 }`: another cymbal's bell clashes); more hits taken as crashes: `crash_pct = 80.0` |
| bass too boomy / still thin | `[tracks.bass] bass_fundamental = { boost_db = 1.0 }` (or 3.0); the low notes' lost fundamental: `restore_db = -6.0` (less) .. `0.0` (more) |
| bass buzzy / gritty | `[tracks.bass] harmonic_clean = { depth_db = 18.0 }` (default 12; `enabled = false` for none); `exciter = { amount_db = -23.0 }` |
| bass clanky / smeared (try) | `[tracks.bass] note_eq = { enabled = true }`: note by note, each note's harmonics up to where they stand out, its attack evened out with the song's others, the rest above 250 Hz down 15 dB (replaces harmonic_clean; `depth_db`, `attack = { max_cut_db = ... }`) |
| stems should sound more "studio" | `[tracks.<stem>] tone_match = { reference = 0.7 }` (0 = the band's own median tone, 1 = raw studio tracks) |
| snare dull | `[tracks.snare] transient = { attack = [0.3, 1.0, 0.3] }` |
| cymbals harsh / dull / thin | `[tracks.cymbals]` EQ: the 3.5 kHz cut, the 7 kHz shelf, the 1.6 kHz body (+2.5 dB); `tone_match = { reference = 0.5 }` for less of the studio tone (hi-hat too) |
| hi-hat choked on kicks | `[tracks.hihat]` duck `depth_db` (default 5) |
| snare buzz in intros and breaks | `[tracks.hihat] kit_gate = { depth_db = 20.0 }` (same for cymbals; default 15); the original recording follows that gate in 1.2–12 kHz (`[mix] kit_gate_band_hz`; `[]` leaves the original untouched) |
| rattle under the hi-hat between its hits (the snare wires with the PA) | `[bed] floor_db = -24.0` (deeper; default −15), `bias_db = 3.0` (less kept as hat; default 6); the hat's attacks thin: `protect_ms = 40.0`; `stems = ["hihat", "cymbals"]` to clean the cymbals too; `enabled = false` for none |
| snare buzz audible in the hats / cymbals | `[buzz] stems = ["snare", "toms", "hihat", "cymbals"]` (take it from them too; it was heard as artifacts, so by default only snare and toms); thinner after it: `stem_floor_db = { hihat = -6.0, cymbals = -6.0 }`; cymbals too far back: `[drum_bus.balance] offsets = { cymbals = 4.0 }` (default 1.5) |
| bass boomy / thin | `[tracks.bass]` the 100 Hz low shelf (default +2.5 dB) |
| a stem sounds different from song to song | its `tone_match = { strength = 0.9 }` (0 = off); `transient = { adapt = true }` for the attack |
| guitars too wide / narrow | `[tracks.guitar] width = 0.75`, `balance = 0.8` (1 = fully centred) |
| two guitars as separate stems | `[stems] pan_split = { enabled = true }` (no new separation needed); place them with `[tracks.guitar_left] pan = …` |
| squashed | `loudness.target_lufs = -16.0` |
| the mix should follow each song's character | `[character] adapt = true` (wall / ballad / build / dynamic / steady tables; off by default so the songs sound alike) |
| songs differ in tone | `[consistency] strength = 0.8`; a song should keep its own: `[files."Vita".consistency] enabled = false` |
| snare buzz still there between hits | `[buzz] stem_between_floor_db = { snare = -28.0, hihat = -24.0 }` (deeper; same keys for cymbals, toms) |
| hats / cymbals lose their decay | `[buzz] own_hit_ms = { hihat = 400.0 }` (keep more after each hit) or a shallower `stem_between_floor_db` |
| a song should keep its own levels | `[files."Vita".balance] enabled = false` |
| a song starts too early / late | `trim.pre_roll_s`, or `[files."Vita".trim] start_s = 12.0` |
| separation artifacts (watery vocals) | smaller stem changes, or `stems.backend = "demucs"` and compare |
| prefer the earlier separation | `[stems] plan = "cascade"` (vocals first, the instruments from what is left; its stems are still cached in `processed/stems/`, the default plan's in `processed/stems_ensemble/`) |
| hear what each separation model did | `stages/<song>/1_raw/models/` (every model's own outputs before averaging; `index.json` lists their weights) |

After changes: the report PNGs show how hard each stage works (a compressor should average
about 1 dB; the limiter should act only a few percent of the time), and the stage files let
you hear exactly where a sound changes.

Background reading, with how this pipeline compares: [docs/RESEARCH.md](docs/RESEARCH.md).

## EQ references

- Kick: [projethomestudio.fr](https://www.projethomestudio.fr/en/settings-equalization-bass-drum/)
- Snare: [eMastered](https://emastered.com/blog/snare-eq), [Music Guy Mixing](https://www.musicguymixing.com/snare-eq/)
- Toms: [eMastered](https://emastered.com/blog/floor-toms-eq), [Music Guy Mixing](https://www.musicguymixing.com/eq-tom/)
- Hi-hat: [eMastered](https://emastered.com/blog/hi-hat-eq); cymbals: [Sounds Heavy](https://www.soundsheavy.com/mixing-techniques/overhead-eq-mixing/)
- Bass: [Music Guy Mixing](https://www.musicguymixing.com/bass-guitar-eq/), [Audio Spectra](https://audiospectra.net/how-to-eq-bass-guitar/)
- Male vocals: [Orpheus Audio Academy](https://www.orpheusaudioacademy.com/how-to-eq-male-vocals/), [Audio Spectra](https://audiospectra.net/mixing-male-vocals/)
- Electric guitar: [Audio Spectra](https://audiospectra.net/how-to-eq-an-electric-guitar-eliminate-guesswork/), [Audio Issues](https://www.audio-issues.com/music-mixing/guitar-frequency-fixes/)
- Tambourine: [Sound Shock Audio](https://soundshockaudio.com/how-to-eq-tambourine/), [Johnny Copland](https://johnnycopland.com/how-to-mix-live-tambourine/)
- Steve Albini: [Sound On Sound](https://www.soundonsound.com/people/steve-albini), [Tape Op](https://tapeop.com/interviews/24/steve-albini), [Bobby Owsinski](https://bobbyowsinskiblog.com/steve-albini-interview/)

## Development

```sh
uv run pytest          # 103 checks, about 2 min: DSP, settings, takes and close tracks, the pipeline end to end, the web app, the separation library's internals
uv run --with remotezip --with requests python tools/build_raw_references.py   # rebuild the raw-track references
```
