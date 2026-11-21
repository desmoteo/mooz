# The web app

A small local web app over the command line: create a project, choose takes and profiles,
remaster, watch it run, change the settings and the profiles, and listen: to the recording,
the master, the stems and every stage in between. It adds no processing of its own. Everything it does can be done with `mooz` and a text editor, and the
files it writes are the ones the command line reads.

Status, 10 November 2026: planned and built the same day, and tried on two real takes (see
Checks). The same day a second pass added the take's page (job 5) and the editing of every
profile, and a third the take's pipeline graph, names on the pads of a take with close tracks,
and the guide inside the app. On 11 November: comparing two profiles blind, with the verdicts
kept in the project (job 6). What is left out is at the end.

```sh
uv run mooz-gui            # http://127.0.0.1:8377, opens the browser
uv run mooz-gui --port 9000 --no-browser
```

## Why a web app

- The pipeline is Python and runs for minutes per take. A page served by a small Python
  server needs no second toolkit, runs on the laptop that has the GPU, and can be opened
  from any browser on it.
- No framework: the server is the standard library's `http.server`, the page is one HTML
  file with plain CSS and JavaScript. Nothing to build, nothing to fetch from the network.
- One new dependency, `tomlkit`: it rewrites `rehearsal.toml` and keeps its comments.

## What the words mean

| word | what it is | where it lives |
|---|---|---|
| project | a settings file and the recordings it names | a `rehearsal.toml`; its `[input] paths` are the recordings |
| take | one moment recorded: a file, or a Zoom's files of that moment | found by `takes.discover`, as on the command line |
| profile | a named sound: a preset on top of the project's settings | the built-in presets, plus `presets/<name>.toml` next to the project's `rehearsal.toml` (a profile of its own, or its version of a built-in one) |
| job | one run: the chosen takes through the chosen profiles | one `mooz` process |

The app keeps one file of its own, the list of known projects
(`~/.config/mooz/projects.json`: a name and the path of a `rehearsal.toml` each).
Removing a project from the list deletes nothing else.

## The six jobs

### 1. Project creation

- Name, and the folder with the recordings, chosen in a folder browser served by the app (a
  browser cannot hand a page a folder's path).
- The takes found there are shown before anything is written: name, tracks (`TrMic`, `Tr2`,
  `TrLR`), length.
- Create writes `<folder>/rehearsal.toml` with `[input] paths = ["."]` and the chosen
  profiles. A folder that already has one is adopted as it is.
- Started in a folder with a `rehearsal.toml`, the app lists it as a project by itself.

### 2. Remaster

- The project's page is its takes, one row each, one column per profile: loudness, loudness
  range and limiter activity from the take's report, a play button for the master and one
  for the A/B file, a link to the report figures and one to the take's page (job 5).
- Each take has an Original button: the recording itself. Going between Original, Play and
  A/B of a take keeps the moment of the song (the recording is entered at the master's trim).
- Tick takes and profiles, choose A/B files or separation only, press Remaster. That is
  `mooz <takes> -c <rehearsal.toml> --preset a --preset b [--ab]`.
- One job at a time: the GPU and the memory do not leave room for two.
- **What is up to date is left as it is.** A take already mastered with these very settings,
  recordings and program, all its files there, is not made again: its row in the job says "up to
  date", and a job where nothing is to do says so. So a job that stopped half-way is started again
  and goes on where it stopped. "Also what is up to date" makes those again too
  (`--again`); the take page's "Remaster this take" always does.
- **A run stopped for memory says so.** The system stops a run that needs more memory than it
  may use with a kill, and the run itself cannot say anything; the job then reads "The system
  stopped the run: it needed more memory than the 15G it may use", with the take it stopped at.
  Until 16 November 2026 the limit was a fixed 4 GB, which a take of 5:40 reaches; it now goes
  with the machine ([DESIGN.md](DESIGN.md), Performance).
- **Mix**, next to Play and A/B: the mix before the master step as a file (`<take>_mix`), nothing
  clipped or limited.
- The masters go next to the recordings (`<recordings>/<output dir>`), not next to the
  `rehearsal.toml`. Two projects over the same recordings therefore share those folders: a
  profile run from one overwrites the other's masters of that profile.
- Two profiles that write to the same folder cannot run in one job (the second would
  overwrite the first): the app says which, before starting.

### 3. Profile settings

- **Which profiles run** by default: `[input] presets`.
- **The project's settings**, the ones the README's tuning table sends people to most: a form
  with the loudness target, the original's share, the group and kit levels, the vocal room,
  the guitars' width, the close tracks' share, the output switches. Each field shows the
  default, and whether the project sets it. Resetting a field removes the key from the file.
- **The whole file** in a text box, for everything else.
- **Profiles**: every one can be edited, in the same form (each field shows the project's
  value, which is what it goes back to) and as its whole file; one Save writes both, the
  form's values on the text.
  - A profile of the project's own is `presets/<name>.toml` next to the `rehearsal.toml`,
    made from an empty one or from a copy of another.
  - A built-in one is never written to. Saving it writes the project's version under the
    same name, which the settings loader finds first; "Back to the built-in" deletes that
    file. Other projects keep the built-in one.
- Nothing is saved that does not load: every save is parsed and resolved against
  `defaults.toml` first, and a typo comes back as the error the command line would give. The
  previous `rehearsal.toml` is kept as `rehearsal.toml.bak`.

### 4. Progress

- The pipeline says where it is: with `REHEARSAL_PROGRESS=1` it writes one JSON object per
  line on its standard output, prefixed `@@` ([progress.py](../mooz/progress.py)).
  Without it nothing changes.

  | event | when | holds |
  |---|---|---|
  | `plan` | once, at the start | every (take, profile) the run will do |
  | `prepare` | before the first take, per take with close tracks | which take is being separated ahead |
  | `separation` | a track's stems are needed | `cached`, or `start`, `step` (model i of n), `done` |
  | `unit` | a take starts under a profile | its index in the plan |
  | `stage` | inside a take | the stage's name and the fraction of the take done |
  | `unit_done` | a take ends | seconds, loudness figures and files, or the error |
  | `end` | the run ends | what failed |

- The fraction comes from one table of stage weights, measured on a real run
  (`progress.STAGES`: the stem chains are 64 % of a take, the mix bus 10 %, the character
  8 %), so the bar moves at about the speed of the work when the stems are cached. A
  separation is shown by its models (3 of 9), not by time.
- The app shows the job's bar, each take's state and stage, the time so far, the run's last
  lines, and once a take has finished an estimate of what is left: the finished takes'
  seconds of work per second of audio, times the audio still to do.
- Cancel stops the process group: the pipeline, a separation in flight, `ffmpeg`. The
  command line removes its work folder on the way out; a separation cut short is done again
  next time (its cache is only marked complete at the end). Stopping the app (Ctrl-C or a
  plain kill) cancels a running job the same way.

### 5. Listening: a take along the pipeline

The take's page (the take's title on the project's page, or Stages in a profile's column)
is the pipeline drawn as a grid, with a pad for every file a run left behind.

| column | what is in it | files |
|---|---|---|
| Recording | the room track, the close tracks, the recorder's own mix-down | the take's files |
| Separated | each stem as the chains get it; where a close track was added, also the room's stem and the close track's alone | `1_merged/`, `1_raw/`, `1_raw_Tr<n>/` |
| Cleaned | the kit's stems after the clean-up, and what it took out | `3_drum_cleanup/` |
| Stem chain | each stem after its chain, and the samples laid under the drums | `2_stems/`, `2_stems/samples/` |
| Drum bus | the kit as one instrument | `4_drum_bus` |
| In the mix | every part as it goes into the mix, the recording at its share too | `5_premix/` |
| Mix | before loudness and the limiter | `5_mix` |
| Master | the master, and the A/B file | next to `stages/` |

- The rows are the recording and the stems, the kit's last: a row is one stem's way through
  the pipeline, and the kit's rows meet in the drum bus. A line runs along each row as far as
  it has files, lilac where it is raw and the rings' colours towards the master.
- **A take with close tracks names its Separated pads.** `merged` (or `room`, where no close
  track was added) is what the chain starts from; `room` and `Tr2` next to it are what it is
  made of, each at its level in the merged stem (the merge's gains, from the report), so the
  three compare by ear. A dashed `Tr2` is a close stem that is not used: bleed. The room's
  voice is the one the merge takes: the dry part and the room's sound together.
- **The pipeline graph**, the page's second tab, draws the same files as boxes and arrows:
  which file goes into which.
  - Layers, left to right: Recording, Separation, Stems, Merged, Cleaned, Stem chain, Drum bus,
    In the mix, Mix, Master. The close tracks are on top, each with its separation and its
    stems; the room track's stems under them, a lane per row of the grid.
  - An arrow carries what the report has for it: the shares of a merged stem (90 % from
    `Tr2`, 10 % from the room), the recording's share in the mix, a part's balance gain.
  - A box plays its file. A dashed box is a step with no file of its own (a separation:
    choosing it lists each model's own output, to play) or a file that is not used.
  - Choosing a box lights what goes into it and what comes out of it, to the recording on
    one side and the master on the other.
  - The left and right arrow keys follow the arrows, up and down go through a layer.
- **Same moment.** Going from pad to pad keeps the moment of the song: the stems and the
  recording are whole files, entered at the take's trim (from its report); the stage files
  and the master start there. So do the profile tabs: the same pad goes on playing under the
  other profile.
- **Same scale.** The stage files are written 12 dB under their real level (`stages.GAIN_DB`).
  With "Levels lined up" (on to begin with) everything else is played 12 dB down too, the
  master included, so two stages compare by ear and the raw drum peaks, above full scale, do
  not clip. Off, every file plays as it is.
- **What the run did there.** Under the grid, for the chosen pad: the part of the take's
  report that belongs to its stage (a stem's chain step by step, the drum bus, the balance,
  the mix bus, the limiter), EQ curves as small bar charts, and the file's path.
- The arrow keys move from pad to pad, along a stem's row or down a stage's column; while
  something plays, the next pad takes over.
- **Where a running take is.** While a job works on the shown take and profile, its stage's
  column is marked, and during the stem chains the stem's row; when it ends, the page reads
  the new files.
- A take that is not mastered with the shown profile has its recording and, if they are
  cached, its stems. "Master this take with <profile>" starts a job for it alone.
- Every separation model's own output (`1_raw/models/` and `1_raw_Tr<n>/models/`, from their
  `index.json`) is listed under the drawing, with the stems that have no row (a close
  track's drums).
- For the chosen pad or box, the other files at the same point are offered next to it (what
  the clean-up took out, the samples, the parts of a merged stem).

### 6. Comparing blind: which of two profiles sounds better

The figures say what a profile did; whether it sounds better is for the ears, and they are
fair only when they do not know which is which. The take's third tab, Compare blind, plays
two profiles' masters of the take as A and B.

- **Which is which is drawn by lot** when the comparison starts, and again for every round.
  The page says it only after the choice.
- **In step, on the same sample.** Both masters are read whole and decoded in the browser
  (`decodeAudioData`), and both start from one call at one moment of the audio clock. A and
  B are two gains, crossed in about 20 ms: no gap, no slip, nothing to tell the sides apart
  but the sound. (The page's ordinary player changes file, with a short gap: fine for
  going from stage to stage, not for this.)
- **Same loudness.** Each is turned down to the quieter one's integrated loudness, from the
  two reports. Masters of one target are within 0.1 LU already; a profile with another
  target is not, and louder would win.
- **A passage on repeat.** Loop from here and Loop to here set the two ends at the playing
  moment; the loop is the audio buffers' own, exact to the sample.
- Keys: `A`, `B`, `X` (the other side), Space, `[` and `]`.
- **The verdict**: A sounds better, B sounds better, or no difference heard, with a few
  words on why. It is written to `listening.json` next to `rehearsal.toml`
  ([listening.py](../mooz/gui/listening.py)): the take, what was behind A and
  behind B, the choice, the note, the loop, and when each of the two masters was made.
- **A master against its own mix.** The list of sides also holds each profile's mix
  (`<take>_mix`, "pro, the mix"): the master step alone, at the same loudness. On the take's page
  the mix plays up to 4 dB louder than the master (both 12 dB under their real level), and the
  louder of two nearly always sounds better; here the louder file is turned down to the other's
  loudness and what is left is what the clipper and the limiter did. The page says by how much
  it turned it down once the choice is made.
- **Another round** draws A and B again and pauses the song, so the change of sides cannot
  be heard as a change of sound.
- **What it adds up to.** The takes' page counts the verdicts per pair of profiles
  ("trial-pro chosen in 5 of 7 rounds, trial-base in 1, no difference heard in 1 (3
  takes)"). A verdict is about the two files that were heard: once one of them is mastered
  again it stays in the list, marked, and is not counted. Take back removes one.
- Two masters of four minutes are about 190 MB decoded; they are let go when the tab or the
  take is left.

### The guide

"How it works", under the projects: one page of the app's own (`static/help.html`) that says
what each part is for, what the stages are, how listening works and where the files are.

Nothing is computed for this page: `gui/stagemap.py` lists the files and reads the report.

## Design

```
browser  ──HTTP, JSON──>  gui/server.py   routes, files with Range (audio seeking)
                              │
                 gui/projects.py           the list of projects; takes, results, settings, profiles
                 gui/stagemap.py           a take's files along the pipeline (the take's page)
                 gui/listening.py          the blind comparisons' verdicts (listening.json)
                 gui/jobs.py               one job: starts the command line, reads its events
                              │
                     python -m mooz.cli … --preset a --preset b
                              │  @@{"ev": "stage", …}
                     pipeline.py, separate.py  ──>  progress.py
```

- **The server does no audio work.** It imports the settings and the take discovery, never
  the pipeline: a job is a child process, so the memory cap, the GPU and a crash stay there.
- **Local only.** It listens on 127.0.0.1. Requests whose `Host` is not that address are
  refused, and so are writes from another site's page (`Origin` check): a web page open in
  the same browser cannot drive it. Files are served only from inside a project's folders,
  and only audio, figures and reports.
- **State.** The projects list and the settings files are on disk; the job is in memory. If
  the app stops, the job is stopped with it.
- **The page** polls the job once a second while one runs. One HTML file, one stylesheet,
  one script, in `mooz/gui/static/`.

Changes outside the app's own folder, all small:

| where | what |
|---|---|
| `cli.py` | `--preset` can be given several times, `default` names the chain with no preset; progress events around the takes; SIGTERM leaves through the normal clean-up |
| `config.py` | a preset is looked for next to the settings file (`presets/<name>.toml`) before the built-in ones |
| `pipeline.py`, `separate.py` | `progress.stage(...)` calls at the stage boundaries |
| `chaincache.py` (11 November) | not for the app alone, but it is what makes a tweak from the app quick: each stem's chain is kept from run to run, and the job's line says "bass, kept" when one is used as it is |

## Checks

- `tests/test_gui.py`: the app on a temporary folder, over real HTTP. A project made from a
  folder of two takes; the settings form writes the file and keeps its comments, a typo is
  refused and the file untouched; a project profile is found by the settings loader; a job
  with a stand-in command goes through its events to done, and a cancelled one stops; files
  outside the project are not served; a foreign `Host` or `Origin` is refused.
- Also there: a profile edited through the form, a built-in one saved as the project's
  version and given back; a take's stage folder, stem caches and master laid out as the
  take's page gets them, each file served through the stage folder's links.
- `tests/test_pipeline.py`: with progress on, the end-to-end run's stage fractions only rise
  and end at 1.
- The page itself, driven in headless Chrome on a throw-away folder (not part of the test
  suite: it needs a browser): a project made through the folder browser, takes unticked, the
  form saved and a field put back to its default, a profile started from a copy, refused with a
  typo and saved without, the whole file refused with an unknown table, the project taken off
  the list. Names are shown as text, never as markup.
- A real job, 10 November 2026: Miramare (4:00) and the short Stanza (0:47) of 5 November, the
  default profile with A/B files, into a throw-away output folder, stems from the cache. Done
  in 5 min 11 s (259 s and 48 s); the masters measure as the ones of 9 November (−14.0 and
  −14.1 LUFS, loudness range 2.8 and 13.3 LU). The stage weights come from this run.
- Cancel on a real run, in the stem chains after 27 s: answered in 0.2 s, no process left, the
  work folder removed. The stage files the take had already written stay; the next run
  overwrites them.
- The take's page and the profile editor, driven in headless Chrome on the band's project
  (read only) and on a throw-away one:
  - Miramare: from the kick's chain at 1:00 to its raw stem, its cleaned stem, the snare, and
    along the row to the master with the arrow keys; each file entered at the same moment
    (within 0.05 s), the files at their real level played at a gain of 0.25 (−12 dB).
  - Riempio i vuoti, whose master starts 1.7 s into the recording: from the master at 0:30 to
    Original, entered at 0:32.
  - The bass's chain under `default`, then the `simple_bass` tab: the same pad goes on there.
  - A master of the short Stanza started from its page into a throw-away folder (59 s): the
    marks went Recording, Separated, the eight stems of the chains in turn, Drum bus, In the
    mix, Mix, Master, and the page filled in when it ended.
  - `albini` changed through the form and the text, saved as the project's version, a typo
    refused, given back to the built-in; a profile made from the form alone and deleted.
- The graph and the guide, driven in headless Chrome on the band's project (read only):
  46 boxes and 55 arrows for Miramare; the bass's chain chosen lights 11 arrows and 12
  boxes; with the arrow keys from there to the merged bass, the room's stem, the guitar's,
  and along its arrows to the master and the A/B file, each entered at the same moment; the
  room's separation lists its 29 model outputs, and one plays. On the grid, the bass's `Tr2`
  pad plays at +1.5 dB against the stage files and the `room` pad at −12 dB: their levels in
  the merged stem.
- `tests/test_gui.py`, for job 6: a verdict is refused unless its two profiles both have a
  master of the take and the choice is one of them; three verdicts are kept oldest first in
  `listening.json`, counted per pair, one taken back; a master made again leaves its verdicts
  listed and uncounted; a `listening.json` that is not ours to read is said and left as it is.
- Comparing blind, driven in headless Chrome on a throw-away project over a linked copy of
  Miramare (11 November 2026; `default` against `simple_bass`, both at −14.0 LUFS):
  - Start read and decoded the two masters (stereo, 48 kHz, 3:59) and both played from one
    start call; the ordinary player, which was on, stopped and its bar went away.
  - `B`, `X` and a click on a side moved the two gains between 1 and 0. Space paused and
    played, also with a side still in focus from the click; on another button it pressed
    that button.
  - A loop set with `[` at 1:00 and `]` at 1:03: over 8 s the position stayed inside it.
  - A note with markup in it was shown as text. After "B sounds better" the page named the
    two sides and the choice, and the file held the take, what was behind A and B, the
    choice, the loop and the note; "Another round" paused the song and drew again.
  - The takes' page added the two rounds up; Take back there left one, on disk too.
  - Leaving the take let the two decoded masters go. At 430 px the view fits the window.
  - Not checked there: the sound itself. That both sides start on one sample follows from
    the single start time, not from a measurement of the output.
- Not tried: the page in Firefox or Safari (the stems are 32-bit float WAV files, which
  Chrome plays).
- 16 November 2026, in headless Chrome on a throw-away project over a linked copy of Lisa dice
  (5:40, the take a run had stopped at): the Mix button plays `…_mix.mp3`; a job on two profiles
  that are up to date ends as "Nothing to do: all 2 masters are up to date", both rows "up to
  date"; Compare blind offers pro, "pro, the mix", default, "default, the mix", starts on the
  newest master and its mix, turns the master down 3.5 dB, and the verdict is kept as `pro:mix`
  over `pro`. A run of that take with the limit set to 2G ended with the message and exit code
  137; with the limit the machine gives (15G) it went through, 4.0 GB at most.
- Not tried: a job that has to separate. The separation's events are tested with a stand-in,
  not on the GPU.

## Left out, on purpose

- No queue of jobs, no history of past jobs beyond the reports on disk.
- No per-song `[files."…"]` editor: the text box covers it.
- No waveform display: the player's own bar is what there is to seek with.
- No mixer: one file plays at a time (two stems together are a job for a DAW, which the
  stage files line up in). Comparing blind is the one place where two files run together,
  and only one of them is heard.
- Comparing blind takes masters only, two at a time: not a stage file, not three profiles at
  once, not the recording (its loudness is another, and it is told apart at once).
- No log-in: it is for the laptop it runs on. Do not put it on a network.
- `--learn-balance`, `--analyze` and the tools under `tools/` stay on the command line.
