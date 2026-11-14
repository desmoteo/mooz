# Research notes: improving stems and remixing them

What published work says about turning separated stems of a room recording into studio-like
tracks and remixing them, compared with what Mooz does ([DESIGN.md](DESIGN.md)).
Compiled October 2026. A few claims rest on abstracts only, where the full text could not be
opened (marked *abstract*).

## What the literature says, and where we stand

| topic | finding | source | here |
|---|---|---|---|
| level balance | Engineers agree closely on the lead vocal level, about −3 LU relative to the mix; drums about −6.5 to −8.5 LU, bass −8 to −9 LU. "Equal loudness" of all tracks is not what engineers do. | De Man, PhD thesis 2017, [pdf](http://www.brechtdeman.com/publications/pdf/PhD-thesis.pdf) | targets learnt from the band's own songs (vocals −8.5, drums −3.1 LU re the remix): consistent, but they keep the rehearsal room's balance |
| vocal level | Listeners prefer vocals louder than in commercial mixes (those averaged −4.6 dB vocal-to-accompaniment). | Knoll & Siedenburg 2022, [link](https://journals.sagepub.com/doi/full/10.1177/20592043221142712) | no such prior |
| dynamics | At equal loudness, mixes with more dynamic range (LDR, PLR, crest factor) are preferred: "err on the side of too little compression". | De Man thesis | many compression stages in series (stem compressors, drum glue + parallel, multiband, glue, clipper, limiter) |
| reverb | The total reverb sits about −14 LU; mixes at −9 LU or more were judged too wet. A just-noticeable change in direct-to-reverberant ratio is about 5–6 dB. Excess vocal reverb is a top complaint. | De Man thesis; De Man & Reiss 2015, [pdf](http://www.brechtdeman.com/publications/pdf/jarp10.pdf) | vocal room turned down ≤ 6 dB (one just-noticeable step); the room level is not measured |
| what reviewers comment on | vocals 31 % of comments, drums 19 % (snare and kick most). | De Man & Reiss 2015 | effort is where it should be |
| stereo | lead vocal, snare and bass centred; panning widens gradually from about 50 to 400 Hz; too wide and too mono both rate low. | De Man thesis | mono below 120 Hz (a hard split), vocals and snare centred |
| spectrum | commercial mixes cluster around one long-term spectral target. | Pestana et al. 2013, [link](https://www.researchgate.net/publication/274511175_Spectral_characteristics_of_popular_commercial_recordings_1950-2010) *abstract* | tonal EQ to a −2 dB/oct target |
| masking | cross-adaptive EQ that cuts a track only where it masks another reduced measured and perceived masking, close to a professional mix. | Hafezi & Reiss 2015, [link](https://www.semanticscholar.org/paper/Autonomous-Multitrack-Equalization-Based-on-Masking-Hafezi-Reiss/3c508c7799a3f68bdcd7b653d5e6e0242af191e4) *abstract* | only kick-keyed ducks; vocals vs guitars not unmasked |
| compression settings | multitrack compressor settings regressed from signal features matched semi-pro mixes. | Ma et al. 2015, [pdf](https://www.eecs.qmul.ac.uk/~josh/documents/2015/Ma%202015%20-%20Intelligent%20Multitrack%20Compression.pdf) | static, style-adjusted settings |
| separate, then mix | separation followed by cross-adaptive gain, EQ, panning and an exciter improved balance and clarity. | Matz, Cano & Abeßer, ISMIR 2015, [link](https://publica.fraunhofer.de/handle/publica/390656) | the same architecture |
| remixing separated stems | vocals could be raised by up to about 6 dB before the separation artifacts became an issue (small perceptible loss). | Wierstorf et al., AES 2017, [link](https://www.researchgate.net/publication/320628858_Perceptual_Evaluation_of_Source_Separation_for_Remixing_Music) *abstract* | per-stem changes are not limited |
| separation artifacts | spectrogram separators lose transients and highs; a generative post-processor was preferred for bass and drums. | Schaffer et al., ISMIR 2022, [arXiv](https://arxiv.org/abs/2208.12387) | transient shaper, exciter; drum samples (now) |
| metrics | SI-SAR predicts ratings of separated drums and bass better than SDR. | 2025 benchmark, [arXiv](https://arxiv.org/abs/2507.06917) | not used |
| source restoration | inverting EQ, compression, reverb, distortion and codecs; best: separation ensemble, then restoration. | Zang et al. 2025, [arXiv](https://arxiv.org/abs/2505.21827); [arXiv](https://arxiv.org/abs/2603.16926) | tested: no gain on these stems (DESIGN §10) |
| effect removal | detect which effects are present, then chain removal models. | Rice et al. 2023 (RemFX), [arXiv](https://arxiv.org/abs/2308.16177) | vocal de-reverb only (nothing else was applied in a rehearsal) |
| bleed reduction | multi-channel interference reduction (Kernel Additive Modelling, adaptive cancellers) trades distortion for separation. | KAMIR 2015, [link](https://www.audiolabs-erlangen.de/resources/MIR/2015-ICASSP-KAMIR/); Clifford & Reiss 2011, [pdf](http://eecs.qmul.ac.uk/~josh/documents/2011/CliffordReiss-2011-DAFx.pdf) | ducks, gates, buzz NMF (the stems could serve as "virtual mics") |
| gates | gate settings should minimise artifacts on the wanted signal, not only maximise bleed reduction. | Terrell et al. 2010, [link](https://link.springer.com/article/10.1155/2010/465417) | threshold from the noise floor; depths relaxed after listening |
| drum reinforcement | phase-align samples, use velocity layers and round-robins, blend rather than replace; bleed causes false triggers. | Sound On Sound, [link](https://www.soundonsound.com/techniques/how-replace-reinforce-recorded-drums); [arXiv](https://arxiv.org/abs/2509.24853) | done: own-hit samples, 2 layers, aligned, velocity-matched, bleed rejected |
| learned mixing | deep models predict mixing parameters; FxNorm normalises stems (loudness, EQ, panning, compression, reverb) before mixing, rated on par with professional mixes. | Steinmetz 2020, [arXiv](https://arxiv.org/abs/2010.10291); FxNorm 2022, [arXiv](https://arxiv.org/abs/2208.11428) | our tone match is FxNorm-like, for EQ only |
| reference mixing | console parameters estimated from a reference song (Diff-MST); any effect chain tuned by search against a production-style embedding (ST-ITO); style transfer robust on separated stems (Koo et al.). | [Diff-MST](https://arxiv.org/abs/2407.08889); [ST-ITO](https://arxiv.org/abs/2410.21233); [Koo et al.](https://ai.sony/publications/Music-Mixing-Style-Transfer-A-Contrastive-Learning-Approach-to-Disentangle-Audio-Effects/) | not used |
| two-stage mixing | mixing within groups, then the groups, beats a flat mix. | 2026, [arXiv](https://arxiv.org/abs/2609.02835) | the same (kit → drum bus → mix) |
| delivery | −16 LUFS for music; −1 dBTP ceiling. | AES TD1008, [pdf](https://aes.org/wp-content/uploads/2024/01/20210924_TD1008_v3.13.pdf) | −16 LUFS (rehearsal.toml), −1 dBTP |

## Improvements, by expected benefit for the effort

1. **Balance priors from engineers.** Blend the band-learnt targets with the published ones
   (vocal ≈ −3 LU re mix, bass −8 to −9, drums −6.5 to −8.5), with a small vocal bias. The
   band medians keep the songs consistent; the priors would fix that they are consistently a
   rehearsal balance (drums loud, voice back). Low effort.
2. **Measure the result and correct.** After the mix: each group's loudness, PLR/LDR,
   side/mid ratio and the room/reverb level (aim ≈ −14 LU), then nudge settings and re-mix,
   as automatic-mixing systems do. Low effort, one extra pass.
3. **Unmask vocals from guitars.** Cut the guitars dynamically in 1–4 kHz only where and
   when the voice is masked (cross-adaptive EQ). Medium effort.
4. **A dynamics budget.** Cap the total gain reduction across all compression stages (or
   drop one, e.g. the multiband) and accept a slightly lower loudness rather than squash.
   Low effort.
5. **Limits on separated stems.** Keep level, EQ and gate moves on separated stems within
   about 6 dB of their separated balance, and avoid wide re-panning, so the summed stems keep
   masking each other's artifacts. Low effort.
6. **Reference-driven tuning.** Tune the chain's parameters by search (ST-ITO style) against
   commercial reference songs or Cambridge-MT mixes. High effort.

**Done** (October 2026): 1 (`[balance] priors`, weight 0.3), 3 (guitars dip 1.5–4 kHz under
the voice's words), 4 in part (gentler multiband ratios), the gradual 50–400 Hz stereo
widening, the raw-multitrack tone reference, and cross-song consistency of levels (balance
strength 0.9) and of tone (`[consistency]`). Not yet: 2 (measure-and-correct loop on the
final mix), 5, 6.

## Where our choices are risky, according to this work

- Targets learnt only from rehearsals reproduce the room's balance (point 1).
- Tone matching only to the band's median bakes the room into the target; the raw-multitrack
  reference (`tone_match.reference`) now offers the studio side of it.
- Many compression stages in series, where listeners prefer more dynamic range (point 4).
- Deep gates and expanders remove the residual that masks separation artifacts (point 5);
  they were already relaxed after listening.
- The hard mono split at 120 Hz is fine, but practice widens gradually from 50 to 400 Hz.
