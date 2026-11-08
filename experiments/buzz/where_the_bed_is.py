# The between-hit bed of the hi-hat stem on Biancaneve: where it is before and after the drum split, whether it is
# cymbal wash, what it follows, where it sits in stereo, where it enters the finished mix. Run from the project root.
import sys, numpy as np, soundfile as sf
from pathlib import Path
from scipy import signal
sys.path.insert(0,".")
from rehearsal_master import augment
R=Path("new_tracks/drive-download-20261107T141948Z-1-001/processed")
D=R/"stems_ensemble/261105_235855 Biancaneve"; M=D/"models"
FS=48000; H=int(0.05*FS)
def rd(p): x,_=sf.read(p,dtype="float32",always_2d=True); return x
raw={n:rd(D/f"{n}.wav") for n in ("hihat","snare","cymbals","kick","bass","guitar","vocals","other")}
hits={nm:augment.drum_hits(raw[nm],FS,lo,hi)[0] for nm,(lo,hi) in {"hihat":(6000,14000),"snare":(150,400),"cymbals":(3000,14000),"kick":(40,150)}.items()}
n=len(raw["hihat"])//H; t=np.arange(n)*0.05
busy=np.zeros(n,bool)
for s in hits.values():
    for x in s//H: busy[max(0,x-1):x+5]=True
late=(t>=54)&(t<268); free=(~busy)&late
hatf=np.zeros(n,bool); hatf[hits["hihat"][hits["hihat"]>54*FS]//H]=True
def fr(x,lo,hi,mono=True):
    y=signal.sosfiltfilt(signal.butter(4,[lo,hi],"band",fs=FS,output="sos"),x.mean(axis=1) if mono else x,axis=0)**2
    m=len(y)//H; y=y[:m*H]
    return 10*np.log10(y.reshape(m,H,*y.shape[1:]).mean(axis=1)+1e-20)
def pad(x): return x[:n*H] if len(x)>=n*H else np.pad(x,((0,n*H-len(x)),(0,0)))
print("=== 1. where the between-hit bed is, before and after the drum split (2-8 kHz, median dB; models/ files are 12 dB down, corrected)")
print(f"{'signal':58} {'between hits':>12} {'at hat hits':>11} {'hit - bed':>9}")
def row(lab,x,off=0.0):
    f=fr(pad(x),2000,8000)[:n]+off; print(f"{lab:58} {np.median(f[free]):12.1f} {np.median(f[hatf]):11.1f} {np.median(f[hatf])-np.median(f[free]):9.1f}")
mix=rd(next((R.parent/"261105_235855 Biancaneve").glob("*TrMic*")))
row("the recording (TrMic)",mix)
row("drums (BS-Roformer-SW, what the drum split gets)",rd(M/"1_mix/BS-Roformer-SW/drums.flac"),12)
row("  hihat: MDX23C-DrumSep (weight 0.25)",rd(M/"4_drums/MDX23C-DrumSep-aufr33-jarredou/hihat.flac"),12)
row("  hihat: mdx23c_drumsep_5stem (weight 0.75)",rd(M/"4_drums/mdx23c_drumsep_5stem_aufr33_jarredou/hihat.flac"),12)
row("  hihat: ensemble (= 1_raw)",raw["hihat"])
row("  snare: MDX23C-DrumSep",rd(M/"4_drums/MDX23C-DrumSep-aufr33-jarredou/snare.flac"),12)
row("  snare: mdx23c_drumsep_5stem",rd(M/"4_drums/mdx23c_drumsep_5stem_aufr33_jarredou/snare.flac"),12)
row("  cymbals: MDX23C-DrumSep",rd(M/"4_drums/MDX23C-DrumSep-aufr33-jarredou/cymbals.flac"),12)
row("  cymbals: mdx23c_drumsep_5stem",rd(M/"4_drums/mdx23c_drumsep_5stem_aufr33_jarredou/cymbals.flac"),12)
row("  cymbals: ensemble",raw["cymbals"])
row("  kick + toms: ensemble",raw["kick"]+rd(D/"toms.wav"))
for nm in ("other","guitar","vocals","bass"): row(f"{nm} (BS-Roformer-SW)",raw[nm])
sum_stems=sum(raw.values())+rd(D/"toms.wav")+rd(D/"vocal_room.wav")
row("all stems summed",sum_stems)
row("recording minus all stems (what the separation drops)",mix[:len(sum_stems)]-sum_stems)

print("\n=== 2. is it cymbal wash? the hi-hat bed against the cymbals stem, frame by frame, and after cymbal hits")
fh=fr(raw["hihat"],2000,8000); fc=fr(raw["cymbals"],2000,8000); fs_=fr(raw["snare"],2000,8000)
print(f"   corr(hihat bed, cymbals bed) over hit-free frames after 54 s: r={np.corrcoef(fh[free],fc[free])[0,1]:+.2f}   corr(hihat bed, snare bed): r={np.corrcoef(fh[free],fs_[free])[0,1]:+.2f}")
cy=hits["cymbals"][hits["cymbals"]>54*FS]
cy_alone=np.array([s for s in cy if np.min(np.abs(hits["hihat"]-s))>0.3*FS])
eh=signal.sosfiltfilt(signal.butter(4,[2000,8000],"band",fs=FS,output="sos"),raw["hihat"].mean(axis=1))**2
k=int(0.02*FS); eh=np.convolve(eh,np.ones(k)/k,"same")
lags=[-300,-100,0,100,300,600,1000,1500,2000]
print(f"   hihat stem 2-8 kHz around cymbal hits with no hat within 300 ms (n={len(cy_alone)}), median dB at ms:", " ".join(f"{L}:{10*np.log10(np.median(eh[np.clip(cy_alone+int(L/1000*FS),0,len(eh)-1)])):.1f}" for L in lags))

print("\n=== 3. does the bed follow anything? hit-free bed per 12 s window vs. the other stems' levels (dB)")
print(f"{'t':>5} {'hihat bed':>9} {'cymb bed':>8} {'snare bed':>9} {'Tr2 voice':>9} {'Tr2 bass':>8} {'guitar':>7} {'kick':>6} {'hats/12s':>8} {'free%':>5}")
tb=rd(R/"stems_close/261105_235855 Biancaneve Tr2/bass.wav"); tv=rd(R/"stems_close/261105_235855 Biancaneve Tr2/vocals.wav")
fv=fr(pad(tv),150,4000)[:n]; fb=fr(pad(tb),40,400)[:n]; fg=fr(raw["guitar"],100,3000); fk=fr(raw["kick"],40,150)
for a in range(54,266,12):
    w=(t>=a)&(t<a+12); m=w&~busy
    if m.sum()<4: print(f"{a:5d}  (no hit-free frames)"); continue
    print(f"{a:5d} {np.median(fh[m]):9.1f} {np.median(fc[m]):8.1f} {np.median(fs_[m]):9.1f} {np.median(fv[w]):9.1f} {np.median(fb[w]):8.1f} {np.median(fg[w]):7.1f} {np.median(fk[w]):6.1f} {((hits['hihat']>a*FS)&(hits['hihat']<(a+12)*FS)).sum():8d} {100*m.sum()/w.sum():5.0f}")

print("\n=== 4. stereo: where do hat hits and the bed sit? L-R level difference and inter-channel coherence, 2-8 kHz, in the hi-hat stem")
f2=fr(raw["hihat"],2000,8000,mono=False)   # (frames, 2)
ild=f2[:,0]-f2[:,1]
def coh(x,frames):
    sel=np.concatenate([x[f*H:(f+1)*H] for f in np.flatnonzero(frames)[:400]])
    b=signal.sosfiltfilt(signal.butter(4,[2000,8000],"band",fs=FS,output="sos"),sel,axis=0)
    return np.corrcoef(b[:,0],b[:,1])[0,1]
hat0=np.zeros(n,bool); hat0[hits["hihat"][hits["hihat"]>54*FS]//H]=True
sn0=np.zeros(n,bool); sn_al=np.array([s for s in hits["snare"] if s>54*FS and np.min(np.abs(hits["hihat"]-s))>0.15*FS]); sn0[sn_al//H]=True
for lab,m in (("hat hits (first 50 ms)",hat0),("snare hits, no hat (first 50 ms)",sn0),("between hits",free)):
    print(f"   {lab:34} L-R {np.median(ild[m]):+5.1f} dB (IQR {np.percentile(ild[m],25):+.1f}..{np.percentile(ild[m],75):+.1f})   L/R correlation {coh(raw['hihat'],m):+.2f}")
f2s=fr(raw["snare"],2000,8000,mono=False); ilds=f2s[:,0]-f2s[:,1]
print(f"   snare stem at snare hits                L-R {np.median(ilds[sn0]):+5.1f} dB")

print("\n=== 5. in the finished mix, where does the between-hit bed come from? 5_premix parts, 2-8 kHz after 54 s (full preset, stage files 12 dB down)")
P=R/"full/stages/261105_235855 Biancaneve/5_premix"; T0=int(1.9*FS)
parts={p.stem:rd(p) for p in P.glob("*.flac")}
tot=sum(parts.values())
def row5(lab,x):
    x=np.concatenate([np.zeros((T0,2),x.dtype),x]); f=fr(pad(x),2000,8000)[:n]
    print(f"   {lab:12} between hits {np.median(f[free]):7.1f}   at hat hits {np.median(f[hatf]):7.1f}")
for k_ in sorted(parts): row5(k_,parts[k_])
row5("all parts",tot)
mixf=rd(R/"full/stages/261105_235855 Biancaneve/5_mix.flac"); row5("5_mix",mixf)

print("\n=== 6. the hat's own tail, learnt before 54 s (no bed) and after: shape re 5 kHz, 90-130 ms after a hat hit")
C=[1250,2000,3150,5000,8000,12500]
def shape(starts,a,b):
    acc=[]
    for s in starts:
        seg=raw["hihat"][s+int(a*FS/1000):s+int(b*FS/1000)].mean(axis=1)
        if len(seg)>100: acc.append(np.abs(np.fft.rfft(seg*np.hanning(len(seg)),n=4096))**2)
    p=np.median(acc,axis=0); f=np.fft.rfftfreq(4096,1/FS)
    v=np.array([10*np.log10(p[(f>=c/2**(1/6))&(f<c*2**(1/6))].mean()+1e-30) for c in C]); return v-v[3]
early=hits["hihat"][(hits["hihat"]>30*FS)&(hits["hihat"]<54*FS)]; lateh=hits["hihat"][hits["hihat"]>54*FS]
print(f"   {'':34}"+" ".join(f"{c:>7}" for c in C))
print(f"   {'hat tail before 54 s (n=%d)'%len(early):34}"+" ".join(f"{v:+7.1f}" for v in shape(early,90,130)))
print(f"   {'hat tail after 54 s (n=%d)'%len(lateh):34}"+" ".join(f"{v:+7.1f}" for v in shape(lateh,90,130)))
print(f"   {'hat attack after 54 s':34}"+" ".join(f"{v:+7.1f}" for v in shape(lateh,0,30)))
bedacc=[]
for f_ in np.flatnonzero(free)[:600]:
    seg=raw["hihat"][f_*H:(f_+1)*H].mean(axis=1); bedacc.append(np.abs(np.fft.rfft(seg*np.hanning(len(seg)),n=4096))**2)
p=np.median(bedacc,axis=0); f=np.fft.rfftfreq(4096,1/FS); v=np.array([10*np.log10(p[(f>=c/2**(1/6))&(f<c*2**(1/6))].mean()+1e-30) for c in C])
print(f"   {'the bed between hits':34}"+" ".join(f"{x:+7.1f}" for x in v-v[3]))
