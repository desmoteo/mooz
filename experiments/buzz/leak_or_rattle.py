# Is the voice-tied part of the hi-hat bed a leak of the voice (coherent with the close voice) or a rattle it sets off
# (incoherent, same shape with the voice on or off)? Run from the project root.
import sys, numpy as np, soundfile as sf
from pathlib import Path
from scipy import signal
sys.path.insert(0,".")
from rehearsal_master import augment, multitrack
R=Path("new_tracks/drive-download-20261107T141948Z-1-001/processed")
D=R/"stems_ensemble/261105_235855 Biancaneve"; FS=48000; H=int(0.05*FS)
def rd(p): x,_=sf.read(p,dtype="float32",always_2d=True); return x
hh=rd(D/"hihat.wav"); sn=rd(D/"snare.wav"); cy=rd(D/"cymbals.wav"); voc=rd(D/"vocals.wav")+rd(D/"vocal_room.wav")
tv=rd(R/"stems_close/261105_235855 Biancaneve Tr2/vocals.wav").mean(axis=1)
d=round(11.54/1000*FS); tvs=-multitrack.shifted(tv,d,len(hh))      # the close voice as the room hears it
hits={nm:augment.drum_hits(x,FS,lo,hi)[0] for nm,x,(lo,hi) in (("hihat",hh,(6000,14000)),("snare",sn,(150,400)),("cymbals",cy,(3000,14000)))}
n=len(hh)//H; t=np.arange(n)*0.05
busy=np.zeros(n,bool)
for s in hits.values():
    for x in s//H: busy[max(0,x-1):x+5]=True
def fr(x,lo,hi):
    y=signal.sosfiltfilt(signal.butter(4,[lo,hi],"band",fs=FS,output="sos"),x)**2; m=len(y)//H
    return 10*np.log10(y[:m*H].reshape(m,H).mean(axis=1)+1e-20)
fv=fr(tvs,150,4000)[:n]; fh=fr(hh.mean(axis=1),2000,8000)[:n]
sel=(~busy)&(t>=54)&(t<180)
on=sel&(fv>-55); off=sel&(fv<-80)
print(f"hit-free frames 54-180 s: voice on {on.sum()}, voice off {off.sum()}   hi-hat bed 2-8 kHz: voice on {np.median(fh[on]):.1f}  off {np.median(fh[off]):.1f} dB")
# 1. coherence (linear relation) between the close voice and the hi-hat stem, over voice-on hit-free frames
def cat(x,frames): return np.concatenate([x[f*H:(f+1)*H] for f in np.flatnonzero(frames)])
a=cat(tvs,on); b=cat(hh.mean(axis=1),on); c=cat(voc.mean(axis=1),on); s_=cat(sn.mean(axis=1),on)
f,Chh=signal.coherence(a,b,FS,nperseg=2048); _,Cvoc=signal.coherence(a,c,FS,nperseg=2048); _,Csn=signal.coherence(a,s_,FS,nperseg=2048)
print("\ncoherence with the close voice (1 = a filtered copy, 0 = unrelated), by band:")
for lo,hi in ((200,800),(800,2000),(2000,4000),(4000,8000),(8000,12000)):
    m=(f>=lo)&(f<hi); print(f"   {lo:5}-{hi:<5} Hz   room vocals stem {Cvoc[m].mean():.2f}   hi-hat stem {Chh[m].mean():.2f}   snare stem {Csn[m].mean():.2f}")
# 2. envelope: does the bed's level follow the voice's level, and with what lag?
print("\nenvelope correlation (dB vs dB, 50 ms frames, voice-on hit-free frames) of the hi-hat bed with the voice, by lag of the bed after the voice:")
idx=np.flatnonzero(on)
for lag in (-4,-2,-1,0,1,2,4,8):
    i=idx[(idx+lag>=0)&(idx+lag<n)]; print(f"   lag {50*lag:+5d} ms  r={np.corrcoef(fv[i],fh[i+lag])[0,1]:+.2f}")
# 3. shape of the bed with the voice on vs off, and of the voice itself in the room
C=[1250,2000,3150,5000,8000,12500]
def shape(x,frames):
    acc=[]
    for f_ in np.flatnonzero(frames)[:800]:
        seg=x[f_*H:(f_+1)*H]; acc.append(np.abs(np.fft.rfft(seg*np.hanning(len(seg)),n=4096))**2)
    p=np.median(acc,axis=0); fq=np.fft.rfftfreq(4096,1/FS)
    v=np.array([10*np.log10(p[(fq>=c/2**(1/6))&(fq<c*2**(1/6))].mean()+1e-30) for c in C]); return v-v[3]
print("\nshape re 5 kHz:"+" "*22+" ".join(f"{c:>7}" for c in C))
for lab,x,m in (("hi-hat bed, voice on",hh.mean(axis=1),on),("hi-hat bed, voice off",hh.mean(axis=1),off),("room vocals stem, voice on",voc.mean(axis=1),on),("close voice (Tr2), voice on",tvs,on)):
    print(f"   {lab:38}"+" ".join(f"{v:+7.1f}" for v in shape(x,m)))
# 4. a simple predictor: bed dB = a + b * voice dB, fit on hit-free frames; how much of the bed's variance it explains
m=sel&(fv>-90)
A=np.column_stack([np.ones(m.sum()),np.clip(fv[m],-80,0)]); coef,res,_,_=np.linalg.lstsq(A,fh[m],rcond=None)
pred=A@coef; print(f"\nbed = {coef[0]:.1f} + {coef[1]:.2f} x voice(dB): residual sd {np.std(fh[m]-pred):.1f} dB vs {np.std(fh[m]):.1f} dB without the voice (hit-free frames 54-180 s)")
