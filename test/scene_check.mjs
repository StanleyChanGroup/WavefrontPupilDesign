import { usafTarget, convolvePsf, psnr } from '../src/scene.js';
import { referenceCircle, psfFromPupil, synthPhase, maskedPhase, PAPER_MODES } from '../src/optics.js';
const N=128, S=256, S2=S*S;
const img=usafTarget(S);
let mn=Infinity,mx=-Infinity; for(const v of img){ if(v<mn)mn=v; if(v>mx)mx=v; }
console.log(`target ${S}x${S}  range [${mn.toFixed(3)}, ${mx.toFixed(3)}]  len ${img.length}`);

// (a) delta PSF must return the input unchanged
const delta=new Float64Array(N*N); delta[64*128+64]=1;
const same=convolvePsf(img,delta,S);
let dmax=0; for(let p=0;p<S2;p++) dmax=Math.max(dmax,Math.abs(same[p]-img[p]));
console.log(`(a) delta-PSF identity      max|out-in| = ${dmax.toExponential(3)}   ${dmax<1e-10?'PASS':'FAIL'}`);

// real aberrated PSF
const circle=referenceCircle(N);
const c=new Float64Array(PAPER_MODES.length);
let s=12345; const rnd=()=>{s=(s+0x6D2B79F5)|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
for(let i=0;i<c.length;i++) c[i]=(rnd()*2-1)*2;
const psf=psfFromPupil(circle,maskedPhase(circle,synthPhase(c,N),N),N);
const blur=convolvePsf(img,psf,S);

// (b) centroid must not move
const cen=a=>{let sx=0,sy=0,t=0; for(let y=0;y<S;y++)for(let x=0;x<S;x++){const v=a[y*S+x];sx+=v*x;sy+=v*y;t+=v;} return [sx/t,sy/t];};
const [ix,iy]=cen(img), [bx,by]=cen(blur);
console.log(`(b) centroid  in (${ix.toFixed(4)}, ${iy.toFixed(4)})  out (${bx.toFixed(4)}, ${by.toFixed(4)})  shift ${Math.hypot(bx-ix,by-iy).toExponential(3)} px   ${Math.hypot(bx-ix,by-iy)<1e-6?'PASS':'FAIL'}`);

// (c) mean brightness preserved
const mean=a=>{let t=0;for(const v of a)t+=v;return t/a.length;};
const r=Math.abs(mean(blur)/mean(img)-1);
console.log(`(c) mean brightness ratio   ${(mean(blur)/mean(img)).toFixed(9)}   rel err ${r.toExponential(3)}   ${r<1e-9?'PASS':'FAIL'}`);
console.log(`    PSNR(blurred, sharp) = ${psnr(blur,img).toFixed(2)} dB  (should be finite and < 40)`);

// timing
const t0=performance.now(); for(let i=0;i<5;i++) convolvePsf(img,psf,S);
console.log(`\none 256 convolution: ${((performance.now()-t0)/5).toFixed(1)} ms`);
const img5=usafTarget(512), t1=performance.now(); convolvePsf(img5,psf,512);
console.log(`one 512 convolution: ${(performance.now()-t1).toFixed(1)} ms`);
