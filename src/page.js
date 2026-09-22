
import { synthPhase, psfFromPupil, simulateMeasurement, asymmetryCorrected,
         autoconvArgmax, throughput, maskedPhase, PAPER_MODES } from './optics.js';
import { base64ToBytes, unpackBitsF64 } from './ladder.js';
import { LADDER_B64, FRAMES } from './ladderdata.js';
import { SHAPES } from './shapesdata.js';
import { testCoeffs, N_TEST_PHASES } from './phasedata.js';
import * as net from './net.js';
import * as Scat from './scatter.js';
import * as Plasma from './plasma.js';
import { createScheduler } from './netqueue.js';
import { CAMERAMAN_B64, CAMERAMAN_SIZE, CAMERAMAN_CREDIT } from './cameramandata.js';
import * as Scene from './scene.js';
const { convolvePsf, psnr, ssim } = Scene;

const N=128, N2=N*N, ALPHA_MAX=0.353933, NW=16, COEFF=2.0;
const bytes=base64ToBytes(LADDER_B64);
const MASKS=FRAMES.map((f,i)=>unpackBitsF64(bytes,i*2048,N,N));
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* ---------- shape drawing (vector, crisp) ---------- */
const VIEW={x:40,y:40,s:48};                       // window on the 128 grid
function path(ctx,sh,W,dx=0,dy=0,flip=false){
  const k=W/VIEW.s, tx=p=>(p-VIEW.x)*k, ty=p=>(p-VIEW.y)*k;
  ctx.beginPath();
  if(sh.kind==='circle'){
    const cx=flip?127-sh.cx:sh.cx, cy=flip?127-sh.cy:sh.cy;
    ctx.arc(tx(cx+dx),ty(cy+dy),sh.r*k,0,Math.PI*2);
  } else {
    sh.pts.forEach(([px,py],i)=>{
      const X=flip?127-px:px, Y=flip?127-py:py;
      const a=tx(X+dx), b=ty(Y+dy);
      i?ctx.lineTo(a,b):ctx.moveTo(a,b);
    });
    ctx.closePath();
  }
}
const isDark=()=>matchMedia('(prefers-color-scheme:dark)').matches
  && document.documentElement.getAttribute('data-theme')!=='light'
  || document.documentElement.getAttribute('data-theme')==='dark';
function fillShape(cv,sh,color,opts={}){
  const g=cv.getContext('2d'), W=cv.width;
  g.clearRect(0,0,W,W); g.fillStyle=css('--sunk'); g.fillRect(0,0,W,W);
  g.fillStyle=color; path(g,sh,W,0,0,opts.flip); g.fill();
  if(opts.stroke){ g.strokeStyle=opts.stroke; g.lineWidth=opts.lw||1.5; g.stroke(); }
}
/* shift of the aligned flip, in 128-grid pixels, wrapped to [-64,64) */
function alignShift(mask){
  const [ay,ax]=autoconvArgmax(mask,N);
  let sy=(ay+1)%N, sx=(ax+1)%N;
  if(sy>=N/2) sy-=N; if(sx>=N/2) sx-=N;
  return {sy,sx};
}

/* ---------- field rendering ---------- */
const CROP_F={x:28,y:28,s:72}, CROP_P={x:40,y:40,s:48};
const tmp=document.createElement('canvas'); tmp.width=N; tmp.height=N;
const tctx=tmp.getContext('2d',{willReadFrequently:true});
const HOT=[[0,[10,12,22]],[.22,[40,28,94]],[.45,[134,32,110]],[.7,[226,82,60]],[.88,[248,178,66]],[1,[255,248,224]]];
const DIV=[[0,[26,88,158]],[.5,[240,240,243]],[1,[178,32,44]]];
const SEP=[[0,[198,45,36]],[.5,[217,138,24]],[1,[18,135,106]]];
function ramp(st,t){ t=Math.max(0,Math.min(1,t));
  for(let i=1;i<st.length;i++){ if(t<=st[i][0]){ const[p0,c0]=st[i-1],[p1,c1]=st[i];
    const u=(t-p0)/((p1-p0)||1);
    return [c0[0]+(c1[0]-c0[0])*u,c0[1]+(c1[1]-c0[1])*u,c0[2]+(c1[2]-c0[2])*u]; } }
  return st[st.length-1][1]; }
const rgb=c=>`rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
function paintField(cv,arr,o={}){
  const {sym=false,gamma=.32,max=null,crop=CROP_F}=o;
  let mx=max; if(mx===null){ mx=0; for(let p=0;p<N2;p++){const v=sym?Math.abs(arr[p]):arr[p]; if(v>mx)mx=v;} }
  mx=mx||1;
  const px=new Uint8ClampedArray(N2*4);
  for(let p=0;p<N2;p++){
    const c=sym?ramp(DIV,.5+.5*Math.max(-1,Math.min(1,arr[p]/mx)))
               :ramp(HOT,Math.pow(Math.max(0,arr[p])/mx,gamma));
    px[p*4]=c[0];px[p*4+1]=c[1];px[p*4+2]=c[2];px[p*4+3]=255;
  }
  tctx.putImageData(new ImageData(px,N,N),0,0);
  const g=cv.getContext('2d'); g.imageSmoothingEnabled=true;
  g.clearRect(0,0,cv.width,cv.height);
  g.drawImage(tmp,crop.x,crop.y,crop.s,crop.s,0,0,cv.width,cv.height);
}

/* ---------- physics ---------- */
function flipPhase(ph){ const o=new Float64Array(N2);
  for(let y=0;y<N;y++) for(let x=0;x<N;x++) o[y*N+x]=-ph[(N-1-y)*N+(N-1-x)];
  return o; }                                   /* twin wavefront; the pupil never moves */
function separation(A,B){ let sa=0,sb=0,d=0;
  for(let p=0;p<N2;p++){ sa+=A[p]; sb+=B[p]; d+=Math.abs(A[p]-B[p]); }
  return d/(sa+sb); }
function mulberry32(a){ return function(){ a|=0;a=a+0x6D2B79F5|0;
  let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296; }; }
/* Wavefronts are the PUBLISHED TEST SPLIT, not fresh draws: testCoeffs(i) returns
   the 33 Zernike coefficients of test phase i, and synthPhase regenerates it exactly. */
function phaseIndices(seed){ const r=mulberry32((seed*2654435761)>>>0);
  const out=[]; for(let k=0;k<NW;k++) out.push(Math.floor(r()*N_TEST_PHASES)); return out; }
const PHASES=()=>phaseIndices(seedBase).map(i=>synthPhase(testCoeffs(i),N));

/* ---------- state ---------- */
const NF=FRAMES.length;
let sel=0, seedBase=1, phases=null, trend=new Array(NF).fill(null), trendRun=0;
/* A pupil picked from the population scatter is not on the ladder, so it overrides the
   slider selection until the slider or track is touched again. */
let picked=null;                 // {id, mask, shape, alpha, thru, area}
const curMask =()=> picked ? picked.mask  : MASKS[sel];
const curShape=()=> picked ? picked.shape : SHAPES[sel];
const curName =()=> picked ? ('pupil '+picked.id) : FRAMES[sel].name;
let session=null;   /* the ORT session; scheduling state lives in netqueue.js */
const SCENE_N=512;
let userImg=null;
/* convolvePsf is circular, so an image that does not tile wraps its own edges back in.
   Taper the border to the image mean over ~8% of the frame to make it effectively tileable. */
function tileSafe(img,S){
  let mean=0; for(const v of img) mean+=v; mean/=img.length;
  const w=Math.max(4,Math.round(S*0.08)), out=Float64Array.from(img);
  const ramp=d=>0.5-0.5*Math.cos(Math.PI*Math.min(1,d/w));
  for(let y=0;y<S;y++) for(let x=0;x<S;x++){
    const t=Math.min(ramp(x),ramp(S-1-x),ramp(y),ramp(S-1-y));
    if(t<1){ const p=y*S+x; out[p]=mean+(img[p]-mean)*t; }
  }
  return out;
}
function imageToScene(bitmap,S){
  const c=document.createElement('canvas'); c.width=c.height=S;
  const g=c.getContext('2d',{willReadFrequently:true});
  const k=Math.max(S/bitmap.width,S/bitmap.height);          // cover-fit, centred
  const w=bitmap.width*k, h=bitmap.height*k;
  g.imageSmoothingQuality='high';
  g.drawImage(bitmap,(S-w)/2,(S-h)/2,w,h);
  const d=g.getImageData(0,0,S,S).data, out=new Float64Array(S*S);
  for(let p=0;p<S*S;p++)                                      // Rec. 709 luma
    out[p]=(0.2126*d[p*4]+0.7152*d[p*4+1]+0.0722*d[p*4+2])/255;
  return tileSafe(out,S);
}
function cameraman(size){
  const bin=atob(CAMERAMAN_B64), S0=CAMERAMAN_SIZE;
  const src=new Float64Array(S0*S0);
  for(let p=0;p<S0*S0;p++) src[p]=bin.charCodeAt(p)/255;
  if(size===S0) return tileSafe(src,size);
  const out=new Float64Array(size*size), k=(S0-1)/(size-1);
  for(let y=0;y<size;y++){
    const sy=y*k, y0=Math.floor(sy), fy=sy-y0, y1=Math.min(S0-1,y0+1);
    for(let x=0;x<size;x++){
      const sx=x*k, x0=Math.floor(sx), fx=sx-x0, x1=Math.min(S0-1,x0+1);
      out[y*size+x]=(1-fy)*((1-fx)*src[y0*S0+x0]+fx*src[y0*S0+x1])
                   +   fy *((1-fx)*src[y1*S0+x0]+fx*src[y1*S0+x1]);
    }
  }
  return tileSafe(out,size);
}
const SCENE_LIST = [
  ...(Scene.SCENES ?? [{ id:'usaf', label:'Resolution target', make:s=>Scene.usafTarget(s) }]),
  { id:'cameraman', label:'Cameraman', make:cameraman, display:'linear', credit:CAMERAMAN_CREDIT },
];
let sceneId = SCENE_LIST[0].id;
let scene = SCENE_LIST[0].make(SCENE_N);
const netCells=()=>[...document.querySelectorAll('.pipe .cell.net, .shot.good canvas')];

/* ---------- track ---------- */
FRAMES.forEach((f,i)=>{
  const b=document.createElement('button');
  b.className='chip'; b.type='button';
  b.setAttribute('aria-pressed', i===0?'true':'false');
  b.title=`${f.name} — α = ${f.alpha_corrected.toFixed(4)}`;
  const cv=document.createElement('canvas'); cv.width=72; cv.height=72;
  const t=document.createElement('span'); t.className='t';
  t.textContent=Math.round(100*f.alpha_corrected/ALPHA_MAX);
  b.append(cv,t); $('track').append(b);
  fillShape(cv,SHAPES[i],Plasma.forAlpha(f.alpha_corrected),{stroke:Plasma.outline(isDark()),lw:1});
  b.addEventListener('click',()=>{ picked=null; sel=i; $('rAlpha').value=i;
    if($('rAlpha2')) $('rAlpha2').value=i; update(); });
});

/* ---------- decomposition hover ---------- */
const tip=$('tip');
/* One routine for both pointers. Touch gets a tap-to-pin behaviour: the label stays until
   you tap elsewhere, because a finger cannot hover. */
let decPinned=false;
function decDetailAt(clientX,clientY){
  const cv=$('cDec'), r=cv.getBoundingClientRect(), W=cv.width;
  const gx=(clientX-r.left)/r.width*W, gy=(clientY-r.top)/r.height*W;
  const g=cv.getContext('2d'), sh=curShape(), {sy,sx}=alignShift(curMask());
  path(g,sh,W); const inP=g.isPointInPath(gx,gy);
  path(g,sh,W,sx,sy,true); const inF=g.isPointInPath(gx,gy);
  if(inP&&inF) return 'symmetric — overlaps its own flip';
  if(inP)      return 'asymmetric — this is what breaks the degeneracy';
  return null;
}
function showTip(txt,x,y){
  if(!txt){ tip.classList.remove('on'); return; }
  tip.textContent=txt; tip.classList.add('on');
  const w=tip.offsetWidth||180;
  tip.style.left=Math.max(6,Math.min(innerWidth-w-6,x+12))+'px';
  tip.style.top=Math.max(6,y-30)+'px';
}
$('cDec').addEventListener('mousemove',e=>{
  if(decPinned) return;                       /* a tap has pinned a label; leave it */
  showTip(decDetailAt(e.clientX,e.clientY),e.clientX,e.clientY);
});
$('cDec').addEventListener('mouseleave',()=>{ if(!decPinned) showTip(null); });
$('cDec').addEventListener('pointerup',e=>{
  if(e.pointerType==='mouse') return;          /* mouse already has hover */
  const txt=decDetailAt(e.clientX,e.clientY);
  decPinned=!!txt; showTip(txt,e.clientX,e.clientY);
});
/* tapping anywhere else dismisses a pinned label */
document.addEventListener('pointerdown',e=>{
  if(decPinned && e.target!==$('cDec')){ decPinned=false; showTip(null); }
},true);

/* ---------- trend chart ---------- */
/* The canvas must match its displayed box or everything inside it is scaled
   anisotropically -- that is what squashed the labels. Size the backing store to the
   element (times devicePixelRatio) and draw in CSS pixels. */
function fitCanvas(cv){
  const r=cv.getBoundingClientRect();
  if(!r.width) return false;
  const dpr=Math.min(window.devicePixelRatio||1, 2);
  const w=Math.round(r.width*dpr), h=Math.round(r.height*dpr);
  if(cv.width!==w||cv.height!==h){ cv.width=w; cv.height=h; }
  const g=cv.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  return true;
}
function drawTrend(){
  const cv=$('trend');
  if(!fitCanvas(cv)) return;
  const W=cv.width/Math.min(window.devicePixelRatio||1,2),
        H=cv.height/Math.min(window.devicePixelRatio||1,2),
        g=cv.getContext('2d');
  const L=52,R=14,T=14,B=34, iw=W-L-R, ih=H-T-B;
  g.clearRect(0,0,W,H);
  const xs=a=>L+(a/0.36)*iw, ys=v=>T+ih-(v/0.6)*ih;
  g.strokeStyle=css('--line'); g.fillStyle=css('--faint');
  g.lineWidth=1; g.font='500 11px '+css('--mono').split(',')[0].replace(/"/g,'');
  for(let v=0;v<=0.6;v+=0.15){                    /* recessive grid, labelled to real values */
    g.beginPath(); g.moveTo(L,ys(v)); g.lineTo(W-R,ys(v)); g.stroke();
    g.textAlign='right'; g.textBaseline='middle';
    g.fillText((v*100).toFixed(0)+'%',L-8,ys(v));
  }
  g.textAlign='center'; g.textBaseline='top';
  for(let a=0;a<=0.36;a+=0.09){
    g.fillText(a.toFixed(2),xs(a),H-B+8);
  }
  g.fillStyle=css('--muted'); g.textAlign='center';
  g.fillText('asymmetry α',L+iw/2,H-B+21);
  g.save(); g.translate(13,T+ih/2); g.rotate(-Math.PI/2);
  g.textAlign='center'; g.textBaseline='middle'; g.fillText('separation',0,0); g.restore();

  const pts=trend.map((v,i)=>v===null?null:[FRAMES[i].alpha_corrected,v])
    .filter(Boolean).sort((a,b)=>a[0]-b[0]).map(([a,v])=>[xs(a),ys(v)]);
  if(pts.length>1){
    g.strokeStyle=css('--line-strong'); g.lineWidth=2; g.beginPath();
    pts.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y)); g.stroke();
  }
  trend.forEach((v,i)=>{
    if(v===null) return;
    /* No ladder point is 'current' when an off-ladder pupil is picked from the scatter --
       enlarging the old slider position would claim the wrong pupil. */
    const x=xs(FRAMES[i].alpha_corrected), y=ys(v), cur=!picked && i===sel;
    g.beginPath(); g.arc(x,y,cur?7:4,0,Math.PI*2);
    g.fillStyle=rgb(ramp(SEP,v/0.6)); g.fill();
    g.lineWidth=2; g.strokeStyle=css('--surface'); g.stroke();   /* 2px surface ring */
  });
  /* The trend chart only holds the ladder pupils, so an off-ladder selection has no point
     to label -- marking the old slider position would claim the wrong pupil. */
  const v = picked ? null : trend[sel];
  if(v!==null){                                    /* direct label: current pupil only */
    const x=xs(FRAMES[sel].alpha_corrected), y=ys(v);
    g.fillStyle=css('--ink'); g.font='500 12px '+css('--mono').split(',')[0].replace(/"/g,'');
    g.textAlign=x>L+iw*0.75?'right':'left'; g.textBaseline='bottom';
    g.fillText(` ${(v*100).toFixed(1)}% `,x+(x>L+iw*0.75?-10:10),y-8);
  }
}
async function buildTrend(){
  /* Snapshot the wavefronts and take a run id. Clicking "New aberration" twice used to
     leave two loops writing into the same trend array from different phase sets, and the
     correlation printed at the end was computed over the mixture. */
  const run=++trendRun, myPhases=phases;
  trend=new Array(NF).fill(null);
  const order=[...Array(NF).keys()];
  for(const i of order){
    let s=0;
    for(const ph of myPhases){
      const gt=maskedPhase(MASKS[i],ph,N), tw=maskedPhase(MASKS[i],flipPhase(ph),N);
      s+=separation(psfFromPupil(MASKS[i],gt,N), psfFromPupil(MASKS[i],tw,N));
    }
    if(run!==trendRun) return;                    /* superseded */
    trend[i]=s/myPhases.length;
    $('tNote').textContent=`${i+1} / ${NF} pupils · ${NW} wavefronts each`;
    drawTrend();
    await new Promise(r=>requestAnimationFrame(r));
    if(run!==trendRun) return;
  }
  const xs=trend.map((_,i)=>FRAMES[i].alpha_corrected), ys=trend;
  const mx=xs.reduce((a,b)=>a+b)/NF, my=ys.reduce((a,b)=>a+b)/NF;
  let sxy=0,sxx=0,syy=0;
  for(let i=0;i<NF;i++){ const dx=xs[i]-mx, dy=ys[i]-my; sxy+=dx*dy; sxx+=dx*dx; syy+=dy*dy; }
  $('tCorr').textContent=`Pearson r = ${(sxy/Math.sqrt(sxx*syy)).toFixed(3)}`;
  $('tNote').textContent=`${NF} pupils × ${NW} wavefronts`;
}

/* ---------- main update ---------- */
const zero=new Float64Array(N2);
function update(){
  const mask=curMask(), f=FRAMES[sel], sh=curShape();
  [...$('track').children].forEach((c,i)=>c.setAttribute('aria-pressed',(!picked&&i===sel)?'true':'false'));

  const alpha=asymmetryCorrected(mask,N), thru=throughput(mask,N);
  const {sy,sx}=alignShift(mask);

  fillShape($('cPupil'),sh,css('--pupil'));
  fillShape($('cFlip'),sh,css('--flip'),{flip:true});
  /* aligned: pupil under its shifted flip */
  {
    const cv=$('cOver'), g=cv.getContext('2d'), W=cv.width;
    g.clearRect(0,0,W,W); g.fillStyle=css('--sunk'); g.fillRect(0,0,W,W);
    g.fillStyle=css('--pupil'); path(g,sh,W); g.fill();
    g.globalAlpha=.62; g.fillStyle=css('--flip'); path(g,sh,W,sx,sy,true); g.fill();
    g.globalAlpha=1;
  }
  /* split: asymmetric yellow, symmetric purple. The boolean runs on a TRANSPARENT
     offscreen canvas -- source-atop against an already-painted ground would clip to
     the ground instead of to the pupil, i.e. purple everywhere. */
  {
    const cv=$('cDec'), g=cv.getContext('2d'), W=cv.width;
    const off=document.createElement('canvas'); off.width=off.height=W;
    const og=off.getContext('2d');
    og.fillStyle=css('--asym'); path(og,sh,W); og.fill();
    og.globalCompositeOperation='source-atop';
    og.fillStyle=css('--sym'); path(og,sh,W,sx,sy,true); og.fill();
    g.clearRect(0,0,W,W); g.fillStyle=css('--sunk'); g.fillRect(0,0,W,W);
    g.drawImage(off,0,0);
  }
  let area=0, symA=0;
  for(let y=0;y<N;y++){ const fy=N-1-(((y-sy)%N+N)%N);
    for(let x=0;x<N;x++){ const p=y*N+x; if(!mask[p]) continue; area++;
      if(mask[fy*N+(N-1-(((x-sx)%N+N)%N))]) symA++; } }

  const pct=100*alpha/ALPHA_MAX;
  $('hAlpha').textContent=alpha.toFixed(4);
  $('hPct').innerHTML=pct.toFixed(1)+'<small>%</small>';
  $('pName').textContent=curName();
  if($('rAlpha2')) $('rAlpha2').value=sel;
  if($('sName'))  $('sName').textContent=curName();
  if($('sAlpha')) $('sAlpha').textContent=alpha.toFixed(3);
  if($('sPct'))   $('sPct').textContent=pct.toFixed(0)+'%';
  if($('sThumb')) fillShape($('sThumb'),sh,Plasma.forAlpha(alpha),{stroke:Plasma.outline(isDark()),lw:1});
  $('lgSym').textContent=`${(100*symA/area).toFixed(1)}%`;
  $('lgAsym').textContent=`${(100*(area-symA)/area).toFixed(1)}%`;
  $('hThru').innerHTML=(thru*100).toFixed(0)+'<small>%</small>';

  /* degeneracy on the first wavefront; the meter reports the average */
  const ph=phases[0];
  const gt=maskedPhase(mask,ph,N), tw=maskedPhase(mask,flipPhase(ph),N);
  const A=psfFromPupil(mask,gt,N), Bp=psfFromPupil(mask,tw,N);
  let pk=0; for(let p=0;p<N2;p++){ if(A[p]>pk)pk=A[p]; if(Bp[p]>pk)pk=Bp[p]; }
  const diff=new Float64Array(N2);
  for(let p=0;p<N2;p++) diff[p]=Math.abs(A[p]-Bp[p]);
  paintField($('cPlus'),A,{max:pk}); paintField($('cMinus'),Bp,{max:pk});
  paintField($('cDiff'),diff,{max:pk});

  let sAvg=0;
  for(const p of phases){
    const g1=maskedPhase(mask,p,N), g2=maskedPhase(mask,flipPhase(p),N);
    sAvg+=separation(psfFromPupil(mask,g1,N),psfFromPupil(mask,g2,N));
  }
  sAvg/=phases.length;
  for(const id of ['mNum','mNum2']){ const e=$(id); if(!e) continue;
    e.textContent=(sAvg*100).toFixed(1)+'%'; e.style.color=rgb(ramp(SEP,sAvg/0.6)); }
  $('mBar').style.width=Math.min(100,sAvg*100)+'%';
  $('mBar').style.background=rgb(ramp(SEP,sAvg/0.6));

  paintField($('cIdeal'),simulateMeasurement(mask,zero,{N}));
  paintField($('cAbb'),simulateMeasurement(mask,gt,{N}));
  fillShape($('cFlat'),sh,css('--line-strong'));   /* flat reference wavefront */
  /* The wavefront colour scale is fixed by the ground-truth aberration and reused for the
     estimate and the residual, so the residual visibly flattens instead of being
     re-normalised back to full saturation. Set here so it holds before the net loads. */
  wScale=0; for(let p=0;p<N2;p++) if(mask[p]){ const a=Math.abs(gt[p]); if(a>wScale) wScale=a; }
  wScale=wScale||1;
  paintField($('cPhase'),gt,{sym:true,crop:CROP_P,max:wScale});
  drawScale(wScale);

  if(window.redrawScatter) window.redrawScatter();
  scheduleScenes(mask,gt);
  drawTrend();
  scheduleNet();
}

/* ---------- the network ---------- */
function setNetState(cls, msg){
  const m=$('mStat'); m.className='mstat '+cls; m.textContent=msg;
}
async function loadNet(){
  if(session) return session;
  setNetState('busy','downloading…');
  try{
    const t0=performance.now();
    session=await net.loadModel(net.MODEL_URL,{
      onProgress:({loaded,total})=>{ if(total) setNetState('busy',
        `downloading ${(100*loaded/total).toFixed(0)}%`); },
    });
    setNetState('on',`ready · ${((performance.now()-t0)/1000).toFixed(1)} s`);
    netCells().forEach(c=>c.classList.remove('idle'));
    netq.flush();   /* run now that a session exists */
    return session;
  }catch(e){
    setNetState('err','model failed to load');
    console.error('[net] load failed:',e);
    throw e;
  }
}
const NET_READOUTS=['scStrehl','scBefore','sStrehl','vRms','vRms2','vPsnrS','vSsim','vPsnrC'];
function clearNetReadouts(){
  for(const id of NET_READOUTS){ const e=$(id); if(e){ e.textContent='—'; e.style.color=''; } }
  for(const id of ['scBar','scBefBar']){ const e=$(id); if(e) e.style.width='0%'; }
}
/* The scheduling rules live in netqueue.js so they can be tested against the real code
   rather than a copy. This supplies only what is specific to the page. */
const netq = createScheduler({
  debounceMs: 180,
  onPending(){
    /* Blank the numbers the moment the pupil changes: leaving them up attributes the
       previous pupil's Strehl to the new one until inference finishes, and if the model
       is not loaded at all it would sit there indefinitely. */
    clearNetReadouts();
    if(session) netCells().forEach(c=>c.classList.add('pending'));
  },
  gather(){
    /* curMask(), not MASKS[sel] -- a pupil picked from the scatter overrides the slider. */
    const mask=curMask(), ph=phases[0];
    const gt=maskedPhase(mask,ph,N);
    return { mask, gt, meas:simulateMeasurement(mask,gt,{N}) };
  },
  predict({mask,meas}){
    if(!session) return Promise.reject(new Error('model not loaded'));
    return net.predictPhase(session,meas,mask);
  },
  commit(pred,{mask,gt}){
    const res=net.residual(gt,pred,mask);
    const predM=maskedPhase(mask,Float64Array.from(pred),N);

    /* Same scale as the ground truth -- see update(). A prediction that overshoots simply
       clips, which is itself informative. */
    paintField($('cPredW'),net.removePiston(predM,mask),{sym:true,crop:CROP_P,max:wScale});
    paintField($('cResW'),net.removePiston(res,mask),{sym:true,crop:CROP_P,max:wScale});
    paintField($('cPredI'),psfFromPupil(mask,predM,N));
    const corrPsf=psfFromPupil(mask,res,N);
    paintField($('cCorrI'),corrPsf);

    const st=net.strehlOf(mask,gt,pred);
    const before=net.strehlOf(mask,gt,new Float32Array(N*N));   /* no correction at all */
    $('scStrehl').textContent=(st*100).toFixed(1)+'%';
    if($('sStrehl')){ $('sStrehl').textContent=(st*100).toFixed(1)+'%';
      $('sStrehl').style.color=rgb(ramp(SEP,st)); }
    $('scStrehl').style.color=rgb(ramp(SEP,st));
    $('scBar').style.width=(st*100)+'%';
    $('scBar').style.background=rgb(ramp(SEP,st));
    $('scBefore').textContent=(before*100).toFixed(1)+'%';
    $('scBefBar').style.width=(before*100)+'%';
    /* Piston is invisible to the image -- a constant phase is a unit-modulus factor
       outside |FFT|^2 -- so report the residual the way the paper's loss does, with
       piston removed. Leaving it in inflates the number without changing the optics. */
    const resNP=net.removePiston(res,mask);
    let rms=0,n=0; for(let p=0;p<N2;p++) if(mask[p]){ rms+=resNP[p]*resNP[p]; n++; }
    const rmsTxt=(n?Math.sqrt(rms/n):0).toFixed(3);
    $('vRms').textContent=rmsTxt+' rad';
    if($('vRms2')) $('vRms2').textContent=rmsTxt+' rad';

    /* only the CORRECTED frame needs the network; the other two are already drawn */
    const co=convolvePsf(scene,corrPsf,SCENE_N);
    paintGrey($('cSceneC'),forDisplay(co),dispRange);
    /* image-domain scores, corrected against this pupil's own diffraction limit */
    const p=psnr(co,refImg), sq=ssim(co,refImg,SCENE_N,1);
    $('vPsnrC').textContent=p.toFixed(2)+' dB';
    if($('vPsnrS')) $('vPsnrS').textContent=p.toFixed(1)+' dB';
    if($('vSsim'))  $('vSsim').textContent=sq.toFixed(3);

    netCells().forEach(c=>c.classList.remove('pending'));
  },
  onError(e){
    if(String(e && e.message)==='model not loaded') return;   /* expected before loading */
    setNetState('err','inference failed'); console.error('[net]',e);
  },
});
function scheduleNet(){ netq.schedule(); }

/* Field of view is set by the SCENE size, not by touching the PSF.
   An earlier version shrank the PSF on its own 128 grid to widen the field. That was
   wrong: at 3x the circle's diffraction core falls to ~1 px FWHM, below what the grid
   can represent, and the relative widths between pupils compress -- the circle-to-most-
   asymmetric ratio read 1.55 instead of its true 1.81, understating the resolution cost
   of a smaller pupil by 14%. Rendering the scene at 512 with the PSF untouched gives the
   same visual sharpening (18.4 dB vs 19.5) with the ratio exact. It costs ~166 ms per
   convolution instead of 16, so the scene panels are debounced. */
let refImg=null, abImg=null, sceneTimer=null, wScale=1;
function scheduleScenes(mask,gt){
  clearTimeout(sceneTimer);
  for(const id of ['cScene','cSceneA']) $(id).classList.add('pending');
  sceneTimer=setTimeout(()=>{ drawScenes(mask,gt);
    for(const id of ['cScene','cSceneA']) $(id).classList.remove('pending'); },110);
}
function drawScenes(mask,gt){
  const sharp=psfFromPupil(mask,new Float64Array(N2),N);
  refImg=convolvePsf(scene,sharp,SCENE_N);
  abImg =convolvePsf(scene,psfFromPupil(mask,gt,N),SCENE_N);
  const rD=forDisplay(refImg); dispRange=rangeOf(rD);
  paintGrey($('cScene'),rD,dispRange);
  paintGrey($('cSceneA'),forDisplay(abImg),dispRange);
  $('vPsnrA').textContent=psnr(abImg,refImg).toFixed(2)+' dB';   /* PSNR on linear values */
}
/* Some scenes (the star field) carry a huge linear dynamic range and need an asinh
   stretch to be visible at all. Apply the scene's own transform, then paint all three
   frames on ONE shared range taken from the reference, so they stay comparable. */
function forDisplay(img){
  const sc=SCENE_LIST.find(s=>s.id===sceneId);
  return (sc && sc.display==='asinh' && Scene.asinhStretch) ? Scene.asinhStretch(img) : img;
}
function rangeOf(img){ let mn=Infinity,mx=-Infinity;
  for(const v of img){ if(v<mn)mn=v; if(v>mx)mx=v; } return [mn,(mx-mn)||1]; }
let dispRange=[0,1];
function drawScale(wmax){
  const cv=$('cScale'); if(!cv) return;
  const g=cv.getContext('2d'), W=cv.width, H=cv.height;
  const im=g.createImageData(W,H);
  for(let x=0;x<W;x++){
    const c=ramp(DIV, x/(W-1));
    for(let y=0;y<H;y++){ const p=(y*W+x)*4;
      im.data[p]=c[0]; im.data[p+1]=c[1]; im.data[p+2]=c[2]; im.data[p+3]=255; }
  }
  g.putImageData(im,0,0);
  $('cbLo').textContent='−'+wmax.toFixed(1)+' rad';
  $('cbHi').textContent='+'+wmax.toFixed(1)+' rad';
}
function paintGrey(cv,img,range){
  const S=Math.round(Math.sqrt(img.length));
  const [mn,r]=range||rangeOf(img);
  const px=new Uint8ClampedArray(S*S*4);
  for(let p=0;p<S*S;p++){ const v=255*Math.max(0,Math.min(1,(img[p]-mn)/r));
    px[p*4]=px[p*4+1]=px[p*4+2]=v; px[p*4+3]=255; }
  const t=document.createElement('canvas'); t.width=t.height=S;
  t.getContext('2d').putImageData(new ImageData(px,S,S),0,0);
  const g=cv.getContext('2d'); g.imageSmoothingEnabled=true;
  g.clearRect(0,0,cv.width,cv.height); g.drawImage(t,0,0,cv.width,cv.height);
}

/* ---------- wiring ---------- */
for(const id of ['rAlpha','rAlpha2']){
  const el=$(id); if(!el) continue;
  el.addEventListener('input',e=>{
    picked=null; sel=+e.target.value;
    for(const other of ['rAlpha','rAlpha2']) if(other!==id && $(other)) $(other).value=sel;
    update();
  });
}

/* the sticky bar only appears once the real controls have scrolled off */
if(window.IntersectionObserver && $('sticky')){
  const ctl=document.querySelector('.demo .ctl');
  if(ctl) new IntersectionObserver(([e])=>{
    const on = !e.isIntersecting && e.boundingClientRect.top < 0;
    const bar = $('sticky');
    bar.classList.toggle('on', on);
    /* `inert` rather than aria-hidden: the bar holds a slider and a button, and
       aria-hidden on a container whose control has focus hides that control from
       assistive technology. inert removes it from focus order AND the a11y tree. */
    bar.inert = !on;
  },{threshold:0}).observe(ctl);
}
for(const id of ['bSeed','bSeed2']){
  const el=$(id); if(!el) continue;
  el.addEventListener('click',()=>{
    seedBase=(seedBase%997)+1; phases=PHASES(); update(); buildTrend();
  });
}
SCENE_LIST.forEach((sc,i)=>{
  const b=document.createElement('button');
  b.className='seg thumb'; b.type='button';
  b.setAttribute('aria-pressed', i===0?'true':'false');
  const cv=document.createElement('canvas'); cv.width=cv.height=104;
  const lb=document.createElement('span'); lb.className='tl'; lb.textContent=sc.label;
  b.append(cv,lb); $('picker').append(b);
  /* preview at 256 -- cheap, and the picker only needs to show what the scene looks like */
  try{
    const small=sc.make(256);
    const shown=(sc.display==='asinh'&&Scene.asinhStretch)?Scene.asinhStretch(small):small;
    paintGrey(cv,shown);
  }catch(err){ console.error('[scene] preview failed for',sc.id,err); }
  b.addEventListener('click',()=>{
    sceneId=sc.id; scene=sc.make(SCENE_N);
    [...$('picker').children].forEach(c=>c.setAttribute('aria-pressed',c===b?'true':'false'));
    update();
  });
});

const fileIn=document.createElement('input');
fileIn.type='file'; fileIn.accept='image/*'; fileIn.style.display='none';
document.body.append(fileIn);
const upBtn=document.createElement('button');
upBtn.className='seg thumb upload'; upBtn.type='button';
upBtn.setAttribute('aria-pressed','false');
upBtn.title='Use a picture of your own — it is read in your browser and never uploaded';
const upCv=document.createElement('canvas'); upCv.width=upCv.height=104;
const upLb=document.createElement('span'); upLb.className='tl'; upLb.textContent='Your image';
(function(){ const g=upCv.getContext('2d');
  g.fillStyle=css('--sunk'); g.fillRect(0,0,104,104);
  g.strokeStyle=css('--faint'); g.lineWidth=6; g.lineCap='round';
  g.beginPath(); g.moveTo(52,34); g.lineTo(52,70); g.moveTo(34,52); g.lineTo(70,52); g.stroke(); })();
upBtn.append(upCv,upLb);
upBtn.addEventListener('click',()=>fileIn.click());
$('picker').append(upBtn);
fileIn.addEventListener('change',async()=>{
  const f=fileIn.files&&fileIn.files[0]; if(!f) return;
  try{
    /* createImageBitmap rejects anything the browser cannot decode -- HEIC from an iPhone
       is the common one. Fall back to an <img>, which accepts whatever the browser can
       render, before giving up. */
    let bmp;
    try { bmp = await createImageBitmap(f); }
    catch { bmp = await new Promise((res,rej)=>{
      const url=URL.createObjectURL(f), im=new Image();
      im.onload=()=>{ URL.revokeObjectURL(url); res(im); };
      im.onerror=()=>{ URL.revokeObjectURL(url); rej(new Error('browser cannot decode '+(f.type||'this file'))); };
      im.src=url;
    }); }
    userImg=imageToScene(bmp,SCENE_N); bmp.close&&bmp.close();
    scene=userImg; sceneId='user';
    upLb.textContent=f.name.length>14?f.name.slice(0,12)+'…':f.name;
    paintGrey(upCv,userImg);
    [...$('picker').children].forEach(c=>c.setAttribute('aria-pressed',c===upBtn?'true':'false'));
    update();
  }catch(err){
    console.error('[scene] could not read image:',err);
    upLb.textContent = /decode/i.test(String(err)) ? 'Unsupported format' : 'Unreadable';
  }
  fileIn.value='';
});

$('bLoad').addEventListener('click',async e=>{
  e.target.disabled=true; e.target.textContent='loading…';
  try{ await loadNet(); $('gate').style.display='none'; }
  catch{ e.target.disabled=false; e.target.textContent='Retry'; }
});
netCells().forEach(c=>c.classList.add('idle'));
$('fNW').textContent=NW;
if($('tNW')) $('tNW').textContent=NW;
phases=PHASES();
if(window.ResizeObserver){ new ResizeObserver(()=>drawTrend()).observe($('trend')); }
window.addEventListener('resize',()=>drawTrend());
update();
buildTrend();


/* ---------- the population scatter ---------- */
(function(){
  const cv=$('scatter'); if(!cv) return;
  const tok=()=>({ink:css('--ink'),muted:css('--muted'),faint:css('--faint'),line:css('--line'),
                  surface:css('--surface'),sunk:css('--sunk'),accent:css('--pupil'),
                  mono:css('--mono').split(',')[0].replace(/"/g,'')});
  let field='loss', geom=null, hover=null, ready=false;
  const marked=FRAMES.map(f=>f.pool_id).filter(i=>i!=null);
  /* A pupil picked from the scatter is not on the ladder, so FRAMES[sel] is not what is
     displayed. Highlight the picked dot itself. */
  const currentPool=()=> picked ? picked.id : (FRAMES[sel] ? FRAMES[sel].pool_id : null);

  function render(){
    if(!ready) return;
    geom=Scat.draw(cv,{field,tokens:tok(),marked,current:currentPool(),hover,dark:isDark()});
  }
  window.redrawScatter=render;

  /* One routine renders the detail card; hover and tap both call it. On touch the card is
     pinned (a finger cannot hover) and the tap also selects that pupil, so the gesture
     does something rather than just previewing. */
  function showDetail(i, clientX, clientY){
    const box=$('scHover');
    if(i==null){ box.hidden=true; return; }
    const d=Scat.get();
    Scat.drawShape($('scShape'),i,Plasma.forAlpha(d.alpha[i]),css('--sunk'),Plasma.outline(isDark()));
    const mark=f=>f===field?' style="color:var(--pupil)"':'';
    $('scVals').innerHTML=
      `α <b>${d.alpha[i].toFixed(4)}</b><br>`+
      `<span${mark('strehl')}>Strehl <b${mark('strehl')}>${(d.strehl[i]*100).toFixed(1)}%</b></span><br>`+
      `<span${mark('loss')}>MSE <b${mark('loss')}>${d.loss[i].toFixed(3)}</b></span><br>`+
      `<span${mark('psfdiff')}>PSF diff <b${mark('psfdiff')}>${d.psfdiff[i].toFixed(3)}</b></span><br>`+
      `throughput <b>${(d.thru[i]*100).toFixed(0)}%</b>`;
    box.hidden=false;
    const r=cv.getBoundingClientRect();
    const w=box.offsetWidth||170, h=box.offsetHeight||90;
    const x=clientX-r.left, y=clientY-r.top;
    box.style.left=Math.max(4,Math.min(r.width-w-4, x+14))+'px';
    box.style.top =Math.max(4,Math.min(r.height-h-4, y-h-10))+'px';
  }

  async function selectPupil(i){
    if(!Scat.masksReady()){
      $('scNote').textContent='loading pupil masks (3 MB)…';
      try{ await Scat.loadMasks(); }
      catch(err){ $('scNote').textContent='could not load pupil masks'; console.error(err); return; }
    }
    const d=Scat.get();
    picked={ id:i, mask:Scat.maskOf(i), shape:{kind:'poly',pts:pairs(Scat.shapeOf(i))},
             alpha:d.alpha[i], thru:d.thru[i] };
    $('scNote').textContent=`showing pupil ${i} — pick another, or move the slider to go back`;
    update();
    const bar=$('sticky');
    if(bar){ bar.classList.remove('flash'); void bar.offsetWidth; bar.classList.add('flash'); }
  }
  function pairs(flat){ const o=[]; for(let k=0;k<flat.length;k+=2) o.push([flat[k],flat[k+1]]); return o; }

  let pinned=false;
  cv.addEventListener('mousemove',e=>{
    if(!geom||pinned) return;
    const r=cv.getBoundingClientRect();
    const i=Scat.pick(geom,e.clientX-r.left,e.clientY-r.top);
    if(i===hover) return;
    hover=i; render(); showDetail(i,e.clientX,e.clientY);
  });
  cv.addEventListener('mouseleave',()=>{ if(!pinned){ hover=null; $('scHover').hidden=true; render(); } });

  cv.addEventListener('pointerup',async e=>{
    if(!geom) return;
    const r=cv.getBoundingClientRect();
    const i=Scat.pick(geom, e.clientX-r.left, e.clientY-r.top, e.pointerType==='mouse'?14:22);
    if(i==null){ if(pinned){ pinned=false; hover=null; $('scHover').hidden=true; render(); } return; }
    hover=i; pinned=e.pointerType!=='mouse';    /* touch pins; mouse keeps hovering */
    render(); showDetail(i,e.clientX,e.clientY);
    await selectPupil(i);
  });
  /* dismiss a pinned card by tapping away */
  document.addEventListener('pointerdown',e=>{
    if(pinned && e.target!==cv){ pinned=false; hover=null; $('scHover').hidden=true; render(); }
  },true);

  const AXES=[['sfLoss','loss'],['sfDiff','psfdiff'],['sfStrehl','strehl']];
  for(const [id,f] of AXES){
    $(id).addEventListener('click',()=>{
      field=f;
      for(const [i2,f2] of AXES) $(i2).setAttribute('aria-pressed', String(f2===f));
      render();
    });
  }
  if(window.ResizeObserver) new ResizeObserver(()=>render()).observe(cv);

  Scat.load().then(d=>{
    ready=true;
    $('scNote').textContent=`${d.n} pupils · ${d.meta.phases_per_pupil} wavefronts each · published evaluation`;
    render();
  }).catch(e=>{ $('scNote').textContent='could not load the population data'; console.error('[scatter]',e); });
})();
