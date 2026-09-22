/**
 * mobile.mjs -- layout and touch regression in a real browser.
 *
 * The DOM stub in smoke.mjs cannot catch layout: it has no box model, so it reports
 * whatever getBoundingClientRect was hard-coded to return. Every mobile defect found so
 * far -- a sticky bar overflowing the viewport, a loading overlay covering its caption, a
 * tooltip running off a 320px screen -- was invisible to it and visible here. This drives
 * headless Firefox over a real HTTP server, so it exercises the code path a visitor gets.
 *
 * Checks, at ten widths from 320 to 1024:
 *   1. no document-level horizontal overflow, before and after the sticky bar appears
 *      (scrolling INSIDE the pupil track, pipeline and citation block is intentional and
 *      is excluded -- see clipped() in the escapee probe)
 *   2. the sticky slider and "New aberration" stay on screen
 *   3. the model-loading overlay fits its panel and does not cover the caption
 *   4. no overflow after live resizing, and the scatter selection survives it
 *   5. touch: a tap pins the detail card, selects that pupil, a tap away dismisses it,
 *      a real finger drag scrolls the page, and nothing cancels touchmove
 *   6. the decomposition tooltip stays inside the viewport at 320px
 *
 * A failing width prints the offending elements rather than just a number, which is what
 * made the original sticky-bar defect findable.
 *
 * Two engines, on purpose:
 *   Firefox for layout, because it is where the reported overflow was first reproduced
 *     (scrollWidth 693 at a 390 viewport, 800 at 768).
 *   Chromium for touch, because Playwright does not support mobile emulation in Firefox.
 *     That is not a silent no-op: with isMobile the Firefox layout viewport widens to fit
 *     overflowing content (a 600px box in a 390 viewport reports innerWidth 608), so
 *     overflow could hide itself, and hasTouch leaves ontouchstart false and
 *     maxTouchPoints 0. Chromium honours both, and its CDP session can dispatch a real
 *     touch drag -- window.scrollBy() only proves programmatic scrolling and would pass
 *     even if a handler blocked the gesture.
 *
 * Run:  node test/mobile.mjs
 *       npm install && npx playwright install firefox chromium
 * Exit: 0 all passed | 1 a check failed | 2 nothing ran (playwright absent)
 */
import { createServer } from 'node:http';
import { statSync, createReadStream } from 'node:fs';
import { join, extname, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

let firefox;
try { ({ firefox } = await import('playwright')); }
catch {
  console.error('mobile.mjs: playwright is not installed.');
  console.error('  npm install && npx playwright install firefox');
  console.error('  (browser tests are optional; the other suites need no dependencies)');
  process.exit(2);
}

const ROOT = normalize(join(dirname(fileURLToPath(import.meta.url)), '..'));
const WIDTHS = [320, 360, 390, 430, 520, 540, 620, 700, 768, 1024];
const MIME = { '.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript',
  '.json':'application/json', '.wasm':'application/wasm', '.onnx':'application/octet-stream',
  '.bin':'application/octet-stream', '.png':'image/png', '.webp':'image/webp', '.svg':'image/svg+xml' };

const srv = createServer((rq, rs) => {
  let f = join(ROOT, normalize(decodeURI(rq.url.split('?')[0])).replace(/^(\.\.[/\\])+/, ''));
  if (f.endsWith('/')) f += 'index.html';
  try {
    const st = statSync(f);
    rs.writeHead(200, { 'content-type': MIME[extname(f)] || 'application/octet-stream', 'content-length': st.size });
    createReadStream(f).pipe(rs);
  } catch { rs.writeHead(404); rs.end(); }
});
await new Promise(r => srv.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${srv.address().port}/`;

let fail = 0;
const ok = (c, n, d = '') => { console.log(`${c ? 'PASS' : 'FAIL'}  ${n}${d ? '   ' + d : ''}`); if (!c) fail++; };
const overflow = p => p.evaluate(() => {
  const r = document.documentElement;
  return { sw: r.scrollWidth, cw: r.clientWidth, inner: innerWidth };
});

/* Which elements actually stick out. An element inside a container with overflow-x
   auto/scroll/hidden is clipped by design -- the pupil track scrolls sideways on purpose
   -- so only un-clipped escapees are real defects. */
const escapees = p => p.evaluate(() => {
  const vw = document.documentElement.clientWidth;
  const clipped = el => {
    for (let q = el.parentElement; q; q = q.parentElement) {
      const o = getComputedStyle(q).overflowX;
      if (o === 'auto' || o === 'scroll' || o === 'hidden') return true;
    }
    return false;
  };
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.right <= vw + 1 || clipped(el)) continue;
    const cls = typeof el.className === 'string' && el.className
      ? '.' + el.className.trim().split(/\s+/).slice(0, 3).join('.') : '';
    out.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${cls}`.slice(0, 46)
      + ` right=${Math.round(r.right)} w=${Math.round(r.width)}`);
  }
  return out.slice(0, 8);
});

const b = await firefox.launch({ headless: true });

/* Chromium is only needed for the touch sections. If it is missing, say which checks did
   not run rather than reporting a pass count that quietly excludes them. */
let chrome = null, chromeWhy = '';
try { const { chromium } = await import('playwright'); chrome = await chromium.launch({ headless: true }); }
catch (e) { chromeWhy = e.message.split('\n')[0]; }
let skipped = 0;
const skip = (n, why) => { console.log(`SKIP  ${n}   ${why}`); skipped++; };
const settled = async p => { await p.evaluate(() => document.fonts?.ready); await p.waitForTimeout(600); };

console.log('=== fresh loads ===');
for (const w of WIDTHS) {
  const p = await b.newPage({ viewport: { width: w, height: 900 } });
  await p.goto(base, { waitUntil: 'load' });
  await settled(p);

  const before = await overflow(p);
  const gate = await p.evaluate(() => {
    /* The overlay explaining the model download sits over the first scene tile. It must
       fit inside its panel and leave the caption readable -- a fixed breakpoint used to
       fail here at 768, where the scene picker beside the grid also takes width. */
    const shot = document.querySelector('.shot.good');
    const g = document.getElementById('gate').getBoundingClientRect();
    const inner = document.getElementById('gate').firstElementChild.getBoundingClientRect();
    const cap = shot.querySelector('.hd').getBoundingClientRect();
    const cv = shot.querySelector('canvas').getBoundingClientRect();
    return { tile: Math.round(cv.width),
             fits: inner.left >= g.left - 1 && inner.right <= g.right + 1
                && inner.top >= g.top - 1 && inner.bottom <= g.bottom + 1,
             clear: g.bottom <= cap.top + 1 };
  });

  await p.evaluate(() => window.scrollTo(0, 1400));
  await p.waitForTimeout(400);
  const after = await overflow(p);
  const barOn = await p.evaluate(() => document.getElementById('sticky').classList.contains('on'));
  const reach = await p.evaluate(() => {
    const btn = document.getElementById('bSeed2'), sl = document.getElementById('rAlpha2');
    if (!btn || !sl) return false;
    const vw = document.documentElement.clientWidth;
    const br = btn.getBoundingClientRect(), sr = sl.getBoundingClientRect();
    return br.right <= vw + 1 && br.left >= -1 && sr.right <= vw + 1 && sr.left >= -1;
  });

  const clean = before.sw <= before.cw + 1 && after.sw <= after.cw + 1;
  ok(clean, `no overflow @${w}`,
    `inner ${before.inner}, ${before.sw}/${before.cw} -> ${after.sw}/${after.cw}${barOn ? ' (bar shown)' : ''}`);
  if (!clean) for (const e of await escapees(p)) console.log(`        ${e}`);
  if (barOn) ok(reach, `sticky controls on screen @${w}`);
  ok(gate.fits && gate.clear, `loading overlay fits, caption clear @${w}`, `tile ${gate.tile}px`);
  await p.close();
}

console.log('\n=== live resize, after interacting ===');
{
  const p = await b.newPage({ viewport: { width: 390, height: 844 } });
  await p.goto(base, { waitUntil: 'load' });
  await settled(p);
  await p.evaluate(() => {
    const s = document.getElementById('rAlpha');
    s.value = '12'; s.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await p.waitForTimeout(400);
  const sel0 = await p.evaluate(() => document.getElementById('pName').textContent);
  for (const w of [844, 390, 320, 768, 390]) {
    await p.setViewportSize({ width: w, height: 844 });
    await p.waitForTimeout(450);
    const o = await overflow(p);
    ok(o.sw <= o.cw + 1, `no overflow after resize to ${w}`, `${o.sw}/${o.cw}`);
    if (o.sw > o.cw + 1) for (const e of await escapees(p)) console.log(`        ${e}`);
  }
  const sel1 = await p.evaluate(() => document.getElementById('pName').textContent);
  ok(sel0 === sel1 && /\d/.test(sel1), 'selection survives resizing', `"${sel0}" -> "${sel1}"`);
  await p.close();
}

console.log('\n=== touch ===');
if (!chrome) skip('all touch checks', `chromium unavailable: ${chromeWhy}`);
else {
  const ctx = await chrome.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  const p = await ctx.newPage();
  await p.goto(base, { waitUntil: 'load' });
  await p.evaluate(() => document.fonts?.ready);
  await p.waitForTimeout(800);

  /* Confirm the emulation actually took. In Firefox these are false/0 even with
     hasTouch, which is why the touch sections run here. */
  const emu = await p.evaluate(() => ({ touch: 'ontouchstart' in window, pts: navigator.maxTouchPoints,
                                        coarse: matchMedia('(pointer:coarse)').matches, iw: innerWidth }));
  ok(emu.touch && emu.pts > 0 && emu.iw === 390, 'touch emulation is in effect',
    `ontouchstart ${emu.touch}, maxTouchPoints ${emu.pts}, coarse ${emu.coarse}, innerWidth ${emu.iw}`);

  await p.evaluate(() => document.querySelector('#scatter').scrollIntoView({ block: 'center' }));
  await p.waitForTimeout(1200);                       // let the scatter fetch its data
  const box = await p.locator('#scatter').boundingBox();

  /* Aim inside the point cloud. At 390px the cloud sits left of centre, so a tap at 0.72
     lands on blank canvas -- that was a harness bug, not a page bug. The later candidates
     are a fallback: if the cloud moves, this still tests tap handling rather than point
     layout. A page with broken touch fails every candidate. */
  const read = () => p.evaluate(() => ({
    card: !document.getElementById('scHover').hidden,
    name: document.getElementById('pName').textContent,
  }));
  let tapped = null;
  for (const [fx, fy] of [[0.32, 0.50], [0.30, 0.30], [0.40, 0.70], [0.25, 0.45], [0.50, 0.75]]) {
    await p.touchscreen.tap(box.x + box.width * fx, box.y + box.height * fy);
    await p.waitForTimeout(600);                      // masks may load
    tapped = await read();
    if (tapped.card) break;
  }
  ok(tapped.card, 'tap shows the detail card', `pupil "${tapped.name}"`);
  ok(/pupil\s*\d+/i.test(tapped.name), 'tap selects that pupil', tapped.name);

  await p.touchscreen.tap(20, box.y - 60);
  await p.waitForTimeout(400);
  ok(await p.evaluate(() => document.getElementById('scHover').hidden), 'tapping away dismisses the card');

  /* A real finger drag, through the browser's input pipeline. Synthetic TouchEvents
     dispatched from page script are untrusted and never scroll, and window.scrollBy only
     shows that programmatic scrolling works -- it would pass even if a touchmove handler
     called preventDefault, or an ancestor set touch-action:none. This drags upward over
     the scatter canvas, the element most likely to swallow the gesture. */
  const cdp = await ctx.newCDPSession(p);
  const drag = async (x, y, dy, steps = 12) => {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
    for (let i = 1; i <= steps; i++) {
      await cdp.send('Input.dispatchTouchEvent',
        { type: 'touchMove', touchPoints: [{ x, y: y + (dy * i) / steps }] });
      await p.waitForTimeout(16);
    }
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  };
  const box2 = await p.locator('#scatter').boundingBox();
  const y0 = await p.evaluate(() => scrollY);
  await drag(box2.x + box2.width / 2, box2.y + box2.height / 2, -260);
  await p.waitForTimeout(700);                        // let any fling settle
  const y1 = await p.evaluate(() => scrollY);
  ok(y1 > y0, 'a finger drag over the canvas scrolls the page', `scrollY ${y0} -> ${y1}`);

  /* The mechanism that would break it, tested directly: an ancestor opting out of
     panning, or a listener cancelling the move. */
  const blockers = await p.evaluate(() => {
    const cv = document.getElementById('scatter');
    const bad = [];
    for (let el = cv; el; el = el.parentElement) {
      const ta = getComputedStyle(el).touchAction;
      if (ta === 'none' || ta === 'pinch-zoom') bad.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}: ${ta}`);
    }
    const ev = new TouchEvent('touchmove', { bubbles: true, cancelable: true });
    cv.dispatchEvent(ev);
    return { bad, cancelled: ev.defaultPrevented };
  });
  ok(blockers.bad.length === 0 && !blockers.cancelled, 'nothing blocks touch panning',
    `touch-action blockers: ${blockers.bad.join('; ') || 'none'}; touchmove cancelled: ${blockers.cancelled}`);

  await ctx.close();
}

console.log('\n=== decomposition tooltip at the narrowest width ===');
if (!chrome) skip('tooltip checks at 320px', `chromium unavailable: ${chromeWhy}`);
else {
  /* The longest label is ~46 monospace characters. With white-space:nowrap it measured
     about 326px, so at a 320px viewport the positioning clamp had no room left and the
     tooltip ran off the right edge (reported x~332). It must wrap instead. */
  const ctx = await chrome.newContext({ viewport: { width: 320, height: 844 }, hasTouch: true, isMobile: true });
  const p = await ctx.newPage();
  await p.goto(base, { waitUntil: 'load' });
  await settled(p);

  /* The long label is the ASYMMETRIC one. The default pupil is a circle, which has no
     asymmetric region, so the tooltip can only ever show the shorter symmetric string --
     a test that taps the default pupil passes even with nowrap restored (verified by
     mutation). Drive the slider to the most asymmetric pupil first. */
  const slots = await p.evaluate(() => {
    const s = document.getElementById('rAlpha');
    s.value = s.max; s.dispatchEvent(new Event('input', { bubbles: true }));
    return s.max;
  });
  await p.waitForTimeout(500);
  await p.evaluate(() => document.getElementById('cDec').scrollIntoView({ block: 'center' }));
  await p.waitForTimeout(400);
  const box = await p.locator('#cDec').boundingBox();

  let worst = null, sawLong = false;
  for (let fy = 0.2; fy <= 0.82; fy += 0.08) {
    for (let fx = 0.12; fx <= 0.9; fx += 0.06) {
      await p.touchscreen.tap(box.x + box.width * fx, box.y + box.height * fy);
      await p.waitForTimeout(90);
      const r = await p.evaluate(() => {
        const t = document.getElementById('tip');
        if (!t.classList.contains('on')) return null;
        const b = t.getBoundingClientRect(), de = document.documentElement;
        return { left: Math.round(b.left), right: Math.round(b.right), w: Math.round(b.width),
                 vw: de.clientWidth, text: t.textContent };
      });
      if (!r) continue;
      if (/^asymmetric/.test(r.text)) sawLong = true;
      if (!worst || r.right > worst.right || r.w > worst.w) worst = r;
    }
  }
  ok(!!worst, 'a tooltip appears on tap',
    worst ? `"${worst.text.slice(0, 26)}..." (slider at ${slots})` : 'none shown');
  ok(sawLong, 'the longest (asymmetric) label was reached',
    sawLong ? '' : 'only the short symmetric label was hit -- the check would be vacuous');
  if (worst) {
    ok(worst.right <= worst.vw + 1 && worst.left >= -1,
      'tooltip stays inside a 320px viewport',
      `left ${worst.left}, right ${worst.right}, viewport ${worst.vw}`);
    ok(worst.w <= worst.vw,
      'tooltip is narrower than the viewport (it wraps)', `width ${worst.w} <= ${worst.vw}`);
  }
  const o = await overflow(p);
  ok(o.sw <= o.cw + 1, 'a shown tooltip causes no page overflow', `${o.sw}/${o.cw}`);
  await ctx.close();
}

await b.close(); if (chrome) await chrome.close(); srv.close();
if (fail) console.log(`\n${fail} FAILURE(S)${skipped ? `, ${skipped} SECTION(S) SKIPPED` : ''}`);
else if (skipped) console.log(`\nno failures, but ${skipped} section(s) were SKIPPED -- not a full pass`);
else console.log('\nALL MOBILE CHECKS PASSED');
process.exit(fail ? 1 : skipped ? 2 : 0);
