/**
 * smoke.mjs -- does the page actually come up?
 *
 * Every other harness tests a module in isolation; none of them load src/page.js. That is
 * how commit 8682109 shipped with an unbalanced brace in the scheduler: the file parses as
 * a CommonJS script (which `node --check` assumes) but not as the ES module the browser
 * loads, so the whole page failed to initialise while the optics and scene suites passed.
 *
 * This runs page.js against a DOM stub sufficient for it to boot, then exercises the two
 * things that have broken repeatedly:
 *   - initialisation: the pupil track builds, readouts populate
 *   - the inference scheduler: changing pupils DURING a slow prediction must end up
 *     displaying the pupil that was selected last, never an earlier one
 *
 * Run:  node test/smoke.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '   ' + detail : ''}`);
  if (!ok) failures++;
};

/* ---------- a DOM stub, only as deep as page.js actually reaches ---------- */

const ctxStub = () => new Proxy({}, {
  get(_, k) {
    if (k === 'canvas') return { width: 300, height: 300 };
    if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h });
    if (k === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h });
    if (k === 'measureText') return () => ({ width: 10 });
    if (k === 'isPointInPath') return () => false;
    if (k === 'setTransform' || k === 'putImageData' || k === 'drawImage') return () => {};
    return () => {};
  },
  set() { return true; },
});

function makeEl(tag = 'div') {
  const el = {
    tagName: tag.toUpperCase(), children: [], style: {}, dataset: {},
    width: 300, height: 300, hidden: false, inert: false, value: '0',
    textContent: '', innerHTML: '', files: null, disabled: false,
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    _attrs: {},
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k] ?? null; },
    addEventListener(type, fn) { (this._ev ||= {})[type] = fn; },
    append(...kids) { kids.forEach(k => { this.children.push(k); k.parentNode = this; }); },
    appendChild(k) { this.append(k); return k; },
    getContext: () => ctxStub(),
    getBoundingClientRect: () => ({ width: 800, height: 300, left: 0, top: 0, right: 800, bottom: 300 }),
    scrollIntoView() {},
    focus() {},
    click() { this._ev?.click?.({ target: this }); },
    querySelector: () => null,
    querySelectorAll: () => [],
    get offsetWidth() { return 200; },
    get offsetHeight() { return 80; },
  };
  return el;
}

const byId = new Map();
const html = readFileSync(join(ROOT, 'index.html'), 'utf8');
for (const m of html.matchAll(/id="([^"]+)"/g)) {
  const tag = /<canvas[^>]*id="$/.test(html.slice(0, m.index)) ? 'canvas' : 'div';
  byId.set(m[1], makeEl(tag));
}

globalThis.document = {
  documentElement: makeEl('html'),
  body: makeEl('body'),
  getElementById: id => byId.get(id) ?? null,
  createElement: tag => makeEl(tag),
  querySelector: sel => byId.get(sel.replace(/^[.#]/, '')) ?? makeEl(),
  querySelectorAll: () => [],
  addEventListener() {},
};
globalThis.window = {
  devicePixelRatio: 1, innerWidth: 1200,
  addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }),
  ResizeObserver: class { observe() {} }, IntersectionObserver: class { observe() {} },
};
globalThis.matchMedia = window.matchMedia;
globalThis.ResizeObserver = window.ResizeObserver;
globalThis.IntersectionObserver = window.IntersectionObserver;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#1B96DC' });
globalThis.requestAnimationFrame = fn => setTimeout(fn, 0);
globalThis.atob = s => Buffer.from(s, 'base64').toString('binary');
globalThis.ImageData = class { constructor(d, w, h) {
  if (typeof d === 'number') { this.width = d; this.height = w; this.data = new Uint8ClampedArray(d * w * 4); }
  else { this.data = d; this.width = w; this.height = h ?? (d.length / 4 / w); } } };
globalThis.URL.createObjectURL ??= () => 'blob:stub';
globalThis.URL.revokeObjectURL ??= () => {};
globalThis.createImageBitmap ??= async () => { throw new Error('no bitmap in the stub'); };
globalThis.fetch = async (u) => {
  const p = join(ROOT, String(u).split('?')[0].replace('./', ''));
  const b = readFileSync(p);
  return { ok: true, json: async () => JSON.parse(b.toString('utf8')),
           arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
           headers: { get: () => String(b.length) }, body: null };
};

/* ---------- 1. does it boot? ---------- */

let booted = true, bootErr = null;
try {
  await import(join(ROOT, 'src/page.js'));
  await new Promise(r => setTimeout(r, 300));   // let the initial update() settle
} catch (e) { booted = false; bootErr = e; }

check('page.js initialises without throwing', booted, bootErr ? String(bootErr).slice(0, 140) : '');

const track = byId.get('track');
check('pupil track is populated', !!track && track.children.length > 0,
  track ? `${track.children.length} chips` : 'no #track');

const alpha = byId.get('hAlpha');
check('asymmetry readout is filled', !!alpha && /\d/.test(alpha.textContent),
  alpha ? `"${alpha.textContent}"` : 'missing');

/* ---------- 2. the REAL scheduler from src/netqueue.js ---------- */
/* Previously this section reimplemented the scheduler, so it could only ever test its own
   copy. It now imports the production module and drives it with a mock predictor, which
   is what makes a future regression in the real code fail here. */

const { createScheduler } = await import(join(ROOT, 'src/netqueue.js'));

function harness({ latency = 120, failOn = null } = {}) {
  let current = 'A';
  const committed = [];
  let predictions = 0;
  const q = createScheduler({
    debounceMs: 20,
    gather: () => ({ pupil: current }),
    predict: async (inputs) => {
      predictions++;
      await new Promise(r => setTimeout(r, latency));
      if (failOn && inputs.pupil === failOn) throw new Error('mock failure');
      return 'result-for-' + inputs.pupil;
    },
    commit: (result, inputs) => committed.push({ result, pupil: inputs.pupil }),
    onError: () => {},
  });
  return {
    select(p) { current = p; q.schedule(); },
    get last() { return committed.length ? committed[committed.length - 1] : null; },
    get committed() { return committed; },
    get predictions() { return predictions; },
    get current() { return current; },
    q,
  };
}
const settle = ms => new Promise(r => setTimeout(r, ms));

{ // change during inference -> the later pupil wins
  const h = harness();
  h.select('A');
  await settle(60);                 // A is mid-flight
  h.select('B');
  await settle(600);
  check('change during inference -> latest pupil committed',
    h.last && h.last.pupil === 'B', `committed ${h.last && h.last.pupil}`);
  check('the superseded result is never committed',
    !h.committed.some(c => c.pupil === 'A'),
    h.committed.map(c => c.pupil).join(',') || 'nothing');
}

{ // the redundancy the reviewer found: absorbing a request must not cause a second run
  const h = harness();
  h.select('A');
  await settle(60);
  h.select('B');                    // absorbed by the in-flight attempt's retry
  await settle(800);
  check('a superseded request does not trigger a redundant prediction',
    h.predictions === 2, `${h.predictions} predictions (A, then the B retry)`);
}

{ // rapid changes -> exactly one commit, for the last pupil
  const h = harness();
  for (const p of ['A', 'B', 'C', 'D']) { h.select(p); await settle(8); }
  await settle(900);
  check('rapid successive changes -> last one wins',
    h.last && h.last.pupil === 'D', `committed ${h.committed.map(c => c.pupil).join(',')}`);
}

{ // a failing predictor must not spin
  const h = harness({ failOn: 'A' });
  h.select('A');
  await settle(600);
  check('a failing prediction stops instead of looping',
    h.predictions === 1 && !h.q.busy, `${h.predictions} prediction(s), busy=${h.q.busy}`);
}

{ /* A request landing DURING commit -- after the version check passed, before busy
     clears -- cannot be absorbed by the retry loop, so only the re-run in finally can
     serve it. Mutation testing found this path untested: deleting that re-run left every
     other check passing. */
  let fired = false;
  let current = 'A';
  const committed = [];
  const q = createScheduler({
    debounceMs: 20,
    gather: () => ({ pupil: current }),
    predict: async () => { await new Promise(r => setTimeout(r, 60)); return 'r'; },
    commit: (_r, inputs) => {
      committed.push(inputs.pupil);
      if (!fired) { fired = true; current = 'C'; q.schedule(); }   // lands mid-commit
    },
    onError: () => {},
  });
  q.schedule();
  await settle(600);
  check('a request arriving during commit is still served',
    committed[committed.length - 1] === 'C', `committed ${committed.join(',')}`);
}

{ /* Timing mirror of the redundancy test above: there the predictor outlives the
     debounce, so the pending timer fires while busy and no-ops. Here the debounce is long
     and the predictor fast, so the RETRY settles first and the timer is still pending
     afterwards. A timer that checks only !busy then starts a second prediction for inputs
     that were just committed. */
  let current = 'A';
  const committed = [];
  let predictions = 0;
  const q = createScheduler({
    debounceMs: 200,
    gather: () => ({ pupil: current }),
    predict: async () => { predictions++; await new Promise(r => setTimeout(r, 20)); return 'r'; },
    commit: (_r, inputs) => committed.push(inputs.pupil),
    onError: () => {},
  });
  q.schedule();
  await settle(205);              // the attempt is running, predictor takes 20 ms
  current = 'B';
  q.schedule();                   // absorbed by the retry; its timer is still pending
  await settle(80);
  const afterRetry = predictions;
  await settle(400);              // let the stale timer fire
  check('a stale debounce timer does not re-predict settled inputs',
    predictions === afterRetry && committed.join(',') === 'B',
    `${afterRetry} -> ${predictions} predictions, committed ${committed.join(',')}`);
}

{ /* The one window only the re-run in finally covers, found by mutation testing:
     a change that lands DURING commit (so the retry loop has already passed its version
     check) whose debounce timer is then cancelled by flush(). Nothing else is left to
     serve it. Every simpler "changed while busy" case is absorbed by the retry loop or
     picked up by the pending timer, which is why removing the re-run passed all of them. */
  let current = 'A';
  const committed = [];
  const q = createScheduler({
    debounceMs: 20,
    gather: () => ({ pupil: current }),
    predict: async () => { await new Promise(r => setTimeout(r, 80)); return 'r'; },
    commit: (_r, inputs) => {
      committed.push(inputs.pupil);
      if (committed.length === 1) { current = 'Z'; q.schedule(); q.flush(); }
    },
    onError: () => {},
  });
  q.schedule();
  await settle(600);
  check('a change during commit with its timer cancelled is still served',
    committed[committed.length - 1] === 'Z', `committed ${committed.join(',')}`);
}

{ // a change arriving AFTER a commit must still run
  const h = harness();
  h.select('A');
  await settle(400);                // A committed
  h.select('B');
  await settle(400);
  check('a change after completion still runs',
    h.last && h.last.pupil === 'B', `committed ${h.committed.map(c => c.pupil).join(',')}`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL CHECKS PASSED');
process.exit(failures ? 1 : 0);
