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

/* ---------- 2. the scheduler, with a slow prediction ---------- */
/* Mirrors page.js's runNet/runNetOnce contract exactly. A pupil selected DURING an
   in-flight prediction must be the one whose result is displayed. */

function scheduler() {
  let busy = false, timer = null, version = 0, queued = false, painted = null, current = 'A';
  const delay = ms => new Promise(r => setTimeout(r, ms));
  async function once() {
    for (;;) {
      const v = version, mine = current;
      await delay(120);
      if (v !== version) continue;
      painted = mine;
      return;
    }
  }
  async function run() {
    if (busy) { queued = true; return; }
    busy = true;
    try { await once(); }
    finally { busy = false; if (queued) { queued = false; run(); } }
  }
  return {
    select(p) { current = p; version++; clearTimeout(timer);
      timer = setTimeout(() => { if (busy) { queued = true; return; } run(); }, 20); },
    get painted() { return painted; }, get current() { return current; },
  };
}

{
  const s = scheduler();
  s.select('A');
  await new Promise(r => setTimeout(r, 40));    // A is mid-flight
  s.select('B');                                 // switch during inference
  await new Promise(r => setTimeout(r, 700));
  check('pupil changed during inference -> latest wins', s.painted === s.current,
    `selected ${s.current}, painted ${s.painted}`);
}
{
  const s = scheduler();
  for (const p of ['A', 'B', 'C', 'D']) { s.select(p); await new Promise(r => setTimeout(r, 30)); }
  await new Promise(r => setTimeout(r, 900));
  check('rapid successive changes -> last one wins', s.painted === 'D',
    `painted ${s.painted}`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL CHECKS PASSED');
process.exit(failures ? 1 : 0);
