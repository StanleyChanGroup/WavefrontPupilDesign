/**
 * scene.js -- three procedural scenes + FFT convolution with a PSF.
 *
 * This reproduces, live, the interaction shown in the paper's Fig. 12
 * (figures/correction_results.png): a test scene convolved with the aberrated PSF,
 * and with the residual PSF after correction. Every scene is generated procedurally,
 * in code, so that no copyrighted scan has to be shipped and no image licence is
 * involved anywhere in the page.
 *
 * ------------------------------------------------------------------------------
 * THE SCENE CATALOGUE -- see SCENES at the bottom of the file
 * ------------------------------------------------------------------------------
 *   usaf     usafTarget()   a USAF-1951-style resolution chart: calibrated bar
 *                           groups and Siemens stars, so the surviving resolution
 *                           can be READ OFF the picture. Documented immediately below.
 *   stars    starField()    the canonical adaptive-optics scene and one of the
 *                           paper's stated applications: a bright star, a binary
 *                           at 0.95 x the Rayleigh limit that merges under
 *                           aberration and splits again after correction, 6.5
 *                           magnitudes of field stars, and a galaxy-like smudge.
 *   terrain  terrain()      an extended natural surface: a hill-shaded, seamless
 *                           crater and ridge field built from self-implemented
 *                           gradient noise, with structure at every scale from the
 *                           frame down to the diffraction limit.
 *
 * SCENES is an ordered array of {id, label, blurb, make(size), display, spec(size)},
 * so a picker can be built without hard-coding any of the three. All three return a
 * Float64Array of length size*size with values in [0, 1]; all three are deterministic
 * (the two seeded ones take opts.seed); all three are verified in
 * test/scenes_verify.mjs and test/SCENE_VERIFICATION.md.
 *
 * Everything is float64 (Number / Float64Array). Images are row-major, flat index
 * p = y*S + x, y = row (0 = top), x = column (0 = left) -- the same convention as
 * src/optics.js.
 *
 * The only dependency is src/optics.js, and only for its radix-2 FFT (fft2 / ifft2),
 * which is exact at any power of two. No other imports, no packages: usable in Node
 * (ESM) and in the browser.
 *
 * ------------------------------------------------------------------------------
 * SCENE 1 -- USAF CHART LAYOUT (see usafBlocks() for the machine-readable table)
 * ------------------------------------------------------------------------------
 * All geometry is authored in "chart units" on a 256 x 256 canvas and scaled by
 * size/256, so a 512 x 512 target is the identical drawing at 2 px per chart unit.
 *
 *   +---------------------------------------------------------+  mid-grey field, 0.5
 *   | ###  TOP BAND (black)  white 3-bar clusters + numerals ###|
 *   |  +----+                                        +----+     |
 *   |  |bar |        CENTRAL CHECKERBOARD SQUARE      |bar |     |
 *   |  |lad |   4 quadrants (black/white) + a 12-spoke|lad |     |
 *   |  |der |   Siemens star + 4 small corner stars   |der |     |
 *   |  +----+                                        +----+     |
 *   | ###  BOTTOM BAND (black)  same, rotated 180 deg         ###|
 *   +---------------------------------------------------------+
 *
 * Bar groups follow the real USAF-1951 frequency law
 *     pitch(group, element) = P0 * 2^-(group + (element-1)/6),   P0 = 32 chart units,
 * so the numerals painted next to each cluster are the honest group/element numbers
 * for that pitch. usafBlocks(size) returns the exact pitch of every drawn block in
 * IMAGE PIXELS, so a panel can state which groups survive a given blur.
 *
 * Pitch ladder actually drawn (chart units == pixels at size 256, doubled at 512):
 *
 *   g,e   pitch px   bar width px   cycles/px   where
 *   0,6    17.959       8.980         0.0557    left strip   (V)
 *   1,1    16.000       8.000         0.0625    right strip  (H)
 *   1,3    12.699       6.350         0.0787    left strip   (H)
 *   1,4    11.314       5.657         0.0884    top band     (V+H)
 *   1,5    10.079       5.040         0.0992    right strip  (V)
 *   1,6     8.980       4.490         0.1114    bottom band  (V+H)
 *   2,1     8.000       4.000         0.1250    left strip   (V)
 *   2,2     7.127       3.564         0.1403    top band     (V+H)
 *   2,3     6.350       3.175         0.1575    right strip  (H)
 *   2,4     5.657       2.828         0.1768    bottom band  (V+H)
 *   2,6     4.490       2.245         0.2227    bottom band  (V+H)
 *   3,1     4.000       2.000         0.2500    top band     (V+H)
 *   3,2     3.564       1.782         0.2806    bottom band  (V+H)
 *   3,4     2.828       1.414         0.3536    top band     (V+H)
 *   4,1     2.000       1.000         0.5000    bottom band  (V+H)  == Nyquist
 *
 * "V" = vertical bars (the pattern is modulated along x, so it probes HORIZONTAL
 * resolution); "H" = horizontal bars (modulated along y). Each bar block is the
 * standard 3 bars of width w separated by gaps of w, 5w long -- a 5w x 5w square
 * with w = pitch/2.
 *
 * The Siemens stars carry a CONTINUOUS frequency sweep. With S bright spokes the
 * tangential period at radius r is
 *     T(r) = 2*pi*r / (2*S) = pi*r / S     (S = 12 here, so T(r) = 0.2618 * r),
 * i.e. a blur whose cutoff period is T washes the star out inside
 *     r_blur = S * T / pi.
 * starPeriodAtRadius() / starRadiusForPeriod() expose both directions.
 *
 * ------------------------------------------------------------------------------
 * CONVOLUTION
 * ------------------------------------------------------------------------------
 * convolvePsf() performs a CIRCULAR convolution on the S x S torus via the FFT:
 *
 *   1. k = psf / sum(psf)                       -- unit sum => mean brightness preserved
 *   2. k is zero-padded into S x S with the PSF's CENTRE pixel (M/2, M/2) -- the
 *      fftshift-centred origin that optics.js psfFromPupil() produces -- placed at
 *      array index (0, 0), each PSF pixel (y, x) landing at ((y-M/2) mod S, (x-M/2) mod S).
 *      This is exactly ifftshift for M == S, and the zero-padded generalisation of it
 *      for M < S. Getting this wrong is THE classic bug here: putting the PSF corner
 *      at (0,0) translates the output by -M/2 px, and putting the PSF centre at
 *      (S/2, S/2) translates it by S/2 px.
 *   3. out = Re{ ifft2( fft2(image) * fft2(k) ) }
 *
 * Consequences, all checked in test/scene_verify.mjs:
 *   - a delta PSF (1 at pixel (M/2, M/2)) returns the input bit-for-bit to ~1e-13;
 *   - a delta displaced by (dy, dx) from the centre translates the image by exactly
 *     (+dy, +dx) -- down and right, as the displaced PSF energy demands;
 *   - sum(out) == sum(image) exactly (unit-sum kernel, circular domain);
 *   - a centro-symmetric PSF leaves the centroid unmoved to ~1e-12 px. An aberrated
 *     PSF DOES move it, by precisely the PSF's own centroid offset from (M/2, M/2) --
 *     that is physics (coma shifts the image), not an indexing error.
 *
 * Circular rather than zero-padded-linear is deliberate: the chart's field is 0.5
 * grey all the way to the border, so zero padding would manufacture a dark frame that
 * bleeds inward under a large blur. On the torus the grey field simply wraps into
 * itself. Chart content is inset 4 chart units from every edge, so the wrapped
 * neighbourhood of any border pixel is plain grey.
 */

import { fft2, ifft2 } from './optics.js';

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

/** The chart is authored on this canvas and scaled to the requested size. */
export const CHART_UNITS = 256;

/** Sizes the generator is designed and verified for. Any power of two >= 64 works. */
export const SUPPORTED_SIZES = Object.freeze([256, 512]);

/** Bar pitch, in chart units, of USAF group 0 / element 1 on this chart. */
export const USAF_PITCH_BASE = 32;

/** Number of BRIGHT spokes in every Siemens star (so 2*S = 24 sectors). */
export const STAR_SPOKES = 12;

/** Supersampling factor used while rasterising; the chart is drawn at size*SS and boxed down. */
const SS = 4;

/** The chart's three tones. Indices into PALETTE are what the raster buffer stores. */
const PALETTE = [0.0, 0.5, 1.0];
const BLACK = 0, GREY = 1, WHITE = 2;

/** Field (background) value. */
export const FIELD_VALUE = PALETTE[GREY];

/* ------------------------------------------------------------------ *
 * USAF-1951 frequency law
 * ------------------------------------------------------------------ */

/**
 * Bar pitch (one full black+white period) in CHART UNITS for a USAF-1951
 * group/element, with this chart's scale P0 = 32 units at group 0, element 1:
 *
 *   pitch(g, e) = 32 * 2^-(g + (e-1)/6)
 *
 * which is the standard f = 2^(g + (e-1)/6) line-pairs-per-mm law with 32 units/mm.
 *
 * @param {number} group    USAF group number (0..4 are drawn here)
 * @param {number} element  USAF element number, 1..6
 * @returns {number} pitch in chart units
 */
export function usafPitch(group, element) {
  return USAF_PITCH_BASE * Math.pow(2, -(group + (element - 1) / 6));
}

/** Tangential period (chart units or px, same unit as r) of a star at radius r. */
export function starPeriodAtRadius(r, spokes = STAR_SPOKES) {
  return (Math.PI * r) / spokes;
}

/** Radius at which a star's tangential period equals T -- inside this it is unresolved. */
export function starRadiusForPeriod(T, spokes = STAR_SPOKES) {
  return (spokes * T) / Math.PI;
}

/* ------------------------------------------------------------------ *
 * Layout  (single source of truth for both the rasteriser and usafBlocks)
 * ------------------------------------------------------------------ */

const MARGIN = 4;          // grey border around everything
const BAND_H = 40;         // height of the black top / bottom bands
const BAND_PAD = 3;        // padding inside a band
const SQUARE = 144;        // side of the central checkerboard square
const STRIP_W = 48;        // width of the left / right grey bar-ladder strips
const FRAME_INSET = 5;     // white/black frame inside the central square
const FRAME_T = 2;
const BIG_STAR_R = 52;
const SMALL_STAR_R = 12;
const SMALL_STAR_OFF = 46; // small-star centre offset from the square's centre

// Clusters painted inside the bands: [group, element]. Each draws a vertical-bar
// block, a horizontal-bar block, and the group/element numerals -- the "III =| 6"
// motif of the reference panel.
const TOP_CLUSTERS = [[1, 4], [2, 2], [3, 1], [3, 4]];
const BOT_CLUSTERS = [[1, 6], [2, 4], [2, 6], [3, 2], [4, 1]];

// Single blocks stacked down the grey side strips: [group, element, orientation].
const LEFT_BLOCKS = [[0, 6, 'V'], [1, 3, 'H'], [2, 1, 'V']];
const RIGHT_BLOCKS = [[1, 1, 'H'], [1, 5, 'V'], [2, 3, 'H']];

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Numeral cell size (one font pixel) for a cluster of this pitch. */
const numeralScale = (pitch) => clamp(Math.round(pitch / 4), 2, 3);

/** Block extent (both dimensions) for a 3-bar group of this pitch: 5 bar widths. */
const blockExtent = (pitch) => 2.5 * pitch;

const BAND_X0 = MARGIN;
const BAND_W = CHART_UNITS - 2 * MARGIN;
const TOP_BAND = { x: BAND_X0, y: MARGIN, w: BAND_W, h: BAND_H };
const BOT_BAND = { x: BAND_X0, y: CHART_UNITS - MARGIN - BAND_H, w: BAND_W, h: BAND_H };
const SQ = {
  x: (CHART_UNITS - SQUARE) / 2,
  y: (CHART_UNITS - SQUARE) / 2,
  w: SQUARE,
  h: SQUARE,
  cx: CHART_UNITS / 2,
  cy: CHART_UNITS / 2,
};
const LEFT_STRIP = { x: MARGIN, y: SQ.y, w: STRIP_W, h: SQ.h };
const RIGHT_STRIP = { x: CHART_UNITS - MARGIN - STRIP_W, y: SQ.y, w: STRIP_W, h: SQ.h };

let _layout = null;

/**
 * Build the whole chart's geometry in chart units. Block origins are snapped to
 * integers so that, at size 256 (1 px per chart unit) and 512 (2 px), every bar
 * edge of an integer-width group lands on a pixel boundary. That matters most for
 * the 2.0-px group: half-pixel placement there would average the bars away into
 * flat grey rather than resolve them.
 */
function layout() {
  if (_layout) return _layout;

  const blocks = [];   // bar blocks, in chart units
  const numerals = []; // {digit, x, y, s, rot180}
  let seq = 0;

  const addBlock = (o) => { blocks.push({ id: o.region + (++seq), ...o }); };

  // ---- bands -------------------------------------------------------------
  const band = (bandBox, clusters, region, rot180) => {
    const innerX = bandBox.x + BAND_PAD;
    const innerY = bandBox.y + BAND_PAD;
    const innerW = bandBox.w - 2 * BAND_PAD;
    const innerH = bandBox.h - 2 * BAND_PAD;

    const specs = clusters.map(([g, e]) => {
      const p = usafPitch(g, e);
      const bx = blockExtent(p);
      const s = numeralScale(p);
      const gap = Math.max(2, 0.3 * p);
      return { g, e, p, bx, s, gap, w: bx + gap + bx + gap + 3 * s };
    });
    const used = specs.reduce((a, c) => a + c.w, 0);
    const slack = (innerW - used) / (specs.length - 1);
    if (slack < 0) throw new Error('scene.js: band clusters overflow the band');

    let x = innerX;
    for (const c of specs) {
      const by = innerY + (innerH - c.bx) / 2;
      // vertical bars, then horizontal bars, then the group/element numerals
      addBlock({ region, group: c.g, element: c.e, orient: 'V', pitch: c.p,
        x: Math.round(x), y: Math.round(by), size: c.bx, polarity: 'white-on-black', rot180 });
      const hx = x + c.bx + c.gap;
      addBlock({ region, group: c.g, element: c.e, orient: 'H', pitch: c.p,
        x: Math.round(hx), y: Math.round(by), size: c.bx, polarity: 'white-on-black', rot180 });
      const nx = hx + c.bx + c.gap;
      const stackH = 11 * c.s;
      const ny = innerY + (innerH - stackH) / 2;
      numerals.push({ digit: c.g, x: Math.round(nx), y: Math.round(ny), s: c.s, rot180 });
      numerals.push({ digit: c.e, x: Math.round(nx), y: Math.round(ny + 6 * c.s), s: c.s, rot180 });
      x += c.w + slack;
    }
  };
  band(TOP_BAND, TOP_CLUSTERS, 'T', false);
  band(BOT_BAND, BOT_CLUSTERS, 'B', true);

  // ---- side strips -------------------------------------------------------
  const strip = (box, list, region) => {
    const specs = list.map(([g, e, orient]) => {
      const p = usafPitch(g, e);
      return { g, e, orient, p, bx: blockExtent(p) };
    });
    const used = specs.reduce((a, c) => a + c.bx, 0);
    const slack = (box.h - used) / (specs.length + 1);
    if (slack < 0) throw new Error('scene.js: strip blocks overflow the strip');
    let y = box.y + slack;
    for (const c of specs) {
      addBlock({ region, group: c.g, element: c.e, orient: c.orient, pitch: c.p,
        x: Math.round(box.x + (box.w - c.bx) / 2), y: Math.round(y), size: c.bx,
        polarity: 'white-on-grey', rot180: false });
      y += c.bx + slack;
    }
  };
  strip(LEFT_STRIP, LEFT_BLOCKS, 'L');
  strip(RIGHT_STRIP, RIGHT_BLOCKS, 'R');

  // ---- stars -------------------------------------------------------------
  const stars = [
    { id: 'star-centre', cx: SQ.cx, cy: SQ.cy, r: BIG_STAR_R, spokes: STAR_SPOKES },
    { id: 'star-tl', cx: SQ.cx - SMALL_STAR_OFF, cy: SQ.cy - SMALL_STAR_OFF, r: SMALL_STAR_R, spokes: STAR_SPOKES },
    { id: 'star-tr', cx: SQ.cx + SMALL_STAR_OFF, cy: SQ.cy - SMALL_STAR_OFF, r: SMALL_STAR_R, spokes: STAR_SPOKES },
    { id: 'star-bl', cx: SQ.cx - SMALL_STAR_OFF, cy: SQ.cy + SMALL_STAR_OFF, r: SMALL_STAR_R, spokes: STAR_SPOKES },
    { id: 'star-br', cx: SQ.cx + SMALL_STAR_OFF, cy: SQ.cy + SMALL_STAR_OFF, r: SMALL_STAR_R, spokes: STAR_SPOKES },
  ];

  _layout = { blocks, numerals, stars, square: SQ, topBand: TOP_BAND, botBand: BOT_BAND,
    leftStrip: LEFT_STRIP, rightStrip: RIGHT_STRIP };
  return _layout;
}

/* ------------------------------------------------------------------ *
 * 3x5 digit font
 * ------------------------------------------------------------------ */

// Row bitmaps, most significant of 3 bits = leftmost column.
const DIGITS = [
  [0b111, 0b101, 0b101, 0b101, 0b111], // 0
  [0b010, 0b110, 0b010, 0b010, 0b111], // 1
  [0b111, 0b001, 0b111, 0b100, 0b111], // 2
  [0b111, 0b001, 0b111, 0b001, 0b111], // 3
  [0b101, 0b101, 0b111, 0b001, 0b001], // 4
  [0b111, 0b100, 0b111, 0b001, 0b111], // 5
  [0b111, 0b100, 0b111, 0b101, 0b111], // 6
  [0b111, 0b001, 0b010, 0b010, 0b010], // 7
  [0b111, 0b101, 0b111, 0b101, 0b111], // 8
  [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

/* ------------------------------------------------------------------ *
 * Rasteriser
 * ------------------------------------------------------------------ */

/**
 * A painter over a device buffer of D x D palette indices, where one chart unit is
 * `f` device pixels. `rot180` rotates subsequent drawing 180 degrees about a box --
 * that is how the bottom band's upside-down cluster (as on the reference panel) is
 * produced without a second set of coordinates.
 */
function makePainter(D, f) {
  const buf = new Uint8Array(D * D);
  let tf = null; // {x0, y0, x1, y1} in chart units

  const mapX = (x, w) => (tf ? tf.x0 + tf.x1 - (x + w) : x);
  const mapY = (y, h) => (tf ? tf.y0 + tf.y1 - (y + h) : y);

  /** Axis-aligned rectangle in chart units; `fill` is a palette index or (x,y)->index. */
  function rect(x, y, w, h, fill) {
    const X = mapX(x, w), Y = mapY(y, h);
    const i0 = clamp(Math.round(Y * f), 0, D), i1 = clamp(Math.round((Y + h) * f), 0, D);
    const j0 = clamp(Math.round(X * f), 0, D), j1 = clamp(Math.round((X + w) * f), 0, D);
    if (typeof fill === 'number') {
      for (let i = i0; i < i1; i++) buf.fill(fill, i * D + j0, i * D + j1);
    } else {
      for (let i = i0; i < i1; i++) {
        const yc = (i + 0.5) / f;
        for (let j = j0; j < j1; j++) buf[i * D + j] = fill((j + 0.5) / f, yc);
      }
    }
  }

  /** Filled disc in chart units; `fill` is (x,y)->index, evaluated per device pixel. */
  function disc(cx, cy, r, fill) {
    const i0 = clamp(Math.floor((cy - r) * f), 0, D), i1 = clamp(Math.ceil((cy + r) * f), 0, D);
    const j0 = clamp(Math.floor((cx - r) * f), 0, D), j1 = clamp(Math.ceil((cx + r) * f), 0, D);
    const r2 = r * r;
    for (let i = i0; i < i1; i++) {
      const yc = (i + 0.5) / f, dy = yc - cy;
      for (let j = j0; j < j1; j++) {
        const xc = (j + 0.5) / f, dx = xc - cx;
        if (dx * dx + dy * dy <= r2) buf[i * D + j] = fill(xc, yc, dx, dy);
      }
    }
  }

  /** One 3x5 digit whose font pixel is `s` chart units; rotated with the transform. */
  function digit(d, x, y, s, fill) {
    const rows = DIGITS[d];
    for (let r = 0; r < 5; r++) {
      const bits = rows[tf ? 4 - r : r];
      for (let c = 0; c < 3; c++) {
        const bit = (bits >> (tf ? c : 2 - c)) & 1;
        if (bit) rect(x + c * s, y + r * s, s, s, fill);
      }
    }
  }

  return {
    buf,
    rect,
    disc,
    digit,
    setRot180(box) { tf = box; },
    clearRot180() { tf = null; },
  };
}

/** Which quadrant tone a point of the central square carries. */
function quadrantTone(x, y) {
  return ((x < SQ.cx) === (y < SQ.cy)) ? BLACK : WHITE;
}
const invertTone = (t) => (t === WHITE ? BLACK : WHITE);

/** Draw one 3-bar group: 3 bars of width pitch/2 separated by gaps of pitch/2, 5 widths long. */
function drawBars(p, x, y, pitch, orient, fill) {
  const w = pitch / 2;
  const L = 5 * w;
  for (let k = 0; k < 3; k++) {
    if (orient === 'V') p.rect(x + 2 * k * w, y, w, L, fill);
    else p.rect(x, y + 2 * k * w, L, w, fill);
  }
}

const _targetCache = new Map();

/**
 * Procedural USAF-1951-style resolution target.
 *
 * Rasterised at `SS`x supersampling and box-filtered down, so straight bar edges that
 * fall on pixel boundaries stay perfectly binary while the star wedges and disc rims
 * get the soft edge a real printed chart has. Values are exactly in [0, 1]; the field
 * is exactly 0.5.
 *
 * @param {number} [size=256]  image side; a power of two, 256 or 512 recommended
 * @param {object} [opts]
 * @param {boolean} [opts.shared=false]  return the cached instance instead of a copy
 *        (faster, but the caller must then treat the array as read-only)
 * @returns {Float64Array} length size*size, row-major, values in [0, 1]
 */
export function usafTarget(size = 256, opts = {}) {
  if (!Number.isInteger(size) || size < 64 || (size & (size - 1)) !== 0) {
    throw new Error(`usafTarget: size ${size} must be a power of two >= 64`);
  }
  let master = _targetCache.get(size);
  if (!master) {
    master = renderTarget(size);
    _targetCache.set(size, master);
  }
  return opts.shared ? master : Float64Array.from(master);
}

function renderTarget(size) {
  const f = (SS * size) / CHART_UNITS; // device pixels per chart unit
  const D = size * SS;
  const L = layout();
  const p = makePainter(D, f);

  // 1. mid-grey field
  p.buf.fill(GREY);

  // 2. the two black bands
  p.rect(TOP_BAND.x, TOP_BAND.y, TOP_BAND.w, TOP_BAND.h, BLACK);
  p.rect(BOT_BAND.x, BOT_BAND.y, BOT_BAND.w, BOT_BAND.h, BLACK);

  // 3. the central square's four quadrants
  const h = SQUARE / 2;
  p.rect(SQ.x, SQ.y, h, h, BLACK);           // top-left
  p.rect(SQ.cx, SQ.y, h, h, WHITE);          // top-right
  p.rect(SQ.x, SQ.cy, h, h, WHITE);          // bottom-left
  p.rect(SQ.cx, SQ.cy, h, h, BLACK);         // bottom-right

  // 4. the frame inside the square, drawn in the opposite tone to whatever it crosses
  const fx = SQ.x + FRAME_INSET, fy = SQ.y + FRAME_INSET;
  const fw = SQUARE - 2 * FRAME_INSET;
  const inv = (x, y) => invertTone(quadrantTone(x, y));
  p.rect(fx, fy, fw, FRAME_T, inv);
  p.rect(fx, fy + fw - FRAME_T, fw, FRAME_T, inv);
  p.rect(fx, fy, FRAME_T, fw, inv);
  p.rect(fx + fw - FRAME_T, fy, FRAME_T, fw, inv);

  // 5. the Siemens stars. Inside the disc a spoke inverts the local quadrant tone and
  //    a gap keeps it, which is what gives the reference panel its pinwheel look.
  for (const s of L.stars) {
    const sect = Math.PI / s.spokes;      // one bright + one dark sector spans 2*sect
    const phase = sect / 2;               // centre a spoke on the +x axis
    p.disc(s.cx, s.cy, s.r, (xc, yc, dx, dy) => {
      const bg = quadrantTone(xc, yc);
      const k = Math.floor((Math.atan2(dy, dx) + Math.PI + phase) / sect);
      return (k & 1) === 0 ? invertTone(bg) : bg;
    });
  }

  // 6. bar blocks
  for (const b of L.blocks) {
    if (b.rot180) p.setRot180(BOT_BAND); else p.clearRot180();
    drawBars(p, b.x, b.y, b.pitch, b.orient, WHITE);
  }
  p.clearRot180();

  // 7. group / element numerals
  for (const n of L.numerals) {
    if (n.rot180) p.setRot180(BOT_BAND); else p.clearRot180();
    p.digit(n.digit, n.x, n.y, n.s, WHITE);
  }
  p.clearRot180();

  return downsample(p.buf, D, size, SS);
}

function downsample(buf, D, S, ss) {
  const out = new Float64Array(S * S);
  const inv = 1 / (ss * ss);
  for (let y = 0; y < S; y++) {
    const base = y * ss;
    for (let x = 0; x < S; x++) {
      const col = x * ss;
      let acc = 0;
      for (let j = 0; j < ss; j++) {
        const row = (base + j) * D + col;
        for (let i = 0; i < ss; i++) acc += PALETTE[buf[row + i]];
      }
      out[y * S + x] = acc * inv;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * The documented bar-group table
 * ------------------------------------------------------------------ */

/**
 * Every 3-bar block actually drawn on the chart, with its honest frequency content,
 * scaled to an image of `size` px. The panel can use `x/y/w/h` to crop a block and
 * `pitchPx` to state which groups survive a given blur.
 *
 * Fields:
 *   id             stable identifier, e.g. "T3"
 *   region         'T' top band, 'B' bottom band, 'L'/'R' side strips
 *   group, element USAF-1951 group and element numbers (painted on the chart for bands)
 *   orient         'V' vertical bars (probes horizontal resolution) | 'H'
 *   pitchPx        one black+white period, in image pixels
 *   barWidthPx     pitchPx / 2
 *   cyclesPerPx    1 / pitchPx   (0.5 == Nyquist)
 *   cyclesPerImage size / pitchPx
 *   x, y, w, h     bounding box of the block in image pixels (w == h == 2.5*pitchPx)
 *   polarity       'white-on-black' (bands) | 'white-on-grey' (side strips)
 *   rot180         true for the bottom band, which is painted upside down
 *
 * @param {number} [size=256]
 * @returns {Array<object>} sorted by descending pitch (coarsest group first)
 */
export function usafBlocks(size = 256) {
  const k = size / CHART_UNITS;
  const L = layout();
  return L.blocks.map((b) => {
    const pitchPx = b.pitch * k;
    let x = b.x, y = b.y;
    if (b.rot180) { // rotate the stored top-left corner about the bottom band
      x = BOT_BAND.x + BOT_BAND.x + BOT_BAND.w - (b.x + b.size);
      y = BOT_BAND.y + BOT_BAND.y + BOT_BAND.h - (b.y + b.size);
    }
    return {
      id: b.id,
      region: b.region,
      group: b.group,
      element: b.element,
      orient: b.orient,
      pitchPx,
      barWidthPx: pitchPx / 2,
      cyclesPerPx: 1 / pitchPx,
      cyclesPerImage: size / pitchPx,
      x: x * k,
      y: y * k,
      w: b.size * k,
      h: b.size * k,
      polarity: b.polarity,
      rot180: b.rot180,
    };
  }).sort((a, b) => b.pitchPx - a.pitchPx);
}

/**
 * The Siemens stars, scaled to an image of `size` px, with the radius/period map.
 *
 * @returns {Array<{id, cx, cy, rPx, spokes, periodAtRimPx, periodAtRadius, radiusForPeriod}>}
 */
export function siemensStars(size = 256) {
  const k = size / CHART_UNITS;
  return layout().stars.map((s) => ({
    id: s.id,
    cx: s.cx * k,
    cy: s.cy * k,
    rPx: s.r * k,
    spokes: s.spokes,
    periodAtRimPx: starPeriodAtRadius(s.r * k, s.spokes),
    periodAtRadius: (r) => starPeriodAtRadius(r, s.spokes),
    radiusForPeriod: (T) => Math.min(s.r * k, starRadiusForPeriod(T, s.spokes)),
  }));
}

/* ------------------------------------------------------------------ *
 * FFT convolution
 * ------------------------------------------------------------------ */

const _convScratch = new Map();
function convScratch(S) {
  let s = _convScratch.get(S);
  if (!s) {
    const n2 = S * S;
    s = {
      ire: new Float64Array(n2), iim: new Float64Array(n2),
      kre: new Float64Array(n2), kim: new Float64Array(n2),
      rows: new Uint8Array(S),
    };
    _convScratch.set(S, s);
  }
  return s;
}

/**
 * Convolve an image with a PSF, on the S x S torus, via the FFT.
 *
 * The PSF is expected in the fftshift-centred layout that optics.js psfFromPupil() /
 * simulateMeasurement() return: its origin -- the pixel a point source images to when
 * the wavefront is flat -- is (M/2, M/2), M being the PSF's side. It is normalised to
 * unit sum, so the convolution preserves mean brightness exactly, and zero-padded into
 * S x S with that origin at index (0, 0), so the output is NOT translated relative to
 * the input. See the module header for the shift conventions this pins down.
 *
 * @param {ArrayLike<number>} image  length size*size
 * @param {ArrayLike<number>} psf    length M*M, M even and <= size
 * @param {number} [size=256]        image side, a power of two
 * @param {object} [opts]
 * @param {boolean} [opts.normalize=true]  divide the PSF by its sum
 * @param {number}  [opts.psfOrigin]       PSF origin along both axes (default M/2)
 * @param {Float64Array} [opts.out]        destination, length size*size
 * @returns {Float64Array} the convolved image, length size*size
 */
export function convolvePsf(image, psf, size = 256, opts = {}) {
  const S = size;
  if (!Number.isInteger(S) || S < 2 || (S & (S - 1)) !== 0) {
    throw new Error(`convolvePsf: size ${S} must be a power of two`);
  }
  const n2 = S * S;
  if (image.length !== n2) {
    throw new Error(`convolvePsf: image has ${image.length} samples, expected ${n2}`);
  }
  const M = Math.round(Math.sqrt(psf.length));
  if (M * M !== psf.length) throw new Error('convolvePsf: psf is not square');
  if (M > S) throw new Error(`convolvePsf: psf ${M}x${M} is larger than the image ${S}x${S}`);
  const { normalize = true, out = null } = opts;
  const origin = opts.psfOrigin === undefined ? M >> 1 : opts.psfOrigin;

  const sc = convScratch(S);
  const { ire, iim, kre, kim, rows } = sc;

  // --- kernel: normalise, zero-pad, and put its origin at index (0, 0) ---
  let sum = 0;
  if (normalize) {
    for (let i = 0; i < psf.length; i++) sum += psf[i];
    if (!(Math.abs(sum) > 0)) throw new Error('convolvePsf: psf sums to zero, cannot normalise');
  } else {
    sum = 1;
  }
  const scale = 1 / sum;
  kre.fill(0); kim.fill(0); rows.fill(0);
  for (let y = 0; y < M; y++) {
    const ty = (((y - origin) % S) + S) % S;
    const src = y * M;
    const dst = ty * S;
    let any = false;
    for (let x = 0; x < M; x++) {
      const v = psf[src + x];
      if (v === 0) continue;
      any = true;
      kre[dst + (((x - origin) % S) + S) % S] = v * scale;
    }
    if (any) rows[ty] = 1;
  }

  // --- forward transforms ---
  ire.set(image); iim.fill(0);
  fft2(ire, iim, S, false);
  fft2(kre, kim, S, false, rows); // rows outside the padded PSF are exactly zero

  // --- spectral product ---
  for (let p = 0; p < n2; p++) {
    const ar = ire[p], ai = iim[p], br = kre[p], bi = kim[p];
    ire[p] = ar * br - ai * bi;
    iim[p] = ar * bi + ai * br;
  }

  // --- inverse; the imaginary part is numerical dust and is discarded ---
  ifft2(ire, iim, S);
  const res = out || new Float64Array(n2);
  res.set(ire);
  return res;
}

/* ------------------------------------------------------------------ *
 * Metrics
 * ------------------------------------------------------------------ */

/** Mean squared error between two equal-length images. */
export function mse(a, b) {
  if (a.length !== b.length) throw new Error(`mse: length ${a.length} != ${b.length}`);
  let se = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; se += d * d; }
  return se / a.length;
}

/**
 * Peak signal-to-noise ratio in dB, the metric the paper's Fig. 12 reports.
 *
 *   PSNR = 10 * log10(peak^2 / MSE)
 *
 * `peak` defaults to 1.0, which is the correct data range for the [0, 1] images this
 * module produces (skimage's peak_signal_noise_ratio with data_range=1.0). Identical
 * images give Infinity.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @param {number} [peak=1]
 * @returns {number} dB
 */
/**
 * Structural similarity, the standard Wang et al. 2004 formulation:
 * 11x11 Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03, and the mean over all
 * fully-covered windows. Matches skimage.metrics.structural_similarity with
 * gaussian_weights=True, sigma=1.5, use_sample_covariance=False.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @param {number} size   side length; both images must be size*size
 * @param {number} [L=1]  dynamic range of the data
 */
export function ssim(a, b, size, L = 1) {
  const R = 5, W = 2 * R + 1, sigma = 1.5;
  const k = new Float64Array(W);
  let ks = 0;
  for (let i = 0; i < W; i++) { k[i] = Math.exp(-((i - R) ** 2) / (2 * sigma * sigma)); ks += k[i]; }
  for (let i = 0; i < W; i++) k[i] /= ks;

  const n = size * size;
  const prod = new Float64Array(n), aa = new Float64Array(n), bb = new Float64Array(n);
  for (let p = 0; p < n; p++) { prod[p] = a[p] * b[p]; aa[p] = a[p] * a[p]; bb[p] = b[p] * b[p]; }

  // separable Gaussian, valid region only (no padding -- padding biases the edges)
  const blur = (src) => {
    const tmp = new Float64Array(n), out = new Float64Array(n);
    for (let y = 0; y < size; y++)
      for (let x = R; x < size - R; x++) {
        let acc = 0;
        for (let i = -R; i <= R; i++) acc += k[i + R] * src[y * size + x + i];
        tmp[y * size + x] = acc;
      }
    for (let y = R; y < size - R; y++)
      for (let x = R; x < size - R; x++) {
        let acc = 0;
        for (let i = -R; i <= R; i++) acc += k[i + R] * tmp[(y + i) * size + x];
        out[y * size + x] = acc;
      }
    return out;
  };

  const mu1 = blur(a), mu2 = blur(b);
  const s11 = blur(aa), s22 = blur(bb), s12 = blur(prod);
  const C1 = (0.01 * L) ** 2, C2 = (0.03 * L) ** 2;
  let sum = 0, count = 0;
  for (let y = R; y < size - R; y++)
    for (let x = R; x < size - R; x++) {
      const p = y * size + x;
      const m1 = mu1[p], m2 = mu2[p];
      const v1 = s11[p] - m1 * m1, v2 = s22[p] - m2 * m2, cv = s12[p] - m1 * m2;
      sum += ((2 * m1 * m2 + C1) * (2 * cv + C2)) / ((m1 * m1 + m2 * m2 + C1) * (v1 + v2 + C2));
      count++;
    }
  return count ? sum / count : 1;
}

export function psnr(a, b, peak = 1) {
  const m = mse(a, b);
  if (m === 0) return Infinity;
  return 10 * Math.log10((peak * peak) / m);
}

/* ================================================================== *
 *
 *  SCENE 2 -- STAR FIELD          (adaptive optics on a night sky)
 *
 * ------------------------------------------------------------------
 * This is the canonical AO scene and one of the paper's stated
 * applications. It is built so that the merge/split of a close binary is
 * the visible headline of the panel:
 *
 *   - ONE BRIGHT ISOLATED STAR (mag 0), far from everything else, so the
 *     PSF's own shape -- Airy rings when sharp, speckle when aberrated --
 *     is readable on a single source.
 *   - A CLOSE BINARY, two equal mag-1.6 components separated by
 *     0.95 x the Rayleigh limit of the reference pupil, i.e.
 *
 *         Rayleigh   = (j11/pi) * N / (2*radius)
 *                    = 1.2196699 * 128 / 38 = 4.10836 px
 *         separation = 0.95 * 4.10836       = 3.90294 px
 *
 *     (N = 128, radius = 19: the GOLDEN.md geometry. Zero-padding that pupil
 *     to 1024 x 1024 samples its PSF at 1/8 px and puts the first dark ring
 *     at 4.125 px, 0.4% from the analytic value -- test/SCENE_VERIFICATION.md
 *     section 0.) That is between the Sparrow limit (~0.77 Rayleigh, where
 *     the dip vanishes even for a perfect pupil) and the Rayleigh limit, so a
 *     diffraction-limited PSF *just* splits the pair and any real aberration
 *     merges it. Measured dip between the two peaks, in the scene as shipped:
 *
 *         diffraction-limited PSF            8.95 %   resolved
 *         2.03 rad RMS aberration, Strehl 9%   --     MERGED, one peak
 *         residual after 88 % correction     7.95 %   resolved again
 *
 *     (10.32 % on an isolated copy of the pair; the sky pedestal under the
 *     shipped scene lifts the trough and takes ~1.4 points off the ratio.)
 *     The merge/split threshold sits near 0.5 of the wavefront left
 *     uncorrected, Strehl ~43 % -- see section 3 of the verification.
 *   - A RANGE OF FAINTER STARS, 6 more named ones and 64 seeded anonymous
 *     ones spanning magnitudes 2.1 to 6.5 -- with the mag-0 star that is a
 *     398:1 range of peak intensity. The image is LINEAR
 *     INTENSITY -- that is what a PSF convolves -- so a display that wants
 *     to show the faint end should tone-map it; asinhStretch() is provided.
 *   - A DIM EXTENDED SMUDGE, an inclined elliptical Gaussian "galaxy" with
 *     a slightly brighter core, peaking at ~4.5% of the bright star, to
 *     carry low-contrast large-scale structure that blur redistributes
 *     rather than destroys.
 *
 * ------------------------------------------------------------------
 * WHY THE POINT SOURCES ARE NOT SINGLE HOT PIXELS
 * ------------------------------------------------------------------
 * A star written as one bright pixel is a delta on the sampling lattice:
 * its spectrum is flat to Nyquist and beyond, it cannot be placed between
 * pixels at all, and everything it carries above the pupil's optical cutoff
 * (f_c = 2*radius/N = 0.297 cycles/px, i.e. periods below 3.37 px) folds
 * back into the passband as aliasing. The convolution is then a function of
 * where the star happened to round to, not of where it is.
 *
 * Every point source here is instead an isotropic Gaussian of
 * sigma = 0.65 px, evaluated at pixel centres about a fractional centre, so
 * the position is sub-pixel exact and the profile rolls off smoothly. The
 * test for "band-limited enough" is shift invariance: imaging a source at
 * x0+t must equal the exact Fourier translation by t of the source imaged at
 * x0. Measured worst case over t in [0,1) (see test/SCENE_VERIFICATION.md):
 *
 *     single hot pixel      25.4 % relative L2 error
 *     Gaussian sigma 0.50    4.2 %
 *     Gaussian sigma 0.65    0.29 %      <- used here
 *     Gaussian sigma 0.80    0.016 %
 *
 * sigma = 0.65 px buys a factor 86 over a hot pixel while still being narrow
 * enough to leave the binary resolved. A fatter source is just extra blur and
 * closes the notch -- the isolated pair's diffraction-limited dip runs
 * 14.06 % at sigma 0.50, 10.32 % at 0.65, 6.93 % at 0.80 and 2.95 % at 1.00 --
 * so 0.65 px is the knee: past it aliasing is already negligible and all that
 * a wider profile buys is a shallower notch.
 *
 * ------------------------------------------------------------------
 * SIZE HANDLING
 * ------------------------------------------------------------------
 * Composition (positions, galaxy axes, star count) is authored in the same
 * CHART_UNITS = 256 canvas the USAF target uses and scaled by size/256, so
 * 512 is the same sky drawn at 2 px per chart unit.
 *
 * The two exceptions are deliberate and are the quantities that only mean
 * anything against the FIXED 128 x 128 PSF grid: the point-source sigma
 * (0.65 px) and the binary separation (3.90294 px). Those are held in
 * ABSOLUTE PIXELS at every size, so the pair sits at 0.95 Rayleigh whether
 * the scene is 256 or 512 and the demo says the same thing either way.
 * ================================================================== */

/** Bounding box of the pupil model these scenes are tuned against (GOLDEN.md). */
export const PUPIL_N = 128;
/** Pupil radius in pixels (GOLDEN.md). */
export const PUPIL_RADIUS = 19;

/** j_{1,1}/pi -- the first zero of 2*J1(x)/x, in units of lambda/D. */
const AIRY_FIRST_ZERO = 1.2196698912665045;

/**
 * Rayleigh resolution limit, in IMAGE PIXELS, of a circular pupil of the given
 * radius inside an N x N bounding box:
 *
 *     lambda/D  = N / (2*radius)            px   (optical cutoff 2*radius/N cyc/px)
 *     Rayleigh  = 1.2196699 * lambda/D      px
 *
 * For the default N = 128, radius = 19 this is 4.10836 px; the first dark ring of
 * psfFromPupil(referenceCircle()), measured at 1/8 px by zero-padding the pupil to
 * 1024 x 1024, lands at 4.125 px -- 0.4 % away.
 */
export function rayleighPx(N = PUPIL_N, radius = PUPIL_RADIUS) {
  return AIRY_FIRST_ZERO * (N / (2 * radius));
}

/** Rayleigh limit for the default pupil: 4.10836 px. */
export const RAYLEIGH_PX = rayleighPx();

/** Optical cutoff frequency of the default pupil, in cycles per image pixel. */
export const PUPIL_CUTOFF_CYC_PX = (2 * PUPIL_RADIUS) / PUPIL_N;

/** Gaussian sigma, in absolute image pixels, of every point source in starField(). */
export const STAR_SIGMA_PX = 0.65;

/**
 * Binary separation as a multiple of the REFERENCE CIRCLE's Rayleigh limit.
 *
 * 0.95 (just inside the circle's limit) is the textbook choice, but it makes the pair
 * unresolvable for most of the demo's pupils for a reason that has nothing to do with
 * wavefront correction: asymmetry is obtained by REMOVING aperture, so those pupils are
 * physically smaller and their diffraction-limited blur is wider. Measured FWHM of the
 * diffraction-limited PSF across the ladder: circle 3.48 px, rising to 7.43 px at the
 * smallest pupil (381 px of area against the circle's 1124) -- 2.1x coarser.
 *
 * 2.80 x Rayleigh = 11.50 px sits clear of the COARSEST pupil in the ladder, so every
 * pupil resolves the pair when its wavefront is flat. Aberration then merges it and
 * correction splits it again, at any point on the slider -- which is the comparison the
 * panel exists to make. The resolution cost of shrinking the pupil is real and is
 * reported separately as throughput; it is deliberately not what this panel tests.
 */
export const BINARY_SEP_RAYLEIGH = 2.80;

/** Binary separation in absolute image pixels: 11.5034 px. */
export const BINARY_SEP_PX = BINARY_SEP_RAYLEIGH * RAYLEIGH_PX;

/** Default seed for the faint-star sprinkle. */
export const STAR_SEED = 8675309;

/** Uniform sky pedestal (linear intensity) under the whole star field. */
export const SKY_LEVEL = 0.0035;

/** Peak of a magnitude-0 point source; leaves headroom for SKY_LEVEL under 1.0. */
const MAG0_PEAK = 0.99;

/** Number of anonymous faint stars sprinkled over the 256-unit canvas. */
const FAINT_STAR_COUNT = 190;

/** Named stars, in chart units. `mag` is an astronomical magnitude: peak = 0.99*10^(-0.4 m). */
const NAMED_STARS = [
  { id: 'A', x: 58.5, y: 63.25, mag: 0.0 },    // the bright isolated one
  { id: 'C', x: 207.75, y: 199.5, mag: 2.1 },
  { id: 'D', x: 121.5, y: 37.75, mag: 2.9 },
  { id: 'E', x: 33.25, y: 152.5, mag: 3.4 },
  { id: 'F', x: 229.5, y: 121.25, mag: 3.9 },
  { id: 'G', x: 155.25, y: 143.75, mag: 3.2 },
  { id: 'H', x: 88.75, y: 111.5, mag: 4.4 },
];

/** The close pair. Centre in chart units; separation and angle are absolute pixels/degrees. */
const BINARY = { id: 'B', x: 176.29, y: 84.61, mag: 1.6, paDeg: 0 };

/** The galaxy-like smudge, in chart units (sigmas) except for its peak intensity. */
const GALAXY = {
  id: 'gal',
  x: 92.0, y: 190.0,
  sigmaMajor: 13.0, sigmaMinor: 6.4, paDeg: 35,
  peak: 0.045,
  corePeak: 0.055, coreSigma: 2.6,
};

/** Magnitude -> peak sample value of an isolated star. */
const magToPeak = (m) => MAG0_PEAK * Math.pow(10, -0.4 * m);

/** mulberry32 -- small, fast, and reproducible in any JS engine. */
function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function assertPow2(fn, size) {
  if (!Number.isInteger(size) || size < 64 || (size & (size - 1)) !== 0) {
    throw new Error(`${fn}: size ${size} must be a power of two >= 64`);
  }
}

/**
 * Add an isotropic Gaussian point source of peak `amp` at the fractional pixel
 * centre (cx, cy), wrapping on the S x S torus. The sample at pixel (x, y) is the
 * profile evaluated at that pixel's CENTRE (x+0.5, y+0.5), which is what makes the
 * position sub-pixel exact rather than snapped to the lattice.
 */
function addPoint(img, S, cx, cy, amp, sigma) {
  const rad = Math.ceil(5 * sigma) + 1;
  const inv = 1 / (2 * sigma * sigma);
  const bx = Math.round(cx - 0.5), by = Math.round(cy - 0.5);
  for (let dy = -rad; dy <= rad; dy++) {
    const y = by + dy;
    const ey = y + 0.5 - cy;
    const row = (((y % S) + S) % S) * S;
    for (let dx = -rad; dx <= rad; dx++) {
      const x = bx + dx;
      const ex = x + 0.5 - cx;
      img[row + (((x % S) + S) % S)] += amp * Math.exp(-(ex * ex + ey * ey) * inv);
    }
  }
}

/** Add an inclined elliptical Gaussian (the galaxy) at (cx, cy); sigmas in pixels. */
function addEllipticalGaussian(img, S, cx, cy, amp, sa, sb, paRad) {
  const c = Math.cos(paRad), s = Math.sin(paRad);
  const rad = Math.ceil(4 * Math.max(sa, sb)) + 1;
  const ia = 1 / (2 * sa * sa), ib = 1 / (2 * sb * sb);
  const bx = Math.round(cx - 0.5), by = Math.round(cy - 0.5);
  for (let dy = -rad; dy <= rad; dy++) {
    const y = by + dy;
    const ey = y + 0.5 - cy;
    const row = (((y % S) + S) % S) * S;
    for (let dx = -rad; dx <= rad; dx++) {
      const x = bx + dx;
      const ex = x + 0.5 - cx;
      const u = ex * c + ey * s;      // along the major axis
      const v = -ex * s + ey * c;     // along the minor axis
      img[row + (((x % S) + S) % S)] += amp * Math.exp(-(u * u * ia + v * v * ib));
    }
  }
}

const _starCache = new Map();
const _starSpecCache = new Map();

/**
 * Build the object list for the star field at `size`, in IMAGE PIXELS.
 *
 * Deterministic: the faint sprinkle comes from mulberry32(seed) drawn in a fixed
 * order, with a rejection rule that keeps faint stars 6 px from each other, 11 px
 * from the named stars and the galaxy core, and 17 px from the binary (so nothing
 * contaminates the dip measurement).
 *
 * @param {number} [size=256]
 * @param {object} [opts]  {seed}
 * @returns {{sizePx, scale, sky, sigmaPx, rayleighPx, cutoffCycPx, stars, binary, galaxy}}
 *   stars[]  {id, kind, x, y, mag, peak}   -- x, y are fractional pixel coordinates
 *   binary   {id, sepPx, sepRayleigh, paDeg, cx, cy, a:{x,y}, b:{x,y}, mag, peak}
 *   galaxy   {id, x, y, sigmaMajorPx, sigmaMinorPx, paDeg, peak, corePeak, coreSigmaPx}
 */
export function starFieldSpec(size = 256, opts = {}) {
  assertPow2('starFieldSpec', size);
  const seed = opts.seed === undefined ? STAR_SEED : opts.seed;
  // Separation is tunable: opts.sepPx wins, else opts.sepRayleigh x the circle's Rayleigh
  // limit, else the module default. It is keyed into the cache so variants do not collide.
  const sepPx = opts.sepPx !== undefined ? opts.sepPx
    : (opts.sepRayleigh !== undefined ? opts.sepRayleigh * RAYLEIGH_PX : BINARY_SEP_PX);
  const key = size + ':' + seed + ':' + sepPx;
  const hit = _starSpecCache.get(key);
  if (hit) return hit;

  const k = size / CHART_UNITS;
  const stars = [];

  for (const s of NAMED_STARS) {
    stars.push({ id: s.id, kind: 'named', x: s.x * k, y: s.y * k,
      mag: s.mag, peak: magToPeak(s.mag) });
  }

  // --- the close pair: centre scales with the canvas, separation does NOT ---
  const th = (BINARY.paDeg * Math.PI) / 180;
  const half = sepPx / 2;
  const bcx = BINARY.x * k, bcy = BINARY.y * k;
  const ax = bcx - Math.cos(th) * half, ay = bcy - Math.sin(th) * half;
  const bx = bcx + Math.cos(th) * half, by = bcy + Math.sin(th) * half;
  const bpeak = magToPeak(BINARY.mag);
  stars.push({ id: 'B1', kind: 'binary', x: ax, y: ay, mag: BINARY.mag, peak: bpeak });
  stars.push({ id: 'B2', kind: 'binary', x: bx, y: by, mag: BINARY.mag, peak: bpeak });
  const binary = {
    id: BINARY.id,
    sepPx,
    sepRayleigh: sepPx / RAYLEIGH_PX,
    rayleighPx: RAYLEIGH_PX,
    paDeg: BINARY.paDeg,
    cx: bcx, cy: bcy,
    a: { x: ax, y: ay },
    b: { x: bx, y: by },
    mag: BINARY.mag,
    peak: bpeak,
  };

  const galaxy = {
    id: GALAXY.id,
    x: GALAXY.x * k, y: GALAXY.y * k,
    sigmaMajorPx: GALAXY.sigmaMajor * k,
    sigmaMinorPx: GALAXY.sigmaMinor * k,
    paDeg: GALAXY.paDeg,
    peak: GALAXY.peak,
    corePeak: GALAXY.corePeak,
    coreSigmaPx: GALAXY.coreSigma * k,
  };

  // --- the faint sprinkle, seeded and rejection-sampled ---
  const rnd = mulberry32(seed);
  const wrapD = (a, b) => { let d = Math.abs(a - b); return d > size / 2 ? size - d : d; };
  const far = (x, y) => {
    for (const s of stars) {
      const lim = s.kind === 'binary' ? 17 : s.kind === 'faint' ? 4.5 : 11;
      if (Math.hypot(wrapD(x, s.x), wrapD(y, s.y)) < lim) return false;
    }
    return Math.hypot(wrapD(x, galaxy.x), wrapD(y, galaxy.y)) >= 11;
  };
  for (let i = 0, guard = 0; i < FAINT_STAR_COUNT && guard < 200 * FAINT_STAR_COUNT; guard++) {
    const x = rnd() * size, y = rnd() * size;
    const mag = 2.8 + 2.9 * Math.sqrt(rnd());   // weighted faint, but bright enough to read
    if (!far(x, y)) continue;
    stars.push({ id: 'f' + i, kind: 'faint', x, y, mag, peak: magToPeak(mag) });
    i++;
  }

  const spec = Object.freeze({
    sizePx: size,
    scale: k,
    sky: SKY_LEVEL,
    sigmaPx: STAR_SIGMA_PX,
    rayleighPx: RAYLEIGH_PX,
    cutoffCycPx: PUPIL_CUTOFF_CYC_PX,
    magRange: [0, 6.5],
    stars,
    binary,
    galaxy,
  });
  _starSpecCache.set(key, spec);
  return spec;
}

/**
 * Procedural star field: 73 point sources in all -- one bright isolated star
 * (mag 0), a close binary at 0.95 x the Rayleigh limit (3.90294 px), 6 more
 * named stars and 64 seeded faint ones spanning magnitudes 2.1 to 6.5 -- plus
 * a dim elliptical galaxy.
 *
 * The result is LINEAR INTENSITY in [0, 1] -- the quantity a PSF actually
 * convolves -- on a uniform sky pedestal of 0.0035. A display wanting the
 * faint end visible should tone-map it; see asinhStretch().
 *
 * @param {number} [size=256]  image side; a power of two, 256 or 512 recommended
 * @param {object} [opts]
 * @param {number}  [opts.seed=STAR_SEED]     seed of the faint sprinkle
 * @param {boolean} [opts.shared=false]       return the cached instance (read-only)
 * @param {string}  [opts.stretch='linear']   'linear' | 'asinh' (a copy either way)
 * @returns {Float64Array} length size*size, row-major, values in [0, 1]
 */
export function starField(size = 256, opts = {}) {
  assertPow2('starField', size);
  const seed = opts.seed === undefined ? STAR_SEED : opts.seed;
  const sepPx = opts.sepPx !== undefined ? opts.sepPx
    : (opts.sepRayleigh !== undefined ? opts.sepRayleigh * RAYLEIGH_PX : BINARY_SEP_PX);
  const key = size + ':' + seed + ':' + sepPx;
  let master = _starCache.get(key);
  if (!master) {
    master = renderStarField(size, { seed, sepPx });
    _starCache.set(key, master);
  }
  const stretch = opts.stretch || 'linear';
  if (stretch === 'asinh') return asinhStretch(master);
  if (stretch !== 'linear') throw new Error(`starField: unknown stretch "${stretch}"`);
  return opts.shared ? master : Float64Array.from(master);
}

function renderStarField(size, opts) {
  const spec = starFieldSpec(size, opts);
  const img = new Float64Array(size * size);
  img.fill(SKY_LEVEL);

  const g = spec.galaxy;
  const pa = (g.paDeg * Math.PI) / 180;
  addEllipticalGaussian(img, size, g.x, g.y, g.peak, g.sigmaMajorPx, g.sigmaMinorPx, pa);
  addEllipticalGaussian(img, size, g.x, g.y, g.corePeak, g.coreSigmaPx, g.coreSigmaPx * 0.8, pa);

  for (const s of spec.stars) addPoint(img, size, s.x, s.y, s.peak, STAR_SIGMA_PX);

  // Overlaps can only add; clamp so the [0, 1] contract holds unconditionally.
  for (let p = 0; p < img.length; p++) if (img[p] > 1) img[p] = 1;
  return img;
}

/**
 * asinh tone map, the standard astronomical stretch: linear near zero, logarithmic
 * far above `soft`, so six magnitudes fit in one display ramp without saturating
 * the bright star.
 *
 *     out = asinh(v / soft) / asinh(1 / soft)
 *
 * Monotone, maps 0 -> 0 and 1 -> 1, so the result is still in [0, 1]. This is a
 * DISPLAY transform: convolve the linear image, then stretch the result.
 *
 * @param {ArrayLike<number>} image
 * @param {object} [opts] {soft=0.0025, out}
 * @returns {Float64Array}
 */
export function asinhStretch(image, opts = {}) {
  const soft = opts.soft === undefined ? 0.0025 : opts.soft;
  const denom = Math.asinh(1 / soft);
  const out = opts.out || new Float64Array(image.length);
  for (let p = 0; p < image.length; p++) {
    const v = image[p] <= 0 ? 0 : image[p];
    out[p] = Math.asinh(v / soft) / denom;
  }
  return out;
}

/**
 * Sample an image along the line joining the two binary components -- the cut a panel
 * should plot to make the merge/split visible, and the measurement behind every dip
 * number quoted above.
 *
 * The scan runs +-`span` separations either side of the pair's midpoint, sampled with
 * a Catmull-Rom interpolator (the imaged field is band-limited well inside Nyquist by
 * the pupil, so interpolation between samples is meaningful). "Resolved" means exactly
 * what Rayleigh means: two distinct interior maxima with a strictly lower minimum
 * between them.
 *
 *     dip = 1 - trough / min(peakA, peakB)
 *
 * @param {ArrayLike<number>} image  an image of this scene -- raw, convolved, whatever
 * @param {number} [size=256]        its side
 * @param {object} [opts] {seed, span=1.5, samples=1201}
 * @returns {{offsets: Float64Array, values: Float64Array, resolved: boolean, dip: number,
 *            peakA: number, peakB: number, trough: number, peakSepPx: number, sepPx: number}}
 *          `offsets` are in pixels from the midpoint, positive towards component B.
 */
export function binaryCut(image, size = 256, opts = {}) {
  assertPow2('binaryCut', size);
  if (image.length !== size * size) {
    throw new Error(`binaryCut: image has ${image.length} samples, expected ${size * size}`);
  }
  const { a, b } = starFieldSpec(size, opts).binary;
  const span = opts.span === undefined ? 1.5 : opts.span;
  const n = opts.samples === undefined ? 1201 : opts.samples;
  const L = Math.hypot(b.x - a.x, b.y - a.y);
  const ux = (b.x - a.x) / L, uy = (b.y - a.y) / L;
  const step = (2 * span * L) / (n - 1);
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const offsets = new Float64Array(n), values = new Float64Array(n);

  const g = (xx, yy) =>
    image[((((yy % size) + size) % size) * size) + (((xx % size) + size) % size)];
  const cr = (p0, p1, p2, p3, t) => 0.5 * ((2 * p1) + (-p0 + p2) * t
    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);

  for (let i = 0; i < n; i++) {
    const off = -span * L + i * step;
    offsets[i] = off;
    const x = mx + ux * off, y = my + uy * off;
    const fxi = Math.floor(x), fyi = Math.floor(y), tx = x - fxi, ty = y - fyi;
    const row = (j) => cr(g(fxi - 1, fyi + j), g(fxi, fyi + j),
      g(fxi + 1, fyi + j), g(fxi + 2, fyi + j), tx);
    values[i] = cr(row(-1), row(0), row(1), row(2), ty);
  }

  const mid = (n - 1) / 2;
  let pa = -Infinity, ia = 0;
  for (let i = 1; i < mid; i++) if (values[i] > pa) { pa = values[i]; ia = i; }
  let pb = -Infinity, ib = 0;
  for (let i = mid + 1; i < n - 1; i++) if (values[i] > pb) { pb = values[i]; ib = i; }
  let tr = Infinity;
  for (let i = ia; i <= ib; i++) if (values[i] < tr) tr = values[i];
  const peak = Math.min(pa, pb);
  const resolved = ia > 1 && ib < n - 2 && ia !== ib && tr < peak * (1 - 1e-12);
  return {
    offsets, values, resolved,
    dip: resolved ? 1 - tr / peak : 0,
    peakA: pa, peakB: pb, trough: tr,
    peakSepPx: (ib - ia) * step,
    sepPx: L,
  };
}

/* ================================================================== *
 *
 *  SCENE 3 -- CRATERED TERRAIN     (an extended natural surface)
 *
 * ------------------------------------------------------------------
 * A hill-shaded lunar-style crater and ridge field: an EXTENDED scene, the
 * opposite limit to the star field's point sources, with real structure at
 * every scale from the whole frame down to the diffraction limit. All of it
 * is generated here -- no image data, no tables.
 *
 * HEIGHT FIELD
 *   h = 0.55 * fBm  +  0.45 * ridged  +  craters
 *
 *   fBm      6 octaves of quintic-interpolated GRADIENT (Perlin) NOISE, periods
 *            128, 64, 32, 16, 8, 4 chart units, amplitude halving each octave.
 *   ridged   5 octaves of 1-|2n-1| squared, periods 64 .. 4 chart units --
 *            this is what makes sharp ridge crests instead of rolling dunes.
 *   warp     both are sampled through a 3-octave domain warp of amplitude
 *            0.09 of the frame, which bends the ridges into meanders instead
 *            of leaving them strung along the lattice axes.
 *   craters  110 of them, radii drawn from an N(>r) ~ r^-2 law between 3.2
 *            and 26 chart units, each a sum of three Gaussians in
 *            d = distance/R:
 *            a bowl -exp(-(d/0.62)^2), a rim +0.85*exp(-((d-1)/0.26)^2) and
 *            an ejecta blanket +0.18*exp(-((d-1.5)/0.6)^2). Gaussians, not
 *            hard edges, so the surface stays smooth and does not alias;
 *            amplitude grows as R^0.35, so big craters are deeper.
 *
 * The gradient at each lattice point is a HASH of its coordinate modulo the
 * octave's period, so every octave is exactly periodic over the frame, and
 * craters wrap with toroidal distance. The whole field therefore TILES
 * SEAMLESSLY -- which matters, because convolvePsf() is circular: a
 * non-tiling scene would show a seam under a large blur.
 *
 * SHADING
 *   The displayed image is not the height but a Lambertian hill-shade of it,
 *   n = normalize(-dh/dx, -dh/dy, 1), out = amb + (1-amb)*max(0, n.l), ambient
 *   0.10, light at azimuth 225 deg -- upper left on screen, so the craters read
 *   as pits and not as domes -- and elevation 32 deg. The gradient is a wrapped
 *   central difference scaled by size/256, so the apparent slope -- and hence
 *   the picture -- is identical at 256 and 512. Shading is then multiplied by
 *   a 3-octave albedo field in [0.70, 1.04] (bright highlands, dark maria) and
 *   mapped by a FIXED affine 1.70*v - 0.296, clamped to [0, 1]. Fixed, not a
 *   per-frame min/max stretch, so the picture does not change when the frame
 *   does; it costs ~0.1 % of pixels clipped at one end or the other.
 *
 * SIZE HANDLING
 *   Authored in CHART_UNITS = 256 like the USAF target: 512 is the same
 *   landscape at 2 px per chart unit. The finest noise octave is 4 chart
 *   units, i.e. 4 px at size 256 -- just above the 3.37 px period the pupil
 *   can still pass -- so the blur/recovery contrast is at its most visible
 *   exactly where the optics decide it.
 * ================================================================== */

/** Default seed for the terrain. */
export const TERRAIN_SEED = 1618033;

/** Terrain tuning constants; every one of them is documented in the block above. */
const TERRAIN = Object.freeze({
  warpOctaves: 3, warpCells: 2, warpAmp: 0.09,
  fbmCells: 2, fbmOctaves: 6, fbmWeight: 0.55,
  ridgeCells: 4, ridgeOctaves: 5, ridgeWeight: 0.45,
  albedoCells: 2, albedoOctaves: 3, albedoBase: 0.70, albedoSwing: 0.34,
  craters: 110, rMin: 3.2, rMax: 26, craterAmp: 0.16, craterRef: 8,
  lightAzDeg: 225, lightElDeg: 32, slopeScale: 4.0, ambient: 0.10,
  gain: 1.70, offset: -0.296,
});

/** 32-bit integer hash of a lattice point -> [0, 1). No tables, no state. */
function latticeHash(i, j, octave, seed) {
  let h = seed | 0;
  h = Math.imul(h ^ (i | 0), 0x27d4eb2d); h ^= h >>> 15;
  h = Math.imul(h ^ (j | 0), 0x85ebca6b); h ^= h >>> 13;
  h = Math.imul(h ^ (octave | 0), 0xc2b2ae35); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Perlin's quintic fade, 6t^5-15t^4+10t^3: C2, so no interpolation creases. */
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

/** The 8 unit gradients a lattice corner can carry: 4 axial, 4 diagonal. */
const R2 = Math.SQRT1_2;
const GRADX = [1, -1, 0, 0, R2, -R2, R2, -R2];
const GRADY = [0, 0, 1, -1, R2, R2, -R2, -R2];

/**
 * GRADIENT (Perlin) noise on a G x G periodic lattice, returned in [0, 1].
 * (x, y) are in LATTICE cells, so the field repeats exactly every G cells --
 * that is what makes the whole scene tile on the convolution torus.
 *
 * Gradient noise rather than value noise on purpose: value noise puts an extremum
 * at every lattice point, which prints a visible axis-aligned quilt through the
 * fBm no matter how much it is warped. Perlin's construction is zero at every
 * lattice point and the extrema fall inside the cells, so the grid does not show.
 */
function gradientNoise(x, y, G, octave, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const fx = x - xi, fy = y - yi;
  const u = fade(fx), v = fade(fy);
  const i0 = ((xi % G) + G) % G, j0 = ((yi % G) + G) % G;
  const i1 = (i0 + 1) % G, j1 = (j0 + 1) % G;
  const g00 = (latticeHash(i0, j0, octave, seed) * 8) | 0;
  const g10 = (latticeHash(i1, j0, octave, seed) * 8) | 0;
  const g01 = (latticeHash(i0, j1, octave, seed) * 8) | 0;
  const g11 = (latticeHash(i1, j1, octave, seed) * 8) | 0;
  const n00 = GRADX[g00] * fx + GRADY[g00] * fy;
  const n10 = GRADX[g10] * (fx - 1) + GRADY[g10] * fy;
  const n01 = GRADX[g01] * fx + GRADY[g01] * (fy - 1);
  const n11 = GRADX[g11] * (fx - 1) + GRADY[g11] * (fy - 1);
  const top = n00 + (n10 - n00) * u;
  const bot = n01 + (n11 - n01) * u;
  // |n| <= sqrt(2)/2 for unit gradients in 2-D, so this lands in [0, 1].
  return 0.5 + Math.SQRT1_2 * (top + (bot - top) * v);
}

/** Fractional Brownian motion; (u, v) in [0,1) frame coordinates. Returns [0, 1]. */
function fbm(u, v, cells, octaves, seed, tag) {
  let sum = 0, amp = 1, norm = 0, G = cells;
  for (let o = 0; o < octaves; o++) {
    sum += amp * gradientNoise(u * G, v * G, G, tag + o, seed);
    norm += amp; amp *= 0.5; G *= 2;
  }
  return sum / norm;
}

/** Ridged multifractal: (1-|2n-1|)^2 summed over octaves. Returns [0, 1]. */
function ridgedNoise(u, v, cells, octaves, seed, tag) {
  let sum = 0, amp = 1, norm = 0, G = cells;
  for (let o = 0; o < octaves; o++) {
    const r = 1 - Math.abs(2 * gradientNoise(u * G, v * G, G, tag + o, seed) - 1);
    sum += amp * r * r;
    norm += amp; amp *= 0.5; G *= 2;
  }
  return sum / norm;
}

const _terrainCache = new Map();
const _terrainHeightCache = new Map();

/**
 * Procedurally generated cratered terrain: a seamless, hill-shaded crater and
 * ridge field with content from the full frame down to ~4 px.
 *
 * @param {number} [size=256]  image side; a power of two, 256 or 512 recommended
 * @param {object} [opts]
 * @param {number}  [opts.seed=TERRAIN_SEED]
 * @param {boolean} [opts.shared=false]  return the cached instance (read-only)
 * @returns {Float64Array} length size*size, row-major, values in [0, 1]
 */
export function terrain(size = 256, opts = {}) {
  assertPow2('terrain', size);
  const seed = opts.seed === undefined ? TERRAIN_SEED : opts.seed;
  const key = size + ':' + seed;
  let master = _terrainCache.get(key);
  if (!master) {
    master = shadeTerrain(terrainHeight(size, seed), size, seed);
    _terrainCache.set(key, master);
  }
  return opts.shared ? master : Float64Array.from(master);
}

/**
 * The raw height field behind terrain(), before shading. Useful for a panel that
 * wants to draw a profile, and for tests. Values are unitless "elevation".
 *
 * @returns {Float64Array} length size*size
 */
export function terrainHeight(size = 256, seed = TERRAIN_SEED) {
  assertPow2('terrainHeight', size);
  const key = size + ':' + seed;
  const hit = _terrainHeightCache.get(key);
  if (hit) return hit;

  const T = TERRAIN;
  const k = size / CHART_UNITS;
  const h = new Float64Array(size * size);

  // --- warped fBm + ridges ---
  for (let y = 0; y < size; y++) {
    const v = (y + 0.5) / size;
    const row = y * size;
    for (let x = 0; x < size; x++) {
      const u = (x + 0.5) / size;
      const wx = fbm(u + 0.31, v + 0.17, T.warpCells, T.warpOctaves, seed, 40) - 0.5;
      const wy = fbm(u + 0.73, v + 0.61, T.warpCells, T.warpOctaves, seed, 60) - 0.5;
      const uu = u + T.warpAmp * wx, vv = v + T.warpAmp * wy;
      h[row + x] = T.fbmWeight * fbm(uu, vv, T.fbmCells, T.fbmOctaves, seed, 0)
                 + T.ridgeWeight * ridgedNoise(uu, vv, T.ridgeCells, T.ridgeOctaves, seed, 20);
    }
  }

  // --- craters, drawn on the torus so the field still tiles ---
  const rnd = mulberry32(seed ^ 0x9e3779b9);
  for (let i = 0; i < T.craters; i++) {
    const u = rnd();
    // N(>r) ~ r^-2  =>  r = rMin / sqrt(1-u), truncated at rMax.
    let r = T.rMin / Math.sqrt(1 - 0.985 * u);
    if (r > T.rMax) r = T.rMax;
    const R = r * k;
    const cx = rnd() * size, cy = rnd() * size;
    const depth = 0.7 + 0.6 * rnd();
    const amp = T.craterAmp * depth * Math.pow(r / T.craterRef, 0.35);
    const ext = Math.ceil(2.6 * R);
    const bx = Math.round(cx - 0.5), by = Math.round(cy - 0.5);
    for (let dy = -ext; dy <= ext; dy++) {
      const yy = by + dy;
      const ey = yy + 0.5 - cy;
      const row = (((yy % size) + size) % size) * size;
      for (let dx = -ext; dx <= ext; dx++) {
        const xx = bx + dx;
        const ex = xx + 0.5 - cx;
        const d = Math.sqrt(ex * ex + ey * ey) / R;
        if (d > 2.6) continue;
        const t1 = d / 0.62, t2 = (d - 1.0) / 0.26, t3 = (d - 1.5) / 0.6;
        const prof = -Math.exp(-t1 * t1) + 0.85 * Math.exp(-t2 * t2) + 0.18 * Math.exp(-t3 * t3);
        h[row + (((xx % size) + size) % size)] += amp * prof;
      }
    }
  }

  _terrainHeightCache.set(key, h);
  return h;
}

/** Lambertian hill-shade + albedo + the fixed affine map into [0, 1]. */
function shadeTerrain(h, size, seed) {
  const T = TERRAIN;
  const out = new Float64Array(size * size);
  const az = (T.lightAzDeg * Math.PI) / 180, el = (T.lightElDeg * Math.PI) / 180;
  const lx = Math.cos(az) * Math.cos(el), ly = Math.sin(az) * Math.cos(el), lz = Math.sin(el);
  const z = T.slopeScale * (size / CHART_UNITS); // slope per CHART UNIT, not per pixel
  for (let y = 0; y < size; y++) {
    const ym = ((y - 1 + size) % size) * size, yp = ((y + 1) % size) * size, row = y * size;
    const v = (y + 0.5) / size;
    for (let x = 0; x < size; x++) {
      const xm = (x - 1 + size) % size, xp = (x + 1) % size;
      const gx = (h[row + xp] - h[row + xm]) * 0.5 * z;
      const gy = (h[yp + x] - h[ym + x]) * 0.5 * z;
      const n = 1 / Math.sqrt(gx * gx + gy * gy + 1);
      const dot = (-gx * lx - gy * ly + lz) * n;
      const shade = T.ambient + (1 - T.ambient) * (dot > 0 ? dot : 0);
      const alb = T.albedoBase + T.albedoSwing
        * fbm((x + 0.5) / size + 0.5, v + 0.25, T.albedoCells, T.albedoOctaves, seed, 80);
      let q = T.offset + T.gain * shade * alb;
      out[row + x] = q < 0 ? 0 : q > 1 ? 1 : q;
    }
  }
  return out;
}

/**
 * Octave table for the terrain, in IMAGE PIXELS, so a panel can say which scales a
 * given blur has removed.
 *
 * @returns {{sizePx, scale, seamless, octaves: Array<{band, periodPx, cyclesPerPx}>,
 *            craterRadiusPx: [number, number], lightAzDeg, lightElDeg}}
 */
export function terrainSpec(size = 256) {
  assertPow2('terrainSpec', size);
  const T = TERRAIN;
  const k = size / CHART_UNITS;
  const octaves = [];
  let G = T.fbmCells;
  for (let o = 0; o < T.fbmOctaves; o++, G *= 2) {
    octaves.push({ band: 'fbm' + o, periodPx: size / G, cyclesPerPx: G / size });
  }
  G = T.ridgeCells;
  for (let o = 0; o < T.ridgeOctaves; o++, G *= 2) {
    octaves.push({ band: 'ridge' + o, periodPx: size / G, cyclesPerPx: G / size });
  }
  octaves.sort((a, b) => b.periodPx - a.periodPx);
  return {
    sizePx: size,
    scale: k,
    seamless: true,
    octaves,
    craterRadiusPx: [T.rMin * k, T.rMax * k],
    craterCount: T.craters,
    lightAzDeg: T.lightAzDeg,
    lightElDeg: T.lightElDeg,
  };
}

/* ================================================================== *
 * The scene catalogue
 * ================================================================== */

/**
 * Every scene this module can generate, in display order. A picker can be built
 * straight off this array; nothing about the three scenes needs to be hard-coded
 * in the page.
 *
 * Each entry:
 *   id       stable key for state / URLs
 *   label    short human name
 *   blurb    one line on what the scene is FOR
 *   make     (size) => Float64Array in [0, 1], length size*size
 *   display  recommended tone map for the canvas: 'linear' | 'asinh'
 *   spec     (size) => the scene's machine-readable geometry, or null
 *
 * @type {ReadonlyArray<{id: string, label: string, blurb: string,
 *                       make: (size?: number) => Float64Array,
 *                       display: string, spec: ((size?: number) => object)|null}>}
 */
export const SCENES = Object.freeze([
  Object.freeze({
    id: 'usaf',
    label: 'USAF-1951 target',
    blurb: 'Calibrated bar groups and Siemens stars: read the resolution off the chart.',
    make: (size = 256) => usafTarget(size),
    display: 'linear',
    spec: (size = 256) => ({ blocks: usafBlocks(size), stars: siemensStars(size) }),
  }),
  Object.freeze({
    id: 'stars',
    label: 'Star field',
    blurb: `Bright star, a ${BINARY_SEP_PX.toFixed(2)} px binary at `
      + `${BINARY_SEP_RAYLEIGH} x Rayleigh, 6.5 mag of field stars, one galaxy.`,
    make: (size = 256) => starField(size),
    display: 'asinh',
    spec: (size = 256) => starFieldSpec(size),
  }),
  Object.freeze({
    id: 'terrain',
    label: 'Cratered terrain',
    blurb: 'Hill-shaded crater and ridge field: broadband natural structure, seamless on the torus.',
    make: (size = 256) => terrain(size),
    display: 'linear',
    spec: (size = 256) => terrainSpec(size),
  }),
]);
