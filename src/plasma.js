/**
 * plasma.js -- matplotlib's `plasma`, the colormap the paper uses to colour pupils by
 * asymmetry (see inference_all_peak.py, `cmap="plasma"`).
 *
 * 32 stops sampled straight from matplotlib 3.10, linearly interpolated in sRGB. That is
 * not perceptually ideal, but it is what matplotlib itself does between its own stops, so
 * the result matches the published figures.
 *
 * Contrast note: the two ends of plasma each disappear against one of our two themes --
 * #0d0887 has the same luminance as the dark ground (contrast 1.00) and #f0f921 is 1.13
 * against the light one. Callers should stroke pupils with `outline` so the shape reads
 * regardless of where on the scale it falls.
 */

const STOPS = [
  '0d0887','220690','310597','3f049c','4e02a2','5b01a5','6700a8','7401a8',
  '8104a7','8d0ba5','9814a0','a21d9a','ad2793','b6308b','bf3984','c7427c',
  'cf4c74','d6556d','dd5e66','e3685f','e97257','ef7c51','f3874a','f79143',
  'fa9e3b','fca934','fdb52e','fdc229','fcd025','f9dd25','f5eb27','f0f921',
].map(h => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]);

/** The largest corrected asymmetry in the published test pool -- the top of the scale. */
export const ALPHA_MAX = 0.353933;

/** @param {number} t in [0,1] -> "rgb(r,g,b)" */
export function plasma(t) {
  const u = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(u)), f = u - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},`
       + `${Math.round(a[1] + (b[1] - a[1]) * f)},`
       + `${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

/** Colour for a pupil of corrected asymmetry `alpha`. */
export function forAlpha(alpha) { return plasma(alpha / ALPHA_MAX); }

/** Same, with an explicit alpha channel -- for the scatter's overlapping dots. */
export function forAlphaA(alpha, a) {
  return plasma(alpha / ALPHA_MAX).replace('rgb(', 'rgba(').replace(')', `,${a})`);
}

/**
 * A neutral hairline that keeps a pupil visible at both ends of the ramp.
 * Dark outline on light grounds, light outline on dark ones.
 */
export function outline(isDark) {
  return isDark ? 'rgba(255,255,255,0.55)' : 'rgba(16,23,32,0.45)';
}
