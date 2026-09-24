/**
 * ladder.js -- reader for assets/ladder.bin (schema "pupil-ladder/1") and for the
 * identically-encoded `bits_b64` mask blobs in golden.json.
 *
 * Documented layout (ladder.json "layout", GOLDEN.md §1.4):
 *   - 26 frames, contiguous; frame f occupies bytes [f*2048, (f+1)*2048)
 *   - 128 x 128 pixels, 1 bit per pixel, 2048 bytes per frame, 53248 bytes total
 *   - row-major pixel index p = y*128 + x, y = 0 is the TOP row, x = 0 the LEFT column
 *   - LSB-first bit order: pixel p lives in byte (p >> 3) at bit (p & 7), bit 0 = value 1
 *   - bit set  => transmissive (mask == 1); bit clear => opaque (mask == 0)
 *
 * Dependency-free: works in Node and in the browser. Pass raw bytes in; use
 * `fetchLadder` if you want the module to do the loading (global fetch).
 */

export const LADDER_LAYOUT = Object.freeze({
  width: 128,
  height: 128,
  bitsPerPixel: 1,
  bytesPerFrame: 2048,
  frameCount: 26,
  totalBytes: 53248,
});

/* ------------------------------------------------------------------ *
 * Bit / base64 decoding
 * ------------------------------------------------------------------ */

const B64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = (() => {
  const t = new Int16Array(256).fill(-1);
  for (let i = 0; i < B64_ALPHABET.length; i++) t[B64_ALPHABET.charCodeAt(i)] = i;
  t['='.charCodeAt(0)] = -2;
  return t;
})();

/**
 * Minimal, environment-independent base64 -> Uint8Array decoder
 * (no atob, no Buffer, so the same code runs in Node and the browser).
 */
export function base64ToBytes(b64) {
  const s = String(b64);
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = B64_LOOKUP[s.charCodeAt(i) & 0xff];
    if (c >= 0) n++;
  }
  const out = new Uint8Array((n * 3) >> 2);
  let acc = 0, bits = 0, o = 0;
  for (let i = 0; i < s.length; i++) {
    const c = B64_LOOKUP[s.charCodeAt(i) & 0xff];
    if (c < 0) continue;
    acc = (acc << 6) | c;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (acc >> bits) & 0xff;
    }
  }
  return out;
}

/**
 * Unpack a 1-bit-per-pixel, LSB-first bitmap into a per-pixel byte mask.
 *
 * @param {Uint8Array} bytes   at least ceil(width*height/8) bytes
 * @param {number} [byteOffset=0]  where this frame starts inside `bytes`
 * @returns {Uint8Array} length width*height, values 0 or 1, flat index p = y*width + x
 */
export function unpackBits(bytes, byteOffset = 0, width = 128, height = 128) {
  const n = width * height;
  const need = byteOffset + ((n + 7) >> 3);
  if (bytes.length < need) {
    throw new Error(`unpackBits: need ${need} bytes, got ${bytes.length}`);
  }
  const out = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    out[p] = (bytes[byteOffset + (p >> 3)] >> (p & 7)) & 1;
  }
  return out;
}

/** Same as `unpackBits`, but produces the Float64Array amplitudes the optics path wants. */
export function unpackBitsF64(bytes, byteOffset = 0, width = 128, height = 128) {
  const n = width * height;
  const out = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    out[p] = (bytes[byteOffset + (p >> 3)] >> (p & 7)) & 1;
  }
  return out;
}

/** Decode a base64 `bits_b64` mask blob (golden.json §1.4) into a 0/1 byte mask. */
export function decodeMaskBase64(b64, width = 128, height = 128) {
  return unpackBits(base64ToBytes(b64), 0, width, height);
}

/**
 * Decode a base64 little-endian float32 blob (`float32_b64`, `noise_float32_b64`).
 * Byte order is handled explicitly, so this is correct on big-endian hosts too.
 */
export function decodeFloat32Base64(b64, count) {
  const bytes = base64ToBytes(b64);
  const n = count ?? (bytes.length >> 2);
  if (bytes.length < n * 4) throw new Error(`decodeFloat32Base64: need ${n * 4} bytes, got ${bytes.length}`);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true); // little-endian
  return out;
}

/* ------------------------------------------------------------------ *
 * Ladder reader
 * ------------------------------------------------------------------ */

function toBytes(src) {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (ArrayBuffer.isView(src)) return new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
  throw new TypeError('readLadder: expected ArrayBuffer, Uint8Array or TypedArray');
}

/**
 * Wrap ladder.bin bytes in a reader.
 *
 * @param {ArrayBuffer|Uint8Array} source  the contents of ladder.bin
 * @param {object} [meta]  optional parsed ladder.json; its `layout` and `frames`
 *                         are used when present (and validated against the bytes)
 * @returns {{
 *   width: number, height: number, bytesPerFrame: number, frameCount: number,
 *   frames: Array|null, bytes: Uint8Array,
 *   pixel(f: number, x: number, y: number): number,
 *   mask(f: number): Uint8Array,
 *   maskF64(f: number): Float64Array,
 *   area(f: number): number,
 *   frameInfo(f: number): object|null
 * }}
 */
export function readLadder(source, meta = null) {
  const bytes = toBytes(source);
  const layout = (meta && meta.layout) || {};
  const width = layout.width ?? LADDER_LAYOUT.width;
  const height = layout.height ?? LADDER_LAYOUT.height;
  const bytesPerFrame = layout.bytes_per_frame ?? layout.bytesPerFrame ?? ((width * height + 7) >> 3);
  const declared = layout.frame_count ?? layout.frameCount ?? null;
  const frameCount = Math.floor(bytes.length / bytesPerFrame);

  if (bytes.length % bytesPerFrame !== 0) {
    throw new Error(`readLadder: ${bytes.length} bytes is not a whole number of ${bytesPerFrame}-byte frames`);
  }
  if (declared != null && declared !== frameCount) {
    throw new Error(`readLadder: ladder.json declares ${declared} frames but the binary holds ${frameCount}`);
  }

  const frames = meta && Array.isArray(meta.frames) ? meta.frames : null;
  const cache = new Map();

  const checkFrame = (f) => {
    if (!Number.isInteger(f) || f < 0 || f >= frameCount) {
      throw new RangeError(`frame ${f} out of range [0, ${frameCount})`);
    }
  };

  return {
    width, height, bytesPerFrame, frameCount, frames, bytes,

    /** Single pixel: 1 = transmissive. y = 0 is the top row, x = 0 the left column. */
    pixel(f, x, y) {
      checkFrame(f);
      const p = y * width + x;
      return (bytes[f * bytesPerFrame + (p >> 3)] >> (p & 7)) & 1;
    },

    /** Frame f as a 0/1 Uint8Array of length width*height (cached). */
    mask(f) {
      checkFrame(f);
      let m = cache.get(f);
      if (!m) { m = unpackBits(bytes, f * bytesPerFrame, width, height); cache.set(f, m); }
      return m;
    },

    /** Frame f as a Float64Array of 0.0/1.0 amplitudes. */
    maskF64(f) {
      checkFrame(f);
      return unpackBitsF64(bytes, f * bytesPerFrame, width, height);
    },

    /** Transmissive pixel count of frame f. */
    area(f) {
      const m = this.mask(f);
      let s = 0;
      for (let i = 0; i < m.length; i++) s += m[i];
      return s;
    },

    /** The ladder.json record for frame f (slot, kind, alpha_corrected, area, ...), if meta was given. */
    frameInfo(f) {
      checkFrame(f);
      if (!frames) return null;
      return frames.find((r) => r.slot === f) ?? frames[f] ?? null;
    },
  };
}

/**
 * Convenience loader for browsers / Node 18+ (uses global fetch).
 * `metaUrl` is optional; when given, ladder.json is fetched and attached.
 */
export async function fetchLadder(binUrl, metaUrl = null) {
  const [binRes, meta] = await Promise.all([
    fetch(binUrl),
    metaUrl ? fetch(metaUrl).then((r) => r.json()) : Promise.resolve(null),
  ]);
  if (!binRes.ok) throw new Error(`fetchLadder: ${binUrl} -> HTTP ${binRes.status}`);
  return readLadder(await binRes.arrayBuffer(), meta);
}
