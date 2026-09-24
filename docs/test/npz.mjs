/**
 * Minimal .npz / .npy reader for the verification harness.
 *
 * golden.npz is a plain ZIP (deflate-compressed here) of .npy members. This reads
 * the central directory, inflates each member with node:zlib, and parses the .npy
 * header. Node built-ins only -- no dependencies.
 */
import { readFileSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';

const DTYPES = {
  '<f8': Float64Array, '<f4': Float32Array,
  '|u1': Uint8Array, '<u1': Uint8Array, '|i1': Int8Array,
  '<i4': Int32Array, '<i8': BigInt64Array, '<u4': Uint32Array,
  '|b1': Uint8Array,
};

function parseNpy(buf) {
  if (buf[0] !== 0x93 || String.fromCharCode(buf[1], buf[2], buf[3], buf[4], buf[5]) !== 'NUMPY') {
    throw new Error('not a .npy file');
  }
  const major = buf[6];
  let headerLen, headerStart;
  if (major === 1) { headerLen = buf.readUInt16LE(8); headerStart = 10; }
  else { headerLen = buf.readUInt32LE(8); headerStart = 12; }
  const header = buf.slice(headerStart, headerStart + headerLen).toString('latin1');
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)[1] === 'True';
  const shapeStr = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)[1].trim();
  const shape = shapeStr === '' ? [] : shapeStr.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const dataStart = headerStart + headerLen;
  const body = buf.slice(dataStart);

  if (descr.startsWith('<U') || descr.startsWith('|S')) {
    // Fixed-width unicode strings (case_ids / mask_ids): UCS-4 LE.
    const w = Number(descr.slice(2));
    const count = shape.reduce((a, b) => a * b, 1);
    const out = [];
    for (let i = 0; i < count; i++) {
      let s = '';
      for (let c = 0; c < w; c++) {
        const cp = body.readUInt32LE((i * w + c) * 4);
        if (cp === 0) break;
        s += String.fromCodePoint(cp);
      }
      out.push(s);
    }
    return { descr, shape, fortran, data: out };
  }

  const Ctor = DTYPES[descr];
  if (!Ctor) throw new Error(`unsupported npy dtype ${descr}`);
  const count = shape.reduce((a, b) => a * b, 1);
  // Copy: the slice is not guaranteed to be element-aligned.
  const bytes = new Uint8Array(body.buffer, body.byteOffset, count * Ctor.BYTES_PER_ELEMENT);
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return { descr, shape, fortran, data: new Ctor(copy.buffer) };
}

/** Read an .npz into { name -> {descr, shape, data} }. */
export function readNpz(path) {
  const b = readFileSync(path);
  let eocd = b.length - 22;
  for (; eocd >= 0; eocd--) if (b.readUInt32LE(eocd) === 0x06054b50) break;
  if (eocd < 0) throw new Error('npz: end-of-central-directory not found');
  const entries = b.readUInt16LE(eocd + 10);
  let p = b.readUInt32LE(eocd + 16);
  const out = {};
  for (let k = 0; k < entries; k++) {
    if (b.readUInt32LE(p) !== 0x02014b50) throw new Error('npz: bad central directory signature');
    const method = b.readUInt16LE(p + 10);
    const csize = b.readUInt32LE(p + 20);
    const nameLen = b.readUInt16LE(p + 28);
    const extraLen = b.readUInt16LE(p + 30);
    const commentLen = b.readUInt16LE(p + 32);
    const lho = b.readUInt32LE(p + 42);
    const name = b.slice(p + 46, p + 46 + nameLen).toString('latin1');
    // Local file header: skip its own (possibly different) name/extra lengths.
    const lNameLen = b.readUInt16LE(lho + 26);
    const lExtraLen = b.readUInt16LE(lho + 28);
    const dataOff = lho + 30 + lNameLen + lExtraLen;
    const raw = b.slice(dataOff, dataOff + csize);
    const member = method === 0 ? raw : inflateRawSync(raw);
    out[name.replace(/\.npy$/, '')] = parseNpy(member);
    p += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}
