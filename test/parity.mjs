// Fixture-parity + timing harness for the J-Lens Explorer worker.
// Extracts the <script id="workerSrc"> block from the app HTML, evals it,
// and checks the JS compute path against the Python golden fixtures
// (top-10 set overlap >= 0.9 per position-layer, both lenses, prompts 1/4/9).
// Also measures the acceptance-criteria timings on this machine.
//
// Run: node test/parity.mjs [--quick]

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const app = join(root, 'app');

const html = readFileSync(join(app, 'jlens-explorer.html'), 'utf-8');
const m = /<script id="workerSrc" type="text\/plain">([\s\S]*?)<\/script>/.exec(html);
if (!m) throw new Error('workerSrc block not found');
const { Engine } = new Function(m[1] + '\nreturn { Engine };')();

const load = (name) => {
  const b = readFileSync(join(app, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

console.log('loading model + tokenizer + lens + vocab…');
const t0 = Date.now();
Engine.loadModel(load('stories110M.bin'));
Engine.loadTokenizer(load('tokenizer.bin'));
Engine.loadLens(load('stories110M.jlens'));
const vocabMeta = JSON.parse(readFileSync(join(app, 'vocab.json'), 'utf-8'));
Engine.loadVocab(vocabMeta);
console.log(`  loaded in ${Date.now() - t0} ms; config`, Engine.model.config.dim,
  'dim,', Engine.model.config.nLayers, 'layers');

const fixtures = JSON.parse(readFileSync(join(app, 'fixtures.json'), 'utf-8'));
const band = fixtures.layers;

// ---------- 1. fixture parity ----------
let worst = { overlap: 1 }, sum = 0, count = 0, fails = 0;
let prunedSum = 0, prunedCount = 0;
for (const fp of fixtures.prompts) {
  const ids = Engine.tokenizer.encode(fp.text, true, false);
  const same = ids.length === fp.token_ids.length && ids.every((t, i) => t === fp.token_ids[i]);
  console.log(`prompt ${fp.eval_index}: tokenization ${same ? 'MATCHES' : 'DIFFERS'} (${ids.length} tokens)`);
  if (!same) {
    console.log('  py:', fp.token_ids.join(' '));
    console.log('  js:', ids.join(' '));
  }
  const cap = Engine.newCapture(ids.length);
  for (let p = 0; p < ids.length; p++) Engine.captureForward(cap, ids[p], p, null);
  Engine.cap = cap;

  for (let li = 0; li < band.length; li++) {
    for (let p = 0; p < ids.length; p++) {
      for (const kind of ['jlens', 'logit']) {
        const fix = new Set(fp[kind][li][p].ids.slice(0, 10));
        const js = Engine.readoutFull(p, li, kind, 10).map(e => e.id);
        const inter = js.filter(id => fix.has(id)).length;
        const overlap = inter / 10;
        sum += overlap; count++;
        if (overlap < 0.9) fails++;
        if (overlap < worst.overlap) worst = { overlap, prompt: fp.eval_index, layer: band[li], pos: p, kind };
      }
      // pruning honesty: pruned top-10 (UI path) vs full top-10 (J-lens)
      const fix = fp.jlens[li][p].ids.slice(0, 10);
      const inPruned = fix.filter(id => Engine.prunedIndex[id] >= 0).length;
      prunedSum += inPruned / 10; prunedCount++;
    }
  }
}
console.log(`\nPARITY: mean top-10 overlap = ${(sum / count).toFixed(4)} over ${count} (pos,layer,lens) cells`);
console.log(`        cells below 0.9: ${fails}  |  worst: ${JSON.stringify(worst)}`);
console.log(`        fixture top-10 coverage by pruned vocab: ${(100 * prunedSum / prunedCount).toFixed(1)}%`);
console.log(fails === 0 ? '        ✅ ACCEPTANCE 1 PASS' : '        ❌ ACCEPTANCE 1 FAIL');

if (process.argv.includes('--quick')) process.exit(fails === 0 ? 0 : 1);

// ---------- 2. analyze timing on a ~200-token story ----------
const story = `Once upon a time there was a little girl named Lily. She had a dog named Max.
One day, Lily and Max went to the beach. Lily built a big sand castle and Max dug a deep hole
near the water. The sun was warm and the waves were soft. Lily found a shiny shell and put it
on top of her castle. Max barked at a small crab that walked sideways across the sand.
Then a big wave came and washed the castle away. Lily was sad. Max saw her tears and brought
her the shiny shell in his mouth. Lily smiled and hugged Max. They built a new castle together,
even bigger than before. When the sun went down, they walked home happy and tired.
Mom made them warm soup and Lily told her all about the beach, the castle, the crab and the
brave little dog who saved the day. Then Lily and Max fell asleep and dreamed of the sea.`;
const ids = Engine.tokenizer.encode(story.replace(/\n/g, ' '), true, false);
console.log(`\nTIMING: analyze path on ${ids.length} tokens…`);
const tA = Date.now();
const cap = Engine.newCapture(ids.length);
for (let p = 0; p < ids.length; p++) Engine.captureForward(cap, ids[p], p, null);
const tFwd = Date.now() - tA;
const tD = Date.now();
for (const l of Engine.lens.layers) Engine.buildD(l);
const tDict = Date.now() - tD;
const tP = Date.now();
Engine.precompute(cap, null);
const tPre = Date.now() - tP;
console.log(`  forward (capture): ${(tFwd / 1000).toFixed(1)} s  (${(tFwd / ids.length).toFixed(0)} ms/token)`);
console.log(`  dictionaries (one-time, 5 layers): ${(tDict / 1000).toFixed(1)} s`);
console.log(`  readout precompute: ${(tPre / 1000).toFixed(1)} s`);
console.log(`  analyze total (forward+precompute): ${((tFwd + tPre) / 1000).toFixed(1)} s`);
console.log(`  ${(tFwd + tPre) / 1000 < 60 ? '✅' : '❌'} ACCEPTANCE 2 (<60 s)  ${(tFwd + tPre) / 1000 < 30 ? '(stretch <30 s ✅)' : ''}`);

// ---------- 3. readout latency (post-capture) ----------
const tR = Date.now();
let n = 0;
for (let p = 0; p < Math.min(50, cap.nPos); p++) { Engine.readout(p, 40); n++; }
const perReadout = (Date.now() - tR) / n;
console.log(`\nreadout latency: ${perReadout.toFixed(1)} ms per token selection (${n} samples)`);
console.log(`  ${perReadout < 100 ? '✅' : '❌'} ACCEPTANCE 3 (<100 ms)`);

// ---------- 4. heat latency ----------
const holeId = Engine.tokenizer.encode(' hole', false, false).slice(-1)[0];
const tH = Date.now();
Engine.heat(holeId);
console.log(`heat map over ${cap.nPos} positions: ${Date.now() - tH} ms`);
