'use strict';

const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const ROOT = path.resolve(__dirname, '..');
const ASSETS_DIR = path.resolve(ROOT, '..', 'static', 'assets');
const BASELINE_PATH = path.resolve(ROOT, '.bundle-baseline.json');

if (!fs.existsSync(ASSETS_DIR)) {
  console.error('ERROR: static/assets/ not found. Run `npm run build` first.');
  process.exit(1);
}

if (!fs.existsSync(BASELINE_PATH)) {
  console.error(`ERROR: ${BASELINE_PATH} not found.`);
  process.exit(1);
}

const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8'));
const files = fs.readdirSync(ASSETS_DIR);

function globToRegex(glob) {
  // Escape all regex special chars except *, then replace * with [^/]*
  const re =
    '^' +
    glob
      .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
      .replace(/\*/g, '[^/]*') +
    '$';
  return new RegExp(re);
}

const budgets = baseline.perChunkBudgetsGzipKB;
const budgetEntries = Object.entries(budgets).map(([glob, kb]) => ({
  glob,
  regex: globToRegex(glob),
  kb,
}));

let totalGzipKB = 0;
const results = [];
let failed = false;

for (const file of files) {
  const fullPath = path.join(ASSETS_DIR, file);
  const stat = fs.statSync(fullPath);
  if (!stat.isFile()) continue;
  const buf = fs.readFileSync(fullPath);
  const gz = zlib.gzipSync(buf);
  const gzipKB = gz.length / 1024;
  totalGzipKB += gzipKB;

  const match = budgetEntries.find((e) => e.regex.test(file));
  if (!match) {
    results.push({ file, gzipKB, budgetKB: null, status: 'UNBUDGETED' });
    continue;
  }
  const ok = gzipKB <= match.kb;
  if (!ok) failed = true;
  results.push({ file, gzipKB, budgetKB: match.kb, status: ok ? 'OK' : 'FAIL' });
}

const totalOK =
  !baseline.totalBudgetGzipKB || totalGzipKB <= baseline.totalBudgetGzipKB;
if (!totalOK) failed = true;

console.log(
  '\n  CHUNK'.padEnd(40),
  'GZIP_KB'.padStart(10),
  'BUDGET_KB'.padStart(12),
  'STATUS'
);
console.log('  ' + '-'.repeat(70));
for (const r of results.sort((a, b) => b.gzipKB - a.gzipKB)) {
  const sizeStr = r.gzipKB.toFixed(1).padStart(10);
  const budgetStr =
    r.budgetKB === null
      ? '      (none)'
      : r.budgetKB.toFixed(0).padStart(12);
  console.log('  ' + r.file.padEnd(38), sizeStr, budgetStr, ' ', r.status);
}
console.log('  ' + '-'.repeat(70));
console.log(
  '  TOTAL'.padEnd(40),
  totalGzipKB.toFixed(1).padStart(10),
  (baseline.totalBudgetGzipKB ?? '(none)').toString().padStart(12),
  ' ',
  totalOK ? 'OK' : 'FAIL'
);

if (failed) {
  console.error('\nBUNDLE SIZE CHECK FAILED');
  process.exit(1);
}
console.log('\nBundle size check passed.');
process.exit(0);
