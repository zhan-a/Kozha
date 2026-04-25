/**
 * Translator end-to-end smoke against the deployed site.
 *
 * Verifies the two prior fixes survive a real production round-trip:
 *
 *   1. Upload pipeline decodes MOV/MP4/WebM/MP3/WAV via ffmpeg.wasm
 *      (Bug A2 — previously the iPhone .mov path threw an Event that
 *      describeError stringified to "[object Event]" in the transcript).
 *   2. Microphone tab boots an onnxruntime-web runtime via the importmap
 *      with a WASM fallback when WebGPU is missing (Bug B — previously
 *      the bare-specifier import threw "Failed to resolve module
 *      specifier 'onnxruntime-web'").
 *
 * The text-input tab and the avatar are out of scope.
 *
 * Currently failing on production (deploy SHA a57d761): both phases
 * surface "Speech model bundle failed to load (Failed to resolve module
 * specifier 'onnxruntime-web/webgpu')" because transformers.js@4.2.0
 * statically imports a third specifier the importmap doesn't list.
 * Per the prompt's "do not change application code" rule, the bug is
 * filed in docs/translator-fix/02-bug-importmap-webgpu.md rather than
 * fixed here. The spec is the canonical reproducer and is expected to
 * pass once the missing importmap entry is added.
 *
 * Test framework note: the repo pins `@playwright/test@1.59.1` (see
 * package.json and the two prior translator-fix specs). The fix prompt
 * suggested 1.49.1; existing-convention takes precedence per the
 * "do not introduce a second test framework alongside the existing one"
 * rule and the prior specs that made the same call.
 *
 * Deploy-wait note: the site exposes a deploy receipt at
 * /data/last_deploy.json (the contrib-fix flow already uses it). It
 * carries the deployed git SHA, so we poll that instead of /healthz
 * which doesn't exist on the EC2 nginx.
 *
 * Run:
 *   # against the deployed site, waiting for HEAD to be live:
 *   npx playwright test tests/translator_input_e2e.spec.ts --reporter=list
 *
 *   # or pin a specific deployed SHA for local iteration:
 *   KOZHA_EXPECTED_SHA=$(curl -s https://kozha-translate.com/data/last_deploy.json | jq -r .sha) \
 *     npx playwright test tests/translator_input_e2e.spec.ts --reporter=list
 */
import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const FIXTURES_DIR = path.join(REPO_ROOT, 'tests', 'fixtures', 'audio-video');
const MOV_FIXTURE = path.join(FIXTURES_DIR, 'iphone-sample.mov');
const WAV_FIXTURE = path.join(FIXTURES_DIR, 'sample.wav');

const BASE_URL = (process.env.KOZHA_BASE_URL || 'https://kozha-translate.com').replace(/\/+$/, '');
const DEPLOY_RECEIPT_URL = `${BASE_URL}/data/last_deploy.json`;
const DEPLOY_WAIT_MS = 5 * 60_000;
const DEPLOY_POLL_MS = 5_000;

// Generate fixtures synchronously at module load if missing. Chromium
// reads --use-file-for-fake-audio-capture lazily when the fake stream
// is constructed, so technically a beforeAll hook would also work, but
// generating ahead of test.use() means the launch-args path always
// points at a real file and the fixture happens to be available for
// other tests too.
function ensureFixture(target: string, ffmpegArgs: ReadonlyArray<string>): void {
  if (fs.existsSync(target)) return;
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const cmd = ['ffmpeg', '-y', ...ffmpegArgs, target]
    .map((s) => `'${s.replace(/'/g, `'\\''`)}'`).join(' ');
  execSync(cmd, { stdio: 'ignore' });
}
ensureFixture(MOV_FIXTURE, [
  '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
  '-c:a', 'aac', '-movflags', '+faststart',
]);
ensureFixture(WAV_FIXTURE, [
  '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
  '-c:a', 'pcm_s16le', '-ac', '1', '-ar', '16000',
]);

// Browser flags. The fake-media flags let the mic tab record from a
// canned WAV without a microphone or a permission prompt; the autoplay
// flag lets the upload tab's hidden <video> probe duration without a
// synthetic user gesture.
test.use({
  launchOptions: {
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${WAV_FIXTURE}`,
      '--autoplay-policy=no-user-gesture-required',
    ],
  },
});

function expectedSha(): string {
  const env = (process.env.KOZHA_EXPECTED_SHA || '').trim();
  if (env) return env;
  // HEAD is what we just pushed. origin/main would be slightly more
  // accurate (it's what GitHub Actions deploys), but HEAD == origin/main
  // immediately after a successful push, and we don't want a stale
  // origin ref to lock the test out of recognising the new deploy.
  return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
}

async function waitForDeploy(target: string): Promise<void> {
  const target7 = target.slice(0, 7);
  const deadline = Date.now() + DEPLOY_WAIT_MS;
  let last: { status?: number; sha?: string; raw?: string } = {};
  while (Date.now() < deadline) {
    try {
      const res = await fetch(DEPLOY_RECEIPT_URL, { cache: 'no-store' });
      last.status = res.status;
      if (res.ok) {
        const body = (await res.json()) as { sha?: string };
        last.sha = body.sha;
        if (body.sha === target) return;
      } else {
        last.raw = await res.text().catch(() => '');
      }
    } catch (err) {
      last.raw = (err as Error).message;
    }
    await new Promise((r) => setTimeout(r, DEPLOY_POLL_MS));
  }
  throw new Error(
    `deploy did not complete within ${Math.round(DEPLOY_WAIT_MS / 1000)}s — ` +
      `expected sha ${target7}, last receipt sha=${(last.sha || '').slice(0, 7)} ` +
      `httpStatus=${last.status} body=${(last.raw || '').slice(0, 200)}`,
  );
}

// One test, two phases, shared context. The upload phase warms both
// ffmpeg.wasm and the Whisper model in IndexedDB, so the mic phase
// transcribes a 1 s clip in a few seconds — the prompt's 30 s mic
// window is realistic only because the upload phase ran first. The
// upload phase itself needs a longer budget on a fresh Playwright
// context (cold ffmpeg.wasm + cold Whisper-tiny ≈ 30–60 s on a typical
// home network); we extend it to 120 s and document the deviation.
test('translator upload + mic e2e on deployed site', async ({ page }, testInfo) => {
  testInfo.setTimeout(DEPLOY_WAIT_MS + 6 * 60_000);

  const consoleAll: string[] = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const failedRequests: string[] = [];
  // Phase markers let the mic-phase assertions filter out errors that
  // were emitted during the upload phase (e.g. an unrelated /api/plan
  // hiccup) and not introduced by the mic boot.
  let phase: 'pre' | 'upload' | 'mic' = 'pre';
  const errorPhase: string[] = [];

  page.on('console', (msg) => {
    const t = msg.type();
    consoleAll.push(`[${t}][${phase}] ${msg.text()}`);
    if (t === 'error') {
      consoleErrors.push(msg.text());
      errorPhase.push(phase);
    }
  });
  page.on('pageerror', (err) => {
    const text = `pageerror: ${String(err)}`;
    consoleAll.push(`[${phase}] ${text}`);
    pageErrors.push(text);
  });
  page.on('requestfailed', (req) => {
    const url = req.url();
    if (/onnxruntime|@huggingface|@ffmpeg/i.test(url)) {
      failedRequests.push(`${url} (${req.failure()?.errorText || 'unknown'})`);
    }
  });
  page.on('response', (resp) => {
    const url = resp.url();
    if (/onnxruntime|@huggingface|@ffmpeg/i.test(url) && resp.status() >= 400) {
      failedRequests.push(`${url} (HTTP ${resp.status()})`);
    }
  });

  await test.step('deploy wait', async () => {
    await waitForDeploy(expectedSha());
  });

  await test.step('upload tab — MOV decode + transcribe', async () => {
    phase = 'upload';
    await page.goto(`${BASE_URL}/app.html`, { waitUntil: 'networkidle' });

    // Switch to the Audio / Video panel via the sidebar button. Selector
    // by data-panel attribute is stable; the visible label "Audio /
    // Video" sits inside a div alongside the icon, so role+name doesn't
    // resolve cleanly here.
    await page.locator('.sidebar-item[data-panel="video"]').click();
    await expect(page.locator('#panel-video')).toBeVisible();

    const fileInput = page.locator('#videoFile');
    const processBtn = page.locator('#videoToSignBtn');
    const transcript = page.locator('#transcription');

    await fileInput.setInputFiles(MOV_FIXTURE);
    // validatePickedFile probes duration via a hidden <video> element
    // before flipping the button enabled. Give it 15 s — it usually
    // resolves in well under 1 s.
    await expect(processBtn).toBeEnabled({ timeout: 15_000 });

    await processBtn.click();

    // Wait window: the prompt asks for "up to 30 s", but on a fresh
    // Playwright context we pay cold ffmpeg.wasm (~30 MB) + cold
    // Whisper-tiny (~40 MB) sequentially before transcription begins.
    // 120 s absorbs both fetches on a typical home network. The bug
    // signals we actually assert (no [object Event], no error text)
    // are independent of how long the wait is.
    const t0 = Date.now();
    try {
      await expect.poll(
        async () => (await transcript.inputValue()).trim(),
        { timeout: 120_000, intervals: [500, 1000, 2000] },
      ).not.toEqual('');
    } catch (err) {
      const status = await page.locator('#videoStatus').innerText().catch(() => '');
      const errBox = await page.locator('#videoError').innerText().catch(() => '');
      throw new Error(
        `upload transcript stayed empty after ${Date.now() - t0}ms\n` +
          `  videoStatus: ${status}\n` +
          `  videoError:  ${errBox}\n` +
          `  console tail:\n${consoleAll.slice(-30).join('\n')}\n` +
          `  underlying: ${(err as Error).message}`,
      );
    }

    const value = (await transcript.inputValue()).trim();
    expect(value, 'upload transcript should be non-empty').not.toEqual('');
    // The fix-related assertions: Bug A1 leaked Event.toString() into
    // the transcript ("[object Event]"); Bug A2's wrong-format error
    // produced literal "error" / "Error" text in the field. Whisper
    // hallucinations on a 1 s 440 Hz sine ("you", "...", "Thanks for
    // watching.") never contain these substrings, so a hit is always
    // a regression.
    expect(value, `transcript leaked "[object Event]": ${value}`)
      .not.toContain('[object Event]');
    expect(value, `transcript leaked "error": ${value}`)
      .not.toContain('error');
    expect(value, `transcript leaked "Error": ${value}`)
      .not.toContain('Error');

    // Per the prompt: "the page console did not log any uncaught error
    // during the run (collect via page.on('pageerror', ...); expect 0)".
    // pageerror fires only on uncaught JS exceptions; console.error
    // entries (e.g. an unrelated /api/plan hiccup on the deployed
    // backend) don't trip it.
    expect(
      pageErrors,
      `unexpected page errors during upload phase:\n${pageErrors.join('\n')}`,
    ).toEqual([]);
  });

  await test.step('microphone tab — record + transcribe', async () => {
    phase = 'mic';

    // Clear the textarea so we observe a mic-driven update, not a
    // leftover from the upload phase.
    await page.locator('#transcription').fill('');

    await page.locator('.sidebar-item[data-panel="microphone"]').click();
    await expect(page.locator('#panel-microphone')).toBeVisible();

    // Wait for the mic-boot script (Bug B fix) to expose a runtime
    // reference. Headless Chromium has no navigator.gpu, so we expect
    // the WASM branch to win and `window.ortWasm` to be set; on a
    // future GPU-enabled runner `window.ort` is also accepted.
    await expect.poll(
      async () => await page.evaluate(() =>
        Boolean((window as any).ort) || Boolean((window as any).ortWasm),
      ),
      { timeout: 30_000, intervals: [200, 500, 1000] },
    ).toBe(true);

    // recordBtn is a single toggle: first click starts via getUserMedia,
    // second click stops. The fake audio stream begins streaming the
    // sample.wav fixture as soon as getUserMedia resolves.
    const recordBtn = page.locator('#recordBtn');
    const transcribeBtn = page.locator('#transcribeBtn');
    const transcript = page.locator('#transcription');

    await recordBtn.click();
    // Button text flips to "■ Stop recording" once MediaRecorder.start()
    // returns. If it never does, getUserMedia probably failed.
    await expect(recordBtn).toHaveText(/Stop recording/, { timeout: 10_000 });

    // Fixture is 1 s; give the recorder a tick beyond that so at least
    // one ondataavailable chunk lands.
    await page.waitForTimeout(1_500);

    await recordBtn.click();
    await expect(recordBtn).toHaveText(/Start recording/, { timeout: 5_000 });

    await transcribeBtn.click();

    // 60 s on a warm cache (the upload phase pre-fetched both the
    // transformers.js bundle and the Whisper-tiny weights). The prompt
    // asks for 30 s; we extend modestly to absorb network jitter on
    // shared CI runners. Still well under the prompt's "up to" cap
    // for an actual transcription.
    const t0 = Date.now();
    try {
      await expect.poll(
        async () => (await transcript.inputValue()).trim(),
        { timeout: 60_000, intervals: [500, 1000, 2000] },
      ).not.toEqual('');
    } catch (err) {
      const status = await page.locator('#micStatus').innerText().catch(() => '');
      throw new Error(
        `mic transcript stayed empty after ${Date.now() - t0}ms\n` +
          `  micStatus: ${status}\n` +
          `  console tail:\n${consoleAll.slice(-30).join('\n')}\n` +
          `  underlying: ${(err as Error).message}`,
      );
    }

    const value = (await transcript.inputValue()).trim();
    expect(value, 'mic transcript should be non-empty').not.toEqual('');
    expect(value, `transcript leaked "[object Event]": ${value}`)
      .not.toContain('[object Event]');

    // Mic-specific console hygiene. The prompt names three error
    // shapes that would mean Bug B regressed:
    //   - module-resolve errors (importmap pin missing or wrong);
    //   - "Failed to load resource" against an onnxruntime URL
    //     (wrong @version in the importmap → 404);
    //   - "WebGPU is not supported" thrown error (the boot script
    //     should branch on navigator.gpu before anything throws).
    const moduleResolveLike = (s: string) =>
      /\b(?:Failed to resolve module specifier|Importing a module script failed|Failed to load module script|Failed to fetch dynamically imported module)\b/i.test(s);
    const ortLoadLike = (s: string) =>
      /onnxruntime/i.test(s) && /Failed to load resource|net::ERR_|HTTP \d{3}/i.test(s);
    const webgpuThrown = (s: string) =>
      /WebGPU is not supported/i.test(s);

    const micRelevant = consoleErrors.filter(
      (e) => moduleResolveLike(e) || ortLoadLike(e) || webgpuThrown(e),
    );
    expect(
      micRelevant,
      `mic phase: ORT/WebGPU/module-resolve console errors:\n${micRelevant.join('\n')}\n\nfull tail:\n${consoleAll.slice(-30).join('\n')}`,
    ).toEqual([]);

    expect(
      failedRequests,
      `mic phase: requests to onnxruntime/transformers/ffmpeg URLs failed:\n${failedRequests.join('\n')}`,
    ).toEqual([]);

    // Pageerror across the whole run — a "WebGPU is not supported"
    // throw would surface here even if the message never reached
    // console.error.
    const webgpuPageError = pageErrors.filter(webgpuThrown);
    expect(
      webgpuPageError,
      `WebGPU thrown error escaped boot guard:\n${webgpuPageError.join('\n')}`,
    ).toEqual([]);

    // The fallback note is the documented WASM-fallback hint, not an
    // error. Confirm it's styled as a hint when visible — a future
    // refactor that promotes it to .field-error styling would be a
    // regression of the "documented fallback path, not an error"
    // contract from the prompt.
    const fallbackNote = page.locator('#micFallbackNote');
    if (await fallbackNote.isVisible()) {
      await expect(fallbackNote).toHaveClass(/field-hint/);
      await expect(fallbackNote).not.toHaveClass(/field-error/);
    }
  });
});
