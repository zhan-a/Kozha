ultrathink

You are running an end-to-end smoke test against the deployed Kozha
translator at https://kozha-translate.com/app.html. Two prior fixes
have landed: the audio/video upload pipeline now decodes
MOV/MP4/WebM/MP3/WAV via ffmpeg.wasm, and the microphone speech
model loads via an importmap with a WASM fallback when WebGPU is
missing.

This task verifies both fixes work on the live deployment, not just
in local dev.

Single objective: a Playwright spec that exercises both inputs end
to end and asserts the user-visible outcomes the bugs masked. The
spec must run after deploy completes.

Do not change application code. Do not touch the avatar. Do not
refactor anything you find in the test path.

## Test framework

Playwright (TypeScript) preferred. Pin the dependency:

    @playwright/test @ 1.49.1

If the repo's test convention is Python + pytest, use
playwright-python @ 1.49.0 at the same path with `.py`. Do not
introduce a second test framework alongside the existing one.

Browsers: install only the Chromium build that ships with that
Playwright version. Firefox and WebKit are out of scope for this
spec — the Firefox WebGPU-absence path is exercised by manual
verification plus the unit test in the prior fix prompt.

## Test path

`tests/translator_input_e2e.spec.ts` (or `.py`).

## Step 1 — fixtures

Reuse the fixtures created during the upload-decode fix:

    tests/fixtures/audio-video/iphone-sample.mov   (1 s, QuickTime)
    tests/fixtures/audio-video/sample.wav          (1 s, PCM 16 kHz)

If they don't exist on disk when the test runs, generate them with
ffmpeg in a beforeAll hook:

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a aac \
           -movflags +faststart \
           tests/fixtures/audio-video/iphone-sample.mov

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a pcm_s16le \
           -ac 1 -ar 16000 \
           tests/fixtures/audio-video/sample.wav

## Step 2 — upload-tab assertion

In the spec:

    1. Launch Chromium with `--autoplay-policy=no-user-gesture-required`
       so any inline media can decode without a synthetic click.
    2. await page.goto('https://kozha-translate.com/app.html', {
           waitUntil: 'networkidle' });
    3. Switch to the Audio / Video Upload tab. Selector by role +
       accessible name; if the tab uses a class, fall back to that.
    4. Set the file input to
       `tests/fixtures/audio-video/iphone-sample.mov` via
       page.setInputFiles.
    5. Click the Process / Process to Sign button.
    6. Wait up to 30 s for the transcript element to be visible AND
       non-empty.
    7. Assertions:
         - transcript is non-empty
         - transcript does not contain "[object Event]"
         - transcript does not contain "error" or "Error"
         - the page console did not log any uncaught error during
           the run (collect via page.on('pageerror', ...); expect 0)

## Step 3 — microphone-tab assertion

Microphone needs a fake media stream. Playwright Chromium accepts
these flags:

    --use-fake-ui-for-media-stream
    --use-fake-device-for-media-stream
    --use-file-for-fake-audio-capture=<absolute path to sample.wav>

Set them on browser launch options:

    chromium.launch({
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-audio-capture=${path.resolve(
            'tests/fixtures/audio-video/sample.wav')}`,
      ],
    });

Then in the spec:

    1. await page.goto(...);
    2. Switch to the Microphone Input tab.
    3. Click Record. The fake audio stream begins.
    4. Wait 1.5 s (the fixture is 1 s; give the recorder a tick).
    5. Click Stop, then click Transcribe.
    6. Wait up to 30 s for the transcript element to populate.
    7. Assertions:
         - transcript is non-empty
         - transcript does not contain "[object Event]"
         - the page console did not log any module-resolve error,
           any "Failed to load resource" for an onnxruntime URL,
           or any "WebGPU is not supported" thrown error
         - if a "Using the WASM speech model" UI note is present
           that's fine — it's the documented fallback path, not an
           error

## Step 4 — wait for deploy

Run the test only after the deploy finishes. If your CI has a
deploy-finished event, hook into that. If not, poll
`https://kozha-translate.com/healthz` (or the equivalent) for the
expected git SHA from `git rev-parse HEAD` before running the spec
— a simple loop with a 5 s sleep, max 5 minutes, is sufficient.

## Acceptance criteria

  1. The spec passes against the deployed site for both upload and
     microphone paths.
  2. No console error or page error during the run, with the sole
     exception of the documented WASM-fallback UI note (which must
     be styled as a hint, not an error).
  3. Both transcript assertions produce non-empty, non-error
     strings within their 30 s windows.

## Reminders

  - Pin Playwright and the Chromium browser bundle. No `@latest`.
  - Run the spec against the live deployment URL, not localhost.
    The whole point is to verify the production path.
  - Do not change application code. If a test failure points to a
    real bug, file it and stop — do not fix it in this prompt.
  - The text-input pipeline is out of scope.
  - The avatar is out of scope.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "test(translator): e2e smoke for upload + mic on deployed site" && git push
