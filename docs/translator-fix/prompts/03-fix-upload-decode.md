ultrathink

You are fixing Bug A2 in the Kozha translator: the audio/video upload
pipeline cannot decode iPhone IMG_NNNN.mov files (and likely fails on
any QuickTime container). The user uploads a .mov, the pipeline
catches an error, the upload returns no transcript. The error-
reporting fix is already in, so the user now sees a specific decode-
failure message — but the upload still doesn't produce a transcript.

Single objective: make uploads of MOV, MP4, WebM, MP3, and WAV files
all produce a transcript through the existing upload pipeline. Add
fixtures and a smoke test that round-trips all five formats locally.

Do not touch the text-input pipeline. Do not touch the microphone
pipeline (Bug B is a separate prompt). Do not touch the avatar.

## Why current decode fails

iPhone clips ship in a QuickTime container holding H.264 video and
AAC audio. The browser-native AudioContext.decodeAudioData rejects
this container outright — it accepts mp4, m4a, wav, ogg, webm in
most browsers, but not the .mov container variant. The audio track
has to be extracted before decode.

## Decode strategy

Use ffmpeg.wasm to extract the audio track to 16-bit PCM WAV at
16 kHz mono. That target is the safest input for in-browser speech
recognisers (Whisper, Moonshine, Distil-Whisper all accept it).
Pin to specific versions:

    @ffmpeg/ffmpeg @ 0.12.10
    @ffmpeg/util   @ 0.12.1

Both pinned exactly. Do not use `@latest`. Load via importmap or
absolute URL — match whatever strategy the rest of the page uses
for ES modules so visitors aren't paying for two import mechanisms.
Do NOT bundle ffmpeg.wasm into the page bundle: the wasm payload is
roughly 30 MB, it must lazy-load only on the first upload.

Implementation outline (the JS is yours; this is the shape):

    1. On first upload, dynamic-import @ffmpeg/ffmpeg and
       @ffmpeg/util. Cache the FFmpeg instance for subsequent
       uploads in the same session.
    2. Probe: `await ffmpeg.exec(['-i', input, '-f', 'null', '-'])`.
       If the probe shows no audio stream, surface "no audio track
       in this file" and stop.
    3. Extract:
         await ffmpeg.exec([
           '-i', input,
           '-vn',           // drop video
           '-ac', '1',      // mono
           '-ar', '16000',  // 16 kHz
           '-f', 'wav',
           output
         ]);
    4. Read the output WAV from FFmpeg's virtual filesystem and hand
       it to the existing decode-and-transcribe path (the path that
       already works for plain WAV uploads — do not duplicate it).

Cancellation: wrap the ffmpeg call with an AbortController so the
user can cancel and free the worker.

Timeouts: if extraction takes longer than 60 s on a 60 s clip,
abort and surface "decode timeout — file too long for in-browser
processing".

Use describeError (already in `public/lib/describe-error.js`) for
any new error sinks you add. Do not undo the prior fix.

## Fixtures

Add these files under `tests/fixtures/audio-video/`. Each must be
≤ 200 KB. Generate with ffmpeg if you don't have a real sample:

    iphone-sample.mov   QuickTime + H.264 + AAC, 1 second
    sample.mp4          MP4 + H.264 + AAC, 1 second
    sample.webm         WebM + VP8 + Opus, 1 second
    sample.mp3          MPEG audio, 1 second, 128 kbps
    sample.wav          PCM 16-bit, 16 kHz mono, 1 second

A 1-second 440 Hz sine tone is sufficient. Generation commands:

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a aac \
           -movflags +faststart \
           tests/fixtures/audio-video/iphone-sample.mov

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:v libx264 \
           -t 1 -pix_fmt yuv420p -c:a aac \
           tests/fixtures/audio-video/sample.mp4

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:v libvpx \
           -c:a libopus tests/fixtures/audio-video/sample.webm

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a libmp3lame \
           -b:a 128k tests/fixtures/audio-video/sample.mp3

    ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a pcm_s16le \
           -ac 1 -ar 16000 tests/fixtures/audio-video/sample.wav

Commit the binaries to the repo. They are small enough.

## Smoke test

Add `tests/translator_upload_smoke.ts` (TypeScript via Vitest is
preferred). If the repo's existing test framework is pytest, add the
equivalent at `tests/translator_upload_smoke.py` instead — match the
existing convention; do not introduce a second framework. Pin the
runner version (Vitest 2.1.8 or pytest 8.3.4 — record the pin in
package.json or pyproject.toml).

The test:

    1. Spins up the translator page in headless Chrome (Playwright
       1.49.1 pinned, no other browser engine).
    2. Switches to the Audio / Video Upload tab.
    3. For each of the five fixtures, uploads it via
       page.setInputFiles, clicks the Process button, waits up to
       30 s for the transcript element to be visible AND non-empty.
    4. Asserts the transcript is non-empty and does not contain
       "error", "Error", "Event", or "[object".

All five fixtures must pass. Partial passes are not acceptance.

## Acceptance criteria

  1. Uploading any of the five fixtures through the page produces a
     transcript without an error in the console.
  2. The smoke test under tests/translator_upload_smoke.{ts,py}
     passes locally for all five fixtures.
  3. ffmpeg.wasm is lazy-loaded on first upload, not at page boot.

## Reminders

  - Pin every third-party version explicitly. No `@latest`.
  - Do not touch the text-input pipeline, the microphone pipeline,
    or the avatar.
  - The error-reporting fix (Bug A1) is already in. Do not undo it;
    rely on describeError for any new error sites you add.
  - Lazy-load ffmpeg.wasm. The 30 MB cost must not hit the page on
    first paint.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "fix(translator): decode mov/mp4/webm/mp3/wav uploads via ffmpeg.wasm" && git push
