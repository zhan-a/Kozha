ultrathink

You are fixing Bug A1 in the Kozha translator: every error the
audio/video upload code path catches is shown to the user as the
literal string "[object Event]". The user reproduces by uploading
IMG_1314.mov from an iPhone (QuickTime container with H.264 video
and AAC audio). The actual underlying error is masked, so we cannot
see whether the failure is a codec issue, a worker error, a fetch
failure, or something else.

Single objective: replace every broken error stringification in the
audio/video upload code path with a `describeError` helper, then
verify the IMG_1314.mov reproduction now shows a specific, actionable
message.

This prompt does not address Bug A2 (the underlying decode failure)
or Bug B (the microphone model loading). Do not touch the text-input
pipeline. Do not touch the avatar.

## Step 1 — create the helper

Create `public/lib/describe-error.js` exporting a single function:

    export function describeError(err) { ... }

The helper must produce a useful, short, human-readable string for
any of these inputs without ever returning "[object Event]" or
"[object Object]":

    - Error instances: use .message; fall back to .name
    - DOMException: return `${name}: ${message}`
    - Event instances:
        * If the event target is a media element (HTMLMediaElement,
          AudioContext) and `target.error` is set, decode the
          MediaError code (1=aborted, 2=network, 3=decode,
          4=src-not-supported) into a short human phrase.
        * If the event is an ErrorEvent, return
          `${message} (at ${filename}:${lineno})` when those fields
          exist.
        * If the event is a MessageEvent from a worker that posted a
          `{type: "error", message: string}` payload, surface the
          message field.
        * Otherwise return `${type} event` (e.g. "abort event").
    - AbortError / AbortSignal triggered: return "operation aborted".
    - Plain strings: return as-is.
    - Plain objects with .message: return the message field.
    - null / undefined: return "unknown error".
    - Anything else: return Object.prototype.toString.call(err)
      stripped of the leading `[object ` and trailing `]`, lowercased.
      Never the raw coercion.

Add a JSDoc comment summarising the contract. Vanilla JS, no
dependencies, no transpilation. The file must be loadable both as an
ES module (`import { describeError } from "/lib/describe-error.js";`)
and via a `<script type="module">`.

## Step 2 — replace broken stringifications

Find every site in the audio/video upload code path that currently
stringifies a caught value into a user-visible string. Patterns to
look for:

    catch (e) { someUiSink(String(e)); }
    catch (e) { someUiSink("" + e); }
    catch (e) { someUiSink(`${e}`); }
    catch (e) { someUiSink(e.toString()); }
    catch (e) { alert(e); }
    fileReader.onerror = e => someUiSink(e);
    worker.onerror = e => someUiSink(e);
    audioElement.onerror = e => someUiSink(e);

For each, replace with `someUiSink(describeError(e))`. Import
describeError at the top of the module (ESM) or via a synchronous
script tag (legacy). Do not change the surrounding control flow.

Stay in the audio/video upload code path. Do not touch:

    - the text-input pipeline (it works)
    - the microphone tab (Bug B is a separate fix)
    - the avatar
    - any unrelated catch block elsewhere in the codebase

## Step 3 — reproduce and verify

Reproduction recipe:

    1. Open the translator page in Chrome stable.
    2. Switch to the Audio / Video Upload tab.
    3. Upload an iPhone .mov file (any IMG_NNNN.mov works as a
       fixture; if you don't have one, generate one with
       `ffmpeg -f lavfi -i sine=frequency=440:duration=1 -c:a aac \
        -movflags +faststart sample.mov`).
    4. Click Process.

Before the fix the page shows "[object Event]" or similar opaque
string. After the fix the page shows a specific message. Examples
of what counts as specific:

    - "media decode failed (codec unsupported?)"
    - "decode timeout after 10s"
    - "worker exited with code 1"
    - "fetch /models/whisper-tiny.onnx failed: 404"
    - "AbortError: signal aborted without reason"

If the message is still opaque, your describeError didn't cover that
input shape — extend it until the IMG_1314.mov reproduction surfaces
something a developer can act on.

## Acceptance criteria

  1. `public/lib/describe-error.js` exists, exports describeError,
     handles every input shape listed in Step 1.
  2. Every broken stringification in the audio/video upload code
     path is routed through describeError.
  3. The IMG_1314.mov reproduction in Chrome stable surfaces a
     specific, non-`[object *]` message.

## Reminders

  - Do not address Bug A2 (the underlying decode). The next prompt
    does that. Your job is the error reporting only — IMG_1314.mov
    can still fail, just with a useful message.
  - Do not touch the text-input pipeline.
  - Do not touch the avatar.
  - Pin no third-party dependency in this prompt; the helper is
    vanilla JS.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "fix(translator): wire describeError into upload error sinks" && git push
