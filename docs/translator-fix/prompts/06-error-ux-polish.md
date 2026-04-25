ultrathink

You are polishing the translator's user-facing error UX. Three prior
fixes have landed: the audio/video upload pipeline decodes MOV / MP4
/ WebM / MP3 / WAV via ffmpeg.wasm, the mic speech model loads via
an importmap with a WASM fallback, and errors no longer read
"[object Event]" — they surface a developer-facing message via
describeError.

This task is the last polish: turn those developer-facing messages
into one-sentence, plain-language explanations the user can act on.

Single objective: every user-visible error from the upload tab and
the microphone tab must read as a one-sentence sentence-case English
explanation with at least one concrete next step. No raw exception
strings, no codes, no jargon ("decodeAudioData", "AbortError",
"NotAllowedError" must not appear in user-facing copy).

Do not touch the text-input pipeline. Do not touch the avatar. Do
not add new design components — use existing tokens.css.

## Map of error categories to user copy

For the upload tab:

    Internal cue (from describeError)  User-visible sentence
    ---------------------------------- ----------------------------------
    media decode failed                "This file uses a codec your
    media format unsupported           browser can't decode in-page.
    codec unsupported                  Try MP4 or WebM, or upload an
                                       audio-only file like MP3."
    decode timeout                     "Decoding took too long. Try a
                                       shorter clip (under 60 seconds)."
    file too large                     "This file is over the 50 MB
                                       upload limit. Try a shorter clip
                                       or compress it first."
    fetch /models/* failed             "We couldn't load the speech
    fetch /onnxruntime-web/* failed    model. Check your connection
                                       and try again."
    worker exited / unreachable        "The browser worker that
                                       processes audio crashed. Reload
                                       the page and try again."
    operation aborted                  "Upload cancelled."
    network error                      "Network error while uploading.
                                       Check your connection and retry."
    anything else                      "Couldn't process this file.
                                       Try a different format or
                                       reload the page."

For the mic tab:

    Internal cue (from describeError)  User-visible sentence
    ---------------------------------- ----------------------------------
    NotAllowedError / permission       "Microphone access was blocked.
    denied                             Allow microphone access in your
                                       browser's site settings and
                                       reload."
    NotFoundError / no device          "No microphone found. Connect a
                                       microphone and reload the page."
    aborted by user                    "Recording cancelled."
    model load failed                  "We couldn't load the speech
                                       model. Check your connection
                                       and try again."
    transcribe failed                  "Couldn't transcribe this clip.
                                       Try recording again, or upload
                                       an audio file instead."
    WebGPU unavailable (note,          "Using the slower WASM speech
    not error)                         model — WebGPU isn't available
                                       in this browser."
    anything else                      "Microphone capture failed.
                                       Reload the page and try again."

## Where to wire it

Add a single mapper module:

    public/lib/error-copy.js

    export function userMessageForUpload(internalMsg) { ... }
    export function userMessageForMic(internalMsg) { ... }

Both take the developer-facing string produced by describeError (an
internal cue from the table above, lower-case substring) and return
a one-sentence user-facing sentence. Use a small switch / keyword-
match — not a regex hairball. The categories above are the universe;
the "anything else" rows are the catch-all.

At every UI sink in the upload pipeline, wrap the existing error
path so the user sees `userMessageForUpload(describeError(e))`
instead of the raw describeError output. Same for the mic pipeline
with `userMessageForMic`.

Keep the developer-facing string available too — log it with
`console.warn('[upload]', describeError(e))` / `console.warn('[mic]
', describeError(e))` so debugging is still possible. The user
copy is what the UI shows; the dev string is what the console
shows.

## Visual treatment

Use the existing error styling already in `tokens.css`. The error
row on the upload tab and the mic tab already exists — just change
what text goes into it. Do not introduce new components, new
colours, or new icons.

The WebGPU-fallback note is informational, not an error. Reuse the
existing hint / notice token, not the error token. If no hint token
exists, use a muted text colour from tokens.css; do not create a
new one.

## Acceptance criteria

  1. Every error message visible to the user in the upload tab and
     the mic tab is a one-sentence sentence-case English string
     with at least one concrete next step.
  2. No raw exception class name, no codec name, no module path
     appears in user-facing copy.
  3. The developer-facing message from describeError is still
     logged to the console for debugging.

## Reminders

  - Use existing tokens.css. No new design tokens, no new
    components.
  - Do not touch the text-input pipeline.
  - Do not touch the avatar.
  - Keep error sites short — one mapper call per sink, no
    scattered if/else chains in the UI code.
  - The mapper takes describeError output (already a clean string),
    not the raw caught value. Don't try to handle Event objects in
    the mapper; that's describeError's job.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "ui(translator): plain-language error copy for upload + mic" && git push
