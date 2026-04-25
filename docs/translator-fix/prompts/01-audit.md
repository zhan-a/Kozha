ultrathink

You are auditing the Kozha translator's input pipeline before any fixes
land. Two production bugs are open:

  Bug A — uploading IMG_1314.mov (an iPhone clip, QuickTime container
          holding H.264 video and AAC audio) produces an error the user
          sees as the literal string "[object Event]". The upload tab
          shows nothing else; the transcript box stays empty.

  Bug B — switching to the Microphone Input tab on Firefox 150 (Linux,
          no WebGPU) fires a module-resolve error in the console. The
          speech-recognition model never loads and the Transcribe
          button is functionally dead.

This task is read-only with exactly one exception: you write the audit
document. Do not change application code. Do not touch the avatar. Do
not refactor the text-input pipeline; it works and is out of scope.

## Files to enumerate

In `docs/translator-fix/01-audit.md`, list every file involved in:

  1. Audio/video upload: the file `<input>`, the JS module that reads
     the file, any web worker that decodes or transcribes, the speech-
     recognition module the audio is handed to, and the server endpoint
     if a server round-trip happens.
  2. Microphone capture: the getUserMedia call, the recorder, any
     encoder, the speech-recognition module.
  3. Speech recognition itself: name the model (Whisper-tiny?
     Moonshine? Distil-Whisper? something else?) and the model file
     location (CDN URL, local /models path, lazy-imported package).

For each file: `path/from/repo/root.js` plus a one-line description of
its role. Group by tab (Upload, Microphone) and indicate shared code.

## Specific facts to find

  - The exact file:line where the `[object Event]` string is produced.
    Grep the audio/video code path for these patterns and list every
    call site:
        `String(<caught-error>)`
        `"" + <caught-error>`
        `` `${<caught-error>}` ``
        `<caught-error>.toString()`
        `alert(<caught-error>)`
        `someUiSink(<caught-error>)` where the sink stringifies
    A non-Error caught value (Event, MessageEvent, ErrorEvent) coerces
    to "[object Event]" via any of the above. Find every such site in
    the upload pipeline.
  - The current onnxruntime-web import strategy. Record:
        - whether the page uses an importmap (search HTML for
          `<script type="importmap">`)
        - whether it uses a bare specifier (`from 'onnxruntime-web'`),
          an absolute URL, or a relative URL
        - the version that resolves today (or "unpinned" if none)
        - whether the import is dynamic (`await import(...)`) or
          static (top-of-file `import`)
  - Which speech model is loaded. State the model name, the file
    location (URL or local path), and the size on disk if known.
    Note whether the path is wrapped behind a feature flag.
  - Whether ffmpeg.wasm is already a dependency. Search package.json,
    importmaps, and `<script src=...>` tags. Record the version if
    present, "not present" if not.
  - For each browser x format combination, mark works / fails. Use a
    small markdown table. Browsers: Chrome stable, Firefox stable,
    Safari stable. Formats: MOV (iPhone), MP4, WebM, MP3, WAV. Plus
    a row for mic capture in each browser.

## Recommended fix strategy

For Bug A, recommend the concrete decoding strategy in two sentences.
The QuickTime container that iPhone clips ship in is not handled by
AudioContext.decodeAudioData; the audio track must be extracted first
(ffmpeg.wasm, MediaSource demux, or a worker that pulls the AAC and
hands it to a Web Codecs AudioDecoder). Pick one, justify, and name
the package version you would pin to.

For Bug B, recommend the concrete import strategy in two sentences.
Firefox today does not expose `navigator.gpu`, so any onnxruntime-web
backend that requires WebGPU is not loadable there; the WASM backend
must be the fallback. Pick: importmap, absolute URL, or both. Name
the onnxruntime-web version you would pin to (a specific number, not
"latest").

## Reminders

  - The text-input pipeline already works. Do not enumerate its
    files, do not recommend changes to it.
  - The avatar is out of scope.
  - This task does not change application code. The only file you
    create or modify is `docs/translator-fix/01-audit.md`.
  - If you discover a third bug while auditing, document it but do
    not fix it; this prompt does not authorise code changes.

## Output budget

250-500 lines in the audit doc. Use H2 section headings per topic
above, bullets under each, monospace for paths, plain markdown
tables.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "docs(translator-fix): audit input pipeline" && git push
