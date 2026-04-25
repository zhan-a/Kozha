/**
 * End-to-end smoke for the contribute pipeline against the deployed
 * site. Prompt 07 of the contrib-fix Worker-1 sequence.
 *
 * One round-trip: deploy-wait → open contribute → BSL → describe →
 * avatar renders → chip swap → submit → status URL resolves to a page
 * showing the gloss text from the description.
 *
 * Default base URL is the production domain. Override with
 * KOZHA_BASE_URL for staging / local testing. The expected commit
 * for the deploy-wait gate comes from KOZHA_EXPECTED_SHA, falling back
 * to ``git rev-parse origin/main``.
 *
 * Skip rather than fail when the production backend has no LLM key
 * (the contributor flow can't draft a SiGML, so the avatar never
 * renders) — see the explicit ``LLMConfigError`` / ``llm_no_key``
 * branches in describeAndWaitForRender.
 */
import { test, expect, Page, Response } from '@playwright/test';
import { execSync } from 'node:child_process';

const BASE_URL = (process.env.KOZHA_BASE_URL || 'https://kozha-translate.com').replace(/\/+$/, '');
const DEPLOY_RECEIPT_URL = `${BASE_URL}/data/last_deploy.json`;

const DEPLOY_WAIT_MS = 8 * 60_000;       // hard cap for the deploy-wait stage
const DEPLOY_POLL_MS = 5_000;
const RENDER_WAIT_MS = 60_000;
const CORRECTION_WAIT_MS = 30_000;
const SUBMIT_WAIT_MS = 30_000;

const GLOSS = 'HELLO';
const DESCRIPTION = 'the sign for hello — wave a flat hand near the temple';

type StageName =
  | 'deploy wait'
  | 'language pick'
  | 'generation'
  | 'correction'
  | 'submit'
  | 'status URL';

class StageError extends Error {
  constructor(public stage: StageName, message: string) {
    super(`[${stage}] ${message}`);
    this.name = 'StageError';
  }
}

function expectedSha(): string {
  const env = (process.env.KOZHA_EXPECTED_SHA || '').trim();
  if (env) return env;
  // origin/main is what GitHub Actions deploys; HEAD will match after a
  // ``git push origin main``. The repo is the working dir for this spec.
  return execSync('git rev-parse origin/main', { encoding: 'utf8' }).trim();
}

async function waitForDeploy(target: string, deadline: number): Promise<void> {
  const target7 = target.slice(0, 7);
  let last: { status?: number; sha?: string; raw?: string } = {};
  while (Date.now() < deadline) {
    try {
      const res = await fetch(DEPLOY_RECEIPT_URL, { cache: 'no-store' });
      last.status = res.status;
      if (res.ok) {
        const body = (await res.json()) as { sha?: string; status?: string };
        last.sha = body.sha;
        if (body.sha === target) return;
      } else {
        last.raw = await res.text().catch(() => '');
      }
    } catch (err) {
      last.raw = (err as Error).message;
    }
    await sleep(DEPLOY_POLL_MS);
  }
  throw new StageError(
    'deploy wait',
    `deploy did not complete within ${Math.round(DEPLOY_WAIT_MS / 1000)}s — ` +
      `expected sha ${target7}, last receipt sha=${(last.sha || '').slice(0, 7)} ` +
      `httpStatus=${last.status} body=${(last.raw || '').slice(0, 200)}`,
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

interface DescribeOutcome {
  ok: boolean;
  noLlmKey: boolean;
  generationErrors: string[];
  raw?: any;
}

function captureDescribeOutcome(page: Page): Promise<DescribeOutcome> {
  // Register the response watcher BEFORE the click that fires the POST.
  // Returns a (non-awaited) promise so the caller can fire the click in
  // between and only ``await`` after.
  //
  // The pipeline can fail two different ways for the no-LLM-key skip:
  //   1. /sessions itself returns 4xx with ``error.code = llm_no_key``
  //      (or ``llm_config_error``) — when the create+describe chain
  //      runs the describe inline.
  //   2. /sessions returns 200 and POST /sessions/<id>/describe returns
  //      200 with ``generation_errors`` carrying ``LLMConfigError``.
  // We watch both URLs and resolve on whichever lands first with
  // information.
  const seen = page.waitForResponse(
    (resp) => {
      const url = resp.url();
      const method = resp.request().method();
      if (method !== 'POST') return false;
      if (/\/api\/chat2hamnosys\/sessions\/[^/]+\/describe$/.test(url)) return true;
      if (/\/api\/chat2hamnosys\/sessions(\?.*)?$/.test(url)) {
        // Only treat /sessions as the terminal frame when it's a hard
        // failure — a 200 means the caller will follow up with /describe
        // and we should keep waiting for that.
        return !resp.ok();
      }
      return false;
    },
    { timeout: RENDER_WAIT_MS },
  );
  return (async () => {
    let resp: Response | null = null;
    try {
      resp = await seen;
    } catch {
      return { ok: false, noLlmKey: false, generationErrors: [] };
    }
    let body: any = null;
    try {
      body = await resp.json();
    } catch {
      body = null;
    }
    const errors = Array.isArray(body?.generation_errors) ? body.generation_errors : [];
    const errCode: string = body?.error?.code || '';
    const detail: string = body?.detail || body?.error?.message || '';
    const errorBlob = (errors.join('\n') + '\n' + errCode + '\n' + detail).toLowerCase();
    const noLlmKey =
      errCode === 'llm_no_key' ||
      errCode === 'llm_config_error' ||
      errorBlob.includes('llmconfigerror') ||
      errorBlob.includes('openai api key') ||
      errorBlob.includes('llm_no_key');
    return { ok: resp.ok(), noLlmKey, generationErrors: errors, raw: body };
  })();
}

async function chipStripFingerprint(page: Page): Promise<string> {
  return await page.evaluate(() => {
    const strip = document.getElementById('sigmlAnnotated');
    if (!strip) return '';
    const chips = Array.from(strip.querySelectorAll('.sigml-chip'));
    return chips
      .map((c) => (c as HTMLElement).getAttribute('data-tag-name') || '')
      .join('|');
  });
}

test('contribute pipeline round-trip on the deployed site', async ({ page }, testInfo) => {
  // Cap the whole test (excluding deploy-wait) to ~5 min as the spec
  // requires. Playwright's per-test timeout includes setup, so size it
  // generously above the sum of the per-stage timeouts.
  testInfo.setTimeout(DEPLOY_WAIT_MS + 5 * 60_000);

  const networkLog: string[] = [];
  page.on('response', (resp) => {
    const url = resp.url();
    if (url.includes('/api/chat2hamnosys/') || url.includes('/data/last_deploy.json')) {
      networkLog.push(`${resp.request().method()} ${resp.status()} ${url}`);
    }
  });
  page.on('requestfailed', (req) => {
    networkLog.push(`FAIL ${req.method()} ${req.url()} — ${req.failure()?.errorText || ''}`);
  });

  try {
    await test.step('deploy wait', async () => {
      const target = expectedSha();
      const deadline = Date.now() + DEPLOY_WAIT_MS;
      await waitForDeploy(target, deadline);
    });

    await test.step('language pick', async () => {
      await page.goto(`${BASE_URL}/contribute.html`, { waitUntil: 'domcontentloaded' });
      // The visible primary picker is a chip strip; clicking a chip
      // forwards to the hidden #pickerSelect and triggers the language
      // header to mount.
      const bslChip = page.locator('.qs__chip[data-lang="bsl"]');
      await expect(bslChip).toBeVisible({ timeout: 15_000 });
      await bslChip.click();
      await expect(page.locator('#langMasthead')).toBeVisible({ timeout: 10_000 });
      await expect(page.locator('#authoring-root')).toBeVisible({ timeout: 10_000 });
    });

    let describeOutcome: DescribeOutcome = { ok: false, noLlmKey: false, generationErrors: [] };

    await test.step('generation', async () => {
      await page.locator('#glossInput').fill(GLOSS);
      await page.locator('#descriptionInput').fill(DESCRIPTION);
      await expect(page.locator('#startAuthoringBtn')).toBeEnabled({ timeout: 5_000 });

      const describeP = captureDescribeOutcome(page);
      await page.locator('#startAuthoringBtn').click();
      describeOutcome = await describeP;

      if (describeOutcome.noLlmKey) {
        // Production has no OPENAI_API_KEY (or it's exhausted). The
        // pipeline parks the draft in awaiting_description with a
        // generation_errors entry; the avatar never plays. Per the
        // prompt we skip rather than report a failure.
        test.skip(
          true,
          `backend reports llm_no_key — generation_errors=${JSON.stringify(describeOutcome.generationErrors)}`,
        );
      }
      if (!describeOutcome.ok) {
        throw new StageError(
          'generation',
          `describe POST did not return OK; generation_errors=${JSON.stringify(describeOutcome.generationErrors)}`,
        );
      }

      // The avatar mount sets data-rendered="true" on the first
      // animactive hook fire from CWASA — see public/contribute-preview.js.
      const canvas = page.locator('#avatarCanvas');
      await expect(canvas).toBeVisible();
      await expect(canvas).toHaveAttribute('data-rendered', 'true', {
        timeout: RENDER_WAIT_MS,
      });
    });

    await test.step('correction', async () => {
      const before = await chipStripFingerprint(page);
      if (!before) {
        throw new StageError('correction', 'chip strip never populated with .sigml-chip elements');
      }

      const firstChip = page.locator('#sigmlAnnotated .sigml-chip').first();
      await expect(firstChip).toBeVisible({ timeout: 10_000 });
      await firstChip.click();

      const picker = page.locator('#sigmlPicker');
      await expect(picker).toBeVisible({ timeout: 5_000 });

      // First non-current option in the picker. Clicking it fires
      // CTX.correct() which posts /correct and updates the SiGML in
      // CTX, which re-renders the chip strip.
      const altOption = page.locator('#sigmlPickerList .sigml-picker-btn:not(.is-current)').first();
      await expect(altOption).toBeVisible({ timeout: 5_000 });
      await altOption.click();

      // Wait for the chip strip to differ from the snapshot we took
      // before the swap. The annotated view re-renders on every
      // CTX.setState({ sigml }) — observed within hundreds of ms in
      // practice, with a 30s budget for backend round-trip.
      await expect
        .poll(
          async () => {
            const after = await chipStripFingerprint(page);
            return after && after !== before ? 'changed' : 'same';
          },
          { timeout: CORRECTION_WAIT_MS, intervals: [200, 500, 1000] },
        )
        .toBe('changed');
    });

    let permanentUrl = '';
    let sessionId = '';

    await test.step('submit', async () => {
      const submitBtn = page.locator('#submissionSubmitBtn');
      await expect(submitBtn).toBeEnabled({ timeout: SUBMIT_WAIT_MS });
      await submitBtn.click();

      const confirmation = page.locator('#submissionConfirmation');
      await expect(confirmation).toBeVisible({ timeout: SUBMIT_WAIT_MS });
      await expect(page.locator('#authoring-root')).toBeHidden();

      permanentUrl = (await page.locator('#confirmationUrl').inputValue()).trim();
      const m = permanentUrl.match(/\/contribute\/status\/([^/?#]+)/);
      if (!m) {
        throw new StageError(
          'submit',
          `permanent URL did not match /contribute/status/<id>: ${permanentUrl}`,
        );
      }
      sessionId = m[1];
      expect(permanentUrl.startsWith(BASE_URL)).toBe(true);
    });

    await test.step('status URL', async () => {
      await page.goto(permanentUrl, { waitUntil: 'domcontentloaded' });
      await expect(page.locator('#statusBody')).toBeVisible({ timeout: 30_000 });
      await expect(page.locator('#statusGloss')).toHaveText(GLOSS, { timeout: 10_000 });
      // The token was written to localStorage at submit time, so the
      // private description should be visible after navigation.
      await expect(page.locator('#statusPrivate')).toBeVisible({ timeout: 10_000 });
      await expect(page.locator('#statusDescription')).toContainText(DESCRIPTION);
      // Defensive: the URL path carries the same id we extracted.
      expect(sessionId.length).toBeGreaterThan(0);
    });
  } catch (err) {
    if (networkLog.length) {
      console.error('--- captured network log ---');
      for (const line of networkLog) console.error(line);
      console.error('--- end captured network log ---');
    }
    throw err;
  }
});
