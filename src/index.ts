/**
 * DocFlare AI - Hono application entrypoint.
 *
 * Routes:
 *   GET    /api/health                   liveness + model info
 *   GET    /api/sites                    list sites with indexing progress            (admin)
 *   POST   /api/sites/index              register a site + queue its sitemap URLs   (admin)
 *   POST   /api/sites/:siteId/process    crawl + embed the next batch of pages      (admin)
 *   GET    /api/sites/:siteId/status     indexing progress                           (admin)
 *   GET    /api/sites/:siteId/queries    recent chat log                             (admin)
 *   DELETE /api/sites/:siteId            remove a site and its vectors               (admin)
 *   POST   /api/chat                     RAG answer (JSON or SSE stream)             (public, origin-checked)
 *   GET    /widget.js                    embeddable widget (also served as a static asset)
 *   GET    /admin                        admin UI (static asset public/admin.html)
 */
import { Hono } from 'hono';
import type { Context, MiddlewareHandler } from 'hono';
import { Db, type SiteRow } from './db/schema';
import {
  AiRateLimitError,
  CHAT_MODEL,
  EMBEDDING_MODEL,
  RERANK_MODEL,
  buildRagMessages,
  embedQuery,
  generateAnswer,
  generateAnswerStream,
  iterateWorkersAiTokens,
  rerankChunks,
  textToWorkersAiStream,
  toSources,
} from './services/ai';
import { CrawlError } from './services/crawler';
import { getProgress, processPendingPages, registerSite, removeSite } from './services/indexer';
import { MAX_TOP_K, diversify, searchChunks } from './services/vectorize';
import { cors, isOriginAllowed, parseAllowedOrigins } from './utils/cors';
import { timingSafeEqual } from './utils/hash';
import {
  HttpError,
  type AppBindings,
  type AppConfig,
  type ChatMessage,
  type ChatRequest,
  type ChatResponse,
  type Env,
  type IndexSiteRequest,
  type Source,
} from './types';

const SITE_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/i;
const MAX_QUESTION_CHARS = 1000;
const MAX_HISTORY_ENTRIES = 10;
const NO_CONTEXT_ANSWER =
  "I couldn't find anything about that in this documentation. Try rephrasing your question or using different keywords.";
/** Without the reranker, over-fetch this many times TOP_K so the per-page cap still has candidates. */
const OVERFETCH_FACTOR = 3;

// ---------------------------------------------------------------------------
// Configuration helpers
// ---------------------------------------------------------------------------

function intVar(value: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number.parseInt(value ?? '', 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function boolVar(value: string | undefined, fallback: boolean): boolean {
  const v = value?.trim().toLowerCase();
  if (!v) return fallback;
  return !['0', 'false', 'no', 'off'].includes(v);
}

function loadConfig(env: Env): AppConfig {
  return {
    maxPages: intVar(env.MAX_PAGES, 200, 1, 5000),
    indexBatchSize: intVar(env.INDEX_BATCH_SIZE, 3, 1, 50),
    topK: intVar(env.TOP_K, 4, 1, 20),
    rerank: boolVar(env.RERANK, true),
    allowedOrigins: parseAllowedOrigins(env.ALLOWED_ORIGINS),
  };
}

async function readJson<T>(c: Context<AppBindings>): Promise<T> {
  try {
    return (await c.req.json()) as T;
  } catch {
    throw new HttpError(400, 'Request body must be valid JSON', 'invalid_json');
  }
}

function requireSiteId(value: unknown): string {
  if (typeof value !== 'string' || !SITE_ID_PATTERN.test(value)) {
    throw new HttpError(
      400,
      'siteId must be 1-64 characters of letters, numbers, "-" or "_"',
      'invalid_site_id',
    );
  }
  return value.toLowerCase();
}

function requireHttpUrl(value: unknown, field: string): URL {
  if (typeof value !== 'string') throw new HttpError(400, `${field} is required`, 'invalid_url');
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new HttpError(400, `${field} must be an absolute URL`, 'invalid_url');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new HttpError(400, `${field} must use http or https`, 'invalid_url');
  }
  return url;
}

function sanitizeHistory(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (m): m is ChatMessage =>
        !!m &&
        typeof m === 'object' &&
        (m.role === 'user' || m.role === 'assistant') &&
        typeof m.content === 'string' &&
        m.content.trim().length > 0,
    )
    .slice(-MAX_HISTORY_ENTRIES);
}

async function requireSite(db: Db, siteId: string): Promise<SiteRow> {
  const site = await db.getSite(siteId);
  if (!site) throw new HttpError(404, `Site "${siteId}" is not registered`, 'site_not_found');
  return site;
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/** Bearer-token guard for management routes. */
const requireAdmin: MiddlewareHandler<AppBindings> = async (c, next) => {
  const expected = c.env.ADMIN_TOKEN;
  if (!expected) {
    throw new HttpError(
      503,
      'ADMIN_TOKEN is not configured. Run `bunx wrangler secret put ADMIN_TOKEN` (or add it to .dev.vars).',
      'admin_token_missing',
    );
  }
  const header = c.req.header('Authorization') ?? '';
  const token = header.startsWith('Bearer ') ? header.slice(7).trim() : '';
  if (!token || !timingSafeEqual(token, expected)) {
    throw new HttpError(401, 'Invalid or missing admin token', 'unauthorized');
  }
  await next();
};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const app = new Hono<AppBindings>();

app.use('*', cors());

app.onError((err, c) => {
  if (err instanceof HttpError) {
    return c.json({ error: err.message, code: err.code }, err.status as 400);
  }
  if (err instanceof AiRateLimitError) {
    c.header('Retry-After', '60');
    return c.json({ error: err.message, code: 'ai_rate_limited' }, 429);
  }
  if (err instanceof CrawlError) {
    return c.json({ error: err.message, code: 'crawl_error' }, 422);
  }
  if (err.name === 'AiError') {
    // Workers AI failures (model deprecated, unavailable, bad input) are surfaced instead of a bare 500.
    console.error(JSON.stringify({ event: 'ai_error', message: err.message }));
    return c.json({ error: `Workers AI error: ${err.message}`, code: 'ai_error' }, 502);
  }
  console.error(JSON.stringify({ event: 'unhandled_error', message: err.message, stack: err.stack }));
  return c.json({ error: 'Internal server error', code: 'internal_error' }, 500);
});

app.notFound((c) => c.json({ error: 'Not found', code: 'not_found' }, 404));

app.get('/api/health', (c) =>
  c.json({
    ok: true,
    service: 'docflare-ai',
    models: { embedding: EMBEDDING_MODEL, chat: CHAT_MODEL, rerank: loadConfig(c.env).rerank ? RERANK_MODEL : null },
    time: new Date().toISOString(),
  }),
);

/**
 * Widget script. In production the static-assets layer usually serves this before the Worker
 * runs; this route is the fallback (e.g. custom routing) and adds explicit caching/CORS headers.
 */
app.get('/widget.js', async (c) => {
  const asset = await c.env.ASSETS.fetch(new Request(new URL('/widget.js', c.req.url), { method: 'GET' }));
  if (!asset.ok) throw new HttpError(404, 'widget.js not found in assets', 'not_found');
  return new Response(asset.body, {
    status: 200,
    headers: {
      'Content-Type': 'application/javascript; charset=utf-8',
      'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
      'Access-Control-Allow-Origin': '*',
    },
  });
});

// --- Management routes -------------------------------------------------------

const sites = new Hono<AppBindings>();
sites.use('*', requireAdmin);

/** All registered sites with their indexing progress (powers the admin UI). */
sites.get('/', async (c) => {
  const db = new Db(c.env.DB);
  const [rows, progress] = await Promise.all([db.listSites(), db.progressAll()]);
  return c.json({
    sites: rows.map((site) => ({
      ...site,
      progress: progress.get(site.id) ?? { siteId: site.id, total: 0, pending: 0, indexed: 0, failed: 0 },
    })),
  });
});

/** Register a site and queue its sitemap for crawling. */
sites.post('/index', async (c) => {
  const config = loadConfig(c.env);
  const body = await readJson<Partial<IndexSiteRequest>>(c);

  const siteId = requireSiteId(body.siteId);
  const sitemap = requireHttpUrl(body.sitemapUrl, 'sitemapUrl');
  const domain = (typeof body.domain === 'string' && body.domain.trim()
    ? body.domain.trim().replace(/^https?:\/\//, '').replace(/[/:].*$/, '')
    : sitemap.hostname
  ).toLowerCase();

  const registered = await registerSite(c.env, siteId, sitemap.toString(), domain, config.maxPages);

  const inline = body.process === true ? await processPendingPages(c.env, siteId, config.indexBatchSize) : null;
  const progress = await getProgress(c.env, siteId);

  return c.json(
    {
      ...registered,
      progress,
      processed: inline,
      next:
        progress.pending > 0
          ? `Pending pages are crawled ${config.indexBatchSize} at a time by the cron trigger. ` +
            `Call POST /api/sites/${siteId}/process repeatedly to finish faster.`
          : 'All pages indexed.',
    },
    202,
  );
});

/** Crawl + embed the next batch of pending pages for a site. */
sites.post('/:siteId/process', async (c) => {
  const config = loadConfig(c.env);
  const siteId = requireSiteId(c.req.param('siteId'));
  await requireSite(new Db(c.env.DB), siteId);

  const batch = await processPendingPages(c.env, siteId, config.indexBatchSize);
  const progress = await getProgress(c.env, siteId);
  return c.json({ ...batch, progress }, batch.rateLimited ? 429 : 200);
});

/** Indexing progress for a site. */
sites.get('/:siteId/status', async (c) => {
  const siteId = requireSiteId(c.req.param('siteId'));
  const site = await requireSite(new Db(c.env.DB), siteId);
  const progress = await getProgress(c.env, siteId);
  return c.json({ site, progress });
});

/** Recent questions and answers for a site (newest first). */
sites.get('/:siteId/queries', async (c) => {
  const siteId = requireSiteId(c.req.param('siteId'));
  const db = new Db(c.env.DB);
  await requireSite(db, siteId);
  const limit = intVar(c.req.query('limit'), 20, 1, 100);
  const queries = await db.listQueries(siteId, limit);
  return c.json({ siteId, queries });
});

/** Delete a site, its pages, chat history and vectors. */
sites.delete('/:siteId', async (c) => {
  const siteId = requireSiteId(c.req.param('siteId'));
  await requireSite(new Db(c.env.DB), siteId);
  const result = await removeSite(c.env, siteId);
  return c.json({ siteId, deleted: true, ...result });
});

app.route('/api/sites', sites);

// --- Chat --------------------------------------------------------------------

/** Encode one custom SSE event. */
function sseEvent(event: string, data: unknown): Uint8Array {
  return new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * Re-emit Workers AI tokens as our own SSE protocol:
 *   event: meta   {id, sources}
 *   event: token  {text}
 *   event: done   {id}
 *   event: error  {message}
 * The full answer is collected and handed to `onComplete` for persistence.
 */
function buildChatStream(
  id: string,
  sources: Source[],
  aiStream: ReadableStream<Uint8Array>,
  onComplete: (answer: string) => Promise<void>,
): { stream: ReadableStream<Uint8Array>; finished: Promise<void> } {
  let resolveFinished: () => void = () => {};
  const finished = new Promise<void>((resolve) => {
    resolveFinished = resolve;
  });

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let answer = '';
      controller.enqueue(sseEvent('meta', { id, sources }));
      try {
        for await (const token of iterateWorkersAiTokens(aiStream)) {
          answer += token;
          controller.enqueue(sseEvent('token', { text: token }));
        }
        controller.enqueue(sseEvent('done', { id }));
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Stream interrupted';
        controller.enqueue(sseEvent('error', { message }));
      } finally {
        controller.close();
        try {
          await onComplete(answer.trim());
        } finally {
          resolveFinished();
        }
      }
    },
  });

  return { stream, finished };
}

app.post('/api/chat', async (c) => {
  const config = loadConfig(c.env);
  const body = await readJson<Partial<ChatRequest>>(c);

  const siteId = requireSiteId(body.siteId);
  const question = typeof body.question === 'string' ? body.question.trim() : '';
  if (!question) throw new HttpError(400, 'question is required', 'invalid_question');
  if (question.length > MAX_QUESTION_CHARS) {
    throw new HttpError(400, `question must be at most ${MAX_QUESTION_CHARS} characters`, 'invalid_question');
  }
  const history = sanitizeHistory(body.history);
  const wantsStream = body.stream === true;

  const db = new Db(c.env.DB);
  const site = await requireSite(db, siteId);

  // Only the site's own domain (plus ALLOWED_ORIGINS) may embed the widget.
  const origin = c.req.header('Origin');
  if (!isOriginAllowed(origin, site.domain, config.allowedOrigins)) {
    throw new HttpError(403, `Origin ${origin} is not allowed for site "${siteId}"`, 'origin_not_allowed');
  }

  // Retrieval: vector candidates → (cross-encoder rerank) → at most MAX_CHUNKS_PER_PAGE per page.
  const queryVector = await embedQuery(c.env.AI, question);
  const candidateCount = config.rerank ? MAX_TOP_K : Math.min(config.topK * OVERFETCH_FACTOR, MAX_TOP_K);
  const candidates = await searchChunks(c.env.VECTOR_INDEX, siteId, queryVector, candidateCount);
  const ranked = config.rerank ? await rerankChunks(c.env.AI, question, candidates) : candidates;
  const chunks = diversify(ranked, config.topK);
  const sources = toSources(chunks);
  const id = crypto.randomUUID();
  const persist = (answer: string) => db.logQuery(id, siteId, question, answer);

  // Generation. Skip the LLM entirely when nothing relevant was retrieved (saves quota).
  if (wantsStream) {
    const aiStream =
      chunks.length === 0
        ? textToWorkersAiStream(NO_CONTEXT_ANSWER)
        : await generateAnswerStream(c.env.AI, buildRagMessages(question, chunks, history, site.domain));

    const { stream, finished } = buildChatStream(id, sources, aiStream, persist);
    c.executionCtx.waitUntil(finished);

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  }

  const answer =
    chunks.length === 0
      ? NO_CONTEXT_ANSWER
      : await generateAnswer(c.env.AI, buildRagMessages(question, chunks, history, site.domain));

  c.executionCtx.waitUntil(persist(answer));
  const response: ChatResponse = { id, answer, sources };
  return c.json(response);
});

// ---------------------------------------------------------------------------
// Worker export: HTTP + cron
// ---------------------------------------------------------------------------

export default {
  fetch: app.fetch,

  /** Cron trigger: crawl the next batch of pending pages across all sites. */
  async scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    const config = loadConfig(env);
    ctx.waitUntil(
      processPendingPages(env, null, config.indexBatchSize).catch((err: unknown) => {
        console.error(JSON.stringify({ event: 'cron.failed', message: err instanceof Error ? err.message : String(err) }));
      }),
    );
  },
} satisfies ExportedHandler<Env>;
