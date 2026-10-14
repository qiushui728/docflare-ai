/**
 * Workers AI helpers: embeddings, RAG prompt construction and text generation.
 * No external API keys - everything runs through the `AI` binding.
 */
import type { ChatMessage, RetrievedChunk, Source } from '../types';

export const EMBEDDING_MODEL = '@cf/baai/bge-small-en-v1.5';
// `@cf/meta/llama-3.1-8b-instruct` was deprecated by Cloudflare on 2026-05-30 (AiError 5028); the fp8 build is the drop-in.
export const CHAT_MODEL = '@cf/meta/llama-3.1-8b-instruct-fp8';
/** Cross-encoder used to re-order the vector-search candidates before they reach the LLM. */
export const RERANK_MODEL = '@cf/baai/bge-reranker-base';
export const EMBEDDING_DIMENSIONS = 384;

/** bge-small accepts up to 100 texts per call. */
const EMBED_BATCH_SIZE = 100;
/** Longest passage handed to the reranker (its window is ~512 tokens for query + passage). */
const RERANK_MAX_CHARS = 2000;
/** Generation settings tuned for factual, grounded answers. */
const MAX_OUTPUT_TOKENS = 768;
const TEMPERATURE = 0.2;
/** Keep the prompt well inside the model's ~8k token context window. */
const MAX_HISTORY_TURNS = 6;
const MAX_HISTORY_CHARS = 1500;
/**
 * Appended to the user's question. Small models follow the language of the chat history (or of the
 * excerpts) over a system rule; a reminder right next to the question is what reliably works.
 */
export const LANGUAGE_REMINDER = '(Answer in the same language as this question.)';

/** Minimal structural view of the AI binding, decoupled from model-specific generics. */
type AiRunner = {
  run(model: string, inputs: Record<string, unknown>): Promise<unknown>;
};

type EmbeddingOutput = { shape?: number[]; data?: number[][] };
/**
 * Text generation output. Llama-family models return `{ response }`; OpenAI-compatible models
 * (GLM, Gemma 4, gpt-oss, ...) return `{ choices: [{ message: { content } }] }` and stream
 * `{ choices: [{ delta: { content } }] }`. Both shapes are accepted so CHAT_MODEL can be swapped.
 */
type TextGenerationOutput = {
  response?: string;
  choices?: Array<{ message?: { content?: string | null }; delta?: { content?: string | null } }>;
};

/** Answer text from either output shape ("" when absent). */
export function outputText(output: TextGenerationOutput | undefined): string {
  if (!output) return '';
  if (typeof output.response === 'string') return output.response;
  const choice = output.choices?.[0];
  return choice?.message?.content ?? choice?.delta?.content ?? '';
}

/** Raised when Workers AI reports a quota / rate limit problem. Mapped to HTTP 429. */
export class AiRateLimitError extends Error {
  constructor(message = 'Workers AI rate limit or daily quota reached. Please try again later.') {
    super(message);
    this.name = 'AiRateLimitError';
  }
}

function isRateLimitError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  // Workers AI error codes 3040 (rate limited) / 3036 (account limited) plus generic phrasing.
  return /rate.?limit|too many requests|\b429\b|\b3040\b|\b3036\b|daily limit|quota|capacity/i.test(msg);
}

function wrapAiError(err: unknown): never {
  if (isRateLimitError(err)) throw new AiRateLimitError();
  throw err instanceof Error ? err : new Error(String(err));
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/** Embed a list of texts (batched). Returns one 384-dim vector per input, in order. */
export async function embedTexts(ai: Ai, texts: string[]): Promise<number[][]> {
  const runner = ai as unknown as AiRunner;
  const vectors: number[][] = [];

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    let output: EmbeddingOutput;
    try {
      output = (await runner.run(EMBEDDING_MODEL, { text: batch })) as EmbeddingOutput;
    } catch (err) {
      wrapAiError(err);
    }
    const data = output.data;
    if (!Array.isArray(data) || data.length !== batch.length) {
      throw new Error(`Embedding model returned ${data?.length ?? 0} vectors for ${batch.length} inputs`);
    }
    for (const vec of data) {
      if (vec.length !== EMBEDDING_DIMENSIONS) {
        throw new Error(`Unexpected embedding size ${vec.length}, expected ${EMBEDDING_DIMENSIONS}`);
      }
      vectors.push(vec);
    }
  }
  return vectors;
}

/** Embed a single query string. */
export async function embedQuery(ai: Ai, text: string): Promise<number[]> {
  const [vector] = await embedTexts(ai, [text]);
  if (!vector) throw new Error('Embedding model returned no vector');
  return vector;
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

/** bge-reranker output: one relevance score (sigmoid, 0-1) per context index. */
type RerankOutput = { response?: Array<{ id?: number; score?: number }> };

/**
 * Re-order candidates with the reranker's scores. Pure and defensive: malformed or empty output
 * leaves the input order untouched; contexts the model did not score keep their vector order after
 * the scored ones. Each returned chunk carries the reranker score in `score`.
 */
export function applyRerank(chunks: RetrievedChunk[], output: RerankOutput | undefined): RetrievedChunk[] {
  const entries = output?.response;
  if (!Array.isArray(entries)) return chunks;
  const scored: RetrievedChunk[] = [];
  const seen = new Set<number>();
  for (const entry of entries) {
    const id = entry?.id;
    const score = entry?.score;
    const chunk = typeof id === 'number' ? chunks[id] : undefined;
    if (!chunk || typeof score !== 'number' || !Number.isFinite(score) || seen.has(id as number)) continue;
    seen.add(id as number);
    scored.push({ ...chunk, score });
  }
  if (scored.length === 0) return chunks;
  scored.sort((a, b) => b.score - a.score);
  chunks.forEach((chunk, i) => {
    if (!seen.has(i)) scored.push(chunk);
  });
  return scored;
}

/**
 * Second retrieval stage: score every candidate against the question with a cross-encoder and sort
 * by that score. Bi-encoder similarity alone tends to favour chunks that merely share vocabulary
 * with the question (e.g. the site name on legal pages); the reranker reads question and passage
 * together and promotes the paragraph that actually answers it. Any failure (rate limit, model
 * error, odd output) falls back to the vector order so chat keeps working.
 */
export async function rerankChunks(ai: Ai, query: string, chunks: RetrievedChunk[]): Promise<RetrievedChunk[]> {
  if (chunks.length < 2) return chunks;
  const runner = ai as unknown as AiRunner;
  let output: RerankOutput;
  try {
    output = (await runner.run(RERANK_MODEL, {
      query,
      contexts: chunks.map((chunk) => ({ text: chunk.metadata.text.slice(0, RERANK_MAX_CHARS) })),
      top_k: chunks.length,
    })) as RerankOutput;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.warn(JSON.stringify({ event: 'rerank.failed', message }));
    return chunks;
  }
  return applyRerank(chunks, output);
}

// ---------------------------------------------------------------------------
// RAG prompt
// ---------------------------------------------------------------------------

/**
 * Curated per-site fact sheet injected into the system prompt (distilled from the customer-service
 * knowledge base, 2026-10-14). Retrieval alone can rank the wrong section (e.g. a question about the
 * personal tiers pulling the institutional tier); these authoritative essentials sit next to the
 * rules so pricing/contacts/boundaries survive any ranking wobble. Matched by domain substring.
 */
const SITE_KB_ESSENTIALS: Array<{ match: string; facts: string }> = [
  {
    match: 'tangoera',
    facts: `- TangoEra is a ~90-second, 6-question assessment producing 7-dimension decision-profile charts: self-reflection and decision support, not a verdict. Never medical/psychological/legal/financial/investment advice; not astrology or fortune-telling; no predictions of future results, no guarantees; statistics show weak correlations only.
- Personal tiers (USD, one-time): Free Chart $0 (matrix + 1 of 7 dimensions, no email required). Starter $49. Full 9-page Profile $199. Standard $399 (4-page snapshot, 7-dim radar, matrix, energy roles, Entry Experience, 3 short-term actions). In-Depth $699 (complete 9-page profile, Bias Spectrum, heatmap, cohort comparison, 30-day actions, free redo within 30 days). Tracking $30/month. Coaching $999 per session (+$250/hr extra). First 20 users get the $49 Starter free with code FEEDBACK20.
- Advisory and Institutional programs are separate B2B offerings, NOT the personal tiers: Advisory from $4,800, Institutional from $49,800, by invitation, scoped in a signed engagement letter.
- Delivery: free snapshot ~2 minutes (2-page PDF by email, check spam). Paid reports delivered 24-48h after valid birth data; every conclusion carries a trace (input + rule).
- Refunds: full refund within 14 days if not delivered (EU withdrawal right); one free redo within 30 days; accuracy refund - if any conclusion cannot be traced, full refund automatically within 48h of the request (reply to the delivery email with the order number).
- Contact: support@tangoera.com (privacy: privacy@tangoera.com); a person reads every message, typically replies within 24 hours; Mon-Fri 9:00am-6:00pm EST.`,
  },
  {
    match: 'acrosstar',
    facts: `- Acrosstar offers 1:1 classical Chinese chrono-psychology (BaZi / GanZhi) consultations by an independent practitioner, for overseas Chinese, founders and tech/creative professionals. It maps innate patterns, talents and life-cycle rhythm for decision insight; not fatalism, no 100%-accuracy claims - only coherent, sincere analysis. No mystique, no fear-selling, no talisman upsells; strictly confidential.
- Tiers (USD, one-time, transparent): Standard $399 - 60-minute 1:1 session (video / voice / IM), personal archetype decoding, current timing-window reading, written synthesis report within 48 hours. In-Depth $699 - adds couple/partnership dynamics reading, multi-year cycle mapping, and a 3-month written Q&A window (included, no extra charge). No hidden repurchases, no discount-code games, no fear-then-pay-more "fixes".
- Booking: leave the form (name, email, tier, format, optional topic) -> pay online (WeChat Pay / Stripe / international cards / bank transfer) -> schedule and format confirmed within 24 hours. Leave birth year-month-day-hour and the key question; the practitioner analyses in advance.
- Formats: video (Zoom / Google Meet), voice (WeChat / Tencent Meeting), or text (Telegram / WhatsApp).
- Birth hour unclear? Year-month-day alone still yields a 75%+ outline of core personality and cycles; 2-3 landmark life events calibrate the hour; completely hourless charts still allow macro-level analysis.
- Contact / direct chat: Telegram @TangoEra_bot; WhatsApp +1 626 781 3770 (David); WeChat ID wxid_2pg1e4g80m5j22; X @webb591673. 7-day no-questions-asked refund promise; 256-bit SSL.
- Never: fatalistic pronouncements ("destined to..."), prescriptive medical/legal/financial advice, talisman or "disaster-fix" product sales, scare readings, or promises to change real-world outcomes.`,
  },
];

/** Resolve the fact sheet for a registered site domain ('' when the domain is unknown). */
export function kbEssentialsFor(domain: string): string {
  const hit = SITE_KB_ESSENTIALS.find((e) => domain.toLowerCase().includes(e.match));
  return hit ? `\nSite knowledge essentials (authoritative facts about this site - use them even when the excerpts are incomplete, and prefer them over any excerpt that conflicts):\n${hit.facts}\n` : '';
}

const SYSTEM_PROMPT = `You are DocFlare, a friendly and precise documentation assistant for {{DOMAIN}}.

Rules:
- Answer ONLY using the documentation excerpts and the site knowledge essentials below. Never guess or use outside knowledge.
- Reply in the language of the user's question (English question -> English answer, Vietnamese question -> Vietnamese answer), even when the excerpts are written in a different language. Translate the facts you use.
- Answer directly, as if you knew the documentation yourself. Do not quote or list the excerpts, do not mention "the context", "the excerpts" or "the provided information", and do not add citation markers such as [1]. The sources are shown to the user separately.
- If neither the excerpts nor the essentials contain the answer, say so briefly and suggest what the user could search for instead.
- Never invent URLs, commands, flags, settings or version numbers that are not in the excerpts or the essentials.
- Be concise. Use Markdown: short paragraphs, bullet lists, and fenced code blocks for commands or code.
{{KB_ESSENTIALS}}
Documentation excerpts:
{{CONTEXT}}`;

/** Convert retrieved chunks into sources (1-based, relevance order, same order as the context). */
export function toSources(chunks: RetrievedChunk[]): Source[] {
  return chunks.map((chunk, i) => ({
    index: i + 1,
    url: chunk.metadata.url,
    title: chunk.metadata.title || chunk.metadata.url,
    score: Number(chunk.score.toFixed(4)),
  }));
}

/** Build the chat messages array for the generation model. */
export function buildRagMessages(
  question: string,
  chunks: RetrievedChunk[],
  history: ChatMessage[] = [],
  domain = 'this website',
): Array<{ role: 'system' | 'user' | 'assistant'; content: string }> {
  const context =
    chunks.length === 0
      ? '(no relevant documentation found)'
      : chunks
          .map((chunk) => {
            const title = chunk.metadata.title || chunk.metadata.url;
            return `Source: ${title} (${chunk.metadata.url})\n${chunk.metadata.text}`;
          })
          .join('\n\n---\n\n');

  const system = SYSTEM_PROMPT.replace('{{DOMAIN}}', domain)
    .replace('{{KB_ESSENTIALS}}', kbEssentialsFor(domain))
    .replace('{{CONTEXT}}', context);

  const trimmedHistory = history
    .slice(-MAX_HISTORY_TURNS)
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_HISTORY_CHARS) }));

  return [
    { role: 'system', content: system },
    ...trimmedHistory,
    { role: 'user', content: `${question}\n\n${LANGUAGE_REMINDER}` },
  ];
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/** Generate a complete answer (non-streaming). */
export async function generateAnswer(
  ai: Ai,
  messages: Array<{ role: string; content: string }>,
): Promise<string> {
  const runner = ai as unknown as AiRunner;
  try {
    const output = (await runner.run(CHAT_MODEL, {
      messages,
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: TEMPERATURE,
    })) as TextGenerationOutput;
    return outputText(output).trim();
  } catch (err) {
    wrapAiError(err);
  }
}

/**
 * Generate an answer as a stream. Workers AI returns Server-Sent Events of the form
 * `data: {"response":"token"}` terminated by `data: [DONE]`.
 */
export async function generateAnswerStream(
  ai: Ai,
  messages: Array<{ role: string; content: string }>,
): Promise<ReadableStream<Uint8Array>> {
  const runner = ai as unknown as AiRunner;
  try {
    const output = await runner.run(CHAT_MODEL, {
      messages,
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: TEMPERATURE,
      stream: true,
    });
    if (!(output instanceof ReadableStream)) {
      // Some environments ignore `stream: true`; fall back to a one-shot stream.
      const text = outputText(output as TextGenerationOutput).trim();
      return textToWorkersAiStream(text);
    }
    return output as ReadableStream<Uint8Array>;
  } catch (err) {
    wrapAiError(err);
  }
}

/** Wrap a plain string into the same SSE shape Workers AI emits, for uniform downstream handling. */
export function textToWorkersAiStream(text: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ response: text })}\n\n`));
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });
}

/**
 * Extract the `response` tokens from a Workers AI SSE byte stream and yield them as strings.
 * Handles chunk boundaries that split a line.
 */
export async function* iterateWorkersAiTokens(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newline: number;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return;
        try {
          const parsed = JSON.parse(payload) as TextGenerationOutput;
          const token = outputText(parsed);
          if (token) yield token;
        } catch {
          // Ignore malformed keep-alive lines.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
