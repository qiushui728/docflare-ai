/**
 * Indexing pipeline: sitemap → queue (D1) → crawl → chunk → embed → Vectorize.
 *
 * Work is done in small batches so each invocation stays inside the Workers Free plan CPU
 * budget. Batches are driven by the cron trigger and/or POST /api/sites/:siteId/process.
 */
import type { Env, ExtractedPage, IndexProgress, TextChunk } from '../types';
import { Db, type PageRow } from '../db/schema';
import { pageIdFor } from '../utils/hash';
import { AiRateLimitError, embedTexts } from './ai';
import { chunkText, extractPage, fetchSitemapUrls, truncateToTokens } from './crawler';
import { chunkVectorId, deletePageVectors, deletePagesVectors, upsertChunks, type ChunkVector } from './vectorize';

export interface RegisterResult {
  siteId: string;
  domain: string;
  sitemapUrl: string;
  /** URLs discovered in the sitemap and queued for crawling. */
  queued: number;
}

export interface PageResult {
  url: string;
  status: 'indexed' | 'failed';
  title?: string;
  chunks?: number;
  error?: string;
}

export interface ProcessResult {
  processed: number;
  indexed: number;
  failed: number;
  /** True when Workers AI rate limiting stopped the batch early; remaining pages stay queued. */
  rateLimited: boolean;
  results: PageResult[];
}

/** Register (or refresh) a site and enqueue every HTML URL found in its sitemap. */
export async function registerSite(
  env: Env,
  siteId: string,
  sitemapUrl: string,
  domain: string,
  maxPages: number,
): Promise<RegisterResult> {
  const db = new Db(env.DB);
  const urls = await fetchSitemapUrls(sitemapUrl, { allowedHost: domain, maxUrls: maxPages });

  await db.upsertSite(siteId, domain, sitemapUrl);
  const pages = await Promise.all(urls.map(async (url) => ({ id: await pageIdFor(url), url })));
  await db.enqueuePages(siteId, pages);

  return { siteId, domain, sitemapUrl, queued: pages.length };
}

/**
 * Token budget for the title / description prepended to embedding inputs. Together with CHUNK_TOKENS
 * (350) this keeps every embedding input under bge-small's 512-token window, also for Vietnamese or
 * CJK titles that tokenize into many word pieces.
 */
const EMBED_TITLE_TOKENS = 40;
const EMBED_DESCRIPTION_TOKENS = 80;

export interface ChunkRecord {
  index: number;
  /** Text stored in Vectorize metadata and shown to the LLM. */
  text: string;
  /** Text actually embedded. */
  embedInput: string;
}

/**
 * Decide what is stored and what is embedded for each chunk.
 * The meta description is prepended to the first chunk and the page title to every embedding input:
 * landing/about copy rarely spells out the site name, so without this "what is <site>?" retrieves the
 * legal pages (which do) instead of the About page. The stored text keeps the full description; the
 * embedding input gets token-capped copies so the chunk itself is never truncated by the model.
 */
export function buildChunkRecords(page: ExtractedPage, chunks: TextChunk[]): ChunkRecord[] {
  const description = page.description.trim();
  const embedTitle = truncateToTokens(page.title.trim(), EMBED_TITLE_TOKENS);
  const embedDescription = truncateToTokens(description, EMBED_DESCRIPTION_TOKENS);
  return chunks.map((chunk, i) => {
    const first = i === 0 && description.length > 0;
    const text = first ? `${description}\n\n${chunk.text}` : chunk.text;
    const body = first && embedDescription ? `${embedDescription}\n\n${chunk.text}` : chunk.text;
    return { index: chunk.index, text, embedInput: embedTitle ? `${embedTitle}\n\n${body}` : body };
  });
}

/** Crawl, chunk, embed and store a single page. */
export async function indexPage(env: Env, page: PageRow): Promise<{ title: string; chunks: number }> {
  const extracted = await extractPage(page.url);
  const chunks = chunkText(extracted.text);
  if (chunks.length === 0) throw new Error('No text chunks produced');

  const records = buildChunkRecords(extracted, chunks);
  const embeddings = await embedTexts(
    env.AI,
    records.map((r) => r.embedInput),
  );

  const vectors: ChunkVector[] = records.map((record, i) => ({
    id: chunkVectorId(page.id, record.index),
    values: embeddings[i] as number[],
    metadata: {
      siteId: page.site_id,
      pageId: page.id,
      url: page.url,
      title: extracted.title,
      chunk: record.index,
      text: record.text,
    },
  }));

  await upsertChunks(env.VECTOR_INDEX, page.site_id, vectors);

  // A re-crawled page may have shrunk; remove stale trailing vectors from the previous run.
  if (page.chunk_count > chunks.length) {
    await deletePageVectors(env.VECTOR_INDEX, page.id, chunks.length, page.chunk_count);
  }

  return { title: extracted.title, chunks: chunks.length };
}

/**
 * Process the next batch of pending pages (for one site, or across all sites when siteId is null).
 * Failures are recorded per page; a Workers AI rate limit stops the batch and leaves pages queued.
 */
export async function processPendingPages(
  env: Env,
  siteId: string | null,
  batchSize: number,
): Promise<ProcessResult> {
  const db = new Db(env.DB);
  const pages = await db.nextPending(siteId, batchSize);
  const result: ProcessResult = { processed: 0, indexed: 0, failed: 0, rateLimited: false, results: [] };

  for (const page of pages) {
    try {
      const { title, chunks } = await indexPage(env, page);
      await db.markPageIndexed(page.id, title, chunks);
      result.indexed++;
      result.results.push({ url: page.url, status: 'indexed', title, chunks });
    } catch (err) {
      if (err instanceof AiRateLimitError) {
        // Leave the page queued; the next cron tick will retry it.
        result.rateLimited = true;
        console.warn(JSON.stringify({ event: 'index.rate_limited', url: page.url }));
        break;
      }
      const message = err instanceof Error ? err.message : String(err);
      await db.markPageFailed(page.id, message);
      result.failed++;
      result.results.push({ url: page.url, status: 'failed', error: message });
      console.warn(JSON.stringify({ event: 'index.page_failed', url: page.url, error: message }));
    }
    result.processed++;
  }

  if (result.processed > 0) {
    const { results: _omit, ...summary } = result;
    console.log(JSON.stringify({ event: 'index.batch', siteId, ...summary }));
  }
  return result;
}

/** Remove a site: its vectors, pages, chat history and the site row itself. */
export async function removeSite(env: Env, siteId: string): Promise<{ vectorsDeleted: number }> {
  const db = new Db(env.DB);
  const pages = await db.listPageVectors(siteId);
  const vectorsDeleted = await deletePagesVectors(env.VECTOR_INDEX, pages);
  await db.deleteSite(siteId);
  return { vectorsDeleted };
}

export async function getProgress(env: Env, siteId: string): Promise<IndexProgress> {
  return new Db(env.DB).progress(siteId);
}
