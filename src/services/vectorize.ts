/**
 * Vectorize wrapper. Each site gets its own namespace, which gives tenant isolation
 * without requiring a metadata index for filtering.
 */
import type { ChunkMetadata, RetrievedChunk } from '../types';

/** Vectorize accepts up to 1000 vectors per upsert; stay conservative to respect payload limits. */
const UPSERT_BATCH_SIZE = 250;
/** Vectorize returns metadata for at most 20 matches when returnMetadata = "all". */
export const MAX_TOP_K = 20;
/** A single page may fill at most this many of the topK slots (keeps sources varied). */
export const MAX_CHUNKS_PER_PAGE = 2;

export interface ChunkVector {
  id: string;
  values: number[];
  metadata: ChunkMetadata;
}

/** Deterministic vector id: `<pageId>-<chunkIndex>` (max 64 bytes). */
export function chunkVectorId(pageId: string, chunkIndex: number): string {
  return `${pageId}-${chunkIndex}`;
}

/** Insert or replace chunk vectors for a site. */
export async function upsertChunks(index: VectorizeIndex, siteId: string, vectors: ChunkVector[]): Promise<number> {
  let written = 0;
  for (let i = 0; i < vectors.length; i += UPSERT_BATCH_SIZE) {
    const batch: VectorizeVector[] = vectors.slice(i, i + UPSERT_BATCH_SIZE).map((v) => ({
      id: v.id,
      values: v.values,
      namespace: siteId,
      metadata: v.metadata,
    }));
    await index.upsert(batch);
    written += batch.length;
  }
  return written;
}

/**
 * Keep the best `topK` chunks while allowing at most `perPage` chunks from the same URL.
 * Input must be sorted by score (descending); order is preserved. When the cap leaves slots
 * empty (few distinct pages matched), the best remaining chunks fill them.
 */
export function diversify<T extends { metadata: { url: string } }>(
  chunks: T[],
  topK: number,
  perPage: number = MAX_CHUNKS_PER_PAGE,
): T[] {
  const perUrl = new Map<string, number>();
  const picked = new Set<T>();
  for (const chunk of chunks) {
    if (picked.size >= topK) break;
    const n = perUrl.get(chunk.metadata.url) ?? 0;
    if (n >= perPage) continue;
    perUrl.set(chunk.metadata.url, n + 1);
    picked.add(chunk);
  }
  for (const chunk of chunks) {
    if (picked.size >= topK) break;
    picked.add(chunk);
  }
  return chunks.filter((c) => picked.has(c));
}

/**
 * Similarity search restricted to one site's namespace. Returns up to `limit` (max MAX_TOP_K)
 * candidates sorted by cosine score; callers rerank and/or `diversify` them.
 */
export async function searchChunks(
  index: VectorizeIndex,
  siteId: string,
  vector: number[],
  limit: number,
): Promise<RetrievedChunk[]> {
  const result = await index.query(vector, {
    topK: Math.min(Math.max(1, limit), MAX_TOP_K),
    namespace: siteId,
    returnValues: false,
    returnMetadata: 'all',
  });

  const chunks: RetrievedChunk[] = [];
  for (const match of result.matches) {
    const metadata = match.metadata as Partial<ChunkMetadata> | undefined;
    if (!metadata || typeof metadata.text !== 'string' || typeof metadata.url !== 'string') continue;
    chunks.push({
      id: match.id,
      score: match.score,
      metadata: {
        siteId: String(metadata.siteId ?? siteId),
        pageId: String(metadata.pageId ?? ''),
        url: metadata.url,
        title: String(metadata.title ?? ''),
        chunk: Number(metadata.chunk ?? 0),
        text: metadata.text,
      },
    });
  }
  return chunks;
}

/** Delete the vectors of a page in the id range [from, to). */
export async function deletePageVectors(
  index: VectorizeIndex,
  pageId: string,
  from: number,
  to: number,
): Promise<void> {
  if (to <= from) return;
  const ids: string[] = [];
  for (let i = from; i < to; i++) ids.push(chunkVectorId(pageId, i));
  await index.deleteByIds(ids);
}

/** Delete every vector that belongs to a list of pages (used when removing a site). */
export async function deletePagesVectors(
  index: VectorizeIndex,
  pages: Array<{ id: string; chunk_count: number }>,
): Promise<number> {
  const ids: string[] = [];
  for (const page of pages) {
    for (let i = 0; i < page.chunk_count; i++) ids.push(chunkVectorId(page.id, i));
  }
  for (let i = 0; i < ids.length; i += UPSERT_BATCH_SIZE) {
    await index.deleteByIds(ids.slice(i, i + UPSERT_BATCH_SIZE));
  }
  return ids.length;
}
