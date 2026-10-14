/**
 * D1 data access layer: row interfaces, SQL statements and a thin repository class.
 * Schema lives in migrations/0000_init.sql.
 */
import type { IndexProgress } from '../types';

export type PageStatus = 'pending' | 'indexed' | 'failed';

export interface SiteRow {
  id: string;
  domain: string;
  sitemap_url: string;
  created_at: string;
}

export interface PageRow {
  id: string;
  site_id: string;
  url: string;
  title: string | null;
  chunk_count: number;
  status: PageStatus;
  error: string | null;
  updated_at: string;
}

export interface QueryRow {
  id: string;
  site_id: string;
  prompt: string;
  response: string | null;
  created_at: string;
}

/** All SQL in one place so it is easy to audit. Positional parameters use ?1, ?2, ... */
export const SQL = {
  upsertSite: `
    INSERT INTO sites (id, domain, sitemap_url)
    VALUES (?1, ?2, ?3)
    ON CONFLICT (id) DO UPDATE SET domain = excluded.domain, sitemap_url = excluded.sitemap_url`,
  getSite: `SELECT * FROM sites WHERE id = ?1`,
  listSites: `SELECT * FROM sites ORDER BY created_at DESC`,
  deleteSite: `DELETE FROM sites WHERE id = ?1`,

  /** Insert a page or reset an existing one back to the queue. */
  enqueuePage: `
    INSERT INTO pages (id, site_id, url, status, error, updated_at)
    VALUES (?1, ?2, ?3, 'pending', NULL, CURRENT_TIMESTAMP)
    ON CONFLICT (id) DO UPDATE SET status = 'pending', error = NULL, updated_at = CURRENT_TIMESTAMP`,
  nextPendingForSite: `
    SELECT * FROM pages WHERE site_id = ?1 AND status = 'pending' ORDER BY updated_at ASC LIMIT ?2`,
  nextPendingAny: `
    SELECT * FROM pages WHERE status = 'pending' ORDER BY updated_at ASC LIMIT ?1`,
  markPageIndexed: `
    UPDATE pages SET title = ?2, chunk_count = ?3, status = 'indexed', error = NULL, updated_at = CURRENT_TIMESTAMP
    WHERE id = ?1`,
  markPageFailed: `
    UPDATE pages SET status = 'failed', error = ?2, updated_at = CURRENT_TIMESTAMP WHERE id = ?1`,
  pageStats: `SELECT status, COUNT(*) AS n FROM pages WHERE site_id = ?1 GROUP BY status`,
  pageStatsAll: `SELECT site_id, status, COUNT(*) AS n FROM pages GROUP BY site_id, status`,
  listPageVectors: `SELECT id, chunk_count FROM pages WHERE site_id = ?1 AND chunk_count > 0`,
  deletePagesForSite: `DELETE FROM pages WHERE site_id = ?1`,

  insertQuery: `INSERT INTO queries (id, site_id, prompt, response) VALUES (?1, ?2, ?3, ?4)`,
  listQueries: `
    SELECT id, site_id, prompt, response, created_at FROM queries
    WHERE site_id = ?1 ORDER BY created_at DESC LIMIT ?2`,
  deleteQueriesForSite: `DELETE FROM queries WHERE site_id = ?1`,
} as const;

/** D1 limits the number of bound parameters per statement; keep batches modest. */
const BATCH_SIZE = 50;

export class Db {
  constructor(private readonly d1: D1Database) {}

  // --- sites ---------------------------------------------------------------

  async upsertSite(id: string, domain: string, sitemapUrl: string): Promise<void> {
    await this.d1.prepare(SQL.upsertSite).bind(id, domain, sitemapUrl).run();
  }

  async getSite(id: string): Promise<SiteRow | null> {
    return this.d1.prepare(SQL.getSite).bind(id).first<SiteRow>();
  }

  async listSites(): Promise<SiteRow[]> {
    const { results } = await this.d1.prepare(SQL.listSites).all<SiteRow>();
    return results;
  }

  /** Remove a site together with its pages and chat history. */
  async deleteSite(id: string): Promise<void> {
    await this.d1.batch([
      this.d1.prepare(SQL.deleteQueriesForSite).bind(id),
      this.d1.prepare(SQL.deletePagesForSite).bind(id),
      this.d1.prepare(SQL.deleteSite).bind(id),
    ]);
  }

  // --- pages ---------------------------------------------------------------

  /** Queue URLs for crawling. Existing rows are reset to "pending" so re-indexing refreshes content. */
  async enqueuePages(siteId: string, pages: Array<{ id: string; url: string }>): Promise<void> {
    for (let i = 0; i < pages.length; i += BATCH_SIZE) {
      const slice = pages.slice(i, i + BATCH_SIZE);
      await this.d1.batch(slice.map((p) => this.d1.prepare(SQL.enqueuePage).bind(p.id, siteId, p.url)));
    }
  }

  async nextPending(siteId: string | null, limit: number): Promise<PageRow[]> {
    const stmt = siteId
      ? this.d1.prepare(SQL.nextPendingForSite).bind(siteId, limit)
      : this.d1.prepare(SQL.nextPendingAny).bind(limit);
    const { results } = await stmt.all<PageRow>();
    return results;
  }

  async markPageIndexed(id: string, title: string, chunkCount: number): Promise<void> {
    await this.d1.prepare(SQL.markPageIndexed).bind(id, title, chunkCount).run();
  }

  async markPageFailed(id: string, error: string): Promise<void> {
    await this.d1.prepare(SQL.markPageFailed).bind(id, error.slice(0, 500)).run();
  }

  async progress(siteId: string): Promise<IndexProgress> {
    const { results } = await this.d1
      .prepare(SQL.pageStats)
      .bind(siteId)
      .all<{ status: PageStatus; n: number }>();
    const counts: Record<PageStatus, number> = { pending: 0, indexed: 0, failed: 0 };
    for (const row of results) counts[row.status] = row.n;
    return {
      siteId,
      total: counts.pending + counts.indexed + counts.failed,
      pending: counts.pending,
      indexed: counts.indexed,
      failed: counts.failed,
    };
  }

  /** Progress for every site in one query, keyed by site id. */
  async progressAll(): Promise<Map<string, IndexProgress>> {
    const { results } = await this.d1
      .prepare(SQL.pageStatsAll)
      .all<{ site_id: string; status: PageStatus; n: number }>();
    const map = new Map<string, IndexProgress>();
    for (const row of results) {
      const entry = map.get(row.site_id) ?? { siteId: row.site_id, total: 0, pending: 0, indexed: 0, failed: 0 };
      entry[row.status] = row.n;
      entry.total += row.n;
      map.set(row.site_id, entry);
    }
    return map;
  }

  async listPageVectors(siteId: string): Promise<Array<{ id: string; chunk_count: number }>> {
    const { results } = await this.d1
      .prepare(SQL.listPageVectors)
      .bind(siteId)
      .all<{ id: string; chunk_count: number }>();
    return results;
  }

  // --- queries -------------------------------------------------------------

  async logQuery(id: string, siteId: string, prompt: string, response: string): Promise<void> {
    await this.d1.prepare(SQL.insertQuery).bind(id, siteId, prompt, response).run();
  }

  async listQueries(siteId: string, limit: number): Promise<QueryRow[]> {
    const { results } = await this.d1.prepare(SQL.listQueries).bind(siteId, limit).all<QueryRow>();
    return results;
  }
}
