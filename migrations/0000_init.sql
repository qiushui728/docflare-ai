-- DocFlare AI - initial D1 schema
-- Apply with: npx wrangler d1 migrations apply docflare-db --remote

-- A "site" is one indexed documentation website (one tenant / one widget).
CREATE TABLE IF NOT EXISTS sites (
  id          TEXT PRIMARY KEY,                       -- user-chosen slug, e.g. "my-docs"
  domain      TEXT NOT NULL,                          -- hostname allowed to embed the widget, e.g. "docs.example.com"
  sitemap_url TEXT NOT NULL,                          -- sitemap used for crawling
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- One row per crawled URL. Doubles as the indexing queue (status = pending/indexed/failed).
CREATE TABLE IF NOT EXISTS pages (
  id          TEXT PRIMARY KEY,                       -- sha256(url) prefix, deterministic for idempotent re-indexing
  site_id     TEXT NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
  url         TEXT NOT NULL,
  title       TEXT,
  chunk_count INTEGER NOT NULL DEFAULT 0,             -- number of vectors stored for this page
  status      TEXT NOT NULL DEFAULT 'pending',        -- pending | indexed | failed
  error       TEXT,                                   -- last crawl/embedding error, if any
  updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (site_id, url)
);

CREATE INDEX IF NOT EXISTS idx_pages_site_status ON pages (site_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages (status, updated_at);

-- Chat history, useful for analytics and spotting documentation gaps.
CREATE TABLE IF NOT EXISTS queries (
  id          TEXT PRIMARY KEY,                       -- UUID
  site_id     TEXT NOT NULL,
  prompt      TEXT NOT NULL,
  response    TEXT,
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queries_site_created ON queries (site_id, created_at);
