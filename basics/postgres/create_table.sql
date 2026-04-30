-- First, create a helper function to strip HTML tags using RegEx
CREATE OR REPLACE FUNCTION strip_html(input text) RETURNS text AS $$
    -- Replaces everything between < and > with a space to prevent merging words
    SELECT regexp_replace($1, '<[^>]*>', ' ', 'g');
$$ LANGUAGE SQL IMMUTABLE;

DROP TABLE articles;

CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    html_content TEXT,
    json_metadata JSONB,
    -- Combined search vector:
    -- 1. Strips HTML tags from the content
    -- 2. Extracts all string values from the JSONB
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(strip_html(html_content), '')) || 
        jsonb_to_tsvector('english', coalesce(json_metadata, '{}'), '["all"]')
    ) STORED
);

-- Optimize with a GIN index
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- Insert HTML and json
INSERT INTO articles (html_content, json_metadata) VALUES
(
    '<h1>Postgres Full Text</h1><p>Learn how to search <b>html</b> easily.</p>', 
    '{"author": "Jane Doe", "tags": ["database", "sql"], "priority": 1}'
),
(
    '<div>Advanced <i>Phrase Search</i> features are built-in.</div>', 
    '{"author": "John Smith", "category": "Tutorial"}'
);
