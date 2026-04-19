-- 1. Create a table with a 'stored' search vector for maximum efficiency
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT,
    -- The generated column combines title and body into a searchable format
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') || 
        setweight(to_tsvector('english', coalesce(body, '')), 'B')
    ) STORED
);

-- 2. Populate with sample data
INSERT INTO articles (title, body) VALUES
('PostgreSQL Full Text Search', 'PostgreSQL provides powerful tools for searching text efficiently.'),
('Mastering Postgres Indexing', 'Learn how to scale your database for high traffic using GIN and GiST.'),
('Phrase Search Techniques', 'Different search techniques, including phrase-based searching, are vital.'),
('Cooking with Postgres', 'A guide to baking delicious database queries.');



