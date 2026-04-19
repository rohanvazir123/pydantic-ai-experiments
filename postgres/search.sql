-- EXAMPLE A: Basic Phrase Search
-- Finds the exact sequence of words regardless of common "stop words" like 'with'
SELECT title FROM articles 
WHERE search_vector @@ phraseto_tsquery('english', 'PostgreSQL Search');

-- EXAMPLE B: Proximity Search (Words within a distance)
-- The <2> operator means 'PostgreSQL' followed by another word, then 'Search'
SELECT title FROM articles 
WHERE search_vector @@ to_tsquery('english', 'PostgreSQL <2> Search');

-- EXAMPLE C: Web-style Search
-- Supports "quotes" for phrases and minus sign for exclusions
SELECT title FROM articles 
WHERE search_vector @@ websearch_to_tsquery('english', '"Full Text Search" -cooking');


-- Create a GIN index on your pre-calculated search vector
CREATE INDEX idx_articles_search_auto ON articles USING GIN (search_vector);

