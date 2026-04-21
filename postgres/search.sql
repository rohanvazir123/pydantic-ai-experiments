SELECT * FROM articles 
WHERE search_vector @@ phraseto_tsquery('english', 'advanced phrase search');

SELECT * FROM articles 
WHERE search_vector @@ to_tsquery('english', 'Jane & html');


