SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'iprep_i1_functional' ;



SELECT table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'iprep_i1_functional'
ORDER BY table_name, ordinal_position;


SELECT 
    tc.table_name, 
    tc.constraint_name, 
    tc.constraint_type, 
    kcu.column_name
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
  AND tc.table_schema = kcu.table_schema
WHERE tc.table_schema = 'iprep_i1_functional'
ORDER BY tc.table_name, tc.constraint_type;

SELECT distinct topic FROM iprep_i1_functional.summary_topics order by topic;
