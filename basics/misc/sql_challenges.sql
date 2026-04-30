-- SQL Crash Course & Challenges
-- Tables: baby_names(year, name, sex, count)
--         world_gdp(country_code, country_name, year, gdp_usd)
--
-- Run in psql:   psql -d postgres -f sql_challenges.sql
-- Or copy-paste individual challenges into any SQL client.
--
-- Each challenge is preceded by:
--   concept  → the SQL feature being practised
--   task     → what to write
--   solution → uncomment to reveal


-- ============================================================
-- BASICS
-- ============================================================

-- ─── CH 01 ── SELECT, WHERE, ORDER BY, LIMIT ─────────────────
-- Concept : Basic SELECT
-- Task    : Return the top 10 female names in 2023, ordered by count descending.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, count
-- FROM   baby_names
-- WHERE  year = 2023 AND sex = 'F'
-- ORDER  BY count DESC
-- LIMIT  10;


-- ─── CH 02 ── Column aliases & expressions ───────────────────
-- Concept : AS, arithmetic expressions
-- Task    : Show name, count, and a new column 'share_pct' = count / SUM(count for that year+sex) * 100.
--           Restrict to female, 2023, top 10.
--           Hint: use a subquery for the denominator.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name,
--        count,
--        round(count * 100.0 /
--              (SELECT sum(count) FROM baby_names WHERE year = 2023 AND sex = 'F'), 2
--             ) AS share_pct
-- FROM   baby_names
-- WHERE  year = 2023 AND sex = 'F'
-- ORDER  BY count DESC
-- LIMIT  10;


-- ─── CH 03 ── BETWEEN, IN, LIKE ──────────────────────────────
-- Concept : Range and pattern filters
-- Task    : Find all female names starting with 'Al' that had more than 1000 births in 2023.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, count
-- FROM   baby_names
-- WHERE  year = 2023
--   AND  sex  = 'F'
--   AND  name LIKE 'Al%'
--   AND  count > 1000
-- ORDER  BY count DESC;


-- ─── CH 04 ── DISTINCT & COUNT ───────────────────────────────
-- Concept : Deduplication, COUNT(DISTINCT ...)
-- Task    : How many distinct names appear in the baby_names table altogether?
--           How many distinct names appear ONLY for females (never for males)?
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- -- Part 1: all names
-- SELECT count(DISTINCT name) AS total_unique_names FROM baby_names;
-- -- Part 2: female-only names
-- SELECT count(DISTINCT name) AS female_only_names
-- FROM   baby_names
-- WHERE  sex = 'F'
--   AND  name NOT IN (SELECT DISTINCT name FROM baby_names WHERE sex = 'M');


-- ============================================================
-- AGGREGATION
-- ============================================================

-- ─── CH 05 ── GROUP BY + aggregate functions ─────────────────
-- Concept : GROUP BY, SUM, AVG, MAX
-- Task    : For each year from 2010 to 2023, show:
--           total_births (sum of count), avg_per_name (avg count), peak (max count).
--           Order by year.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT year,
--        sum(count)              AS total_births,
--        round(avg(count), 1)    AS avg_per_name,
--        max(count)              AS peak_count
-- FROM   baby_names
-- WHERE  year BETWEEN 2010 AND 2023
-- GROUP  BY year
-- ORDER  BY year;


-- ─── CH 06 ── HAVING ─────────────────────────────────────────
-- Concept : Filter AFTER aggregation with HAVING
-- Task    : Which names have appeared (had at least 1 birth) in MORE than 100 different years?
--           Show name, sex, and the number of distinct years they appeared.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, sex, count(DISTINCT year) AS years_active
-- FROM   baby_names
-- GROUP  BY name, sex
-- HAVING count(DISTINCT year) > 100
-- ORDER  BY years_active DESC, name;


-- ─── CH 07 ── CASE WHEN inside aggregation ───────────────────
-- Concept : Conditional aggregation
-- Task    : For each year 2000–2023, show the total births split by sex:
--           columns year, male_births, female_births, total_births.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT year,
--        sum(CASE WHEN sex = 'M' THEN count ELSE 0 END) AS male_births,
--        sum(CASE WHEN sex = 'F' THEN count ELSE 0 END) AS female_births,
--        sum(count)                                      AS total_births
-- FROM   baby_names
-- WHERE  year BETWEEN 2000 AND 2023
-- GROUP  BY year
-- ORDER  BY year;


-- ============================================================
-- SUBQUERIES & CTEs
-- ============================================================

-- ─── CH 08 ── Scalar subquery ────────────────────────────────
-- Concept : Subquery in WHERE
-- Task    : Find female names in 2023 whose count is above the
--           average count of ALL female names in 2023.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, count
-- FROM   baby_names
-- WHERE  year = 2023 AND sex = 'F'
--   AND  count > (SELECT avg(count) FROM baby_names WHERE year = 2023 AND sex = 'F')
-- ORDER  BY count DESC;


-- ─── CH 09 ── CTE (WITH clause) ──────────────────────────────
-- Concept : Common Table Expression
-- Task    : Using a CTE, find the #1 female name for each year from 2010 to 2023.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- WITH ranked AS (
--     SELECT year, name, count,
--            rank() OVER (PARTITION BY year ORDER BY count DESC) AS rnk
--     FROM   baby_names
--     WHERE  sex = 'F' AND year BETWEEN 2010 AND 2023
-- )
-- SELECT year, name, count
-- FROM   ranked
-- WHERE  rnk = 1
-- ORDER  BY year;


-- ─── CH 10 ── EXISTS ─────────────────────────────────────────
-- Concept : EXISTS / NOT EXISTS
-- Task    : Find female names in the 2023 top-10 that were NOT in the 2000 top-10.
--           (Names that rose to the top since 2000.)
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- WITH top10_2023 AS (
--     SELECT name FROM baby_names WHERE year=2023 AND sex='F' ORDER BY count DESC LIMIT 10
-- ),
-- top10_2000 AS (
--     SELECT name FROM baby_names WHERE year=2000 AND sex='F' ORDER BY count DESC LIMIT 10
-- )
-- SELECT t23.name
-- FROM   top10_2023 t23
-- WHERE  NOT EXISTS (SELECT 1 FROM top10_2000 t00 WHERE t00.name = t23.name);


-- ============================================================
-- JOINS
-- ============================================================

-- ─── CH 11 ── Self-join ───────────────────────────────────────
-- Concept : JOIN a table to itself
-- Task    : Show the count for 'Emma' (female) in 2000 alongside her count in 2023
--           in a single row: name, count_2000, count_2023.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT a.name,
--        a.count AS count_2000,
--        b.count AS count_2023
-- FROM   baby_names a
-- JOIN   baby_names b
--        ON  a.name = b.name AND a.sex = b.sex
-- WHERE  a.year = 2000
--   AND  b.year = 2023
--   AND  a.name = 'Emma'
--   AND  a.sex  = 'F';


-- ─── CH 12 ── JOIN two tables ────────────────────────────────
-- Concept : INNER JOIN between baby_names and world_gdp
-- Task    : For the USA (country_code = 'USA'), show total US births and GDP side-by-side
--           for each year from 2000 to 2022.
--           Columns: year, total_births, gdp_usd.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT b.year,
--        sum(b.count)  AS total_births,
--        g.gdp_usd
-- FROM   baby_names b
-- JOIN   world_gdp  g ON b.year = g.year AND g.country_code = 'USA'
-- WHERE  b.year BETWEEN 2000 AND 2022
-- GROUP  BY b.year, g.gdp_usd
-- ORDER  BY b.year;


-- ─── CH 13 ── LEFT JOIN & NULLs ─────────────────────────────
-- Concept : LEFT JOIN; spotting missing data
-- Task    : Which years in baby_names (all years) have NO matching row in world_gdp
--           for the USA? (Years before World Bank coverage or after latest data.)
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT DISTINCT b.year
-- FROM   baby_names b
-- LEFT   JOIN world_gdp g ON b.year = g.year AND g.country_code = 'USA'
-- WHERE  g.year IS NULL
-- ORDER  BY b.year;


-- ============================================================
-- WINDOW FUNCTIONS
-- ============================================================

-- ─── CH 14 ── ROW_NUMBER ─────────────────────────────────────
-- Concept : ROW_NUMBER() OVER (PARTITION BY … ORDER BY …)
-- Task    : Assign a rank to every name in 2023 within its sex group,
--           ordered by count descending. Show rows where rank <= 3.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, sex, count,
--        row_number() OVER (PARTITION BY sex ORDER BY count DESC) AS rnk
-- FROM   baby_names
-- WHERE  year = 2023
-- QUALIFY row_number() OVER (PARTITION BY sex ORDER BY count DESC) <= 3;
-- -- If QUALIFY is not supported (PostgreSQL doesn't natively):
-- WITH r AS (
--     SELECT name, sex, count,
--            row_number() OVER (PARTITION BY sex ORDER BY count DESC) AS rnk
--     FROM   baby_names WHERE year = 2023
-- )
-- SELECT * FROM r WHERE rnk <= 3;


-- ─── CH 15 ── RANK vs DENSE_RANK ─────────────────────────────
-- Concept : Difference between RANK and DENSE_RANK
-- Task    : For female names in 2023, show RANK and DENSE_RANK for the top 15.
--           Notice where the two diverge (ties).
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT name, count,
--        rank()       OVER (ORDER BY count DESC) AS rnk,
--        dense_rank() OVER (ORDER BY count DESC) AS dense_rnk
-- FROM   baby_names
-- WHERE  year = 2023 AND sex = 'F'
-- ORDER  BY count DESC
-- LIMIT  15;


-- ─── CH 16 ── LAG ────────────────────────────────────────────
-- Concept : LAG() — previous row value
-- Task    : For 'Olivia' (female), show year, count, and the count from the
--           previous year (lag_count), and the absolute change.
--           Only show years 2010–2023.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT year, count,
--        lag(count) OVER (ORDER BY year)              AS lag_count,
--        count - lag(count) OVER (ORDER BY year)      AS yoy_change
-- FROM   baby_names
-- WHERE  name = 'Olivia' AND sex = 'F' AND year BETWEEN 2010 AND 2023
-- ORDER  BY year;


-- ─── CH 17 ── Running total ───────────────────────────────────
-- Concept : SUM() OVER (ORDER BY …) — cumulative window
-- Task    : For the USA GDP, show each year's gdp_usd and a running
--           cumulative total of GDP since 1960.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- SELECT year, gdp_usd,
--        sum(gdp_usd) OVER (ORDER BY year)   AS cumulative_gdp
-- FROM   world_gdp
-- WHERE  country_code = 'USA' AND gdp_usd IS NOT NULL
-- ORDER  BY year;


-- ─── CH 18 ── NTILE ──────────────────────────────────────────
-- Concept : NTILE(n) — divide rows into n equal buckets
-- Task    : Split all 2023 female names into 4 quartiles by count.
--           Show the min and max count in each quartile.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- WITH buckets AS (
--     SELECT name, count,
--            ntile(4) OVER (ORDER BY count) AS quartile
--     FROM   baby_names
--     WHERE  year = 2023 AND sex = 'F'
-- )
-- SELECT quartile, min(count) AS min_count, max(count) AS max_count, count(*) AS names
-- FROM   buckets
-- GROUP  BY quartile
-- ORDER  BY quartile;


-- ============================================================
-- ADVANCED / MIXED
-- ============================================================

-- ─── CH 19 ── GDP growth rate ────────────────────────────────
-- Concept : LAG + arithmetic for year-over-year % change
-- Task    : For the top-5 economies by 2022 GDP, calculate the average
--           annual GDP growth rate between 2000 and 2022.
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- WITH top5 AS (
--     SELECT country_code
--     FROM   world_gdp
--     WHERE  year = 2022 AND gdp_usd IS NOT NULL
--     ORDER  BY gdp_usd DESC LIMIT 5
-- ),
-- yoy AS (
--     SELECT g.country_code, g.year, g.gdp_usd,
--            lag(g.gdp_usd) OVER (PARTITION BY g.country_code ORDER BY g.year) AS prev_gdp
--     FROM   world_gdp g
--     JOIN   top5 ON top5.country_code = g.country_code
--     WHERE  g.year BETWEEN 2000 AND 2022 AND g.gdp_usd IS NOT NULL
-- )
-- SELECT country_code,
--        round(avg((gdp_usd - prev_gdp) / prev_gdp * 100)::numeric, 2) AS avg_annual_growth_pct
-- FROM   yoy
-- WHERE  prev_gdp IS NOT NULL
-- GROUP  BY country_code
-- ORDER  BY avg_annual_growth_pct DESC;


-- ─── CH 20 ── Most consistent name ───────────────────────────
-- Concept : stddev inside groupby, ordering by derived metric
-- Task    : Among names that have appeared in every year from 1950 to 2023,
--           find the 10 with the lowest standard deviation in their annual count.
--           (These are the "most stable" names — consistently popular.)
-- ─────────────────────────────────────────────────────────────
-- YOUR QUERY HERE



-- Solution:
-- WITH always_present AS (
--     SELECT name, sex
--     FROM   baby_names
--     WHERE  year BETWEEN 1950 AND 2023
--     GROUP  BY name, sex
--     HAVING count(DISTINCT year) = 2023 - 1950 + 1   -- 74 years
-- ),
-- stats AS (
--     SELECT b.name, b.sex,
--            round(stddev(b.count)::numeric, 0)  AS count_stddev,
--            round(avg(b.count)::numeric, 0)      AS count_avg
--     FROM   baby_names b
--     JOIN   always_present ap ON ap.name = b.name AND ap.sex = b.sex
--     WHERE  b.year BETWEEN 1950 AND 2023
--     GROUP  BY b.name, b.sex
-- )
-- SELECT name, sex, count_avg, count_stddev
-- FROM   stats
-- ORDER  BY count_stddev
-- LIMIT  10;
