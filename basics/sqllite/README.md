

## Downloading data

You can get massive, high-quality, public CSV datasets for free from dedicated data repositories and government portals.
Depending on your interests, the following platforms offer direct links to download datasets or stream them directly via URL into your Python code.

## 1. General & High-Volume Data Repositories

* [Kaggle Datasets](https://www.kaggle.com/datasets): The most popular community platform for data science. You can find millions of community-uploaded CSVs spanning sports statistics, e-commerce, movie ratings, and financial trends.
* [Our World in Data GitHub](https://github.com/owid/owid-datasets): Clean, well-structured historical datasets covering energy, global health, population growth, and economics. You can click on any .csv file, hit "Raw", and copy the URL directly into your code.
* [Awesome Public Datasets GitHub](https://github.com/awesomedata/awesome-public-datasets): A massive, topic-organized directory of open-source datasets across topics like agriculture, biology, climate, and education.

## 2. Government & Institutional Open Data (Largest Scale)

* [Data.gov](https://www.data.gov/): The home of the U.S. Government’s open data. It hosts hundreds of thousands of massive datasets covering climate change, consumer complaints, transportation, and healthcare.
* [NASA Open Data Portal](https://data.nasa.gov/): Direct access to aerospace data, satellite imagery metrics, meteorite landings, and space exploration tracking records.
* [EU Open Data Portal](https://data.europa.eu/en): Official public data published by the institutions and bodies of the European Union, great for international economic and geographic tracking.

## 3. Quick Test Datasets (Copy-and-Paste Ready)
If you just want to test your script right now with reliable, public URLs, you can swap out the PUBLIC_CSV_URL in your code with any of these:

* Global COVID-19 Tracking Data: https://githubusercontent.com
* Massive Airport Location Codes: https://github.io
* Historical Olympic Games Results: https://githubusercontent.com

## Tip for Using GitHub Data
When fetching CSVs from GitHub, make sure the URL contains ://githubusercontent.com instead of github.com. The standard URL downloads an HTML page about the file, which will crash your Python script. Clicking the "Raw" button on GitHub provides the true link.
Would you like help:

* Finding a specific topic (like finance, real estate, or sports)?
* Writing a Python script to search and auto-download zip files from these platforms?



## Directly run this on SQLLite command prompt

### First launch it
sqlite3 llm_metrics.db


### Let it rip

-- 1. Switch to CSV mode so SQLite knows how to parse commas
.mode csv

-- 2. Import your file into a table named 'llm_usage'
.import genai_llm_usage_dataset_1000.csv llm_usage

-- 3. Switch back to a clean visual table layout for reading query results
.mode table
.headers on


-- Inspect schema
.schema llm_usage


-- Get total number of rows
SELECT COUNT(*) FROM llm_usage;

-- Find most heavily used models
SELECT model_name, COUNT(*), SUM(total_tokens)
FROM llm_usage
GROUP BY model_name
ORDER BY COUNT(*) DESC;


-- Create new users table


CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    department TEXT,
    tier TEXT DEFAULT 'free'
);


-- create api keys

CREATE TABLE api_keys (
    key_id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_string TEXT NOT NULL UNIQUE,
    status TEXT DEFAULT 'active',
    owner_id INTEGER,
    FOREIGN KEY (owner_id) REFERENCES users(user_id) ON DELETE CASCADE

);

-- add more data

-- Insert some users

INSERT INTO users (username, department, tier) VALUES ('alice_dev', 'Engineering', 'premium');
INSERT INTO users (username, department, tier) VALUES ('bob_data', 'Analytics', 'free');
INSERT INTO users (username, department, tier) VALUES ('charlie_pm', 'Product', 'premium');

-- Insert API keys linked to those users (owner_id matches user_id)
INSERT INTO api_keys (api_key_string, owner_id) VALUES ('sk-live-1111', 1);
INSERT INTO api_keys (api_key_string, owner_id) VALUES ('sk-test-2222', 1);
INSERT INTO api_keys (api_key_string, owner_id) VALUES ('sk-live-3333', 2);


-- inner join


SELECT users.username, users.department, api_keys.api_key_string, api_keys.status
FROM users
JOIN api_keys ON users.user_id = api_keys.owner_id;


-- data aggregation

SELECT users.username, users.department, api_keys.api_key_string, api_keys.status
FROM users
JOIN api_keys ON users.user_id = api_keys.owner_id;



SELECT type, name, tbl_name, sql 
FROM sqlite_master 
WHERE type='table';


PRAGMA table_info(llm_usage);
.tables

