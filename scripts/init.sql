-- This script runs only when the database is first created
-- Grant specific privileges to the user
GRANT ALL PRIVILEGES ON DATABASE vidhi_ai_db TO vidhi_user;

-- Ensure the user owns the public schema
ALTER SCHEMA public OWNER TO vidhi_user;