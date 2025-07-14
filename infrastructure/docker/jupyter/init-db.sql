-- Database initialization script for JupyterHub and MLflow
-- Creates necessary databases and users with appropriate permissions

-- Create MLflow database and user
CREATE DATABASE mlflow;
CREATE USER mlflow WITH PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Create additional databases for notebook services
CREATE DATABASE notebooks;
CREATE USER notebooks WITH PASSWORD 'notebooks';
GRANT ALL PRIVILEGES ON DATABASE notebooks TO notebooks;

-- Create analytics database for dbt integration
CREATE DATABASE analytics;
CREATE USER analytics WITH PASSWORD 'analytics';
GRANT ALL PRIVILEGES ON DATABASE analytics TO analytics;

-- Grant additional permissions for MLflow user
\c mlflow;
GRANT USAGE ON SCHEMA public TO mlflow;
GRANT CREATE ON SCHEMA public TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow;

-- Grant additional permissions for notebooks user
\c notebooks;
GRANT USAGE ON SCHEMA public TO notebooks;
GRANT CREATE ON SCHEMA public TO notebooks;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO notebooks;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO notebooks;

-- Grant additional permissions for analytics user
\c analytics;
GRANT USAGE ON SCHEMA public TO analytics;
GRANT CREATE ON SCHEMA public TO analytics;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO analytics;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO analytics;

-- Switch back to jupyterhub database
\c jupyterhub;

-- Create additional schema for user management
CREATE SCHEMA IF NOT EXISTS user_management;
GRANT USAGE ON SCHEMA user_management TO jupyterhub;
GRANT CREATE ON SCHEMA user_management TO jupyterhub;

-- Create table for user profiles
CREATE TABLE IF NOT EXISTS user_management.user_profiles (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    full_name VARCHAR(255),
    preferred_environment VARCHAR(50) DEFAULT 'ml',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for user sessions
CREATE TABLE IF NOT EXISTS user_management.user_sessions (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration_minutes INTEGER,
    resources_used JSONB
);

-- Create table for notebook execution logs
CREATE TABLE IF NOT EXISTS user_management.notebook_executions (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    notebook_path VARCHAR(500) NOT NULL,
    execution_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_end TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running',
    error_message TEXT,
    output_size_bytes INTEGER,
    execution_metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_username ON user_management.user_profiles(username);
CREATE INDEX IF NOT EXISTS idx_user_sessions_username ON user_management.user_sessions(username);
CREATE INDEX IF NOT EXISTS idx_user_sessions_started_at ON user_management.user_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_notebook_executions_username ON user_management.notebook_executions(username);
CREATE INDEX IF NOT EXISTS idx_notebook_executions_start ON user_management.notebook_executions(execution_start);

-- Grant permissions on user management tables
GRANT ALL ON user_management.user_profiles TO jupyterhub;
GRANT ALL ON user_management.user_sessions TO jupyterhub;
GRANT ALL ON user_management.notebook_executions TO jupyterhub;
GRANT ALL ON ALL SEQUENCES IN SCHEMA user_management TO jupyterhub;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for user_profiles table
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE
    ON user_management.user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user profile
INSERT INTO user_management.user_profiles (username, email, full_name, preferred_environment)
VALUES ('admin', 'admin@example.com', 'Administrator', 'devops')
ON CONFLICT (username) DO NOTHING;

-- Create view for user activity summary
CREATE OR REPLACE VIEW user_management.user_activity_summary AS
SELECT 
    p.username,
    p.full_name,
    p.preferred_environment,
    p.created_at as user_created_at,
    COUNT(DISTINCT s.session_id) as total_sessions,
    SUM(s.duration_minutes) as total_session_minutes,
    COUNT(DISTINCT ne.notebook_path) as unique_notebooks_executed,
    COUNT(ne.id) as total_notebook_executions,
    MAX(s.started_at) as last_session_start,
    MAX(ne.execution_start) as last_notebook_execution
FROM user_management.user_profiles p
LEFT JOIN user_management.user_sessions s ON p.username = s.username
LEFT JOIN user_management.notebook_executions ne ON p.username = ne.username
GROUP BY p.username, p.full_name, p.preferred_environment, p.created_at;

-- Grant permissions on the view
GRANT SELECT ON user_management.user_activity_summary TO jupyterhub;

-- Create function to log user session
CREATE OR REPLACE FUNCTION user_management.log_user_session(
    p_username VARCHAR(255),
    p_session_id VARCHAR(255),
    p_environment VARCHAR(50)
) RETURNS INTEGER AS $$
DECLARE
    session_record_id INTEGER;
BEGIN
    INSERT INTO user_management.user_sessions (username, session_id, environment)
    VALUES (p_username, p_session_id, p_environment)
    RETURNING id INTO session_record_id;
    
    RETURN session_record_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to end user session
CREATE OR REPLACE FUNCTION user_management.end_user_session(
    p_session_id VARCHAR(255)
) RETURNS BOOLEAN AS $$
DECLARE
    session_start TIMESTAMP;
    duration_calc INTEGER;
BEGIN
    SELECT started_at INTO session_start
    FROM user_management.user_sessions
    WHERE session_id = p_session_id AND ended_at IS NULL;
    
    IF session_start IS NOT NULL THEN
        duration_calc := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - session_start)) / 60;
        
        UPDATE user_management.user_sessions
        SET ended_at = CURRENT_TIMESTAMP,
            duration_minutes = duration_calc
        WHERE session_id = p_session_id;
        
        RETURN TRUE;
    END IF;
    
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Create function to log notebook execution
CREATE OR REPLACE FUNCTION user_management.log_notebook_execution(
    p_username VARCHAR(255),
    p_notebook_path VARCHAR(500),
    p_status VARCHAR(50) DEFAULT 'completed',
    p_error_message TEXT DEFAULT NULL,
    p_output_size_bytes INTEGER DEFAULT NULL,
    p_execution_metadata JSONB DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    execution_record_id INTEGER;
BEGIN
    INSERT INTO user_management.notebook_executions (
        username, notebook_path, execution_end, status, 
        error_message, output_size_bytes, execution_metadata
    )
    VALUES (
        p_username, p_notebook_path, CURRENT_TIMESTAMP, p_status,
        p_error_message, p_output_size_bytes, p_execution_metadata
    )
    RETURNING id INTO execution_record_id;
    
    RETURN execution_record_id;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION user_management.log_user_session TO jupyterhub;
GRANT EXECUTE ON FUNCTION user_management.end_user_session TO jupyterhub;
GRANT EXECUTE ON FUNCTION user_management.log_notebook_execution TO jupyterhub;

-- Create schema for monitoring metrics
CREATE SCHEMA IF NOT EXISTS monitoring;
GRANT USAGE ON SCHEMA monitoring TO jupyterhub;
GRANT CREATE ON SCHEMA monitoring TO jupyterhub;

-- Create table for system metrics
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for application metrics
CREATE TABLE IF NOT EXISTS monitoring.application_metrics (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    username VARCHAR(255),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for metrics tables
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON monitoring.system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_app_metrics_service_name_time ON monitoring.application_metrics(service_name, metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_app_metrics_username ON monitoring.application_metrics(username);

-- Grant permissions on monitoring tables
GRANT ALL ON monitoring.system_metrics TO jupyterhub;
GRANT ALL ON monitoring.application_metrics TO jupyterhub;
GRANT ALL ON ALL SEQUENCES IN SCHEMA monitoring TO jupyterhub;

-- Create function to record metrics
CREATE OR REPLACE FUNCTION monitoring.record_metric(
    p_metric_name VARCHAR(255),
    p_metric_value DOUBLE PRECISION,
    p_tags JSONB DEFAULT NULL,
    p_service_name VARCHAR(255) DEFAULT NULL,
    p_username VARCHAR(255) DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    IF p_service_name IS NULL THEN
        INSERT INTO monitoring.system_metrics (metric_name, metric_value, tags)
        VALUES (p_metric_name, p_metric_value, p_tags);
    ELSE
        INSERT INTO monitoring.application_metrics (service_name, metric_name, metric_value, username, tags)
        VALUES (p_service_name, p_metric_name, p_metric_value, p_username, p_tags);
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on monitoring function
GRANT EXECUTE ON FUNCTION monitoring.record_metric TO jupyterhub;

-- Create cleanup function for old metrics (retention policy)
CREATE OR REPLACE FUNCTION monitoring.cleanup_old_metrics(
    p_retention_days INTEGER DEFAULT 30
) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean up old system metrics
    DELETE FROM monitoring.system_metrics 
    WHERE timestamp < (CURRENT_TIMESTAMP - INTERVAL '1 day' * p_retention_days);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old application metrics
    DELETE FROM monitoring.application_metrics 
    WHERE timestamp < (CURRENT_TIMESTAMP - INTERVAL '1 day' * p_retention_days);
    
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on cleanup function
GRANT EXECUTE ON FUNCTION monitoring.cleanup_old_metrics TO jupyterhub;

-- Final message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Created databases: mlflow, notebooks, analytics';
    RAISE NOTICE 'Created schemas: user_management, monitoring';
    RAISE NOTICE 'Created tables for user profiles, sessions, notebook executions, and metrics';
    RAISE NOTICE 'Created functions for logging and monitoring';
END $$; 