-- Brain-Forge Multi-Modal BCI Platform
-- PostgreSQL Database Initialization Script
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;
-- Create schemas
CREATE SCHEMA IF NOT EXISTS brain_forge;
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS analytics;
-- Set default schema
SET search_path TO brain_forge,
    public;
-- Users and sessions table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(100) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'created',
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE
);
-- Hardware configuration tables
CREATE TABLE IF NOT EXISTS hardware_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    device_type VARCHAR(50) NOT NULL,
    -- 'omp', 'kernel', 'accelerometer'
    device_id VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    calibration_data JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Raw data storage (time-series optimized)
CREATE TABLE IF NOT EXISTS raw_data (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    device_type VARCHAR(50) NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    channel_id VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB,
    PRIMARY KEY (
        time,
        session_id,
        device_type,
        device_id,
        channel_id
    )
);
-- Convert to hypertable for time-series optimization
SELECT create_hypertable('raw_data', 'time', if_not_exists => TRUE);
-- Processed data storage
CREATE TABLE IF NOT EXISTS processed_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    processing_type VARCHAR(50) NOT NULL,
    -- 'filtered', 'epoched', 'features', etc.
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Transfer learning models
CREATE TABLE IF NOT EXISTS transfer_learning_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    architecture JSONB,
    weights BYTEA,
    metadata JSONB,
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Model training history
CREATE TABLE IF NOT EXISTS model_training_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES transfer_learning_models(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    training_config JSONB,
    metrics JSONB,
    loss_history JSONB,
    validation_metrics JSONB,
    epoch_count INTEGER,
    training_time_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Experiments schema tables
CREATE TABLE IF NOT EXISTS experiments.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    protocol JSONB,
    status VARCHAR(20) DEFAULT 'planned',
    created_by UUID REFERENCES brain_forge.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);
CREATE TABLE IF NOT EXISTS experiments.experiment_sessions (
    experiment_id UUID REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    session_id UUID REFERENCES brain_forge.sessions(id) ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, session_id)
);
-- Analytics schema tables
CREATE TABLE IF NOT EXISTS analytics.analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    input_data JSONB,
    parameters JSONB,
    status VARCHAR(20) DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);
-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_hardware_configs_session_id ON hardware_configs(session_id);
CREATE INDEX IF NOT EXISTS idx_hardware_configs_device_type ON hardware_configs(device_type);
CREATE INDEX IF NOT EXISTS idx_raw_data_session_id ON raw_data(session_id);
CREATE INDEX IF NOT EXISTS idx_raw_data_device_type ON raw_data(device_type);
CREATE INDEX IF NOT EXISTS idx_raw_data_time_desc ON raw_data(time DESC);
CREATE INDEX IF NOT EXISTS idx_processed_data_session_id ON processed_data(session_id);
CREATE INDEX IF NOT EXISTS idx_processed_data_type ON processed_data(processing_type);
CREATE INDEX IF NOT EXISTS idx_models_active ON transfer_learning_models(is_active);
CREATE INDEX IF NOT EXISTS idx_models_type ON transfer_learning_models(model_type);
CREATE INDEX IF NOT EXISTS idx_training_history_model_id ON model_training_history(model_id);
CREATE INDEX IF NOT EXISTS idx_training_history_session_id ON model_training_history(session_id);
-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = CURRENT_TIMESTAMP;
RETURN NEW;
END;
$$ language 'plpgsql';
CREATE TRIGGER update_users_updated_at BEFORE
UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_models_updated_at BEFORE
UPDATE ON transfer_learning_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
-- Insert default admin user (password: 'admin123' - CHANGE IN PRODUCTION)
INSERT INTO users (username, email, password_hash, full_name, role)
VALUES (
        'admin',
        'admin@brain-forge.local',
        '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3L6nHx.LPu',
        -- admin123
        'Brain-Forge Administrator',
        'admin'
    ) ON CONFLICT (username) DO NOTHING;
-- Create sample configuration data
INSERT INTO transfer_learning_models (name, model_type, architecture, metadata)
VALUES (
        'EEGNet-8,2',
        'eeg_classification',
        '{"layers": [{"type": "Conv2D", "filters": 8}, {"type": "DepthwiseConv2D"}, {"type": "Dense", "units": 4}]}',
        '{"description": "Compact CNN for EEG classification", "input_shape": [1, 64, 256], "output_classes": 4}'
    ) ON CONFLICT DO NOTHING;
-- Grant permissions
GRANT USAGE ON SCHEMA brain_forge TO PUBLIC;
GRANT USAGE ON SCHEMA experiments TO PUBLIC;
GRANT USAGE ON SCHEMA analytics TO PUBLIC;
GRANT SELECT,
    INSERT,
    UPDATE,
    DELETE ON ALL TABLES IN SCHEMA brain_forge TO PUBLIC;
GRANT SELECT,
    INSERT,
    UPDATE,
    DELETE ON ALL TABLES IN SCHEMA experiments TO PUBLIC;
GRANT SELECT,
    INSERT,
    UPDATE,
    DELETE ON ALL TABLES IN SCHEMA analytics TO PUBLIC;
GRANT USAGE,
    SELECT ON ALL SEQUENCES IN SCHEMA brain_forge TO PUBLIC;
GRANT USAGE,
    SELECT ON ALL SEQUENCES IN SCHEMA experiments TO PUBLIC;
GRANT USAGE,
    SELECT ON ALL SEQUENCES IN SCHEMA analytics TO PUBLIC;
-- Performance optimization
ANALYZE;
-- Display initialization completion
SELECT 'Brain-Forge database initialized successfully!' as status;