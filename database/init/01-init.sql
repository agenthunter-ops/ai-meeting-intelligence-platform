-- Database Initialization Script for AI Meeting Intelligence Platform
-- =================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Set timezone
SET timezone = 'UTC';

-- Create custom types
DO $$ BEGIN
    CREATE TYPE task_state AS ENUM ('PENDING', 'PROGRESS', 'SUCCESS', 'FAILURE', 'RETRY', 'REVOKED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE meeting_type AS ENUM ('standup', 'planning', 'retrospective', 'one_on_one', 'all_hands', 'client_call', 'interview', 'general');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE insight_type AS ENUM ('action_item', 'decision', 'sentiment', 'summary', 'topic', 'risk');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create tables (these will be created by SQLAlchemy, but we can add custom indexes)
-- The actual table creation is handled by the Python application

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at);
CREATE INDEX IF NOT EXISTS idx_meetings_processing_status ON meetings(processing_status);
CREATE INDEX IF NOT EXISTS idx_meetings_type_date ON meetings(meeting_type, meeting_date);

CREATE INDEX IF NOT EXISTS idx_segments_meeting_sequence ON segments(meeting_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_segments_times ON segments(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker);

CREATE INDEX IF NOT EXISTS idx_insights_meeting_type ON insights(meeting_id, type);
CREATE INDEX IF NOT EXISTS idx_insights_status ON insights(status);
CREATE INDEX IF NOT EXISTS idx_insights_priority ON insights(priority);

CREATE INDEX IF NOT EXISTS idx_tasks_state_created ON tasks(state, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_progress ON tasks(progress);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_meetings_title_fts ON meetings USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_segments_text_fts ON segments USING gin(to_tsvector('english', text));

-- Create function for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_meetings_updated_at BEFORE UPDATE ON meetings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_segments_updated_at BEFORE UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_insights_updated_at BEFORE UPDATE ON insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for calculating segment duration
CREATE OR REPLACE FUNCTION calculate_segment_duration()
RETURNS TRIGGER AS $$
BEGIN
    NEW.duration = NEW.end_time - NEW.start_time;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for segment duration
CREATE TRIGGER calculate_segment_duration_trigger BEFORE INSERT OR UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION calculate_segment_duration();

-- Create function for updating meeting statistics
CREATE OR REPLACE FUNCTION update_meeting_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update meeting statistics when segments or insights change
    IF TG_TABLE_NAME = 'segments' THEN
        UPDATE meetings 
        SET total_segments = (
            SELECT COUNT(*) FROM segments WHERE meeting_id = NEW.meeting_id
        ),
        total_words = (
            SELECT COALESCE(SUM(array_length(string_to_array(text, ' '), 1)), 0) 
            FROM segments WHERE meeting_id = NEW.meeting_id
        )
        WHERE id = NEW.meeting_id;
    ELSIF TG_TABLE_NAME = 'insights' THEN
        UPDATE meetings 
        SET total_insights = (
            SELECT COUNT(*) FROM insights WHERE meeting_id = NEW.meeting_id
        )
        WHERE id = NEW.meeting_id;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for meeting statistics
CREATE TRIGGER update_meeting_stats_segments AFTER INSERT OR UPDATE OR DELETE ON segments
    FOR EACH ROW EXECUTE FUNCTION update_meeting_stats();

CREATE TRIGGER update_meeting_stats_insights AFTER INSERT OR UPDATE OR DELETE ON insights
    FOR EACH ROW EXECUTE FUNCTION update_meeting_stats();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO meeting_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO meeting_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO meeting_user;

-- Insert initial data (optional)
INSERT INTO meetings (title, meeting_type, processing_status, created_at) 
VALUES ('Welcome to AI Meeting Intelligence', 'general', 'completed', CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
END $$; 