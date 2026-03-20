-- Add daemon_managed column if it doesn't exist
ALTER TABLE system_weaknesses 
ADD COLUMN IF NOT EXISTS daemon_managed BOOLEAN DEFAULT TRUE;

-- Add index for better query performance
CREATE INDEX IF NOT EXISTS idx_weaknesses_severity ON system_weaknesses(severity);
CREATE INDEX IF NOT EXISTS idx_weaknesses_fixed ON system_weaknesses(fixed_at);
