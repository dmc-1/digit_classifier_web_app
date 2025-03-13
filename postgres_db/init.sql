CREATE TABLE IF NOT EXISTS history (
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    pred VARCHAR(1),
    label VARCHAR(1)
)