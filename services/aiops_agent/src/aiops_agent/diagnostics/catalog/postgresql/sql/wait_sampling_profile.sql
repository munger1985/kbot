SELECT
    profile.event_type AS wait_event_type,
    profile.event AS wait_event,
    CAST(profile.queryid AS text) AS query_id,
    profile.count AS sample_count
FROM pg_wait_sampling_profile AS profile
ORDER BY profile.count DESC, profile.event_type, profile.event
LIMIT :limit
