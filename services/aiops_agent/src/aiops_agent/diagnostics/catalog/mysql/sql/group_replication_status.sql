SELECT
    channel_name,
    member_id,
    member_host,
    member_port,
    member_state,
    member_role,
    member_version
FROM performance_schema.replication_group_members
ORDER BY member_id
