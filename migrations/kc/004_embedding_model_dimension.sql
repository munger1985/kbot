-- KBot 3.5: one text embedding model per Collection, one physical vector dimension.
-- KBOT_MD_MODELS is the shared platform model catalog and already exists.

ALTER TABLE KBOT_MD_MODELS ADD (
    EMBEDDING_DIMENSION NUMBER(10)
);

ALTER TABLE KBOT_MD_MODELS ADD CONSTRAINT CK_MD_MODEL_EMBED_DIM
    CHECK (EMBEDDING_DIMENSION IS NULL OR EMBEDDING_DIMENSION > 0);

ALTER TABLE KBOT_MD_MODELS ADD CONSTRAINT CK_MD_TEXT_EMBED_DIM_REQUIRED
    CHECK (CATEGORY <> 2 OR EMBEDDING_DIMENSION IS NOT NULL);

-- Deployment validation must populate text embedding model dimensions and verify
-- they equal configuration/base.toml [embed].dimensions before V2 Collections
-- are created or enabled. Non-embedding model rows keep this column NULL.
