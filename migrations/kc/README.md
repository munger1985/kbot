# Knowledge Core migrations

Apply scripts in numeric order through the deployment migration runner. These are
the Knowledge Core-owned migrations for the 4.0 runtime and target the same
APEX Schema used by the other Apps. No script in this directory may alter or
drop legacy `KB/File/TxtChunk` objects.

`001_kc_roots.sql` creates Collection, Collection Binding and Ingestion Receipt.
`002_kc_ingestion_aggregates.sql` creates Bundle, immutable Revision/Document
facts, Parse View and Job records. Both scripts must complete before V2 intake
work is deployed.
`003_kc_evidence.sql` creates parser-produced Evidence. It must be applied
before enabling V2 Parser evidence callbacks.

`004_embedding_model_dimension.sql` adds the explicit vector dimension to the
shared model catalog. Every text embedding model bound by a Collection must be
populated with the dimension fixed by `base.toml`.

`005_kc_retrieval_index.sql` adds the single text vector and model identity
snapshot used by the KC `INDEX` job.

`006_kc_discovery_object.sql` adds Bundle/Document profile projections used by
the first retrieval stage. Profile text is a discovery signal, not citation
content.

`007_kc_relation.sql` adds revision-scoped, Evidence-backed relations. Relation
rows are staged first and cannot cross a Collection or Domain.
