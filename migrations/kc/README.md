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

`004_embedding_model_dimension.sql` records the deployment-order boundary:
Model Serving owns the dimension and UUIDv7 model identity, so KC migrations
must not alter its catalog.

`005_kc_retrieval_index.sql` adds the single text vector, UUIDv7 model identity
and served-model-name snapshot used by the KC `INDEX` job.

`006_kc_discovery_object.sql` adds Bundle/Document profile projections used by
the first retrieval stage. Profile text is a discovery signal, not citation
content.

`007_kc_relation.sql` adds revision-scoped, Evidence-backed relations. Relation
rows are staged first and cannot cross a Collection or Domain.
