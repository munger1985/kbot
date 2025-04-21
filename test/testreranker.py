import oci


config = oci.config.from_file( )
region = 'ap-osaka-1'
model_id='cohere.rerank-multilingual-v3.1'
compartment_id='ocid1.compartment.oc1..aaaaaaaau5q457a7teqkjce4oenoiz6bmc4g3s74a5543iqbm7xwplho44fq'

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240))
batch_size=3
inputs= [
    "you good",

    "citrus,",
    "fruit,"
]
input_text='apple'


rerank_text_response = generative_ai_inference_client.rerank_text(
    rerank_text_details=oci.generative_ai_inference.models.RerankTextDetails(
        input=input_text,
        compartment_id=compartment_id,
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            serving_type="ON_DEMAND",
            model_id=model_id),
        documents=inputs ,
        # top_n=690,
        # is_echo=False,
        # max_chunks_per_document=618
    )
    )

print(rerank_text_response.data)
for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i + batch_size]

    rerank_text_detail = oci.generative_ai_inference.models.RerankTextDetails()
    rerank_text_detail.input = input_text
    rerank_text_detail.documents = batch
    rerank_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        serving_type="ON_DEMAND",
        model_id=model_id
    )
    rerank_text_detail.compartment_id = compartment_id
    rerank_response = rerank_generative_ai_inference_client.rerank_text(rerank_text_detail)
    # print(f"Processed batch {i // batch_size + 1} of {(len(inputs) - 1) // batch_size + 1}")
    adjusted_results = []
    for rank in rerank_response.data.document_ranks:
        adjusted_result = {
            "document": rank.document,
            "index": i + rank.index,
            "relevance_score": rank.relevance_score
        }
        adjusted_results.append(adjusted_result)