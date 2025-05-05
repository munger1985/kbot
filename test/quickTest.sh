url="http://localhost:8093"
model="OCI-meta.llama-3.1-405b-instruct"
#VECTOR_STORE_DICT = [
#    'faiss',
#    'oracle',
#    'adb',
#    'opensearch',
#    'heatwave'
#]

vector_store_type="oracle"
response=$(curl  -s -X 'POST' \
  "${url}/knowledge_base/create_knowledge_base" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "knowledge_base_name": "ssTestss",
  "knowledge_base_info": "this is about bank",
  "vector_store_type": "'"$vector_store_type"'",
  "embed_model": "bge_m3"
}' |jq '.code'  )

if [ "$response" -eq 200 ]; then
  echo "create KB is successful (HTTP 200)"
else
  echo "create KB failed (HTTP $response)"
fi

####################################################################

response=$(curl   -s   -X 'POST' \
   "${url}/knowledge_base/upload_from_urls" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "urls": [
    "https://python.langchain.com/docs/integrations/vectorstores/opensearch/"
  ],
  "knowledge_base_name": "ssTestss",
  "batch_name": "./",
  "max_depth": 1,
  "chunk_size": 500,
  "chunk_overlap": 50
}' |jq '.code')

if [ "$response" -eq 200 ]; then
  echo "upload docs from url is successful (HTTP 200)"
else
  echo "upload docs from url failed (HTTP $response)"
fi

####################################################################


response=$(curl -o /dev/null -s -w "%{http_code}" -X 'POST' \
  "${url}/chat/with_rag" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user": "Demo",
  "ask": "opensearch'\''s usage",
  "kb_name": "ssTestss",
  "llm_model": "'"$model"'",
  "prompt_name": "rag_default",
  "rerankerModel": "bgeReranker",
  "reranker_topk": 2,
  "score_threshold": 0.6,
  "vector_store_limit": 10,
  "search_type": "vector"
}')

if [ "$response" -eq 200 ]; then
  echo "RAG is successful (HTTP 200)"
else
  echo "RAG failed (HTTP $response)"
fi
####################################################################

response=$(curl -o /dev/null -s -w "%{http_code}" -X 'POST' \
    "${url}/knowledge_base/delete_kb" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "knowledge_base_name": "ssTestss",
  "stub": "for json body, no need to input "
}' )

if [ "$response" -eq 200 ]; then
  echo "delete KB is successful (HTTP 200)"
else
  echo "delete KB failed (HTTP $response)"
fi


####################################################################


curl -X 'POST' \
  "${url}/chat/translate" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Kbot testing translation with genai == OK ",
  "llm_model": "'"$model"'",
  "language": "Chinese"
}'
