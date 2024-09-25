from langchain_community.document_loaders import  Docx2txtLoader



loader =  Docx2txtLoader('信息化建设项目管理规定_脱敏_全文(1).docx')
xxxx = loader.load()
x=1


# put it in kb_llm_api.py can test single method

if __name__ == '__main__':
    # ask_conversational_rag('how to add security list','genai','s','s')
    SYSTEM_PROMPT = """You are a contact center bot of Changi Airport. Your name is Max. Your task is to help airport customers to provide them best customer service through answering he customer queries. Use the below given context to answer the customer queries. If there is anything that you cannot answer, or you think is inappropriate to answer, simply reply as, "Sorry, I cannot help you with that."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT_template = B_SYS + SYSTEM_PROMPT + E_SYS
    context_instruction_template = "CHAT HISTORY: {chat_history}\n----------\nCONTEXT: {context}\n----------\n\nInstructions:1. Answer only from the given context.\n             2: Do not generate any new content out of this context.\n             3: Your answer should not include any harmful, unethical, violent, racist, sexist, pornographic, toxic, discriminatory, blasphemous, dangerous, or illegal content.\n             4: Please ensure that your responses are socially unbiased and positive in nature.\n             5: Ensure length of the answer  is within 300 words.\n\nNow, Answer the following question: {question}\n"
    # context_instruction_template ="\n----------\nCONTEXT: {context}\n----------\n\nInstructions:1. Answer only from the given context.\n             2: Do not generate any new content out of this context.\n             3: Your answer should not include any harmful, unethical, violent, racist, sexist, pornographic, toxic, discriminatory, blasphemous, dangerous, or illegal content.\n             4: Please ensure that your responses are socially unbiased and positive in nature.\n             5: Ensure length of the answer  is within 300 words.\n\nNow, Answer the following question: {question}\n"
    init_p = '<s>' + B_INST + SYSTEM_PROMPT_template + context_instruction_template + E_INST
    prompt_template = init_p

    # prompt = PromptTemplate(template=prompt_template, input_variables=["context",  "question"])
    # p1=prompt.format(question='what is ppg in airport',
    #                              context='ppg is a white guide dog in this airport'
    #                            )
    llm = config.MODEL_DICT.get('genai')
    diagnosis_prompt = ChatPromptTemplate(
        messages=[
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(init_p),
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot for CHangi airport, your name is mimi"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # diagnosis_memory = ConversationSummaryBufferMemory(llm=llm, memory_key="diagnosis_chat_history", max_token_limit=16000,
    #                                                 return_messages=True)

    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key='answer')

    conversation = ConversationChain(
        llm=llm,
        # We set a low k=2, to only keep the last 2 interactions in memory
        memory=memory,
        verbose=True,
        input_key="input",
        output_key="answer",
        prompt=prompt
    )
    question = 'how to create certificate in oci'

    vector_res_arr, llm_context = makeSimilarDocs(question, 's')
    pt = f"CONTEXT: {llm_context}\n----------\n\nInstructions:1. Answer only from the given context.\n             2: Do not generate any new content out of this context.\n             3: Your answer should not include any harmful, unethical, violent, racist, sexist, pornographic, toxic, discriminatory, blasphemous, dangerous, or illegal content.\n             4: Please ensure that your responses are socially unbiased and positive in nature.\n             5: Ensure length of the answer  is within 300 words.\n\nNow, Answer the following question: {question}\n"
    inputs = {"input": pt}

    response = conversation.invoke(inputs)
    print(response.get('answer'))
    inputs = {"input": "hi, im bob, i am 23, what is your name"}
    conversation.memory.clear()
    response = conversation.invoke(inputs)
    print(response.get('answer'))

    inputs = {"input": "where did i ask you to create certificate in? "}

    response = conversation.invoke(inputs)
    print(response.get('answer'))

    response = conversation.invoke(inputs)

    vector_res_arr, llm_context = makeSimilarDocs('how to create certificate in oci', 's')

    s = conversation.invoke({"question": 'how to create certificate in oci', "context": llm_context})
    s = conversation.invoke({"question": 'how to create certificate in oci', "context": llm_context})

    s = conversation.invoke({"question": 'how to create certificate in oci', "context": llm_context})
    s = conversation('what is ppg')
    s = conversation('what is color of ppg')

    # dd = memory2.load_memory_variables({})
    vector_res_arr, llm_context = makeSimilarDocs('how to create certificate in oci', 's')
    p1 = prompt.format(question='how to create certificate in oci',
                       context=llm_context
                       )
    # s=conversation_with_summary(p1)

    # s=conversation_with_summary('what is ppg')
    # s=conversation_with_summary('what is the task you are serving')
    # s=conversation_with_summary('what is ppg in airport')

    # s=conversation_with_summary.invoke({'question':'how to add security list in oci',
    #                              'context':'open console, click on the security list create button',
    #                              'chat_history':'ppg is a china dog ,color is white'})

    # s=conversation_with_summary.invoke({'question':'what is ppg'})
    sad = 2

