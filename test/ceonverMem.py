from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2, return_messages=True, output_key='output')

memory.save_context(inputs={"input": "dsdad"}, outputs={"output": f'found some context: \n'})

memory.save_context(inputs={"input": "123"}, outputs={"output": f'found some context: \n'})
memory.save_context(inputs={"input": "444"}, outputs={"output": f'found some context: \n'})
dd= memory.load_memory_variables({})
print(dd)




ff='asdasd:123'
# 注意：旋转后的图片可能需要进行裁剪或填充以保持原始尺寸，这取决于 rotate 函数的参数设置
if not ff.endswith('/'):
    ff+= '/'


text = "This is a very long string that needs to be split into smaller documents..."

# Split the text into chunks of 1000 characters
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_text(text)
from langchain_community.document_loaders import TextLoader
# Create documents from the chunks
loader = TextLoader.from_texts(text)
documents = loader.load()



print(ff+'asdasdsad')

