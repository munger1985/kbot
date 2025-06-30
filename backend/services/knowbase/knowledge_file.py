import os
import time
from typing import Optional

from backend.core.log.logger import  logger


class KnowledgeFile:

    def __init__(self,
            # knowledge_base_name: str='knowledge_base_name',          
            # filename: str='default_filename',
            # filepath:str = 'default_filepath',
            # type: str = 'file'
            file_path: str, 
            batch_name: str = './', 
            chunk_size: int = 10000, 
            chunk_overlap: int = 0，
            **kwargs):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。

        '''
        # self.kb_name = kwargs.get('knowledge_base_name')
        # self.filename =kwargs.get('filename')
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.filepath =kwargs.get('filepath')
        if self.type == 'webpage':
            self.ext = '.url'
            # doc_path = get_content_root(knowledge_base_name)
            # url_subpath = os.path.join(doc_path, 'webpages')
            # self.filepath = self.filename
            # loader = WebBaseLoader(filename)
            logger.info(self.filepath)
        # elif type=='audio':
        #     filename = filename.replace('/','-')
        #     self.filepath=get_file_path(knowledge_base_name, filename)
        #     self.ext = os.path.splitext(filename)[-1].lower()
        else:
            self.ext = os.path.splitext(kwargs.get('filename'))[-1].lower() if kwargs.get('filename') else \
                os.path.splitext(kwargs.get('filepath'))[-1].lower()
            # filename = filename.replace('/', '-')
            if not kwargs.get('filepath'):
                self.kbPath = get_kb_path(kwargs.get('knowledge_base_name'))
                fileDiskPath = Path(get_content_root(
                    kwargs.get('knowledge_base_name'))) / self.filename
                if not fileDiskPath.parent.exists:
                    fileDiskPath.parent.mkdir(parents=True)
                self.filepath = str(fileDiskPath)
        if self.ext not in SUPPORTED_EXTS:
            logger.warning(f"暂未支持的文件格式 {self.ext}")
        self.docs = None
        self.texts = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME


        self.file_path = file_path
        self.batch_name = batch_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.filepath = os.path.join(batch_name, file_path)
        self.ext = os.path.splitext(file_path)[1]
        self.knowledge_base_name = os.path.basename(os.path.dirname(file_path))
        self.full_text = ''

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    batch_name = './'
    status = 'success'
    msg = ''
    filename: Optional[str] = None
    filepath: Optional[str] = None
    knowledge_base_name: Optional[str] = None
    ext: Optional[str] = None
    type: str = 'file'
    full_text: str = ''

    def get_mtime(self):
        if self.ext == '.url':
            now = time.time()
            return now
        return os.path.getmtime(self.filepath)

    def get_loader(self, loader_name: str, filepath: str):
        if filepath.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(
                file_path=filepath,
                strategy='hi_res',
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy="by_title",  # section-based chunking
                multipage_sections=False,
                max_characters=self.chunk_size,  # max size of chunks
                new_after_n_chars=self.chunk_size,  # preferred size of chunks
                # combine_text_under_n_chars=100,  # smaller chunks < 100 chars will be combined into a larger chunk
                overlap=self.chunk_overlap,  # overlap between chunks
                mode='elements',
                image_output_dir_path='./figures_unstructured_pdf'
            )
        elif loader_name == 'UnstructuredWordDocumentLoader':
            # loader = Docx2txtLoader(filepath)
            loader = UnstructuredWordDocumentLoader(
                file_path=filepath,
                # 基本参数
                mode="elements",  # 文档处理模式: "single"(整个文档作为一个文档), "elements"(按元素分割), "paged"(按页分割)
                strategy="hi_res",  # 处理策略: "fast"(快速但精度较低) 或 "hi_res"(高精度但较慢)

                # 分块策略参数
                chunking_strategy="by_title",  # 分块策略: "by_title"(按标题分块), "by_page"(按页分块) 或 None
                multipage_sections=False,  # 是否允许章节跨页
                max_characters=self.chunk_size,  # 块的最大字符数
                new_after_n_chars=self.chunk_size,  # 达到此字符数后创建新块
                overlap=self.chunk_overlap,  # 块之间的重叠字符数

                # 表格处理参数
                infer_table_structure=True,  # 是否推断表格结构

                # 图像处理参数
                include_page_breaks=False,  # 是否包含页面分隔符
                include_metadata=True,  # 是否包含元数据

                # 高级选项
                # encoding="utf-8",  # 文件编码
                paragraph_grouper=None,  # 自定义段落分组函数
                headers_to_metadata=False,  # 是否将标题转换为元数据
                skip_infer_table_types=None,  # 跳过推断的表格类型列表
            )
        elif filepath.lower().endswith(".wav"):
            loader = ociSpeechASRLoader(
                self.namespace, self.bucket, self.objectName, self.lang)
        elif loader_name == 'ImageOCRLoader':
            loader = ppOCR(file_path=self.filepath)
        else:
            loader = TextLoader(filepath, autodetect_encoding=True)
        return loader

    def get_size(self):
        if self.ext == '.url':
            return 0  # url not saved to local
        return os.path.getsize(self.filepath)

    def file2docs(self):
        loader = self.get_loader(self.document_loader_name, self.filepath)
        self.docs = loader.load()
        return self.docs
        

    def docs2texts(
            self,
            docs: List[Document] = None,
            text_splitter: TextSplitter = None,
    ):
        if not docs:
            return []
        new_docs: List[Document] = []
        # 对原始数据做清洗，去掉多余的空格等。
        for doc in docs:
            doc.page_content = doc_clean(doc.page_content)
            new_docs.append(doc)
        self.texts = text_splitter.split_documents(new_docs)
        return self.texts

    def url2Docs(
            self,
            text_splitter: TextSplitter = None,
    ):

        loader = WebBaseLoader(self.filepath, encoding='utf-8')
        docs = loader.load()
        docs_transformed = Html2TextTransformer().transform_documents(docs)
        texts = text_splitter.split_documents(docs_transformed)
        return texts

    def file2text(
            self,
            text_splitter: TextSplitter = None,
            lang='en'
    ):
        '''
        for uploading files from web
        :param text_splitter:
        :param lang: when there are some images or audios, specify the language
        :return: chunked langchain documents
        '''
        if self.ext == '.wav':  #### just audios

            # asr to text .
            #  for asr , need to upload  to oci buckets first。
            asrFilePosixPath = Path(self.kbPath) / \
                               Path('asr') / Path(self.filename + '.txt')
            if not asrFilePosixPath.parent.exists():
                asrFilePosixPath.parent.mkdir(parents=True)
            with open(str(asrFilePosixPath), "w") as f:
                f.write(self.full_text)
            document = Document(page_content=self.full_text)
            document.metadata['source'] = self.filepath
            self.docs = [document]

        elif self.document_loader_name == 'ImageOCRLoader':  ## just images
            fileTextOneString = ppOCR(self.filepath, lang)
            ocrFilePosixPath = Path(self.kbPath) / \
                               Path('ocr') / Path(self.filename + '.txt')
            if not ocrFilePosixPath.parent.exists():
                ocrFilePosixPath.parent.mkdir(parents=True)
            with open(str(ocrFilePosixPath), "w") as f:
                f.write(fileTextOneString)
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            self.docs = [document]

        else:  ### if this is just docs
            self.docs = self.file2docs()
        ### because we reckon unstructuredXXXloader is smarter, no need splitter
        if self.ext in ['.pdf', '.doc', '.docx']:
            self.texts = []
            for doc in self.docs:
                doc.metadata.pop('last_modified', None)
                doc.metadata.pop('text_as_html', None)
                doc.metadata.pop('file_directory', None)
                doc.metadata.pop('filename', None)
                doc.metadata.pop('languages', None)
                doc.metadata.pop('orig_elements', None)
                self.texts.append(doc)
        else:
            self.texts = self.docs2texts(docs=self.docs,
                                         text_splitter=text_splitter)
        self.full_text = "\n".join(doc.page_content for doc in self.docs)

        copy2Graphrag(self)

        return self.texts

    def rebuild_file2text(
            self,
            text_splitter: TextSplitter = None,
    ):
        '''
        when recreating the whole vectorStore, scanning all the text files beneath the content Root dir of a certain knowledge base
        :param text_splitter:
        :return: chunked langchain documents
        '''
        if self.ext == '.wav':
            # asr to text .
            asrFilePosixPath = Path(self.kbPath) / \
                               Path('asr') / Path(self.filename + '.txt')
            with open(str(asrFilePosixPath), "r") as f:
                fileTextOneString = f.read()
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            docs = [document]
        elif self.document_loader_name == 'ImageOCRLoader':
            ocrFilePosixPath = Path(self.kbPath) / \
                               Path('ocr') / Path(self.filename + '.txt')
            with open(str(ocrFilePosixPath), "r") as f:
                fileTextOneString = f.read()
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            docs = [document]
        else:
            docs = self.file2docs()

        self.texts = self.docs2texts(docs=docs,
                                     text_splitter=text_splitter)
        return self.texts
