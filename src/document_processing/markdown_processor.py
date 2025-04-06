from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm

class MarkdownProcessor:
    """
    Class for loading and processing markdown files into chunks suitable for embedding.
    """
    def __init__(self, folder_path, file_pattern="**/*.md"):
        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.loaded_docs = []
        self.chunks = []
        self.split_headers = [
            ("#", "Level 1"),
            ("##", "Level 2"),
            ("###", "Level 3"),
            ("####", "Level 4"),
        ]

    def load_markdown_files(self):
        """Load markdown files from the specified directory."""
        loader = DirectoryLoader(
            path=self.folder_path,
            glob=self.file_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        self.loaded_docs = loader.load()
        print(f"Documents loaded: {len(self.loaded_docs)}")
        return self.loaded_docs

    def extract_chunks(self):
        """
        Extract chunks from loaded markdown documents, preserving header hierarchy.
        """
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.split_headers, 
            strip_headers=False
        )
        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, 
            chunk_overlap=30
        )

        total_chunks = 0
        for doc in tqdm(self.loaded_docs, desc="Processing documents"):
            markdown_sections = markdown_splitter.split_text(doc.page_content)

            for section in markdown_sections:
                # Keep metadata from the original document and add section headers
                section.metadata = {
                    **doc.metadata,
                    **section.metadata,
                    "source_filename": doc.metadata.get('source', 'unknown')
                }

            chunks = chunk_splitter.split_documents(markdown_sections)
            self.chunks.extend(chunks)
            print(f"Markdown sections for this doc: {len(markdown_sections)}")
            print(f"Chunks for this doc: {len(chunks)}")
            
            total_chunks += len(chunks)

        print(f"Total chunks created: {total_chunks}")
        return self.chunks 