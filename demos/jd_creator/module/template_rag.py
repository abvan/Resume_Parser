import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from utils.llm_client import GroqLLMClient
from PyPDF2 import PdfReader  # for PDF reading


class TemplateRAG:
    def __init__(self, persist_directory: str = None):
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.Client(
            settings=chromadb.config.Settings(
                persist_directory=persist_directory if persist_directory else None
            )
        )
        self.collection = self.client.create_collection(
            name="jd_templates",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.llm = GroqLLMClient()

    def _extract_text_from_file(self, file_path: str) -> str:
        """Read PDF or TXT and return plain text."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            raise ValueError("Only PDF or TXT files are supported.")

    def load_reference_template(self, text: str):
        if not text.strip():
            raise ValueError("Reference JD text is empty.")
        chunks = self._chunk_text(text, chunk_size=512)
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[f"ref_{i}"],
                documents=[chunk]
            )

    def adapt_jd_to_template(self, generated_jd: str, top_k: int = 3) -> str:
        if self.collection.count() == 0:
            raise ValueError("No reference JD loaded. Please call load_reference_template first.")
        retrieved = self.collection.query(
            query_texts=[generated_jd],
            n_results=top_k
        )
        retrieved_context = "\n".join(retrieved["documents"][0])
        system_prompt = (
            "You are an expert HR content editor. Adapt the given generated job description "
            "to match the tone, style, and structure of the provided reference job description."
        )
        user_prompt = (
            f"Reference JD Style Context:\n{retrieved_context}\n\n"
            f"Generated JD:\n{generated_jd}\n\n"
            "Rewrite the Generated JD so that it closely matches the tone and style of the Reference JD, "
            "while keeping all factual content intact."
        )
        adapted_jd = self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
        return adapted_jd

    def compare_to_template(self, generated_jd: str, uploaded_file_path: str) -> str:
        """Full process: read file → embed → adapt JD."""
        reference_text = self._extract_text_from_file(uploaded_file_path)
        self.load_reference_template(reference_text)
        return self.adapt_jd_to_template(generated_jd)

    def clear_store(self):
        self.client.delete_collection(name="jd_templates")
        self.collection = self.client.create_collection(
            name="jd_templates",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512):
        words = text.split()
        chunks, current_chunk = [], []
        current_length = 0
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks