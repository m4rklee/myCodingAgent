"""
A-mem-sys 生产版自研复刻（不 import 官方包）。

对齐 https://github.com/WujiangXu/A-mem-sys 的 agentic_memory/：
  - MemoryNote + ChromaRetriever + JSON schema 单调用进化 + UUID links
供 harness MEMORY_MODE="official" 使用；评测 robust 版见 official_eval + amem_official.py。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
EVO_THRESHOLD = 100

ANALYZE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

EVOLUTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                "new_tags_neighborhood": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "required": [
                "should_evolve", "actions", "suggested_connections",
                "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVOLUTION_SYSTEM_PROMPT = """
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories (each line starts with memory_id):
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
2.1 If choose to strengthen the connection, which memory should it be connected to? Use the memory_id from the neighbors above. Can you give the updated tags of this memory?
2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
"should_evolve": True or False,
"actions": ["strengthen", "update_neighbor"],
"suggested_connections": ["memory_id_1", "memory_id_2", ...],
"tags_to_update": ["tag_1",..."tag_n"],
"new_context_neighborhood": ["new context",...,"new context"],
"new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}
"""


class MemoryNote:
    """对齐 A-mem-sys MemoryNote（构造时不调 LLM）。"""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[List] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


def _note_metadata(note: MemoryNote) -> Dict[str, Any]:
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "context": note.context,
        "evolution_history": note.evolution_history,
        "category": note.category,
        "tags": note.tags,
    }


class ChromaRetriever:
    """对齐 A-mem-sys retrievers.ChromaRetriever。"""

    def __init__(self, collection_name: str = "memories",
                 model_name: str = DEFAULT_EMBED_MODEL):
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.model_name = model_name
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def _build_enhanced_document(self, document: str, metadata: Dict) -> str:
        enhanced = document
        ctx = metadata.get("context", "")
        if ctx and ctx != "General":
            enhanced += f" context: {ctx}"
        keywords = metadata.get("keywords", [])
        if keywords:
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except json.JSONDecodeError:
                    keywords = []
            if keywords:
                enhanced += f" keywords: {', '.join(keywords)}"
        tags = metadata.get("tags", [])
        if tags:
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = []
            if tags:
                enhanced += f" tags: {', '.join(tags)}"
        return enhanced

    def add_document(self, document: str, metadata: Dict, doc_id: str) -> None:
        enhanced = self._build_enhanced_document(document, metadata)
        processed: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed[key] = json.dumps(value)
            else:
                processed[key] = str(value)
        processed["enhanced_content"] = enhanced
        self.collection.add(
            documents=[enhanced],
            metadatas=[processed],
            ids=[doc_id],
        )

    def delete_document(self, doc_id: str) -> None:
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass

    def search(self, query: str, k: int = 5) -> Dict:
        results = self.collection.query(query_texts=[query], n_results=k)
        if "metadatas" in results and results["metadatas"]:
            for i in range(len(results["metadatas"])):
                if not isinstance(results["metadatas"][i], list):
                    continue
                for j in range(len(results["metadatas"][i])):
                    meta = results["metadatas"][i][j]
                    if not isinstance(meta, dict):
                        continue
                    for key, value in list(meta.items()):
                        if isinstance(value, str) and (
                                value.startswith("[") or value.startswith("{")):
                            try:
                                meta[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                pass
        return results


class SysLLMController:
    """包装 OpenAI 兼容 client，对齐官方 JSON schema 调用。"""

    SYSTEM_MESSAGE = "You must respond with a JSON object."

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def get_completion(self, prompt: str, response_format: dict,
                       temperature: float = 1.0,
                       max_tokens: Optional[int] = None) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("JSON schema call failed (%s), retrying without schema", e)
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                max_tokens=max_tokens or 1000)
            return (resp.choices[0].message.content or "").strip()


class OfficialMemorySystem:
    """对齐 A-mem-sys AgenticMemorySystem。"""

    def __init__(self, client, model: str,
                 model_name: str = DEFAULT_EMBED_MODEL,
                 evo_threshold: int = EVO_THRESHOLD):
        self.model_name = model_name
        self.memories: Dict[str, MemoryNote] = {}
        self.llm = SysLLMController(client, model)
        try:
            temp = ChromaRetriever(collection_name="memories", model_name=model_name)
            temp.client.reset()
        except Exception as e:
            logger.warning("ChromaDB reset: %s", e)
        self.retriever = ChromaRetriever(collection_name="memories",
                                         model_name=model_name)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def analyze_content(self, content: str) -> Dict:
        prompt = (
            "Generate a structured analysis of the following content by:\n"
            "1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)\n"
            "2. Extracting core themes and contextual elements\n"
            "3. Creating relevant categorical tags\n\n"
            "Format the response as a JSON object:\n"
            "{\n"
            '  "keywords": [],\n'
            '  "context": "",\n'
            '  "tags": []\n'
            "}\n\n"
            "Content for analysis:\n" + content
        )
        try:
            response = self.llm.get_completion(prompt, response_format=ANALYZE_JSON_SCHEMA)
            return json.loads(response)
        except Exception as e:
            logger.error("analyze_content failed: %s", e)
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        if time is not None:
            kwargs["timestamp"] = time
        note = MemoryNote(content=content, **kwargs)
        needs_analysis = (
            not note.keywords
            or note.context == "General"
            or not note.tags
        )
        if needs_analysis:
            analysis = self.analyze_content(content)
            if not note.keywords:
                note.keywords = analysis.get("keywords", [])
            if note.context == "General":
                note.context = analysis.get("context", "General")
            if not note.tags:
                note.tags = analysis.get("tags", [])

        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_document(note.content, _note_metadata(note), note.id)

        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        if not self.memories:
            return False, note
        try:
            neighbors_text, memory_ids = self.find_related_memories(
                note.content, k=5)
            if not neighbors_text or not memory_ids:
                return False, note

            prompt = EVOLUTION_SYSTEM_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(memory_ids),
            )
            response = self.llm.get_completion(
                prompt, response_format=EVOLUTION_JSON_SCHEMA)
            response_json = json.loads(response)
            should_evolve = response_json.get("should_evolve", False)

            if should_evolve:
                for action in response_json.get("actions", []):
                    if action == "strengthen":
                        note.links.extend(
                            response_json.get("suggested_connections", []))
                        new_tags = response_json.get("tags_to_update")
                        if new_tags:
                            note.tags = new_tags
                    elif action == "update_neighbor":
                        new_ctx = response_json.get(
                            "new_context_neighborhood", [])
                        new_tags_nb = response_json.get(
                            "new_tags_neighborhood", [])
                        for i in range(min(len(memory_ids), len(new_tags_nb))):
                            mid = memory_ids[i]
                            if mid not in self.memories:
                                continue
                            neighbor = self.memories[mid]
                            if i < len(new_tags_nb):
                                neighbor.tags = new_tags_nb[i]
                            if i < len(new_ctx):
                                neighbor.context = new_ctx[i]
                            self.memories[mid] = neighbor
            return should_evolve, note
        except Exception as e:
            logger.error("process_memory failed: %s", e)
            return False, note

    def consolidate_memories(self) -> None:
        self.retriever = ChromaRetriever(
            collection_name="memories", model_name=self.model_name)
        for memory in self.memories.values():
            self.retriever.add_document(
                memory.content, _note_metadata(memory), memory.id)

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        if not self.memories:
            return "", []
        try:
            results = self.retriever.search(query, k)
            memory_str = ""
            memory_ids: List[str] = []
            ids_batch = (results.get("ids") or [[]])[0]
            metas_batch = (results.get("metadatas") or [[]])[0]
            for i, doc_id in enumerate(ids_batch):
                if i >= len(metas_batch):
                    continue
                metadata = metas_batch[i]
                memory_str += (
                    f"memory_id:{doc_id}\t"
                    f"talk start time:{metadata.get('timestamp', '')}\t"
                    f"memory content: {metadata.get('content', '')}\t"
                    f"memory context: {metadata.get('context', '')}\t"
                    f"memory keywords: {str(metadata.get('keywords', []))}\t"
                    f"memory tags: {str(metadata.get('tags', []))}\n"
                )
                memory_ids.append(doc_id)
            return memory_str, memory_ids
        except Exception as e:
            logger.error("find_related_memories: %s", e)
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        if not self.memories:
            return ""
        results = self.retriever.search(query, k)
        memory_str = ""
        ids_batch = (results.get("ids") or [[]])[0]
        metas_batch = (results.get("metadatas") or [[]])[0]
        for i, doc_id in enumerate(ids_batch[:k]):
            if i >= len(metas_batch):
                continue
            metadata = metas_batch[i]
            memory_str += (
                f"talk start time:{metadata.get('timestamp', '')}\t"
                f"memory content: {metadata.get('content', '')}\t"
                f"memory context: {metadata.get('context', '')}\t"
                f"memory keywords: {str(metadata.get('keywords', []))}\t"
                f"memory tags: {str(metadata.get('tags', []))}\n"
            )
            links = metadata.get("links", [])
            if isinstance(links, str):
                try:
                    links = json.loads(links)
                except json.JSONDecodeError:
                    links = []
            j = 0
            for link_id in links:
                if link_id in self.memories and j < k:
                    neighbor = self.memories[link_id]
                    memory_str += (
                        f"talk start time:{neighbor.timestamp}\t"
                        f"memory content: {neighbor.content}\t"
                        f"memory context: {neighbor.context}\t"
                        f"memory keywords: {str(neighbor.keywords)}\t"
                        f"memory tags: {str(neighbor.tags)}\n"
                    )
                    j += 1
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.memories:
            return False
        note = self.memories[memory_id]
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(note.content, _note_metadata(note), memory_id)
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self.memories:
            return False
        del self.memories[memory_id]
        self.retriever.delete_document(memory_id)
        return True

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, k)
        memories = []
        ids_batch = (results.get("ids") or [[]])[0]
        dist_batch = (results.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids_batch):
            memory = self.memories.get(doc_id)
            if memory:
                entry = {
                    "id": doc_id,
                    "content": memory.content,
                    "context": memory.context,
                    "keywords": memory.keywords,
                    "tags": memory.tags,
                    "score": dist_batch[i] if i < len(dist_batch) else 0.0,
                }
                memories.append(entry)
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.memories:
            return []
        try:
            results = self.retriever.search(query, k)
            memories: List[Dict[str, Any]] = []
            seen_ids: set = set()
            ids_batch = (results.get("ids") or [[]])[0]
            metas_batch = (results.get("metadatas") or [[]])[0]
            dist_batch = (results.get("distances") or [[]])[0]
            if not ids_batch:
                return []

            for i, doc_id in enumerate(ids_batch[:k]):
                if doc_id in seen_ids or i >= len(metas_batch):
                    continue
                metadata = metas_batch[i]
                memory_dict = {
                    "id": doc_id,
                    "content": metadata.get("content", ""),
                    "context": metadata.get("context", ""),
                    "keywords": metadata.get("keywords", []),
                    "tags": metadata.get("tags", []),
                    "timestamp": metadata.get("timestamp", ""),
                    "category": metadata.get("category", "Uncategorized"),
                    "is_neighbor": False,
                }
                if i < len(dist_batch):
                    memory_dict["score"] = dist_batch[i]
                memories.append(memory_dict)
                seen_ids.add(doc_id)

            neighbor_count = 0
            for memory in list(memories):
                if neighbor_count >= k:
                    break
                links = memory.get("links", [])
                if not links:
                    mem_obj = self.memories.get(memory["id"])
                    if mem_obj:
                        links = mem_obj.links
                if isinstance(links, str):
                    try:
                        links = json.loads(links)
                    except json.JSONDecodeError:
                        links = []
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                "id": link_id,
                                "content": neighbor.content,
                                "context": neighbor.context,
                                "keywords": neighbor.keywords,
                                "tags": neighbor.tags,
                                "timestamp": neighbor.timestamp,
                                "category": neighbor.category,
                                "is_neighbor": True,
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            return memories[:k]
        except Exception as e:
            logger.error("search_agentic: %s", e)
            return []
