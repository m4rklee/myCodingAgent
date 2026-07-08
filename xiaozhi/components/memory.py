"""Agent 记忆系统：文件持久化的记忆条目 + LLM 提取/整合/检索。

参考 A-Mem (Agentic Memory for LLM Agents, NeurIPS 2025) 增强：
- 记忆写入时 LLM 抽取 keywords/context/tags 结构化属性；
- 新记忆入库自动建链（向量近邻 + LLM 决策）、反向更新关联记忆的 context/tags；
- 召回三级规则 + LLM 语义双路融合 + links 多跳邻居扩展，返回结构化上下文；
- 超阈值 LLM 合并去重、清理过时记忆。

与原版差异：``memory_dir`` / ``client`` / ``model`` 通过构造函数注入，不引用全局。
向量/持久化档（hybrid/persist/official）的第三方依赖惰性加载，lite 档零依赖。
"""

import time
import json
import hashlib
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from xiaozhi.json_utils import extract_json_array
from xiaozhi.message_utils import extract_message_text

MEMORY_TYPES = ["user", "feedback", "project", "reference"]

# ── A-Mem（NeurIPS 2025, Agentic Memory）风格增强的总开关 ──
# 参考 A-Mem：为记忆生成 LLM 结构化属性(keywords/context/tags)、自动建链(links)、
# 记忆进化，并在召回时带出 linked 邻居（多跳召回）。
# 关掉此开关即回退到原有的「规则三级排序 + LLM 选下标」双路召回，行为完全不变。
ENABLE_AMEM = True

# 建链/进化时取的候选邻居数（对应 A-Mem 的 top-k）
AMEM_LINK_TOPK = 5
# frontmatter 中以列表形式存储的字段（写入/解析时按 JSON 处理）
_LIST_FIELDS = ("keywords", "tags", "links")

# ── 召回模式开关 ──
# "lite"     : 纯规则三级排序 + LLM 选下标（零向量库依赖，写入/检索最轻量）
# "hybrid"   : dense 向量 + BM25 稀疏 + 规则 三路归一化融合召回（精度模式，检索时现算向量）
# "persist"  : ChromaDB 持久化向量库（入库即存向量）+ 三路融合，小智增量版
# "official"      : A-mem-sys 生产版（ChromaDB + UUID links + JSON schema 进化，见 amem_sys.py）
# "official_eval" : AgenticMemory robust 评测版（in-memory 向量 + 3-call 纯文本 + 整数 links）
# 官方 A-Mem 仅单路 dense 向量；hybrid/persist 用「单路→多路」互补争取更强召回。
# 设回 "lite" 即彻底回退到零依赖行为。
MEMORY_MODE = "lite"

# hybrid 三路融合权重（dense 偏语义、bm25 补精确关键词、rule 补 tag/新鲜度）
HYBRID_W_DENSE = 0.5
HYBRID_W_BM25 = 0.3
HYBRID_W_RULE = 0.2
# hybrid 用的 embedding 模型（与官方 A-Mem 同款，保证公平对比）
HYBRID_EMBED_MODEL = "all-MiniLM-L6-v2"
# persist 档：ChromaDB 持久化目录（惰性创建，仅 persist 模式使用）
CHROMA_SUBDIR = ".chroma"
CHROMA_COLLECTION = "xiaozhi_memory"
# 官方 / 小智：每积累 N 次成功进化触发 consolidate（仅重建向量索引，无 LLM 合并）
AMEM_EVO_THRESHOLD = 100

# 中文常见虚词，分词后剔除以降低噪声（规则召回用）
_ZH_STOPWORDS = {
    "的", "了", "和", "是", "我", "你", "他", "她", "它", "在", "有", "个",
    "these", "那", "这", "就", "都", "也", "要", "把", "被", "给", "请",
}
# 记忆类型的中文近义词，用于「type 命中」这一级排序
_TYPE_HINTS = {
    "user": {"偏好", "喜欢", "习惯", "名字", "称呼", "preference", "user"},
    "feedback": {"指引", "反馈", "要求", "希望", "以后", "feedback", "应该"},
    "project": {"项目", "文件", "路径", "代码", "任务", "project", "端口"},
    "reference": {"链接", "地址", "文档", "参考", "reference", "url", "dashboard"},
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> set:
    """中英文混合分词：英文/数字按词，中文按单字（轻量、无第三方依赖）。

    参考 pico-custom 的规则召回思路：不引入 embedding，保持透明可解释。
    """
    text = str(text).lower()
    tokens = set(_TOKEN_RE.findall(text))
    # 中文逐字加入（去掉已被英文正则消费的字符后剩下的 CJK）
    for ch in re.findall(r"[一-鿿]", text):
        tokens.add(ch)
    return {t for t in tokens if t not in _ZH_STOPWORDS and len(t) >= 1}


@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _minmax(scores):
    """把一组分数 min-max 归一化到 [0,1]；全相等时返回全 0。"""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    """dense 向量 + BM25 稀疏 检索器（惰性加载，仅 hybrid 模式使用）。

    - dense：SentenceTransformer 对「记忆全文拼接」编码，与 query 向量算余弦。
    - bm25 ：对记忆语料做 BM25，补精确关键词/低频词的召回。
    lite 模式下本类完全不被实例化，不引入任何加载开销。
    """

    def __init__(self, model_name: str = HYBRID_EMBED_MODEL):
        self.model_name = model_name
        self._model = None  # 惰性加载

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _doc_text(f: dict) -> str:
        """拼接记忆的可检索全文（对齐官方 enhanced_document：正文+属性一起编码）。"""
        parts = [f.get("name", ""), f.get("description", ""), f.get("body", ""),
                 f.get("context", ""), " ".join(f.get("keywords", [])),
                 " ".join(f.get("tags", []))]
        return " ".join(p for p in parts if p)

    def dense_scores(self, query: str, files: list) -> list:
        """query 与各记忆的余弦相似度（min-max 归一化到 [0,1]）。"""
        try:
            from sentence_transformers.util import cos_sim
            model = self._get_model()
            docs = [self._doc_text(f) for f in files]
            q_emb = model.encode([query], convert_to_tensor=True)
            d_emb = model.encode(docs, convert_to_tensor=True)
            sims = cos_sim(q_emb, d_emb)[0].tolist()
            return _minmax(sims)
        except Exception:
            return [0.0] * len(files)

    def bm25_scores(self, query: str, files: list) -> list:
        """BM25 稀疏检索分（min-max 归一化到 [0,1]）。"""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [list(_tokenize(self._doc_text(f))) for f in files]
            if not any(corpus):
                return [0.0] * len(files)
            bm25 = BM25Okapi(corpus)
            raw = list(bm25.get_scores(list(_tokenize(query))))
            return _minmax(raw)
        except Exception:
            return [0.0] * len(files)


class SimpleEmbeddingRetriever:
    """官方评测版 in-memory 向量检索（与 memory_layer.SimpleEmbeddingRetriever 同构）。"""

    def __init__(self, model_name: str = HYBRID_EMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self.corpus: List[str] = []
        self.embeddings = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def add_documents(self, documents: List[str]) -> None:
        import numpy as np
        model = self._get_model()
        if not self.corpus:
            self.corpus = list(documents)
            self.embeddings = model.encode(self.corpus)
        else:
            new_embeddings = model.encode(documents)
            self.corpus.extend(documents)
            self.embeddings = (new_embeddings if self.embeddings is None
                               else np.vstack([self.embeddings, new_embeddings]))

    def search(self, query: str, k: int = 5) -> List[int]:
        """返回 corpus 下标列表（降序相似度），对齐官方 retriever.search。"""
        if not self.corpus:
            return []
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            q_emb = self._get_model().encode([query])[0]
            sims = cosine_similarity([q_emb], self.embeddings)[0]
            top_k = min(k, len(self.corpus))
            return list(np.argsort(sims)[-top_k:][::-1])
        except Exception:
            return []

    def reset(self) -> None:
        self.corpus = []
        self.embeddings = None


class ChromaVectorStore:
    """ChromaDB 持久化向量库（惰性加载，仅 persist 模式使用）。

    对齐官方 A-Mem 的 ChromaRetriever：入库即把「enhanced_document（正文+
    context+keywords+tags）」编码存盘，检索走 collection.query，避免每次检索
    现算全部记忆向量。embedding 复用与 hybrid 同款的 SentenceTransformer
    （all-MiniLM-L6-v2），保证 dense 分数与 hybrid 档一致、对比公平。

    lite/hybrid 模式下本类完全不被实例化，不引入 chromadb 依赖。
    """

    def __init__(self, persist_dir, model_name: str = HYBRID_EMBED_MODEL):
        self.persist_dir = str(persist_dir)
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list) -> list:
        return self._get_model().encode(texts, convert_to_numpy=True).tolist()

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            # 用 cosine 距离；embedding 由我们显式传入（不走 chroma 默认 EF）
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        return self._collection

    def upsert(self, filename: str, doc_text: str) -> None:
        """按 filename 为 id upsert 一条记忆向量（写入/改写记忆时调用）。"""
        try:
            col = self._get_collection()
            emb = self._embed([doc_text])
            col.upsert(ids=[filename], embeddings=emb, documents=[doc_text])
        except Exception:
            pass

    def delete(self, filename: str) -> None:
        try:
            self._get_collection().delete(ids=[filename])
        except Exception:
            pass

    def reset(self) -> None:
        """清空整个 collection（整合记忆时用，随后按新记忆重新 upsert）。"""
        try:
            self._get_collection()  # 确保 _client 已连接
            self._client.delete_collection(CHROMA_COLLECTION)
            self._collection = None  # 下次 _get_collection 重新 get_or_create
        except Exception:
            self._collection = None

    def query(self, query: str, top_k: int) -> dict:
        """向量检索，返回 {filename: dense_score(1-距离)}，已按余弦相似度。"""
        try:
            col = self._get_collection()
            n = col.count()
            if n == 0:
                return {}
            q_emb = self._embed([query])
            res = col.query(query_embeddings=q_emb, n_results=min(top_k, n))
            ids = (res.get("ids") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            # cosine 距离 → 相似度（1-距离），越大越相关
            return {fid: 1.0 - float(d) for fid, d in zip(ids, dists)}
        except Exception:
            return {}


class AgentMemory:
    """Agent 记忆系统：工作上下文、情景经验、语义知识、个性化记忆。"""

    def __init__(self, client, model: str, memory_dir):
        self.working_context: List[MemoryItem] = []
        self.episodic_experience: List[MemoryItem] = []
        self.semantic_knowledge: List[MemoryItem] = []
        self.personalized_memory: List[MemoryItem] = []
        # 工作区文件新鲜度：path -> {"sha256", "summary", "mtime"}
        # 用于“记住即不必重读”：再次读取未变更的文件时提示复用摘要。
        self.file_summaries: Dict[str, Dict[str, Any]] = {}
        self.client = client
        self.model = model
        # 依赖注入：记忆目录/索引随实例，不引用全局
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_index = self.memory_dir / "MEMORY.md"
        self._hybrid = None  # HybridRetriever，惰性创建（仅 hybrid 模式）
        self._chroma = None  # ChromaVectorStore，惰性创建（仅 persist 模式）
        self._sys = None  # OfficialMemorySystem（仅 official 模式）
        self._official_retriever = None  # official_eval in-memory 向量库
        self._official_order: List[str] = []  # official_eval 插入序
        self._evo_cnt = 0  # 进化计数（official_eval / 小智档；达阈值触发 index rebuild）

    def _mode_is_sys(self) -> bool:
        return MEMORY_MODE == "official"

    def _mode_is_eval(self) -> bool:
        return MEMORY_MODE == "official_eval"

    def _get_sys(self):
        if self._sys is None:
            from xiaozhi.components.amem_sys import OfficialMemorySystem
            self._sys = OfficialMemorySystem(self.client, self.model)
        return self._sys

    def _get_official_retriever(self) -> SimpleEmbeddingRetriever:
        if self._official_retriever is None:
            self._official_retriever = SimpleEmbeddingRetriever()
        return self._official_retriever

    def _get_hybrid(self):
        if self._hybrid is None:
            self._hybrid = HybridRetriever()
        return self._hybrid

    def _get_chroma(self):
        if self._chroma is None:
            self._chroma = ChromaVectorStore(self.memory_dir / CHROMA_SUBDIR)
        return self._chroma

    def add_working_context(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.working_context.append(
            MemoryItem(content=content, metadata=metadata or {})
        )

    @staticmethod
    def extract_text(content):
        """Backward-compatible alias for extract_message_text."""
        return extract_message_text(content)

    def _parse_formatter(self, text: str):
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        meta = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
        return meta, parts[2].strip()

    def _rebuild_index(self):
        lines = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_formatter(raw)
            name = meta.get("name", f.stem)
            desc = meta.get("description", body.split("\n")[0][:80])
            # 写绝对路径，模型可直接用 read_file 读取，无需自己猜测目录
            lines.append(f"- [{name}]({f.resolve()}) - {desc}")
        self.memory_index.write_text("\n".join(lines) + "\n" if lines else "")

    def read_memory_index(self):
        if not self.memory_index.exists():
            return ""
        text = self.memory_index.read_text().strip()
        return text if text else ""

    def memory_read(self, filename):
        path = self.memory_dir / filename
        if not path.exists():
            return None
        return path.read_text()

    def memory_search(self):
        pass

    def memory_list(self):
        items = self._memory_list_unsorted()
        if self._mode_is_eval() and self._official_order:
            by_file = {f["filename"]: f for f in items}
            return [by_file[fn] for fn in self._official_order if fn in by_file]
        return sorted(items, key=lambda x: x["filename"])

    def _memory_list_unsorted(self):
        result = []
        for f in self.memory_dir.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_formatter(raw)
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            item = {
                "filename": f.name,
                "name": meta.get("name", f.stem),
                "description": meta.get("description", ""),
                "type": meta.get("type", "user"),
                "body": body,
                "mtime": mtime,
            }
            item["context"] = meta.get("context", "")
            item["created"] = meta.get("created", "")
            for field_name in _LIST_FIELDS:
                raw_val = meta.get(field_name, "")
                parsed = []
                if raw_val:
                    try:
                        parsed = json.loads(raw_val)
                    except (json.JSONDecodeError, TypeError):
                        parsed = []
                item[field_name] = parsed if isinstance(parsed, list) else []
            result.append(item)
        return result

    def _official_all_memories(self) -> list:
        """official 档按插入序返回记忆（对齐官方 list(memories.values())）。"""
        if self._official_order:
            by_file = {f["filename"]: f for f in self._memory_list_unsorted()}
            return [by_file[fn] for fn in self._official_order if fn in by_file]
        return self.memory_list()

    @staticmethod
    def _official_add_document_text(mem: dict) -> str:
        """官方 add_note 的 enhanced_document 格式。"""
        return ("content:" + mem.get("body", "") + " context:" + (mem.get("context") or "") +
                " keywords: " + ", ".join(mem.get("keywords") or []) +
                " tags: " + ", ".join(mem.get("tags") or []))

    def _official_index_note(self, filename: str) -> None:
        """将单条记忆追加进 official in-memory 向量库（先进化后入库）。"""
        mem = next((f for f in self._memory_list_unsorted()
                    if f["filename"] == filename), None)
        if mem is None:
            return
        self._get_official_retriever().add_documents([self._official_add_document_text(mem)])

    def add_note_sys(self, content: str, time: str = None, **kwargs) -> str:
        """A-mem-sys 生产版 add_note（纯 content，返回 memory UUID）。"""
        return self._get_sys().add_note(content, time=time, **kwargs)

    def add_note_eval(self, name, mem_type, description, body,
                          keywords=None, tags=None, context="", created=None):
        """AgenticMemory robust 评测版 add_note（Markdown + 先进化后入库）。"""
        fp = self.memory_write(
            name, mem_type, description, body,
            keywords=keywords, tags=tags, context=context, created=created,
            defer_official_index=True)
        evolved = False
        try:
            evo = self._evolve_memory_official(fp.name)
            evolved = bool(evo.get("links") or evo.get("updated_neighbors"))
        except Exception:
            pass
        if fp.name not in self._official_order:
            self._official_order.append(fp.name)
        self._official_index_note(fp.name)
        if evolved:
            self._bump_evo_and_maybe_consolidate()
        return fp

    def memory_write(self, name, mem_type, description, body,
                     keywords=None, tags=None, context="", links=None, created=None,
                     defer_official_index=False):
        memory_name = name.lower().replace(" ", "-")
        filepath = self.memory_dir / f"{memory_name}.md"
        lines = ["---", f"name: {name}", f"description: {description}", f"type: {mem_type}"]
        # A-Mem 增强字段：仅在开启且有值时写入，保持旧记忆文件格式向后兼容
        if ENABLE_AMEM:
            # created：对齐官方 A-Mem 的 timestamp，显式落 frontmatter，
            # 供结构化检索上下文直接喂给作答 LLM（LoCoMo 半数是时序题）。
            # 传入 created 则复用（改写记忆时保留原时间），否则取当前时间。
            if created is None:
                created = datetime.now().isoformat(timespec="seconds")
            lines.append(f"created: {created}")
            if context:
                lines.append(f"context: {context}")
            if keywords:
                lines.append(f"keywords: {json.dumps(keywords, ensure_ascii=False)}")
            if tags:
                lines.append(f"tags: {json.dumps(tags, ensure_ascii=False)}")
            if links:
                lines.append(f"links: {json.dumps(links, ensure_ascii=False)}")
        lines.append("---")
        filepath.write_text("\n".join(lines) + f"\n\n{body}\n")
        self._rebuild_index()
        # persist 档：ChromaDB 入库即 upsert；official_eval 档用 in-memory retriever（见 add_note_eval）
        if ENABLE_AMEM and MEMORY_MODE == "persist":
            doc = HybridRetriever._doc_text({
                "name": name, "description": description, "body": body,
                "context": context or "",
                "keywords": keywords or [], "tags": tags or [],
            })
            self._get_chroma().upsert(filepath.name, doc)
        return filepath

    def _update_memory_links(self, filename: str, new_links: list) -> None:
        """给已存在的记忆文件合并写入 links（建链用），保留其余字段不变。"""
        path = self.memory_dir / filename
        if not path.exists():
            return
        meta, body = self._parse_formatter(path.read_text())
        existing = []
        if meta.get("links"):
            try:
                existing = json.loads(meta["links"])
            except (json.JSONDecodeError, TypeError):
                existing = []
        merged = list(dict.fromkeys(existing + new_links))  # 去重保序
        self.memory_write(
            name=meta.get("name", path.stem),
            mem_type=meta.get("type", "user"),
            description=meta.get("description", ""),
            body=body,
            keywords=self._json_field(meta, "keywords"),
            tags=self._json_field(meta, "tags"),
            context=meta.get("context", ""),
            links=merged,
            created=meta.get("created") or None,  # 保留原始创建时间
        )

    @staticmethod
    def _json_field(meta: dict, key: str) -> list:
        try:
            return json.loads(meta.get(key, "")) if meta.get(key) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _recent_query_text(self, messages, max_turns=3, limit=2000):
        """取最近若干条 user 消息拼成召回 query。"""
        recent_texts = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = extract_message_text(msg.get("content", ""))
                if content:
                    recent_texts.append(content)
                if len(recent_texts) >= max_turns:
                    break
        return " ".join(reversed(recent_texts))[:limit]

    def find_related_memories(self, query: str, k: int = 5):
        """官方 find_related_memories：向量 top-k + 邻居串（含 memory index），供进化用。"""
        all_memories = self._official_all_memories()
        if not all_memories:
            return "", []
        indices = self._get_official_retriever().search(query, k)
        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + str(m.get("created", "")) +
                "\t memory content: " + m.get("body", "") +
                "\t memory context: " + m.get("context", "") +
                "\t memory keywords: " + str(m.get("keywords", [])) +
                "\t memory tags: " + str(m.get("tags", [])) + "\n"
            )
        return memory_str, indices

    @staticmethod
    def _format_memory_raw_line(mem: dict) -> str:
        """官方 find_related_memories_raw 单行格式（LoCoMo 评测用）。"""
        return (
            "talk start time:" + str(mem.get("created", "")) + "\t"
            "memory content: " + mem.get("body", "") + "\t"
            "memory context: " + str(mem.get("context", "")) + "\t"
            "memory keywords: " + str(mem.get("keywords", [])) + "\t"
            "memory tags: " + str(mem.get("tags", [])) + "\n"
        )

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """官方检索上下文：official→A-mem-sys；official_eval→robust；其余→小智原生召回+官方格式。"""
        if self._mode_is_sys():
            return self._get_sys().find_related_memories_raw(query, k)
        if self._mode_is_eval():
            return self._find_related_memories_raw_eval(query, k)
        return self._find_related_memories_raw_xiaozhi(query, k)

    def _find_related_memories_raw_xiaozhi(self, query: str, k: int = 5) -> str:
        """小智 lite/hybrid/persist：走原生 select_relevant_memories，输出官方 raw 格式。"""
        msgs = [{"role": "user", "content": query}]
        selected = self.select_relevant_memories(msgs, max_items=k)
        if not selected:
            return ""
        by_file = {f["filename"]: f for f in self.memory_list()}
        memory_str = ""
        seen: set = set()
        for fname in selected[:k]:
            mem = by_file.get(fname)
            if not mem or fname in seen:
                continue
            seen.add(fname)
            memory_str += self._format_memory_raw_line(mem)
            j = 0
            for link in mem.get("links", []):
                if j >= k:
                    break
                nb = by_file.get(link) if isinstance(link, str) else None
                if nb and link not in seen:
                    seen.add(link)
                    memory_str += self._format_memory_raw_line(nb)
                    j += 1
        return memory_str

    def _find_related_memories_raw_eval(self, query: str, k: int = 5) -> str:
        all_memories = self._official_all_memories()
        if not all_memories:
            return ""
        indices = self._get_official_retriever().search(query, k)
        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            j = 0
            memory_str += (
                "talk start time:" + str(m.get("created", "")) +
                "memory content: " + m.get("body", "") +
                "memory context: " + m.get("context", "") +
                "memory keywords: " + str(m.get("keywords", [])) +
                "memory tags: " + str(m.get("tags", [])) + "\n"
            )
            for neighbor in m.get("links", []):
                if not isinstance(neighbor, int) or neighbor >= len(all_memories):
                    continue
                nb = all_memories[neighbor]
                memory_str += (
                    "talk start time:" + str(nb.get("created", "")) +
                    "memory content: " + nb.get("body", "") +
                    "memory context: " + nb.get("context", "") +
                    "memory keywords: " + str(nb.get("keywords", [])) +
                    "memory tags: " + str(nb.get("tags", [])) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    def generate_query_official(self, question: str) -> str:
        """官方 generate_query_llm：把问题扩成逗号分隔关键词再检索。"""
        from xiaozhi.components import amem_official as AO
        prompt = (
            "Given the following question, generate several keywords separated by commas.\n\n"
            f"Question: {question}\n\nKeywords:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=200)
            return AO.parse_keywords_response(
                (resp.choices[0].message.content or "").strip())
        except Exception:
            return question

    def answer_question_official(self, question: str, category: int, gold_answer: str,
                                 context: str, temperature_c5: float = 0.5) -> str:
        """官方 category-specific 作答 prompt（对齐 test_advanced_robust）。"""
        from xiaozhi.components import amem_official as AO
        if category == 5:
            opts = (["Not mentioned in the conversation", gold_answer]
                    if random.random() < 0.5
                    else [gold_answer, "Not mentioned in the conversation"])
            user_prompt = (
                f"Based on the context: {context}, answer the following question. {question}\n\n"
                f"Select the correct answer: {opts[0]} or {opts[1]} Short answer:"
            )
            temperature = temperature_c5
        elif category == 2:
            user_prompt = (
                f"Based on the context: {context}, answer the following question. "
                "Use DATE of CONVERSATION to answer with an approximate date.\n"
                "Please generate the shortest possible answer, using words from the "
                "conversation where possible, and avoid using any subjects.\n\n"
                f"Question: {question} Short answer:"
            )
            temperature = 0.7
        elif category == 3:
            user_prompt = (
                f"Based on the context: {context}, write an answer in the form of a "
                "short phrase for the following question. Answer with exact words from "
                f"the context whenever possible.\n\nQuestion: {question} Short answer:"
            )
            temperature = 0.7
        else:
            user_prompt = (
                f"Based on the context: {context}, write an answer in the form of a "
                "short phrase for the following question. Answer with exact words from "
                f"the context whenever possible.\n\nQuestion: {question} Short answer:"
            )
            temperature = 0.7
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=200, temperature=temperature)
            return AO.parse_plain_text_answer(
                (resp.choices[0].message.content or "").strip())
        except Exception:
            return ""

    def rebuild_persist_index(self) -> None:
        """persist 档：对齐 A-Mem consolidate，重置 Chroma 并按全量记忆重建向量索引。"""
        if MEMORY_MODE != "persist":
            return
        chroma = self._get_chroma()
        chroma.reset()
        for m in self.memory_list():
            doc = HybridRetriever._doc_text({
                "name": m["name"],
                "description": m["description"],
                "body": m["body"],
                "context": m.get("context", ""),
                "keywords": m.get("keywords", []),
                "tags": m.get("tags", []),
            })
            chroma.upsert(m["filename"], doc)

    def _bump_evo_and_maybe_consolidate(self) -> None:
        """进化成功后累计计数，达 evo_threshold 时触发 consolidate（A-Mem 语义）。"""
        self._evo_cnt += 1
        if self._evo_cnt % AMEM_EVO_THRESHOLD == 0:
            self.consolidate_memories()

    def rebuild_official_index(self) -> None:
        """official_eval：重建 in-memory 向量索引（无 LLM）。"""
        if not self._mode_is_eval():
            return
        retriever = SimpleEmbeddingRetriever(HYBRID_EMBED_MODEL)
        for m in self._official_all_memories():
            metadata_text = (f"{m.get('context', '')} "
                             f"{' '.join(m.get('keywords') or [])} "
                             f"{' '.join(m.get('tags') or [])}")
            retriever.add_documents([m.get("body", "") + " , " + metadata_text])
        self._official_retriever = retriever

    def _rule_rank_memories(self, query: str, files: list):
        """规则召回：三级排序，返回按相关性降序的 (filename, score) 列表。

        参考 pico-custom 的透明召回思路，适配小智无 tags 的记忆格式：
          1) type 命中：query 中出现该记忆 type 的近义词（近似 tag 精确匹配）
          2) 关键词重叠：query 分词 ∩ (name+description+body) 分词 的数量
          3) 新鲜度：记忆文件 mtime，越新越靠前
        仅返回至少有一项命中的记忆（score 元组用于排序）。
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        ranked = []
        for f in files:
            mem_tokens = _tokenize(f["name"] + " " + f["description"] + " " + f["body"])
            keyword_overlap = len(query_tokens & mem_tokens)
            type_hits = _TYPE_HINTS.get(f["type"], set())
            type_match = int(bool(query_tokens & type_hits))
            if keyword_overlap == 0 and type_match == 0:
                continue
            score = (type_match, keyword_overlap, f.get("mtime", 0.0))
            ranked.append((score, f["filename"]))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _rule_scores_normalized(self, query: str, files: list) -> list:
        """把规则三级排序压成单一可融合分（min-max 归一化，与 files 顺序对齐）。

        规则分本是元组 (type_match, keyword_overlap, mtime)，这里线性合成一个标量：
        type_match 权重最高、其次关键词重叠、mtime 仅作极小的新鲜度加成。
        """
        query_tokens = _tokenize(query)
        raw = []
        for f in files:
            if not query_tokens:
                raw.append(0.0)
                continue
            mem_tokens = _tokenize(f["name"] + " " + f["description"] + " " + f["body"]
                                   + " " + " ".join(f.get("keywords", []))
                                   + " " + " ".join(f.get("tags", [])))
            overlap = len(query_tokens & mem_tokens)
            type_hits = _TYPE_HINTS.get(f["type"], set())
            type_match = 1 if (query_tokens & type_hits) else 0
            raw.append(type_match * 10 + overlap)  # type 命中给强加成
        return _minmax(raw)

    def _hybrid_rank_memories(self, query: str, files: list):
        """dense + BM25 + 规则 三路归一化加权融合，返回按分降序的 (filename, score)。"""
        dense = self._get_hybrid().dense_scores(query, files)
        bm25 = self._get_hybrid().bm25_scores(query, files)
        rule = self._rule_scores_normalized(query, files)
        fused = []
        for i, f in enumerate(files):
            score = float(HYBRID_W_DENSE * dense[i] + HYBRID_W_BM25 * bm25[i]
                          + HYBRID_W_RULE * rule[i])
            fused.append((score, f["filename"]))
        fused.sort(key=lambda x: x[0], reverse=True)
        # 过滤掉三路全 0 的（完全无关）
        return [(s, fn) for s, fn in fused if s > 1e-9]

    def _persist_rank_memories(self, query: str, files: list):
        """persist 档：ChromaDB 持久向量(dense) + BM25 + 规则 三路融合。

        与 hybrid 的唯一区别是 dense 来自持久化向量库的 query（不现算全部记忆），
        大规模下更快；BM25/规则仍现算（语料小、成本低）。向量库缺失某条时
        该条 dense 记 0，由 BM25/规则兜底。"""
        chroma_scores = self._get_chroma().query(query, top_k=max(len(files), 1))
        dense = _minmax([chroma_scores.get(f["filename"], 0.0) for f in files])
        bm25 = self._get_hybrid().bm25_scores(query, files)
        rule = self._rule_scores_normalized(query, files)
        fused = []
        for i, f in enumerate(files):
            score = float(HYBRID_W_DENSE * dense[i] + HYBRID_W_BM25 * bm25[i]
                          + HYBRID_W_RULE * rule[i])
            fused.append((score, f["filename"]))
        fused.sort(key=lambda x: x[0], reverse=True)
        return [(s, fn) for s, fn in fused if s > 1e-9]

    def _official_rank_memories(self, query: str, files: list):
        """official 档：官方 A-Mem 忠实复刻 —— 纯单路 dense 向量检索。

        对齐官方 SimpleEmbeddingRetriever.search：只用 ChromaDB 持久向量的
        余弦相似度取 top-k，**无 BM25、无规则、无 LLM 选下标**。这是与官方
        同类可比的唯一实现（persist 档在此基础上多叠 BM25+规则融合作为增量）。"""
        chroma_scores = self._get_chroma().query(query, top_k=max(len(files), 1))
        ranked = [(chroma_scores.get(f["filename"], 0.0), f["filename"]) for f in files]
        ranked = [(s, fn) for s, fn in ranked if s > 1e-9]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _rank_neighbors(self, query: str, others: list):
        """建链选邻居（差距B）：向量优先，规则兜底。

        - persist/official：用持久化向量库 query 取 dense 排序（官方做法）。
        - hybrid ：现算 dense 排序。
        - lite   ：无向量依赖，回退规则三级排序。
        返回按相关性降序的 (score, filename)。"""
        if MEMORY_MODE in ("persist",):
            scores = self._get_chroma().query(query, top_k=len(others))
            ranked = [(scores.get(f["filename"], 0.0), f["filename"]) for f in others]
            ranked = [(s, fn) for s, fn in ranked if s > 1e-9]
            if ranked:
                ranked.sort(key=lambda x: x[0], reverse=True)
                return ranked
        elif self._mode_is_eval():
            all_memories = self._official_all_memories()
            if all_memories:
                indices = self._get_official_retriever().search(query, k=len(all_memories))
                ranked = []
                for i in indices:
                    if i < len(all_memories):
                        ranked.append((1.0, all_memories[i]["filename"]))
                if ranked:
                    return ranked
        elif MEMORY_MODE == "hybrid":
            dense = self._get_hybrid().dense_scores(query, others)
            ranked = [(dense[i], others[i]["filename"]) for i in range(len(others))]
            ranked = [(s, fn) for s, fn in ranked if s > 1e-9]
            if ranked:
                ranked.sort(key=lambda x: x[0], reverse=True)
                return ranked
        return self._rule_rank_memories(query, others)

    def select_relevant_memories(self, messages, max_items=5):
        """规则召回 + LLM 召回 叠加，并（开启 A-Mem 时）带出 linked 邻居。

        先跑规则三级排序（快、零 token、确定性），再用 LLM 补充语义相关项；
        LLM 失败时以规则召回兜底。规则命中优先，LLM 结果去重后追加。
        A-Mem：命中记忆的 links 邻居一并带出（多跳召回），邻居额外占 max_items 之外的名额。
        """
        files = self.memory_list()
        if not files:
            return []

        recent = self._recent_query_text(messages)
        if not recent.strip():
            return []

        if self._mode_is_sys():
            keywords = self.generate_query_official(recent)
            if not self.find_related_memories_raw(keywords, k=max_items):
                return []
            results = self._get_sys().search_agentic(keywords, k=max_items)
            return [r["id"] for r in results]
        elif self._mode_is_eval():
            keywords = self.generate_query_official(recent)
            if not self.find_related_memories_raw(keywords, k=max_items):
                return []
            all_memories = self._official_all_memories()
            indices = self._get_official_retriever().search(keywords, max_items)
            return [all_memories[i]["filename"] for i in indices
                    if i < len(all_memories)]
        elif MEMORY_MODE == "persist":
            fused = self._persist_rank_memories(recent, files)
            selected = [fname for _, fname in fused[:max_items]]
        elif MEMORY_MODE == "hybrid":
            fused = self._hybrid_rank_memories(recent, files)
            selected = [fname for _, fname in fused[:max_items]]
        else:
            # 轻量模式（默认）：规则三级排序 + LLM 选下标 双路
            # ① 规则召回
            rule_ranked = self._rule_rank_memories(recent, files)
            selected = [fname for _, fname in rule_ranked[:max_items]]
            # ② LLM 召回（补充规则漏掉的语义相关项）
            llm_selected = self._llm_select_memories(recent, files)
            for fname in llm_selected:
                if fname not in selected:
                    selected.append(fname)
                if len(selected) >= max_items:
                    break
            selected = selected[:max_items]

        # ③ A-Mem 邻居扩展（official/official_eval 已在 find_related_memories_raw 内扩展）
        if ENABLE_AMEM and MEMORY_MODE not in ("official", "official_eval"):
            by_file = {f["filename"]: f for f in files}
            neighbors = []
            for fname in selected:
                mem = by_file.get(fname)
                if not mem:
                    continue
                for link in mem.get("links", []):
                    if link not in selected and link not in neighbors and link in by_file:
                        neighbors.append(link)
            # 邻居数量上限：不超过 max_items，避免上下文膨胀
            selected = selected + neighbors[:max_items]

        return selected

    def _llm_select_memories(self, recent: str, files: list) -> list:
        """LLM 相关性判断；异常时返回空（由规则召回兜底）。"""
        catalog_lines = []
        for i, f in enumerate(files):
            catalog_lines.append(f"{i}: {f['name']} - {f['description']}")
        catalog = "\n".join(catalog_lines)

        prompt = (
            "Given the recent conversation and the memory catalog below, "
            "select the indices of memories that are clearly relevant. "
            "Return ONLY a JSON array of integers, e.g. [0, 3]. "
            "If none are relevant, return [].\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"Memory catalog:\n{catalog}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            indices = extract_json_array(text)
            if indices is not None:
                selected = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(files):
                        selected.append(files[idx]["filename"])
                return selected
        except Exception:
            pass
        return []

    def load_memories(self, messages: list):
        if ENABLE_AMEM and (self._mode_is_sys() or self._mode_is_eval()):
            recent = self._recent_query_text(messages)
            if not recent.strip():
                return ""
            keywords = self.generate_query_official(recent)
            raw = self.find_related_memories_raw(keywords, k=5)
            if not raw:
                return ""
            return "##相关记忆\n\n" + raw
        selected_files = self.select_relevant_memories(messages)
        if not selected_files:
            return ""

        parts = ["##相关记忆"]
        for filename in selected_files:
            content = self.memory_read(filename)
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def render_structured_context(self, filenames: list) -> str:
        """把召回的记忆渲染成官方 A-Mem 风格的结构化多字段上下文。

        对齐官方 find_related_memories：每条记忆显式列出
        talk time / content / context / keywords / tags —— 尤其把创建时间
        提到明面（LoCoMo 半数是时序题），而不是把 timestamp 埋在正文里让
        作答 LLM 自己去抠。ENABLE_AMEM 关闭时回退到纯正文拼接（旧行为）。
        """
        by_file = {f["filename"]: f for f in self.memory_list()}
        blocks = []
        for fn in filenames:
            mem = by_file.get(fn)
            if not mem:
                continue
            if not ENABLE_AMEM:
                blocks.append(mem["body"])
                continue
            fields = []
            if mem.get("created"):
                fields.append(f"talk time: {mem['created']}")
            fields.append(f"content: {mem['body']}")
            if mem.get("context"):
                fields.append(f"context: {mem['context']}")
            if mem.get("keywords"):
                fields.append(f"keywords: {', '.join(mem['keywords'])}")
            if mem.get("tags"):
                fields.append(f"tags: {', '.join(mem['tags'])}")
            blocks.append("\t".join(fields))
        return "\n---\n".join(blocks)

    def extract_memories(self, messages):
        dialogue_parts = []
        for msg in messages[-10:]:
            role = msg.get("role", "?")
            content = extract_message_text(msg.get("content", ""))
            if content.strip():
                dialogue_parts.append(f"{role}: {content}")
        dialogue = "\n".join(dialogue_parts)

        if not dialogue.strip():
            return

        existing = self.memory_list()
        existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

        # A-Mem 增强：让同一次提取调用额外产出 keywords/context/tags（零额外 LLM 调用）
        amem_fields = ""
        if ENABLE_AMEM:
            amem_fields = (
                "- keywords: 3-8 个最能代表内容的关键词（名词/概念，避免人名和时间）\n"
                "- context: 一句话概括主题/领域\n"
                "- tags: 3-6 个用于分类检索的标签\n"
            )
        prompt = (
            "从对话中提取用户偏好、约束或者项目事实\n"
            "返回格式为JSON列表。每一项包含字段：{name, type, description, body"
            + (", keywords, context, tags" if ENABLE_AMEM else "") + "}\n"
            "- name: 短标识符，用'-'连接（如 'user-preference-tabs'）"
            "- type: 类型，取值为 'user'（用户偏好）, 'feedback'（指引）, 'project'（项目事实）或'reference'（外部指向）\n"
            "- description: 一行总结，用于索引查找\n"
            "- body: Markdown格式的全部细节描述\n"
            + amem_fields +
            "如果没有新的记忆或已经被现有记忆涵盖，返回[]\n\n"
            f"现有记忆：\n{existing_desc}\n\n"
            f"对话：\n{dialogue[:4000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=1000
            )
            text = (response.choices[0].message.content or "").strip()
            items = extract_json_array(text)
            if not items:
                return
            count = 0
            new_filenames = []
            for mem in items:
                name = mem.get("name", f"memory_{int(time.time())}")
                mem_type = mem.get("type", "user")
                desc = mem.get("description", "")
                body = mem.get("body", "")
                if desc and body:
                    if ENABLE_AMEM and self._mode_is_sys():
                        self.add_note_sys(body)
                        count += 1
                        continue
                    kw = mem.get("keywords") if ENABLE_AMEM else None
                    tg = mem.get("tags") if ENABLE_AMEM else None
                    ctx = mem.get("context", "") if ENABLE_AMEM else ""
                    if ENABLE_AMEM and self._mode_is_eval():
                        fp = self.add_note_eval(
                            name, mem_type, desc, body,
                            keywords=kw if isinstance(kw, list) else None,
                            tags=tg if isinstance(tg, list) else None,
                            context=ctx if isinstance(ctx, str) else "")
                        new_filenames.append(fp.name)
                    else:
                        fp = self.memory_write(name, mem_type, desc, body,
                                               keywords=kw if isinstance(kw, list) else None,
                                               tags=tg if isinstance(tg, list) else None,
                                               context=ctx if isinstance(ctx, str) else "")
                        new_filenames.append(fp.name)
                    count += 1
            if count:
                print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
            if ENABLE_AMEM and self._mode_is_eval():
                for fname in new_filenames:
                    self.evolve_memory(fname)
        except Exception:
            pass

    # ── A-Mem 建链 + 记忆进化 ──
    # 参考 A-Mem 的 process_memory：新记忆入库后，取 top-k 相关邻居交给 LLM，
    # 一次调用同时决定：① 该新记忆连向哪些邻居(strengthen)；
    # ② 是否反向更新邻居的 tags(update_neighbor)。
    # 候选邻居用现有规则召回获取（代替 A-Mem 的向量检索），不引入向量库依赖。

    def evolve_memory(self, new_filename: str) -> dict:
        """为新记忆分析与既有记忆的关联，写入 links，并可更新邻居 context/tags。

        返回 {"links": [...], "updated_neighbors": [...]}；异常或无邻居时安全返回空。
        official 档分派到 _evolve_memory_official（AgenticMemory robust 3-call 复刻）。
        """
        if self._mode_is_eval():
            return self._evolve_memory_official(new_filename)

        result = {"links": [], "updated_neighbors": []}
        if not ENABLE_AMEM:
            return result

        files = self.memory_list()
        new_mem = next((f for f in files if f["filename"] == new_filename), None)
        if new_mem is None:
            return result

        # 候选邻居：优先向量选邻居（差距B：官方用向量 top-k，链质量更高），
        # persist/hybrid 有 dense 时走向量，否则规则兜底（lite 零依赖）。
        query = " ".join([new_mem["name"], new_mem["description"],
                          " ".join(new_mem.get("keywords", []))])
        others = [f for f in files if f["filename"] != new_filename]
        if not others:
            return result
        ranked = self._rank_neighbors(query, others)
        neighbors = [dict(f) for _, fname in ranked[:AMEM_LINK_TOPK]
                     for f in others if f["filename"] == fname]
        if not neighbors:
            return result

        neighbor_desc = "\n".join(
            f"[{i}] {n['name']}: {n['description']} | context={n.get('context', '')} | tags={n.get('tags', [])}"
            for i, n in enumerate(neighbors)
        )
        prompt = (
            "你是记忆进化代理。给定一条新记忆和它的若干近邻记忆，判断如何组织它们。\n"
            f"新记忆：{new_mem['name']}: {new_mem['description']}\n"
            f"关键词：{new_mem.get('keywords', [])}\n"
            f"近邻记忆（每行以 [序号] 开头）：\n{neighbor_desc}\n\n"
            "返回 JSON 对象：\n"
            "{\n"
            '  "connect_indices": [与新记忆语义相关、应建立链接的近邻序号],\n'
            '  "neighbor_updates": {"序号": {"context": "更新后的一句话上下文", "tags": ["更新后的tag", ...]}}\n'
            "  // 仅需更新的近邻；不改的字段保持与原值一致；无需更新时 neighbor_updates 为空\n"
            "}\n"
            "只连接真正相关的；无相关近邻时 connect_indices 返回 []。"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            text = (response.choices[0].message.content or "").strip()
            data = self._extract_json_object(text)
            if not data:
                return result

            # ① strengthen：把新记忆连向选中的邻居（双向）
            connect_idx = data.get("connect_indices", []) or []
            linked_files = []
            for idx in connect_idx:
                if isinstance(idx, int) and 0 <= idx < len(neighbors):
                    linked_files.append(neighbors[idx]["filename"])
            if linked_files:
                self._update_memory_links(new_filename, linked_files)
                for lf in linked_files:
                    self._update_memory_links(lf, [new_filename])  # 反向连接
                result["links"] = linked_files

            # ② update_neighbor：更新邻居 context 和 tags（差距D：官方两者都更新）
            # 兼容旧格式 neighbor_tag_updates（仅 tags）。
            neighbor_updates = data.get("neighbor_updates", {}) or {}
            if not neighbor_updates and data.get("neighbor_tag_updates"):
                neighbor_updates = {k: {"tags": v}
                                    for k, v in data["neighbor_tag_updates"].items()}
            for idx_str, upd in neighbor_updates.items():
                try:
                    idx = int(idx_str)
                except (ValueError, TypeError):
                    continue
                if not (0 <= idx < len(neighbors)) or not isinstance(upd, dict):
                    continue
                n = neighbors[idx]
                new_tags = upd.get("tags") if isinstance(upd.get("tags"), list) else n.get("tags")
                new_ctx = upd.get("context") if isinstance(upd.get("context"), str) else n.get("context", "")
                self.memory_write(
                    name=n["name"], mem_type=n["type"],
                    description=n["description"], body=n["body"],
                    keywords=n.get("keywords"), tags=new_tags,
                    context=new_ctx, links=n.get("links"),
                    created=n.get("created") or None)  # 保留原始创建时间
                result["updated_neighbors"].append(n["filename"])

            if result["links"] or result["updated_neighbors"]:
                print(f"  \033[36m[A-Mem] {new_filename}: linked {len(result['links'])}, "
                      f"evolved {len(result['updated_neighbors'])} neighbors\033[0m")
                self._bump_evo_and_maybe_consolidate()
        except Exception:
            pass
        return result

    # ── AgenticMemory robust 评测版进化（仅 official_eval 档）──
    # 逐行对齐 memory_layer_robust.py 的 process_memory：向量取 top-k 邻居 →
    # 最多 3 次顺序 LLM 调用（决策 → strengthen → update_neighbor），
    # 用 vendored amem_official.py 的官方 prompts + parsers，保证与官方等价。
    # 与 lite/hybrid/persist 的差异：单向连接（只连新记忆，不反向连邻居）、
    # 建链 query 用 content（正文）而非元数据、决策分 4 种标签。

    def analyze_content_official(self, content: str) -> dict:
        """官方 AgenticMemory note construction（official_eval 档）。"""
        from xiaozhi.components import amem_official as AO
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user",
                           "content": AO.ANALYZE_CONTENT_PROMPT.format(content=content)}])
            analysis = AO.parse_analyze_content(
                (resp.choices[0].message.content or "").strip(), content)
            if not analysis["keywords"]:
                retry = self.client.chat.completions.create(
                    model=self.model, temperature=0.3,
                    messages=[{"role": "user",
                               "content": AO.FOCUSED_KEYWORDS_PROMPT.format(content=content)}])
                analysis["keywords"] = AO._parse_list_items(
                    (retry.choices[0].message.content or "").strip())
            return AO.validate_analysis_result(analysis, content)
        except Exception:
            return {"keywords": AO._heuristic_keywords(content),
                    "context": AO._heuristic_context(content),
                    "tags": AO._heuristic_keywords(content, 3)}

    def _evolve_memory_official(self, new_filename: str) -> dict:
        """官方 process_memory 忠实复刻（3 次顺序调用；links 为整数下标）。"""
        from xiaozhi.components import amem_official as AO

        result = {"links": [], "updated_neighbors": []}
        new_mem = next((f for f in self._memory_list_unsorted()
                        if f["filename"] == new_filename), None)
        if new_mem is None:
            return result

        # 新记忆尚未入 _official_order / retriever（对齐官方 add_note 时序）
        all_memories = self._official_all_memories()
        if not all_memories:
            return result
        neighbor_str, indices = self.find_related_memories(
            new_mem.get("body", ""), k=AMEM_LINK_TOPK)
        if not indices:
            return result

        def _complete(prompt, temperature=None):
            kw = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            if temperature is not None:
                kw["temperature"] = temperature
            resp = self.client.chat.completions.create(**kw)
            return (resp.choices[0].message.content or "").strip()

        try:
            decision_prompt = AO.EVOLUTION_DECISION_PROMPT.format(
                context=new_mem.get("context", ""), content=new_mem.get("body", ""),
                keywords=new_mem.get("keywords", []), nearest_neighbors_memories=neighbor_str)
            decision = AO.parse_evolution_decision(_complete(decision_prompt))
            if decision["decision"] == "NO_EVOLUTION":
                return result
            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            if should_strengthen:
                strengthen_prompt = AO.STRENGTHEN_DETAILS_PROMPT.format(
                    content=new_mem.get("body", ""), keywords=new_mem.get("keywords", []),
                    nearest_neighbors_memories=neighbor_str)
                strengthen = AO.parse_strengthen_details(_complete(strengthen_prompt))
                linked_indices = [c for c in strengthen["connections"]
                                  if isinstance(c, int) and 0 <= c < len(all_memories)]
                new_tags = strengthen["tags"] if strengthen["tags"] else new_mem.get("keywords")
                if linked_indices or strengthen["tags"]:
                    merged = list(dict.fromkeys(
                        list(new_mem.get("links", [])) + linked_indices))
                    self.memory_write(
                        name=new_mem["name"], mem_type=new_mem["type"],
                        description=new_mem["description"], body=new_mem["body"],
                        keywords=new_mem.get("keywords"), tags=new_tags,
                        context=new_mem.get("context", ""), links=merged,
                        created=new_mem.get("created") or None,
                        defer_official_index=True)
                    result["links"] = linked_indices

            if should_update:
                update_prompt = AO.UPDATE_NEIGHBORS_PROMPT.format(
                    content=new_mem.get("body", ""), context=new_mem.get("context", ""),
                    nearest_neighbors_memories=neighbor_str,
                    max_neighbor_idx=len(indices) - 1, neighbor_count=len(indices))
                updates = AO.parse_update_neighbors(_complete(update_prompt), len(indices))
                for i in range(min(len(indices), len(updates))):
                    upd = updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(all_memories):
                        continue
                    n = all_memories[memorytmp_idx]
                    new_tags = upd["tags"] if upd["tags"] else n.get("tags")
                    new_ctx = upd["context"] if upd["context"] else n.get("context", "")
                    self.memory_write(
                        name=n["name"], mem_type=n["type"], description=n["description"],
                        body=n["body"], keywords=n.get("keywords"), tags=new_tags,
                        context=new_ctx, links=n.get("links"),
                        created=n.get("created") or None,
                        defer_official_index=True)
                    result["updated_neighbors"].append(n["filename"])

            if result["links"] or result["updated_neighbors"]:
                print(f"  \033[36m[A-Mem official] {new_filename}: linked "
                      f"{len(result['links'])}, evolved {len(result['updated_neighbors'])}\033[0m")
        except Exception:
            pass
        return result

    @staticmethod
    def _extract_json_object(text: str) -> dict:
        """从模型输出里提取第一个完整 JSON 对象（括号配对扫描，容忍夹带文字）。"""
        start = text.find("{")
        if start < 0:
            return {}
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}

    def consolidate_memories(self):
        """A-Mem consolidate：仅重建向量索引，不 LLM 合并/删记忆。

        由 evo_threshold 在成功进化后触发；official 档在 amem_sys.add_note 内触发。
        lite/hybrid 检索现算向量，无需 rebuild。
        """
        if self._mode_is_sys():
            return
        if self._mode_is_eval():
            n = len(self._official_all_memories())
            self.rebuild_official_index()
            print(f"\n\033[33m[Memory: consolidated index, {n} memories]\033[0m")
            return
        if MEMORY_MODE == "persist":
            n = len(self.memory_list())
            self.rebuild_persist_index()
            print(f"\n\033[33m[Memory: consolidated index, {n} memories]\033[0m")

    # ── 工作区文件新鲜度校验（“记住即不必重读”）──
    #
    # 参考 pico-custom 的 file_freshness 机制：读过的工作区文件按内容 sha256
    # 记摘要；再次读取时若 sha256 未变，说明文件没动，可复用上次摘要，
    # 避免把同一文件的完整内容反复塞进上下文 / 反复读盘。

    @staticmethod
    def _file_sha256(path) -> Optional[str]:
        try:
            p = Path(path)
            if not p.is_file():
                return None
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None

    @staticmethod
    def _summarize_content(content: str, limit: int = 180) -> str:
        """把文件读取结果压成一行短摘要，只用于“提醒读过什么”。"""
        lines = [ln.strip() for ln in str(content).splitlines() if ln.strip()]
        if not lines:
            return "(empty)"
        summary = " | ".join(lines[:3])
        return summary[:limit]

    def record_file_read(self, path: str, content: str) -> None:
        """登记一次文件读取：存内容 sha256 + 短摘要。"""
        sha = self._file_sha256(path)
        if sha is None:
            return
        self.file_summaries[str(path)] = {
            "sha256": sha,
            "summary": self._summarize_content(content),
            "mtime": datetime.now().isoformat(timespec="seconds"),
        }

    def is_file_fresh(self, path: str) -> bool:
        """该文件此前读过且内容未变（sha256 一致）→ True。"""
        record = self.file_summaries.get(str(path))
        if not record:
            return False
        return record["sha256"] == self._file_sha256(path)

    def get_file_summary(self, path: str) -> Optional[str]:
        record = self.file_summaries.get(str(path))
        return record["summary"] if record else None

    def invalidate_stale_file_summaries(self) -> list:
        """清理已变更/已删除的文件摘要，返回被失效的路径列表。"""
        invalidated = []
        for path in list(self.file_summaries.keys()):
            if self.file_summaries[path]["sha256"] != self._file_sha256(path):
                self.file_summaries.pop(path, None)
                invalidated.append(path)
        return invalidated

    def render_file_summaries(self, max_items: int = 6) -> str:
        """渲染“近期读过的文件摘要”给 system prompt，仅保留仍新鲜的条目。"""
        lines = []
        for path, record in list(self.file_summaries.items())[-max_items:]:
            if record["sha256"] == self._file_sha256(path):
                lines.append(f"- {path}: {record['summary']}")
        if not lines:
            return ""
        return "近期读过的文件（内容未变，无需重复读取）：\n" + "\n".join(lines)
