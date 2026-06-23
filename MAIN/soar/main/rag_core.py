import json
import os
import re
import time
import hashlib
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote

from paths import PROJECT_ROOT

import dashscope
import faiss
import jieba
import numpy as np
import ollama
from PIL import Image
from rank_bm25 import BM25Okapi

try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None

# ========= Global config =========
# 必须通过环境变量 DASHSCOPE_API_KEY 配置（Docker 由 .env 注入；勿将密钥提交到 Git）
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or ""
# 对话模型（DashScope）；须与控制台「模型监控/API 示例」中的 model 一致，否则报 Model not exist
API_GENERATION_MODEL = os.environ.get(
    "DASHSCOPE_CHAT_MODEL",
    os.environ.get("API_GENERATION_MODEL", "qwen2.5-32b-instruct"),
)


def _safe_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return default


def _safe_float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw)
        if value >= 0:
            return value
    except (TypeError, ValueError):
        pass
    return default


EMBEDDING_PROVIDER = (os.environ.get("EMBEDDING_PROVIDER") or "ollama").strip().lower()
# Ollama 文本嵌入模型（本地）
OLLAMA_EMB_MODEL = os.environ.get(
    "OLLAMA_EMBEDDING_MODEL",
    os.environ.get("EMB_MODEL", "bge-m3"),
)
# DashScope 文本嵌入模型（联网）
DASHSCOPE_EMBEDDING_MODEL = (os.environ.get("DASHSCOPE_EMBEDDING_MODEL") or "text-embedding-v4").strip()
DASHSCOPE_EMBEDDING_DIM = _safe_int_env("DASHSCOPE_EMBEDDING_DIM", 1024)
EMBEDDING_MAX_RETRIES = _safe_int_env("EMBEDDING_MAX_RETRIES", 3)
EMBEDDING_RETRY_BACKOFF_SECONDS = _safe_float_env("EMBEDDING_RETRY_BACKOFF_SECONDS", 1.0)

TOP_K = 50
BM25_TOP_K = 50
RRF_K = 60
RRF_CANDIDATES = 20
FINAL_TOP_K = 10

RERANK_MODEL_NAME = str(PROJECT_ROOT / "models" / "BAAI" / "bge-reranker-v2-m3")

RERANK_THRESHOLD = -2.0
RERANK_GAP_THRESHOLD = 3.5

CONTEXT_MAX_CHARS = 12000
MAX_IMGS_PER_CHUNK = 3
MAX_TABLES_PER_CHUNK = 2
MAX_TOTAL_IMAGES = 6
MAX_TOTAL_TABLES = 3

DEFAULT_IMAGE_DIRS = [
    str(PROJECT_ROOT / "main" / "static" / "img"),
    str(PROJECT_ROOT / "SpaceRAG" / "img"),
    str(PROJECT_ROOT / "SpaceRAG" / "images"),
    str(PROJECT_ROOT / "SpaceRAG" / "table"),
    str(PROJECT_ROOT / "SpaceRAG" / "tables"),
    str(PROJECT_ROOT / "ITSoftwareRAG" / "assets" / "images"),
    str(PROJECT_ROOT / "SmartManufacturingRAG_v2" / "assets" / "images"),
    str(PROJECT_ROOT / "半导体_RAG" / "assets" / "images"),
    str(PROJECT_ROOT / "军工_RAG" / "assets" / "images"),
    str(PROJECT_ROOT / "pre_data" / "professional" / "img"),
    str(PROJECT_ROOT / "pre_data" / "secret" / "img"),
    str(PROJECT_ROOT / "pre_data" / "product" / "img"),
    str(PROJECT_ROOT / "pre_data" / "employee" / "img"),
    str(PROJECT_ROOT / "pre_data" / "space" / "img"),
]

DEFAULT_TABLE_DIRS = [
    str(PROJECT_ROOT / "SpaceRAG" / "table"),
    str(PROJECT_ROOT / "SpaceRAG" / "tables"),
    str(PROJECT_ROOT / "SpaceRAG" / "assets" / "table"),
    str(PROJECT_ROOT / "SpaceRAG" / "assets" / "tables"),
    str(PROJECT_ROOT / "ITSoftwareRAG" / "assets" / "tables"),
    str(PROJECT_ROOT / "SmartManufacturingRAG_v2" / "assets" / "tables"),
    str(PROJECT_ROOT / "半导体_RAG" / "assets" / "tables"),
    str(PROJECT_ROOT / "军工_RAG" / "assets" / "tables"),
    str(PROJECT_ROOT / "pre_data" / "space" / "table"),
    str(PROJECT_ROOT / "pre_data" / "space" / "tables"),
]

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant good at answering questions based on the provided knowledge base.
<instruction>
- Use the retrieved chunks closely.
- Answer in Chinese.
- If chunks contain "[相关图片路径: ...]", list them at the end of answer as markdown image links.
- If chunks contain "[相关表格路径: ...]", list them at the end of answer as markdown links.
- Use Markdown format.
</instruction>"""

SOURCE_PRIVACY_GUARDRAIL = """<source_privacy_guardrail>
- 文字回答中禁止暴露数据来源痕迹：不得出现具体文件名、文档标题、内部标签、水印语、路径、页码、chunk 编号、附件名、截图名、目录结构等可追溯来源的信息。
- 若检索片段包含类似“XX文件/严禁外传/内部资料/仅供内部”等字样，不得原文复述，不得引用为证据句。
- 需要表达依据时，统一改写为“根据相关内部资料”或“根据现有内部资料综合判断”，不得写“在《XX文档》中”“原文显示”等表达。
- 对“未出现/未提及”类结论，统一写法为：“检索知识库，未发现其在……方面开展实质性业务。”，禁止使用“全文未提及”。
- 推荐表述格式：先给结论，再写“根据相关内部资料，得出……（结论/判断）”。
- 输出前必须自检并重写：若答案中出现《》、文件、严禁、外传、页、chunk、.pdf、.doc、/、\\ 等来源痕迹，不得直接输出。
- 若用户明确要求提供来源、原文出处、页码、路径或文档名，礼貌拒绝披露，并继续提供不含来源痕迹的结论性摘要。
</source_privacy_guardrail>"""


def _build_industry_system_prompt(
    kb_name: str,
    focus: str,
    fallback_reference: str,
    evidence_dimensions: str,
    extra_guidance: str = "",
) -> str:
    extra_line = f"- {extra_guidance}\n" if extra_guidance else ""
    return f"""你是“{kb_name}知识库”问答助手。
<instruction>
- 聚焦{focus}。
- 优先回答公司主营产品、技术路线、商业模式、市场空间、竞争格局、客户验证、财务/融资进展和主要风险。
- 输出风格默认使用“结论 -> 依据 -> 风险/不确定性”。
- 回答语言：中文。
- 若上下文含“相关图片路径/相关表格路径”，在答案末尾给出 Markdown 链接。
- 当用户问“某公司市场规模”时，按以下口径回答：
  1) 优先给该公司对应“所在细分赛道”的市场规模（时间+金额+口径）；
  2) 若无公司直连口径，必须给“细分赛道市场规模 + 上游/大行业市场规模”作为替代；
  3) 若有估值/市占率/融资轮次，单列“资本与份额”；
  4) 明确区分“已披露数据”与“基于检索片段的行业映射”。

{extra_line}- 当检索片段中出现“内部保密标识”“禁止外传”，自动不显示“内部保密标识”“禁止外传”，但是语句逻辑要通顺。

- 当目标公司在片段中命中弱或未命中时：
  - 不要只返回“查无信息”。
  - 必须输出“替代答案包”：主营业务（若可推断）+ 细分行业规模 + 大行业规模 + 可得估值/份额（无数据则写“现有资料未披露”）。
  - 若主营业务也无法从片段推断，再明确说明“知识库未覆盖该公司”，并给出最相关的{fallback_reference}。

- 证据要点必须来自检索片段，尽量量化（{evidence_dimensions}）；证据仅做结论性归纳，不得出现文件名、页码、路径或 chunk 编号等来源信息。
</instruction>"""


def _build_industry_suggestion_prompt(domain: str) -> str:
    return f"基于以下资料和问题，生成3个与“{domain}”相关的后续提问。"

SYSTEM_PROMPT_BY_CATEGORY = {
    "space": """你是“商业航天知识库”问答助手。
<instruction>
- 聚焦商业航天产业链、公司业务、技术路线、市场格局、融资与估值信息。
- 输出风格必须“业务口径优先”，先直接给可落地结论，不要先写判别过程或元解释。
- 禁止出现这类开场：如“X本身不直接定义市场规模”“未出现/无法判断所以无法回答”等流程化判别话术。
- 禁止出现这类涉及内部保密文件话语：多份投资报告（蓝箭航天、格思航天、上海垣信、中科宇航）均标注“内部保密标识”；“惠华”以“内部保密标识”等形式高频出现。
- 回答语言：中文；默认结构“结论 -> 证据 -> 风险/不确定性”。

- 当用户问“某公司市场规模”时，按以下口径回答：
  1) 优先给该公司对应“所在细分赛道”的市场规模（时间+金额+口径）；
  2) 若无公司直连口径，必须给“细分赛道市场规模 + 上游/大行业市场规模”作为替代；
  3) 若有估值/市占率/融资轮次，单列“资本与份额”；
  4) 明确区分“已披露数据”与“基于检索片段的行业映射”。

- 当检索片段中出现“内部保密标识”“禁止外传”，自动不显示“内部保密标识”“禁止外传”，但是语句逻辑要通顺。

- 当目标公司在片段中命中弱或未命中时：
  - 不要只返回“查无信息”。
  - 必须输出“替代答案包”：主营业务（若可推断）+ 细分行业规模 + 大行业规模 + 可得估值/份额（无数据则写“现有资料未披露”）。
  - 若主营业务也无法从片段推断，再明确说明“知识库未覆盖该公司”，并给出最相关的行业规模参考。

- 证据要点必须来自检索片段，尽量量化（时间、金额、比例、型号）；证据仅做结论性归纳，不得出现文件名、页码、路径或 chunk 编号等来源信息。
- 若上下文含“相关图片路径/相关表格路径”，在答案末尾给出 Markdown 链接。
</instruction>""",
    "semiconductor": _build_industry_system_prompt(
        kb_name="半导体",
        focus="半导体材料、设备、设计、制造、封测、功率器件、模拟/数字芯片、传感器与下游应用等细分赛道",
        fallback_reference="半导体细分赛道规模参考",
        evidence_dimensions="时间、金额、产能、良率、制程、客户、产品参数或国产化比例",
        extra_guidance="需要明确区分设计、制造、设备、材料和封测环节，避免把上下游口径混用。",
    ),
    "military_special": _build_industry_system_prompt(
        kb_name="军工",
        focus="航空航天装备、军工电子、动力系统、武器装备、军工信息化、军用芯片与元器件、关键材料及配套制造等细分赛道",
        fallback_reference="军工细分赛道规模参考",
        evidence_dimensions="时间、金额、型号、客户验证、军工资质、配套层级、批产进展或国产替代比例",
        extra_guidance="优先区分整机、分系统、核心部件和配套材料/工艺环节，并说明资质壁垒、定点关系和批产节奏。",
    ),
    "it_software": """你是“IT软件知识库”问答助手。
<instruction>
- 聚焦基础软件、工业软件、网络安全、数据库、大数据、企业服务、视频协同与行业数字化等细分赛道。
- 优先回答公司主营产品、技术路线、商业模式、客户结构、市场空间、竞争格局、国产替代进展、财务/融资动态和主要风险。
- 输出风格默认使用“结论 -> 依据 -> 风险/不确定性”。
- 回答语言：中文。
- 当用户问“某公司市场规模”时，按以下口径回答：
  1) 优先给该公司对应“所在细分赛道”的市场规模（时间+金额+口径）；
  2) 若无公司直连口径，必须给“细分赛道市场规模 + 上游/大行业市场规模”作为替代；
  3) 若有估值/市占率/融资轮次，单列“资本与份额”；
  4) 明确区分“已披露数据”与“基于检索片段的行业映射”。

- 当检索片段中出现“内部保密标识”“禁止外传”，自动不显示“内部保密标识”“禁止外传”，但是语句逻辑要通顺。

- 当目标公司在片段中命中弱或未命中时：
  - 不要只返回“查无信息”。
  - 必须输出“替代答案包”：主营业务（若可推断）+ 细分行业规模 + 大行业规模 + 可得估值/份额（无数据则写“现有资料未披露”）。
  - 若主营业务也无法从片段推断，再明确说明“知识库未覆盖该公司”，并给出最相关的软件细分赛道参考。

- 证据要点必须来自检索片段，尽量量化（时间、金额、比例、客户、产品或技术指标）；证据仅做结论性归纳，不得出现文件名、页码、路径或 chunk 编号等来源信息。
- 若上下文含“相关图片路径/相关表格路径”，在答案末尾给出 Markdown 链接。
</instruction>""",
    "internet": _build_industry_system_prompt(
        kb_name="互联网",
        focus="产业互联网、平台经济、企业服务、物联网、数字营销、线上渠道与互联网基础设施等细分赛道",
        fallback_reference="互联网或平台服务细分赛道参考",
        evidence_dimensions="时间、金额、流量、用户数、ARPU、客户数量、GMV或付费转化指标",
        extra_guidance="涉及平台类公司时，优先区分 ToC、ToB 与平台撮合模式，并说明收入驱动因素。",
    ),
    "communication_electronics_hardware": _build_industry_system_prompt(
        kb_name="通信与电子硬件",
        focus="通信设备、光通信、射频、连接器、模组、电子元器件、终端硬件与上游核心器件等细分赛道",
        fallback_reference="通信与电子硬件细分赛道规模参考",
        evidence_dimensions="时间、金额、产品型号、性能指标、客户认证、ASP、出货量或市占率",
        extra_guidance="要区分通信设备、电子元器件和终端硬件的产业链位置，避免把软件能力当成硬件壁垒。",
    ),
    "intelligent_manufacturing": """你是“智能制造知识库”问答助手。
<instruction>
- 聚焦智能制造、先进制造、智能装备、汽车、电气电力、轨道交通与高端装备等细分赛道。
- 优先回答产业链位置、核心产品、技术路线、市场空间、竞争格局、客户验证、财务/融资进展和主要风险。
- 输出风格默认使用“结论 -> 依据 -> 风险/不确定性”。
- 回答语言：中文。
- 若上下文含“相关图片路径/相关表格路径”，在答案末尾给出 Markdown 链接。
- 当用户问“某公司市场规模”时，按以下口径回答：
  1) 优先给该公司对应“所在细分赛道”的市场规模（时间+金额+口径）；
  2) 若无公司直连口径，必须给“细分赛道市场规模 + 上游/大行业市场规模”作为替代；
  3) 若有估值/市占率/融资轮次，单列“资本与份额”；
  4) 明确区分“已披露数据”与“基于检索片段的行业映射”。

- 当检索片段中出现“内部保密标识”“禁止外传”，自动不显示“内部保密标识”“禁止外传”，但是语句逻辑要通顺。

- 当目标公司在片段中命中弱或未命中时：
  - 不要只返回“查无信息”。
  - 必须输出“替代答案包”：主营业务（若可推断）+ 细分行业规模 + 大行业规模 + 可得估值/份额（无数据则写“现有资料未披露”）。
  - 若主营业务也无法从片段推断，再明确说明“知识库未覆盖该公司”，并给出最相关的行业规模参考。

- 证据要点必须来自检索片段，尽量量化（时间、金额、比例、型号）；证据仅做结论性归纳，不得出现文件名、页码、路径或 chunk 编号等来源信息。
- 若上下文含“相关图片路径/相关表格路径”，在答案末尾给出 Markdown 链接。
</instruction>""",
    "new_materials": _build_industry_system_prompt(
        kb_name="新材料",
        focus="先进金属材料、无机非金属材料、高分子材料、复合材料、电子化学品、膜材料与前沿新材料等细分赛道",
        fallback_reference="新材料细分赛道规模参考",
        evidence_dimensions="时间、金额、性能指标、成本、产能、良率、下游认证或替代率",
        extra_guidance="优先说明材料性能、量产能力、认证周期和下游应用，不要只给概念性描述。",
    ),
    "energy_environmental": _build_industry_system_prompt(
        kb_name="能源环保",
        focus="新能源、新能源汽车、储能、氢能、可再生能源、节能环保与资源综合利用等细分赛道",
        fallback_reference="能源环保细分赛道规模参考",
        evidence_dimensions="时间、金额、装机量、产能、效率、成本、补贴政策或渗透率",
        extra_guidance="涉及新能源时，优先区分发电侧、储能侧、用能侧和环保治理侧的商业逻辑。",
    ),
    "consumer_modern_services": _build_industry_system_prompt(
        kb_name="消费品与现代服务",
        focus="消费品品牌、零售渠道、餐饮服务、物流交运、现代服务、跨境电商、宠物经济与本地生活等细分赛道",
        fallback_reference="消费与现代服务细分赛道参考",
        evidence_dimensions="时间、金额、门店数、GMV、客单价、复购率、渠道结构或区域扩张数据",
        extra_guidance="优先判断品牌力、渠道力、供应链效率和单店/单客模型，不要只停留在产品介绍。",
    ),
}

SUGGESTION_PROMPT_BY_CATEGORY = {
    "space": "基于以下资料和问题，生成3个与“商业航天技术、产业、融资或竞争格局”相关的后续提问。",
    "semiconductor": _build_industry_suggestion_prompt("半导体产业链、技术壁垒、国产替代、客户验证或竞争格局"),
    "military_special": _build_industry_suggestion_prompt("军工装备、军工电子、配套层级、批产验证、资质壁垒或竞争格局"),
    "it_software": "基于以下资料和问题，生成3个与“IT软件产品、技术路线、市场空间、客户结构或竞争格局”相关的后续提问。",
    "internet": _build_industry_suggestion_prompt("互联网平台、企业服务、用户增长、商业模式或竞争格局"),
    "communication_electronics_hardware": _build_industry_suggestion_prompt("通信设备、电子硬件、核心器件、客户验证或竞争格局"),
    "intelligent_manufacturing": "基于以下资料和问题，生成3个与“智能制造产业链、技术路线、客户验证或竞争格局”相关的后续提问。",
    "new_materials": _build_industry_suggestion_prompt("材料性能、下游应用、量产认证、成本结构或竞争格局"),
    "energy_environmental": _build_industry_suggestion_prompt("新能源装机、储能、环保治理、政策驱动或竞争格局"),
    "consumer_modern_services": _build_industry_suggestion_prompt("品牌、渠道、单店模型、用户增长或竞争格局"),
}

SOURCE_LEAK_PATTERNS = [
    re.compile(r"《[^》]{1,120}》"),
    re.compile(r"XX文件\s*严禁外传"),
    re.compile(r"严禁外传|仅供内部|页脚|页码"),
    re.compile(r"chunk\s*编号?|chunk\s*\d+", re.IGNORECASE),
    re.compile(r"fileName|doc_name", re.IGNORECASE),
    re.compile(r"\.pdf|\.docx?|\.pptx?|\.xlsx?", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^ \n]+"),
    re.compile(r"/media/[^\s\])]+"),
]


class RAG:
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        kb_category: str = "employee",
        image_source_dirs: List[str] = None,
        table_source_dirs: List[str] = None,
        skip_load: bool = False,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.kb_category = (kb_category or "employee").strip().lower()
        self.image_source_dirs = image_source_dirs if image_source_dirs else DEFAULT_IMAGE_DIRS
        self.table_source_dirs = table_source_dirs if table_source_dirs else DEFAULT_TABLE_DIRS
        self.skip_load = skip_load

        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.bm25 = None
        self.load_error = None

        if (not self.skip_load) and FlagReranker is not None:
            print(f"Loading reranker: {RERANK_MODEL_NAME} ...")
            try:
                self.reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)
                print("Reranker loaded (FP16).")
            except Exception as e:
                print(f"[ERROR] Failed to load reranker: {e}")
                self.reranker = None
        else:
            self.reranker = None

        if self.skip_load:
            self.load_error = "RAG resources not loaded (generation-only mode)."
        else:
            self._load_resources()

    def _load_resources(self):
        print(f"Loading KB index: {os.path.basename(self.index_path)} ...")
        try:
            if not Path(self.index_path).exists():
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
            self.index = self._read_faiss_index_compat(self.index_path)

            if Path(self.metadata_path).exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    for line in f:
                        self.metadata.append(json.loads(line))

            tokenized_corpus = [jieba.lcut(doc.get("text", "")) for doc in self.metadata]
            # 空知识库占位索引：允许启动，但不启用关键词检索（避免 BM25 触发除零）。
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
            print("KB loaded.")

        except Exception as e:
            self.load_error = str(e)
            print(f"KB load failed: {e}")

    def _read_faiss_index_compat(self, index_path: str):
        """
        Windows + non-ASCII path compatibility:
        faiss.read_index(path) may fail on Chinese paths. Fallback to reading
        bytes via Python and deserializing in-memory.
        """
        try:
            return faiss.read_index(index_path)
        except Exception:
            with open(index_path, "rb") as f:
                raw = f.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            return faiss.deserialize_index(arr)

    def _resolve_from_dirs(self, raw_path: str, source_dirs: List[str]) -> Optional[Path]:
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p
        filename = p.name
        parts = list(p.parts)
        for source_dir in source_dirs:
            candidate = Path(source_dir) / filename
            if candidate.exists():
                return candidate
            # Old indexes may store absolute paths from another machine.
            # Try matching by the tail parts to recover nested files robustly.
            if parts:
                max_tail = min(len(parts), 6)
                for keep in range(max_tail, 1, -1):
                    rel_tail = Path(*parts[-keep:])
                    nested = Path(source_dir) / rel_tail
                    if nested.exists():
                        return nested
        return None

    def _resolve_image_absolute(self, img_path: str) -> Optional[Path]:
        return self._resolve_from_dirs(img_path, self.image_source_dirs)

    def _resolve_table_absolute(self, table_path: str) -> Optional[Path]:
        return self._resolve_from_dirs(table_path, self.table_source_dirs)

    def _media_file_signature(self, abs_path: Path) -> str:
        try:
            st = abs_path.stat()
            h = hashlib.sha1()
            with abs_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return f"{st.st_size}:{h.hexdigest()}"
        except Exception:
            return f"path:{str(abs_path.resolve()).lower()}"

    def _prefer_media_path(self, old_path: Path, new_path: Path) -> Path:
        old_name = old_path.name.lower()
        new_name = new_path.name.lower()
        old_generic = bool(re.search(r"_image_\d{1,4}", old_name))
        new_generic = bool(re.search(r"_image_\d{1,4}", new_name))
        if old_generic and not new_generic:
            return new_path
        if new_generic and not old_generic:
            return old_path
        if len(new_path.name) > len(old_path.name):
            return new_path
        return old_path

    def _to_media_web_path(self, abs_path: Path) -> str:
        try:
            rel = abs_path.resolve().relative_to(PROJECT_ROOT.resolve())
            rel_str = str(rel).replace("\\", "/")
            return "/media/" + quote(rel_str, safe="/._-")
        except Exception:
            return "/media/" + quote(abs_path.name, safe="._-")

    def _extract_page_hint_from_text(self, text: str) -> Optional[int]:
        if not text:
            return None
        patterns = [
            r"(?:Slide|slide)\s*(\d+)",
            r"(?:Page|page)\s*(\d+)",
            r"第\s*(\d+)\s*页",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    def _extract_page_hint_from_media_path(self, media_path: str) -> Optional[int]:
        if not media_path:
            return None
        filename = Path(media_path).name
        patterns = [
            r"_p(\d+)_",
            r"(?:slide|page|页|p)(\d+)",
            r"_(\d{1,4})\.",
        ]
        for pat in patterns:
            m = re.search(pat, filename, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    def _question_tokens(self, question: str) -> List[str]:
        if not question:
            return []
        tokens = [t.strip().lower() for t in jieba.lcut(question) if len(t.strip()) >= 2]
        # 去重但保持顺序
        seen = set()
        ordered = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            ordered.append(t)
        return ordered

    def _rank_media_paths(
        self,
        media_paths: List[str],
        question_tokens: List[str],
        chunk_text: str,
        chunk_page: Optional[int],
        seen_paths: set,
        max_pick: int,
    ) -> List[str]:
        if not media_paths:
            return []

        if chunk_page is None:
            chunk_page = self._extract_page_hint_from_text(chunk_text)
        scored = []
        for idx, p in enumerate(media_paths):
            base = Path(p).name.lower()
            score = 0.0

            # 1) 优先未出现过，避免连续问答总是重复同图。
            if p not in seen_paths:
                score += 100.0

            # 2) 问题词与文件名匹配（弱相关但有效）。
            if question_tokens:
                hit = sum(1 for t in question_tokens if t in base)
                score += hit * 8.0

            # 3) 页码接近度（当文件名中可解析页码时）。
            media_page = self._extract_page_hint_from_media_path(p)
            if chunk_page is not None and media_page is not None:
                delta = abs(chunk_page - media_page)
                score += max(0.0, 20.0 - delta * 4.0)

            # 4) 保留顺序偏好（同分时优先前面的素材）。
            score += max(0.0, 5.0 - idx * 0.1)
            scored.append((score, idx, p))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [p for _, _, p in scored[:max_pick]]

    def _is_useful_image(self, img_path: str) -> bool:
        if not img_path:
            return False
        name_lower = Path(img_path).name.lower()
        bad_keywords = ["icon", "logo", "watermark", "avatar", "bg", "background"]
        if any(k in name_lower for k in bad_keywords):
            return False
        abs_p = self._resolve_image_absolute(img_path)
        if abs_p is None:
            return False
        try:
            size_kb = abs_p.stat().st_size / 1024
            if size_kb < 10:
                return False
            with Image.open(abs_p) as im:
                if im.width < 50 or im.height < 50:
                    return False
            return True
        except Exception:
            return False

    def _is_useful_table(self, table_path: str) -> bool:
        if not table_path:
            return False
        abs_p = self._resolve_table_absolute(table_path)
        if abs_p is None:
            return False
        if abs_p.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".webp",
            ".svg",
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".md",
            ".txt",
            ".json",
            ".pdf",
        }:
            return False
        try:
            return abs_p.stat().st_size > 0
        except Exception:
            return False

    def _get_query_embedding(self, text: str) -> np.ndarray:
        try:
            if EMBEDDING_PROVIDER == "dashscope":
                last_error = None
                for attempt in range(EMBEDDING_MAX_RETRIES + 1):
                    response = dashscope.TextEmbedding.call(
                        model=DASHSCOPE_EMBEDDING_MODEL,
                        input=[text],
                        dimension=DASHSCOPE_EMBEDDING_DIM,
                        text_type="query",
                    )
                    status_code = getattr(response, "status_code", None)
                    if status_code == HTTPStatus.OK:
                        output = getattr(response, "output", {}) or {}
                        embeddings = output.get("embeddings") if isinstance(output, dict) else getattr(output, "embeddings", None)
                        if not embeddings:
                            raise RuntimeError("DashScope embedding returned empty embeddings")
                        first = embeddings[0]
                        vector = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
                        if not isinstance(vector, list) or not vector:
                            raise RuntimeError("DashScope embedding vector missing")
                        return np.array(vector, dtype=np.float32).reshape(1, -1)

                    retryable = status_code in {
                        HTTPStatus.TOO_MANY_REQUESTS,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        HTTPStatus.BAD_GATEWAY,
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        HTTPStatus.GATEWAY_TIMEOUT,
                    }
                    last_error = RuntimeError(
                        "DashScope embedding failed: "
                        f"status={status_code} code={getattr(response, 'code', '')} "
                        f"message={getattr(response, 'message', '')}"
                    )
                    if (not retryable) or attempt >= EMBEDDING_MAX_RETRIES:
                        raise last_error
                    time.sleep(EMBEDDING_RETRY_BACKOFF_SECONDS * (2 ** attempt))

                if last_error is not None:
                    raise last_error
                raise RuntimeError("DashScope embedding retry loop exited unexpectedly")

            resp = ollama.embeddings(model=OLLAMA_EMB_MODEL, prompt=text)
            return np.array(resp["embedding"], dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print(f"Embedding error: {e}")
            fallback_dim = DASHSCOPE_EMBEDDING_DIM if EMBEDDING_PROVIDER == "dashscope" else 1024
            return np.zeros((1, fallback_dim), dtype=np.float32)

    def _vector_search(self, query_emb: np.ndarray) -> Tuple[List[float], List[int]]:
        distances, idxs = self.index.search(query_emb, TOP_K)
        return distances[0].tolist(), idxs[0].tolist()

    def _kw_search(self, query: str) -> Tuple[List[float], List[int]]:
        if self.bm25 is None or not self.metadata:
            return [], []
        tokenized_query = jieba.lcut(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_idxs = np.argsort(doc_scores)[-BM25_TOP_K:][::-1]
        top_n_scores = [doc_scores[i] for i in top_n_idxs]
        return top_n_scores, top_n_idxs.tolist()

    def _rrf_fusion(self, vector_results, keyword_results) -> List[Tuple[int, float]]:
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(vector_results[1]):
            if idx == -1:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (RRF_K + rank + 1)
        for rank, idx in enumerate(keyword_results[1]):
            if idx == -1:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (RRF_K + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates or self.reranker is None:
            return candidates

        pairs = [[query, doc.get("text", "")[:800]] for doc in candidates]
        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = score

        return sorted(candidates, key=lambda x: x.get("rerank_score", -999), reverse=True)

    def _build_context(self, ranked_docs: List[Dict[str, Any]], question: str) -> Tuple[str, List[Dict[str, Any]]]:
        parts = []
        refs = []
        seen_images = set()
        seen_tables = set()
        question_tokens = self._question_tokens(question)
        total_images = 0
        total_tables = 0

        filtered_docs = []
        is_using_reranker = self.reranker is not None and len(ranked_docs) > 0 and "rerank_score" in ranked_docs[0]
        top_score = ranked_docs[0].get("rerank_score", 0) if is_using_reranker and ranked_docs else 0

        for doc in ranked_docs:
            if is_using_reranker:
                current_score = doc.get("rerank_score", -99)
                if current_score < RERANK_THRESHOLD:
                    continue
                if (top_score - current_score) > RERANK_GAP_THRESHOLD:
                    continue
            filtered_docs.append(doc)

        final_docs = [ranked_docs[0]] if (not filtered_docs and ranked_docs) else filtered_docs[:FINAL_TOP_K]

        for rank, m in enumerate(final_docs, start=1):
            raw_img_paths = m.get("image_paths", []) or []
            filtered_paths = [p for p in raw_img_paths if self._is_useful_image(p)]
            all_img_abs_paths: List[Path] = []
            for p in filtered_paths:
                abs_img = self._resolve_image_absolute(p)
                if abs_img is not None:
                    all_img_abs_paths.append(abs_img)
            unique_img_abs_by_sig: Dict[str, Path] = {}
            for abs_img in all_img_abs_paths:
                sig = self._media_file_signature(abs_img)
                if sig not in unique_img_abs_by_sig:
                    unique_img_abs_by_sig[sig] = abs_img
                else:
                    unique_img_abs_by_sig[sig] = self._prefer_media_path(unique_img_abs_by_sig[sig], abs_img)
            all_img_web_paths: List[str] = [self._to_media_web_path(p) for p in unique_img_abs_by_sig.values()]
            remaining_img_budget = max(0, MAX_TOTAL_IMAGES - total_images)
            per_chunk_img_limit = min(MAX_IMGS_PER_CHUNK, remaining_img_budget) if remaining_img_budget > 0 else 0
            img_web_paths = self._rank_media_paths(
                media_paths=all_img_web_paths,
                question_tokens=question_tokens,
                chunk_text=m.get("text", "") or "",
                chunk_page=m.get("page_no"),
                seen_paths=seen_images,
                max_pick=per_chunk_img_limit,
            )

            raw_table_paths = m.get("table_paths", []) or []
            filtered_table_paths = [p for p in raw_table_paths if self._is_useful_table(p)]
            all_table_web_paths: List[str] = []
            for p in filtered_table_paths:
                abs_table = self._resolve_table_absolute(p)
                if abs_table is not None:
                    all_table_web_paths.append(self._to_media_web_path(abs_table))
            table_web_paths = self._rank_media_paths(
                media_paths=all_table_web_paths,
                question_tokens=question_tokens,
                chunk_text=m.get("text", "") or "",
                chunk_page=m.get("page_no"),
                seen_paths=seen_tables,
                max_pick=min(MAX_TABLES_PER_CHUNK, max(0, MAX_TOTAL_TABLES - total_tables)),
            )

            unique_context_imgs = []
            for img_path in img_web_paths:
                if img_path not in seen_images:
                    seen_images.add(img_path)
                    unique_context_imgs.append(img_path)
                    total_images += 1
            unique_context_tables = []
            for table_path in table_web_paths:
                if table_path not in seen_tables:
                    seen_tables.add(table_path)
                    unique_context_tables.append(table_path)
                    total_tables += 1

            display_score = m.get("rerank_score", m.get("rrf_score", 0))
            refs.append(
                {
                    "ref_id": rank,
                    "file_path": m.get("file_path"),
                    "doc_name": Path(m.get("file_path", "")).name if m.get("file_path") else "",
                    "preview_text": (m.get("text", "") or "")[:100],
                    # Return globally de-duplicated media to the frontend.
                    "image_paths": unique_context_imgs,
                    "table_paths": unique_context_tables,
                    "score": float(display_score),
                    "type": m.get("type", "text"),
                }
            )

            text_content = m.get("text", "")
            if unique_context_imgs:
                text_content += "\n\n[相关图片路径: " + ", ".join(unique_context_imgs) + "]"
            if unique_context_tables:
                text_content += "\n\n[相关表格路径: " + ", ".join(unique_context_tables) + "]"

            chunk_xml = (
                f'<chunk fileId="{m.get("chunk_id")}" '
                f'fileName="{Path(m.get("file_path", "")).name}" '
                f'score="{display_score:.4f}">'
                f"{text_content}"
                f"</chunk>"
            )
            parts.append(chunk_xml)

        if not parts:
            ctx = "<retrieved_chunks>No relevant information found.</retrieved_chunks>"
        else:
            ctx = "<retrieved_chunks>\n" + "\n".join(parts) + "\n</retrieved_chunks>"

        if len(ctx) > CONTEXT_MAX_CHARS:
            ctx = ctx[:CONTEXT_MAX_CHARS] + "\n<!-- Context truncated -->"

        return ctx, refs

    def _get_system_prompt(self) -> str:
        base_prompt = SYSTEM_PROMPT_BY_CATEGORY.get(self.kb_category, DEFAULT_SYSTEM_PROMPT)
        return f"{base_prompt}\n\n{SOURCE_PRIVACY_GUARDRAIL}"

    def _get_suggestion_prefix(self) -> str:
        return SUGGESTION_PROMPT_BY_CATEGORY.get(
            self.kb_category,
            "基于以下资料和问题，生成3个简短的后续提问。",
        )

    def _contains_source_leak(self, text: str) -> bool:
        if not text:
            return False
        return any(p.search(text) for p in SOURCE_LEAK_PATTERNS)

    def _sanitize_answer_text(self, answer: str) -> str:
        if not answer:
            return answer

        text = answer
        # 优先做短语级替换，尽量保留业务结论。
        phrase_replacements = [
            (r"仅出现在《[^》]{1,120}》[^。；\n]*“[^”\n]*严禁外传[^”\n]*”", "根据相关内部资料，相关材料带有内部保密标识"),
            (r"在《[^》]{1,120}》[^。；\n]*(?:标注|显示|写有)[^。；\n]*", "根据相关内部资料，相关材料带有内部保密标识"),
            (r"全文未(?:提及|显示)", "检索知识库，未发现"),
            (r"惠华文件\s*严禁外传", "内部保密标识"),
            (r"《[^》]{1,120}》", "相关内部资料"),
        ]
        for pattern, repl in phrase_replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # 句子级兜底：若仍含来源痕迹，则整句改写为统一保密表达。
        parts = re.split(r"(\n+)", text)
        sanitized_parts: List[str] = []
        for part in parts:
            if not part or part.isspace():
                sanitized_parts.append(part)
                continue
            if self._contains_source_leak(part):
                sanitized_parts.append("根据相关内部资料，得出上述结论。")
            else:
                sanitized_parts.append(part)

        text = "".join(sanitized_parts)
        text = re.sub(r"(根据相关内部资料，得出上述结论。){2,}", "根据相关内部资料，得出上述结论。", text)
        return text

    def _build_generation_messages(self, context: str, question: str) -> List[Dict[str, str]]:
        system_prompt = self._get_system_prompt()
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _stringify_generation_content(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                text = self._stringify_generation_content(item)
                if text:
                    parts.append(text)
            return "".join(parts)
        if isinstance(value, dict):
            for key in ("text", "content"):
                if key in value:
                    return self._stringify_generation_content(value.get(key))
            for key in ("message", "delta"):
                if key in value:
                    return self._stringify_generation_content(value.get(key))
            return ""
        for attr in ("text", "content", "message", "delta"):
            if hasattr(value, attr):
                return self._stringify_generation_content(getattr(value, attr))
        return ""

    def _extract_generation_text(self, response: Any) -> str:
        choice = None
        try:
            choice = response.output.choices[0]
        except Exception:
            pass

        if choice is None and isinstance(response, dict):
            output = response.get("output") or {}
            choices = output.get("choices") or []
            if choices:
                choice = choices[0]

        if choice is None:
            return ""

        if isinstance(choice, dict):
            for key in ("message", "delta", "content"):
                if key in choice:
                    return self._stringify_generation_content(choice.get(key))
            return ""

        for attr in ("message", "delta", "content"):
            if hasattr(choice, attr):
                return self._stringify_generation_content(getattr(choice, attr))
        return ""

    def _extract_generation_status_code(self, response: Any) -> Optional[int]:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return int(status_code)
        if isinstance(response, dict):
            code = response.get("status_code")
            if code is not None:
                return int(code)
        return None

    def _extract_generation_error(self, response: Any) -> str:
        message = getattr(response, "message", None)
        if not message and isinstance(response, dict):
            message = response.get("message")
        message = str(message or "未知错误")
        return (
            f"模型服务出错: {message} "
            f"（当前 model={API_GENERATION_MODEL}，请在 .env 的 DASHSCOPE_CHAT_MODEL 改为控制台已开通的模型名）"
        )

    def _merge_stream_piece(self, accumulated: str, piece: str) -> str:
        if not piece:
            return accumulated
        if accumulated and piece.startswith(accumulated):
            return piece
        return accumulated + piece

    def iter_answer_dashscope(self, context: str, question: str) -> Generator[str, None, None]:
        messages = self._build_generation_messages(context, question)
        raw_answer = ""
        last_sanitized = ""

        try:
            responses = dashscope.Generation.call(
                model=API_GENERATION_MODEL,
                messages=messages,
                result_format="message",
                temperature=0.7,
                stream=True,
                incremental_output=True,
            )

            saw_stream_response = False
            for response in responses:
                saw_stream_response = True
                status_code = self._extract_generation_status_code(response)
                if status_code == HTTPStatus.OK:
                    piece = self._extract_generation_text(response)
                    if not piece:
                        continue
                    raw_answer = self._merge_stream_piece(raw_answer, piece)
                    sanitized = self._sanitize_answer_text(raw_answer)
                    if sanitized and sanitized != last_sanitized:
                        last_sanitized = sanitized
                        yield sanitized
                else:
                    error_text = self._extract_generation_error(response)
                    if error_text and error_text != last_sanitized:
                        yield error_text
                    return

            if saw_stream_response and last_sanitized:
                return
        except TypeError:
            pass
        except Exception:
            pass

        fallback_answer = self._ask_llm_dashscope(context, question)
        if fallback_answer and fallback_answer != last_sanitized:
            yield fallback_answer

    def _ask_llm_dashscope(self, context: str, question: str) -> str:
        messages = self._build_generation_messages(context, question)

        try:
            response = dashscope.Generation.call(
                model=API_GENERATION_MODEL,
                messages=messages,
                result_format="message",
                temperature=0.7,
            )
            if response.status_code == HTTPStatus.OK:
                raw_answer = self._extract_generation_text(response)
                return self._sanitize_answer_text(raw_answer)
            return self._extract_generation_error(response)
        except Exception:
            return "API 服务连接失败。"

    def _generate_suggestions_dashscope(self, context: str, question: str) -> List[str]:
        short_ctx = context[:800] if context else ""
        prompt = (
            f"{self._get_suggestion_prefix()}\n"
            f"资料: {short_ctx}...\n问题: {question}\n"
            f"要求: 只输出3行问题，不要序号。"
        )
        try:
            response = dashscope.Generation.call(
                model=API_GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                temperature=0.7,
            )
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                return [re.sub(r"^[\d\.\-\s]+", "", line.strip()) for line in content.split("\n") if len(line) > 4][:3]
            return ["请详细介绍一下？", "有什么特点？", "应用场景是什么？"]
        except Exception:
            return ["请详细介绍一下？", "有什么特点？", "应用场景是什么？"]

    def chat(self, question: str) -> Dict[str, Any]:
        if self.load_error:
            return {"answer": f"错误: {self.load_error}", "references": [], "related_questions": []}
        if self.index is None:
            return {"answer": "索引未初始化", "references": [], "related_questions": []}

        q_emb = self._get_query_embedding(question)
        vec_dists, vec_idxs = self._vector_search(q_emb)
        bm25_scores, bm25_idxs = self._kw_search(question)

        rrf_ranked = self._rrf_fusion((vec_dists, vec_idxs), (bm25_scores, bm25_idxs))

        candidate_docs = []
        for idx, rrf_score in rrf_ranked[:RRF_CANDIDATES]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            doc_item = self.metadata[idx].copy()
            doc_item["rrf_score"] = rrf_score
            candidate_docs.append(doc_item)

        final_ranked_docs = self._rerank_candidates(question, candidate_docs)
        context, refs = self._build_context(final_ranked_docs, question)

        answer_text = self._ask_llm_dashscope(context, question)
        related_qs = self._generate_suggestions_dashscope(context, question)

        return {
            "answer": answer_text,
            "references": refs,
            "related_questions": related_qs,
        }
