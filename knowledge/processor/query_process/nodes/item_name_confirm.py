import json
import re
from typing import Dict, Any, Optional, List, Tuple
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names

logger = logging.getLogger(__name__)
config = QueryConfig()

class ItemNameExtractor:
    """商品名提取器：基于用户原始问题和历史对话提取商品名"""
    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        Returns:
            {"item_names": [...], "rewritten_query": "..."}
        """
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        # 1. 获取 LLM 客户端
        llm_client = AIClients.get_llm_openai(response_format=True)
        if llm_client is None:
            return result

        # 2. 组装提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text if history_text else "暂无上下文", query=original_query)
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        # 3. LLM 调用
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        llm_content = llm_response.content.strip()

        # 4. 判断 LLM 的输出
        if not llm_content.strip():
            return result

        try:
            # 5. 清洗和解析
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    # 用于清洗数据
    def _clean_parse(self, llm_response) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)

            # 2.1 清洗 item_names
            raw_item_names = parsed_llm_result.get("item_names")
            if not isinstance(raw_item_names, list):
                cleaned_item_names = []
            else:
                cleaned_item_names = [rwa_item for rwa_item in raw_item_names if rwa_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get("rewritten_query")
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": cleaned_item_names, "rewritten_query": clean_rewritten_query}
        except Exception as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:

    # 用于整合下面三个方法
    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """执行匹配、对齐、过滤三步流程"""
        # 1. 查询向量数据库
        search_results: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_results)

        # 3. 分数差异过滤（仅当 confirmed 有多个时）
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed,search_results)

        return confirmed, options

    # 与milvus数据库里的信息进行混合向量匹配，返回匹配到的item_name信息并返回相似分数的对象
    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
        根据 LLM 提取的商品名，查询向量数据库

        Returns:
            每个 item_name 下的查询结果
            Dict 格式: {"extracted_name": "LLM提取的商品名",
                        "matches": [{"item_name": "向量库商品名", "score": 分数值}]}
        """
        # 1. 定义最终搜索结果
        search_results = []

        # 2. 获取 milvus_client
        milvus_clent = StorageClients.get_milvus_client()
        if milvus_clent is None:
            return []

        # 3. 获取嵌入模型
        embedding_model = AIClients.get_bge_m3_client()
        if embedding_model is None:
            logger.error(f"获取嵌入模型失败")
            return []

        # 4. 嵌入 item_name 获取稠密、稀疏向量
        hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        # 5. 遍历 LLM 提取的所有商品名
        for index, extract_item_name in enumerate(item_names):
            # 5.1 创建混合检索的请求
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result["dense"][index],
                sparse_vector=hybrid_embedding_result["sparse"][index],
            )

            # 5.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_clent,
                collection_name=config.item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5,0.5),
                norm_score=True,
                output_fields=["item_name"]
            )

            # 5.3 解析混合检索请求的结果对象
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]} for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }

            # 5.4 将构建好查询结果放入最终搜索结果中
            search_results.append(item_name_search_result)

        return search_results

    # 用于将向量匹配的结果筛选出置信度高的部分，也就是score分高的，以及其余合格的
    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        根据向量数据库检索到的商品名，分类到 confirmed 或 options

        Returns:
            confirmed: 确认的商品名列表，传给下游多路检索
            options: 候选商品名列表，用于询问用户
        """
        # 1. 定义两个容器用于返回结果
        confirmed = []
        options = []

        # 2. 遍历向量数据库查询到的所有结果
        for item in search_results:
            # 2.0 获取 LLM 提取的商品名
            extracted_name = item.get("extracted_name")

            # 2.1 对某一商品名下找到相似的 item_name 按分数值降序
            matches = sorted(
                item.get("matches") or [],
                key=lambda x: x["score"],
                reverse=True,
            )

            # 2.2 获取 matches 中分数值能进入 confirmed 容器阈值的对象
            high = [m for m in matches if m.get('score') >= 0.7]

            # 3. 询问是否能进入 confirmed 中
            if high:
                # 3.1 准备找最精准的那一个
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)

                # 场景 A: 找到了精确匹配
                if extract:
                    picked = extract['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)

                # 场景 B: 只有一条高置信结果
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)

                # 场景 C: 多条相似，加入 options 让用户选择
                else:
                    for h in high[:3]:
                        picked = h.get('item_name')
                        if picked not in confirmed and picked not in options:
                            options.append(picked)

            # 4. 询问是否能进入 options 中
            else:
                mid = [
                    m for m in matches
                    if m['score'] >= 0.6
                    and m.get('item_name') not in confirmed
                    and m.get('item_name') not in options
                ]

                if mid:
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)

        return confirmed, options[:3]

    # 用于提取出confirmed中分数最高的一个，设置最大分数间隔，用于去除分数偏差太多的item
    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]) -> List[str]:
        """
        将误判的 item_name 从 confirmed 中剔除

        Args:
            confirmed: 初步确认的商品名列表
            search_results: 向量检索结果

        Returns:
            过滤后的 confirmed 列表
        """
        # 1. 获取每个商品名在向量数据库中的最高分数
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get("matches")
            for m in matches:
                score = m.get("score")
                item_name = m.get("item_name")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        # 2. 对 item_name_score 进行排序
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)

        # 3. 取出分数值最大的（作为基准）
        max_item_name_score = sorted_item_name_score[0][1]

        # 4. 保留分数差在阈值内的商品名
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点
    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 历史回填
    """
    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()


    def process(self, state):

        session_id = state['session_id']
        original_query = state['original_query']

        # 1.获取历史对话信息
        chat_history = get_recent_messages(session_id, limit=10)
        history_text = ""
        for msg in chat_history:
            role = msg.get('role')
            content = msg.get('text', "")
            history_text += f"{role}: {content}\n"

        # 2.LLM提取商品名
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get("item_names")
        rewritten_query = clean_llm_result.get("rewritten_query")

        # 3.查询向量数据库 && 过滤
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 5. 决策分支，更新 state
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # 6. 历史回填
        if confirmed:
            # 找出还没有赋予商品名称的对话
            ids_to_update = [
                str(msg["_id"]) for msg in chat_history if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        # 7.将历史对话写入 state，供下游使用
        for history in chat_history:
            history['item_names'] = confirmed
        state['history'] = chat_history

        return state

    # 用于更新state状态,当确认列表有数据时更新state，当只有选择列表时，更新answer继续询问用户，当两个都没有时返回无法识别信息
    def _decide(self, state, item_names, confirmed, options, rewritten_query):
        """根据对齐结果更新 state"""

        if confirmed:
            # 确认成功：更新状态，继续检索流程
            state['rewritten_query'] = rewritten_query
            state['item_names'] = item_names

        elif options:
            # 有候选：设置选择提示，流程中断
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )

        else:
            # 无法识别：设置提示，流程中断
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    test_state: QueryGraphState = {
        "session_id": "sdadada11",
        "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
    }
    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")