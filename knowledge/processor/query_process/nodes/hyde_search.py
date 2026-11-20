from typing import Tuple, List
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

logger = logging.getLogger(__name__)

class HyDeSearchNode(BaseNode):

    name = "hyde_search"

    def process(self, state):
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus 客户端
        embedding_model = AIClients.get_bge_m3_client()
        milvus_client = StorageClients.get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_bge_m3_hybrid_vectors(
            model=embedding_model,
            embedding_documents=[embedding_document],
        )

        if not embedding_result:
            return state

        # 5. 获取 item_name 的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filtered_expr
        )

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=['chunk_id', 'content', 'item_name', 'title']
        )

        if not reps or not reps[0]:
            return state

        # 8. 只更新 hyde_embedding_chunks
        return {'hyde_embedding_chunks': reps[0]}

    # 参数校验
    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        return rewritten_query, item_names

    # 生成假设性文档
    def _generate_hy_document(self, validated_query, validate_item_names):
        # 1. 获取 LLM 客户端
        llm_client = AIClients.get_llm_openai(response_format=False)

        # 2. 判断
        if llm_client is None:
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=validate_item_names,
            rewritten_query=validated_query,
        )
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            # 4. 获取 AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', '').strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    # 获取 item_name 的过滤表达式
    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        """构建商品名过滤表达式"""
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"



if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    import json
    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    for r in result.get('hyde_embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))

    print("-" * 60)
    print("测试完成")