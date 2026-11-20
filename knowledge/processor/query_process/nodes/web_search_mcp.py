import asyncio
import json
from typing import Tuple, List, Dict, Any

from agents.mcp import MCPServerSse, MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):

    name = "web_search_mcp"

    def process(self, state):
        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 只更新 state 的 web_search_docs
        return {'web_search_docs': mcp_result}

    # 参数校验
    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数。

        Args:
            state: 查询图状态。

        Returns:
            校验后的 rewritten_query 和 item_names。

        Raises:
            StateFieldError: 参数校验失败时抛出。
        """
        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    # 创建mcp_client 并且让客户端执行工具
    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """异步执行 MCP 网络搜索。

        Args:
            validated_rewritten_query: 校验后的查询文本。

        Returns:
            搜索结果列表，每项包含 title、url、snippet。
        """
        # 1.获取mcp服务端
        async with MCPServerStreamableHttp(
            name="网络搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                "timeout": 300,
                "terminate_on_close": True
            },
            max_retry_attempts=2,
            cache_tools_list=True
        ) as client:

            # 3.执行mcp工具
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": validated_rewritten_query,
                    "count": 3
                }
            )

        # 4.解析执行完返回的3条结果
        if not execute_tool_result:
            return []
        if not execute_tool_result.content[0]:
            return []

        # 4.1 获取 TextContent 对象的 text
        text_content_text: str = execute_tool_result.content[0].text
        if not text_content_text:
            return []

        # 4.2 反序列化
        try:
            text_content_text: Dict[str, Any] = json.loads(text_content_text)

            # a) 获取 pages
            pages = text_content_text.get("pages", "")
            if not pages:
                return []

            search_result = []
            # b) 遍历得到每一个结果
            for page in pages:
                snippet = page.get("snippet", "").strip()
                title = page.get("title", "").strip()
                url = page.get("url", "").strip()
                search_result.append({"snippet": snippet, "title": title, "url": url})

            # c) 最终返回
            return search_result
        except Exception as e:
            self.logger.error("反序列MCP结果失败")
            return []



if __name__ == '__main__':
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    mcp_search = WebSearchMcpNode()
    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))