import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import sse_generator


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(title="Query Service", description="知识库查询服务")

    # 允许跨域（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # 允许所有来源
        allow_credentials=True,   # 允许携带凭证
        allow_methods=["*"],      # 允许所有方法
        allow_headers=["*"],      # 允许所有头部
    )

    # 静态文件挂载（前端页面）# http://localhost:8001/front/chat.html
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 注册路由
    register_routes(app)

    return app


def register_routes(app: FastAPI):

    @app.get("/chat")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ):
        # 1. 解析参数
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        # 2. 提交任务
        service.submit_query(task_id, request.is_stream)

        # 3. 根据模式选择处理方式
        if request.is_stream:
            # 流式模式：后台任务 + SSE
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            return StreamSubmitResponse(
                message="Query submitted", session_id=session_id, task_id=task_id
            )
        else:
            # 非流式模式：同步执行
            service.run_query_graph(task_id, session_id, request.query, False)
            answer = service.get_answer(task_id)
            return QueryResponse(message="处理完成", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """SSE 实时返回结果"""
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream"
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        """查询当前会话历史记录"""
        items = service.get_history(session_id, limit)
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        """清除会话历史记录"""
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001, log_level="info")