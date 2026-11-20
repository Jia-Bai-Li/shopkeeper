import json
import os

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.nodes import ducment_split
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_rec_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.参数校验
        file_title,chunks,item_name_chunk_k,item_name_chunk_size = self._validate_state(state)

        # 2.构建识别上下文
        final_context = self._prepare_item_name_recognition_context(chunks,item_name_chunk_k,item_name_chunk_size)

        # 3.调用LLM大模型识别商品名称
        result = self._recognition_name(file_title,final_context)
        item_name = json.loads(result)['item_name']

        # 4.向量嵌入：将商品名称转化为稠密和稀疏向量
        dense_vector,sparse_vector = self._embedding_item_name(item_name)

        # 5.保存到Milvus
        self._insert_milvus(file_title,item_name,dense_vector,sparse_vector,self.config.item_name_collection)

        # 6.回填商品名称到每个切片当中,以及更新state
        self._fill_item_name(item_name,chunks,state)

        return state

    # 参数校验
    def _validate_state(self, state):

        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title,chunks,item_name_chunk_k,item_name_chunk_size

    # 构建识别上下文
    def _prepare_item_name_recognition_context(self, chunks, item_name_chunk_k, item_name_chunk_size):
        total = 0
        final_context = []

        for idx, chunk in enumerate(chunks[:item_name_chunk_k]):
            # 判断chunk对象类型
            if not isinstance(chunk, dict):
                continue

            # 获取内容
            chunk_content = chunk.get("content")
            context = f"【切片】-{idx}-{chunk_content}"

            # 判断是否为一个完整的chunk，如果字数超过item_name_chunk_size就舍弃最后一个
            if total + len(context) > item_name_chunk_size:
                break
            total += len(context)
            final_context.append(context)

        return "\n".join(final_context)

    # 调用LLM大模型识别商品名称
    def _recognition_name(self, file_title, final_context) -> str:

        try:
            # 1. 获取LLM客户端（不需要JSON格式）
            llm_client = AIClients.get_llm_openai(response_format=True)

            # 2. 构建提示词
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title,
                context=final_context,
            )
            system_prompt = ITEM_NAME_SYSTEM_PROMPT

            # 3. LLM调用
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            # 4. 获取并校验结果
            llm_result = llm_response.content.strip()

            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    # 向量嵌入：将商品名称转化为稠密和稀疏向量
    def _embedding_item_name(self, item_name):
        try:
            # 1. 获取嵌入模型客户端
            bge_m3_client = AIClients.get_bge_m3_client()

            vector_result = bge_m3_client.encode_documents([item_name])

            # 2. 获取稠密向量
            dense_vector = vector_result["dense"][0].tolist()

            # 3. 获取稀疏向量
            start_index = vector_result["sparse"].indptr[0]
            end_index = vector_result["sparse"].indptr[1]
            token_id = vector_result["sparse"].indices[start_index:end_index].tolist()
            weight = vector_result["sparse"].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None

    # 保存到Milvus
    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        # 1. 向量有效性检查（为空则跳过，不去创建Milvus连接）
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return

        # 2. 获取 Milvus 客户端（单独 try）
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return

        # 3. 数据操作（单独 try）
        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection,milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }

            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    # 如果没有表那就按照规定创建表结构
    def _create_item_name_collection(self, item_name_collection, milvus_client):

        # 添加表结构
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 添加索引结构
        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector",
                              index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector",
                              index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_param)
        self.logger.info(f"集合 {item_name_collection} 创建成功并构建了索引")

    # 回填商品名称到每个切片当中,以及更新state
    def _fill_item_name(self, item_name, chunks, state):
        for chunk in chunks:
            chunk["item_name"] = item_name

        back_chunk = DocumentSplitNode()
        back_chunk._back_chunks(state, chunks)

        state["item_name"] = item_name

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content,
        "file_dir": r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir"
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")