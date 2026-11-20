from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from pathlib import Path
import json

class BgeEmbeddingChunksNode(BaseNode):

    name = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.参数校验
        validate_chunks,config = self._validate_get_inputs(state)

        # 2.获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, "embedding_batch_size", 16)

        # 3. 准备分批嵌入
        total_length = len(validate_chunks)
        final_chunks = []

        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validate_chunks[i:i+embedding_batch_chunk_size]

            # 拼接要嵌入的内容，向量嵌入，把嵌入的向量注入到chunk中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 4. 更新&返回state
        state["chunks"] = final_chunks

        return state


    # 参数校验
    def _validate_get_inputs(self, state):
        config = get_config()

        self.log_step("step1", "参数校验")

        # 1. 获取chunks
        chunks = state.get("chunks")

        # 2. 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        # 3. 返回chunks
        self.logger.info(f"嵌入的块数：{len(chunks)}")
        return chunks, config

    # 完整批次处理方法
    def _process_batch_chunks(self, batch, star_index, total_length):
        self.log_step("step2", f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}")

        # 1. 循环处理所有chunk的要嵌入的内容拼接：将商品名称和chunk内容拼接到一块提高识别度
        embedding_contents = []
        for chunk in batch:
            content = chunk.get("content")
            item_name = chunk.get("item_name")
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        # 2. 批量嵌入
        try:
            bge_m3_model = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)

            if not embedding_result:
                self.logger.warning(f"嵌入后的结果不存在...")
                return batch
        except Exception as e:
            self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
            return batch

        # 3. 循环处理所有chunk的向量以及注入到每一个chunk中
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            ind_ptr = csr_array.indptr
            start_ind_ptr = csr_array.indptr[index]
            end_ind_ptr = csr_array.indptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))

            # 3.3 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch



# ================================================================== #
#                        测试代码                                     #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()

    base_temp_dir = Path(r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir")

    input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")