import os.path, logging
import shutil
from datetime import datetime
from typing import Tuple
import uuid
from fastapi import UploadFile
from minio import Minio

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import kb_import_graph_app
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, update_task_status

logger = logging.getLogger(__name__)

class ImportFileService:

    # 获取带日期的基础文件目录
    def _get_date_dir(self):
        return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))

    def process_upload_file(self, file: UploadFile) -> Tuple[str, str, str]:
        """
        处理上传文件：
        1. 生成 task_id，构建归档目录
        2. 标记 upload_file 节点为运行中
        3. 保存文件到本地磁盘
        4. 同步上传到 MinIO
        5. 标记 upload_file 节点完成
        6. 返回 (task_id, file_dir, import_file_path)
        """
        date_dir = self._get_date_dir()
        task_id = str(uuid.uuid4().hex[:8])
        file_dir = os.path.join(date_dir,task_id)

        # 更新task状态为进行
        add_running_task(task_id, "upload_file")

        # 将文件上传到指定目录并且返回文件绝对地址
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # 将文件信息上传到minion上保存
        self._save_upload_file_to_minio(import_file_path, file.filename)

        # 更新task状态为完成
        add_done_task(task_id, "upload_file")

        return task_id, file_dir, import_file_path

    # 将文件上传到指定目录并且返回文件绝对地址
    def _save_upload_file_to_local(self, file, file_dir):
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)

        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        return import_file_path

    # 将文件信息上传到minion上保存
    def _save_upload_file_to_minio(self, import_file_path, filename):
        # 获取minion对象
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败 原因:{str(e)}")
            return

        # 获取minion相关参数
        bucket_name = os.getenv("MINIO_BUCKET_NAME","")
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 开始上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败 原因：{str(e)}")

    # 开始主流程
    def run_import_graph(self, task_id, file_dir, import_file_path):
        # 开始运行主langgraph程序
        try:
            # 更新task状态
            update_task_status(task_id, "processing")

            # 构建初始化state信息
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 开始执行langgraph程序并且更新到达每个node时的task状态
            for event in kb_import_graph_app.stream(global_graph_init_status):
                for node_name, state in event.items():
                    print(f"[{task_id}] Completed Node: {node_name}")

            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")
            print(f"[{task_id}] Error: {e}")
