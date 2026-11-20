from pathlib import Path
import json
from mcp.server.fastmcp.exceptions import ValidationError

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 获取文件或者目录信息
        self.log_step("step1","获取文件或目录")
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        self.log_step("step2","判断上传文件是否存在")
        # 判断上传文件是否为空
        if not import_file_path:
            raise ValidationError(f"import_file_path的值为空",self.name)

        # 获取文件后缀名并且变成小写
        path = Path(import_file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在：{import_file_path}",self.name)
        suffix = path.suffix.lower()

        # 不同文件类型做不同修改
        self.log_step("step3","判断文件类型")
        if suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        else:
            raise ValidationError(f"不支持的文件类型：{suffix}",self.name)

        # 获取文件标题
        file_name = path.stem

        # 更新状态并返回
        state["file_title"] = file_name

        return state


if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\atguigu\项目\2.资料\pdf文档\doc\万用表RS-12的使用.pdf"
    file_dir = r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir"
    state = {
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    entry = EntryNode(state)
    processed_state = entry(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))