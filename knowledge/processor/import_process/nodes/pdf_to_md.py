import subprocess
import json
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 输入文件校验
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)

        # 使用mineru将pdf转成md
        process_code = self.execute_mineru(import_file_path_obj, file_dir_path_obj)
        if process_code == 0:
            self.logger.info("PDF转换成功")
        else:
            self.logger.error("PDF转换失败")
            raise PdfConversionError("PDF转换失败", self.name)

        # 获取md文件路径
        md_path = self._get_md_paths(import_file_path_obj, file_dir_path_obj)

        # 修改状态返回
        state['md_path'] = md_path
        return state

    # 文件校验工具
    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:
            raise ValidationError("输入的文件路径为空", self.name)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise ValidationError("输入的文件路径不存在", self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_path_obj = Path(file_dir)

        return import_file_path_obj, file_dir_path_obj


    # 使用mineru将pdf转成md
    def execute_mineru(self, import_file_path_obj:Path, file_dir_path_obj:Path) -> int:
        proc = subprocess.Popen(
            args=["mineru", "-p", import_file_path_obj, "-o", file_dir_path_obj, "--source", "local", "--backend", "pipeline"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=-1,
        )

        for line in proc.stdout:
            print(line.rstrip())

        process_code = proc.wait()

        return process_code

    # 获取md文件路径
    def _get_md_paths(self, import_file_path_obj:Path, file_dir_path_obj:Path) -> str:
        file_name = import_file_path_obj.stem
        md_path = file_dir_path_obj/file_name/"auto"/f"{file_name}.md"

        return str(md_path)


if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\atguigu\项目\2.资料\pdf文档\doc\万用表RS-12的使用.pdf"
    file_dir = r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir"
    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    pdf_to_md_node = PdfToMdNode()
    processed_state = pdf_to_md_node(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))