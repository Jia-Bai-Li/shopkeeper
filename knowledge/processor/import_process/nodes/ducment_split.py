import json
import os
import re
from typing import Tuple, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1，获取参数
        md_content,file_title,max_content_length,min_content_length = self._get_inputs(state)

        # 2.根据标题切割
        sections = self._split_by_headings(md_content,file_title)

        # 3.长切短合
        final_chunks = self.split_and_merge(sections,max_content_length,min_content_length)

        # 组装
        chunks = self._assemble_chunk(final_chunks)

        # 更新state
        state["chunks"] = chunks

        # 日志
        self._log_summary(md_content, chunks, max_content_length)

        # 备份
        self._back_chunks(state, chunks)

        # 返回
        return state

    # 获取参数
    def _get_inputs(self, state) -> Tuple[str, str, int, int]:
        self.log_step("step1", "切分文档的参数校验以及获取...")

        config = get_config()

        md_content = state.get("md_content")

        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        file_title = state.get("file_title")

        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content,file_title,config.max_content_length,config.min_content_length

    # 根据标题切割
    def _split_by_headings(self, md_content, file_title) -> List[dict]:
        self.log_step("step2", "根据标题进行切分...")

        # 1. 定义变量
        in_fence = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ""
        hierarchy = [""]*7

        # 2. 定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 切分
        content_lines = md_content.split("\n")

        # 用于封装section对象
        def _flush():
            """封装 section 对象"""
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1,0,-1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                return sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })

        for content_line in content_lines:
            # 判断是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None

            if match:
                _flush()

                level = len(match.group(1))
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title

                # 清空下级标题
                for i in range(level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []

            else:
                # 除了标题行以外全都收集起来
                body_lines.append(content_line)

        _flush()

        return sections

    # 长切短合
    def split_and_merge(self, sections, max_content_length, min_content_length):
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_section = self.merge_short_section(current_sections, min_content_length)

        # 3. 返回
        return final_section

    # 切分所有section中超出字数的部分
    def split_long_section(self, section, max_content_length):
        self.log_step("step3", "进行长内容的切分")

        # 1. 获取 section 对象属性
        title = section.get("title")
        body = section.get("body")
        file_title = section.get("file_title")
        parent_title = section.get("parent_title")

        # 2. 判断表格
        if "<table>" in body:
            self.logger.info("检测到了表格数据...")
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 3. 对标题做校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:50]

        # 4. 拼接 title 前缀
        title_prefix = f"{title}\n\n"

        # 5. 计算总长度
        total_length = len(title_prefix) + len(body)

        # 6. 判断是否需要切分
        if total_length <= max_content_length:
            return [section]

        # 7. 计算 body 可用的长度
        body_length = max_content_length - len(title_prefix)

        if body_length <= 0:
            return [section]

        # 8. 使用 RecursiveCharacterTextSplitter 切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=False
        )
        texts = text_splitter.split_text(body)

        # 8.3 判断切分结果
        if len(texts) <= 1:
            return [section]

        sub_sections = []
        for index, text in enumerate(texts):
            sub_sections.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"  # 标记序号
            })

        return sub_sections

    # 切分完后过小的section合并在一起
    def merge_short_section(self, current_sections, min_content_length):
        # 1. 定义变量
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历以及合并
        for next_section in current_sections[1:]:
            # 同源检查
            same_parent = (current_section["parent_title"] == next_section["parent_title"])

            if same_parent and len(current_section.get("body")) < min_content_length:
                # body的合并
                current_section['body'] = (current_section['body'].rstrip() + "\n\n" + next_section['body'].lstrip())

                # 更新 current_title
                current_section['title'] = next_section['parent_title']
                current_section['part'] = 0
            else:
                # 将原来 current_section 进行封箱
                final_sections.append(current_section)
                # 更新 next_section
                current_section = next_section

        # 最后一个（封装起来）
        final_sections.append(current_section)

        # 4. 对所有 section 的 part 做处理
        part_counter = {}
        result = []

        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get("parent_title")
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section["part"] = new_part
                final_section["title"] = final_section["title"] + f"{new_part}"

            result.append(final_section)

        return result

    # 组装
    def _assemble_chunk(self, final_chunks):
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            # 1. 获取 chunk 的信息
            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3. 判断 part 是否存在
            if "part" in chunk:
                assemble_chunk["part"] = chunk.get("part")

            chunks.append(assemble_chunk)

        return chunks

    # 打印日志
    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    # 备份chunks
    def _back_chunks(self, state, chunks):
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir","")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir,exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"E:\\atguigu\\项目\\2.资料\\pdf文档\\doc\\temp_dir\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir"
    }
    document_node.process(state)


