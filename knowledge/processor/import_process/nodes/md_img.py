import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from threading import Lock
from typing import Tuple, List, Dict, Deque
import json

from minio import Minio

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ValidationError, \
    ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str      # 最近的章节标题
    pre_text: str     # 图片上方的正文内容
    post_text: str    # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名，如 "abc123.jpg"
    path: str                # 图片完整路径
    context: ImageContext    # 在 MD 中的上下文



# 文件读取、路径校验、备份
class MdFileHandler:

    def __init__(self,logger: Logger,node_name: str):
        self.logger = logger
        self.node_name = node_name

    # 读取md内容
    def read_md(self,state: ImportGraphState) -> Tuple[str,Path,Path]:
        md_path = state["md_path"]
        if not md_path:
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md_path文件不存在",self.node_name)

        with open(md_path_obj,'r',encoding="utf-8") as f:
            md_content = f.read()

        images_dir = md_path_obj.parent / "images"

        return md_content,md_path_obj,images_dir

    # 替换原本的md内容
    def backup(self, md_path_obj, new_md_content):
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")

        try:
            with open(new_file_path,'w',encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except Exception as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )


# 扫描图片目录、提取上下文
class ImageScanner:

    def __init__(self,logger: Logger,node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content, images_dir, image_extensions, context_length):
        """
        扫描MD文档，找到每一个图片的上下文字的内容并且封装放到ImageInfo类里
        :param md_content: 完整MD文档信息
        :param images_dir: 图片所在路径
        :param image_extensions: 图片类型
        :param context_length: 上下文长度
        :return:
        """

        # 定义返回的数据结构
        image_info_list: List[ImageInfo] = []

        for image_path in images_dir.iterdir():
            # 如果不是文件跳过
            if not image_path.is_file():
                continue

            # 如果不满足后缀跳过
            if not image_path.suffix in image_extensions:
                continue

            # 查找上下文，如果是表格则没有上下文返回None
            ctx = self._find_context(image_path.name,md_content,context_length)
            if ctx is None:
                continue

            image_info_list.append(ImageInfo(name=image_path.name, path=image_path, context=ctx))

        return image_info_list

    # 查找上下文工具
    def _find_context(self, img_name, md_content, max_chars) -> ImageContext | None:
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        md_lines = md_content.split("\n")

        for line_idx, line in enumerate(md_lines):
            if not pattern.match(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above(md_lines,line_idx)
            prev_content = md_lines[prev_boundary + 1: line_idx]
            img_pre = self._extract_limited_context(prev_content,max_chars, direction="front")

            # 向下：找下一个标题，取图片到标题之间的内容作为下文
            next_boundary = self._find_heading_below(md_lines,line_idx)
            post_content = md_lines[line_idx + 1 : next_boundary]
            img_post = self._extract_limited_context(post_content,max_chars, direction="end")

            return ImageContext(heading=prev_title,pre_text=img_pre,post_text=img_post)

        return None

    @staticmethod
    def _find_heading_above(md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i

        return "", -1


    @staticmethod
    def _extract_limited_context(lines: List[str], max_chars: int, direction: str) -> str:
        """按段落分割，按 direction 方向贪心装填，保持段落完整性。"""
        current_paragraph: List[str] = []
        paragraphs: List[str] = []

        for line in lines:
            #line.strip(): 去除字符串首尾的空白字符(空格、制表符、换行符等)
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()#就近原则

        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if (total + len(para) > max_chars) and selected:#至少有个段落
                break
            selected.append(para)
            total += len(para)

        if direction == "front":
            selected.reverse()#与原文顺序一致，利于VLM

        return "\n\n".join(selected)#折行并空一行

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        for i in range(from_idx + 1,len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i

        return len(md_lines)


# VLM 摘要生成、速率限制
class VLMSummarizer:

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name
        self.request_timestamps: Deque[float] = deque()
        self.lock: Lock = Lock()

    def summarize_all(self, document_title, image_list, vl_model, requests_per_minute) -> Dict[str, str]:
        self.logger.info("【step_3】提取图片摘要")

        summaries: Dict[str, str] = {}

        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(
                f"VLM 不可用，跳过图片摘要生成: {e}"
            )
            for img in image_list:
                summaries[img.name] = "图片描述"
            return summaries

        for img in image_list:
            # 滑动窗口控制发送速率
            self._enforce_rate_limit(self.request_timestamps,requests_per_minute)

            # 使用vlm解释出一张图片的内容描述
            summaries[img.name] = self.summarize_one(client,vl_model,document_title,img)

        self.logger.info(f"生成 {len(summaries)} 张图片摘要")

        return summaries

    # 滑动窗口控制发送速率
    def _enforce_rate_limit(self, request_timestamps, requests_per_minute, window: int = 60):
        with self.lock:
            now = time.time()
            while request_timestamps and now - request_timestamps[0] >= window:
                request_timestamps.popleft()

            if len(request_timestamps) >= requests_per_minute:
                # 计算出还需要多久第一个进入队列的请求才会被移除
                sleep_dur = window - (now - request_timestamps[0])
                if sleep_dur > 0:
                    self.logger.info(
                        f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                    )
                    time.sleep(sleep_dur)
                now = time.time()
                while request_timestamps and now - request_timestamps[0] >= window:
                    request_timestamps.popleft()

            request_timestamps.append(now)

    # 使用vlm解释出一张图片的内容描述
    def summarize_one(self, client, vl_model, document_title, img) -> str:
        parts = [p for p in (img.context.heading,img.context.pre_text,img.context.post_text) if p]
        final_context = "\n".join(parts) if parts else "暂无可用上下文"

        try:
            with open(img.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.path}: {e}")
            return "图片描述"


# MinIO 上传、MD 内容替换
class ImageUploader:

    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def upload_and_replace(self, document_name, md_content, imageinfo_list, summaries, minio_bucket,
                           minio_endpoint) -> str:

        remote_urls:Dict[str, str] = self._upload_all(imageinfo_list,document_name,minio_bucket,minio_endpoint)

        new_md_content = self._replace_in_md(remote_urls=remote_urls,md_content=md_content,summaries=summaries)
        return new_md_content

    # 上传图片到minio
    def _upload_all(self, imageinfo_list, document_name, minio_bucket, minio_endpoint) -> Dict[str, str]:
        remote_urls:Dict[str, str] = {}

        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for image_info in imageinfo_list:
                remote_urls[image_info.name] = image_info.path
            return remote_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(bucket_name=minio_bucket,
                                         object_name=f"{document_name}/{image_info.name}",
                                         file_path=image_info.path)
                remote_url = f"{minio_endpoint}/{minio_bucket}/{document_name}/{image_info.name}"
                remote_urls[image_info.name] = remote_url
            except Exception as e:
                remote_urls[image_info.name] = image_info.path

        return remote_urls

    # 替换md文件中images描述信息
    def _replace_in_md(self, remote_urls, md_content, summaries) -> str:
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(original_path).name
            for img_name, summary in summaries.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class MarkDownImageNode(BaseNode):
    name = "md_img_node"

    def __init__(self):
        super().__init__()
        self.file_handler = MdFileHandler(self.logger,self.name)
        self.scanner = ImageScanner(self.logger,self.name)
        self.summarizer = VLMSummarizer(self.logger,self.name)
        self.uploader = ImageUploader(self.logger,self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 读取文件返回md内容，md地址和md图片地址
        md_content,md_path_obj,images_dir = self.file_handler.read_md(state)

        state["md_content"] = md_content

        if not images_dir.exists():
            self.logger.warning(
                f"文件 {md_path_obj.name} 暂无图片要处理"
            )
            # state["md_content"] = md_content
            return state

        # 扫描图片，提取上下文
        imageinfo_list:List[ImageInfo] = self.scanner.scan_img_dir(md_content,images_dir,image_extensions=self.config.image_extensions,context_length=self.config.img_content_length,)

        # VLM生成摘要
        summaries = self.summarizer.summarize_all(
            document_title = md_path_obj.stem,
            image_list = imageinfo_list,
            vl_model = self.config.vl_model,
            requests_per_minute = self.config.requests_per_minute
        )

        # 上传，替换
        new_md_content:str = self.uploader.upload_and_replace(document_name=md_path_obj.stem,
                                md_content=md_content,
                                imageinfo_list=imageinfo_list,
                                summaries=summaries,
                                minio_bucket=self.config.minio_bucket,
                                minio_endpoint=self.config.get_minio_base_url(),)

        # 备份
        self.file_handler.backup(md_path_obj,new_md_content)

        # 更新返回状态
        return state




if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\atguigu\项目\2.资料\pdf文档\doc\万用表RS-12的使用.pdf"
    file_dir = r"E:\atguigu\项目\2.资料\pdf文档\doc\temp_dir"
    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir,
        "md_path": 'E:\\atguigu\\项目\\2.资料\\pdf文档\\doc\\temp_dir\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md'
    }

    md = MarkDownImageNode()
    processed_state = md(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))