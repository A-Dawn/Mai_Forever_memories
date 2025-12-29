#!/usr/bin/env python3
"""
子进程工作脚本 - 执行知识库重建任务
独立于插件主模块，避免Pickling问题
"""

import json
import sys
import os
from pathlib import Path

# 修复Windows编码问题 - 设置为UTF-8
if os.name == 'nt':  # Windows
    # 设置控制台编码为UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # 重新设置sys.stdout和sys.stderr的编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def rebuild_and_build_in_subprocess(raw_paragraphs: dict, triple_list_data: dict) -> bool:
    """子进程中执行的重建和KG构建任务"""
    try:
        # 修复导入路径问题 - 添加项目根目录到Python路径
        script_dir = Path(__file__).parent.resolve()  # plugins/mai_forever_memories
        project_root = script_dir.parent.parent.resolve()  # MaiBot根目录
        src_path = (project_root / "src").resolve()

        # 使用绝对路径添加到sys.path
        src_path_str = str(src_path)
        project_root_str = str(project_root)

        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        # 导入必要的模块
        from chat.knowledge.embedding_store import EmbeddingManager
        from chat.knowledge.kg_manager import KGManager

        embed_manager = EmbeddingManager()
        try:
            embed_manager.load_from_file()
        except Exception:
            # 允许首次导入时不存在文件
            pass

        kg_manager = KGManager()
        try:
            kg_manager.load_from_file()
        except Exception:
            pass

        # 存储并重建索引
        embed_manager.store_new_data_set(raw_paragraphs, triple_list_data)
        embed_manager.rebuild_faiss_index()
        embed_manager.save_to_file()

        # 构建 KG 并保存
        kg_manager.build_kg(triple_list_data, embed_manager)
        kg_manager.save_to_file()

        return True
    except Exception as exc:
        print(f"子进程执行失败: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 3:
        print("用法: python subprocess_worker.py <raw_file> <triple_file>", file=sys.stderr)
        sys.exit(1)

    raw_file = sys.argv[1]
    triple_file = sys.argv[2]

    # 读取输入文件
    try:
        with open(raw_file, 'r', encoding='utf-8') as f:
            raw_paragraphs = json.load(f)
        with open(triple_file, 'r', encoding='utf-8') as f:
            triple_list_data = json.load(f)
    except Exception as e:
        print(f"读取输入文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 执行任务
    success = rebuild_and_build_in_subprocess(raw_paragraphs, triple_list_data)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()