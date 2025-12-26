#!/usr/bin/env python3
"""
子进程工作脚本，用于执行知识库重建和KG构建任务。
避免在主进程中pickle函数导致的序列化问题。
"""

import json
import sys
from pathlib import Path

def main():
    """主函数，执行重建和KG构建任务"""
    if len(sys.argv) != 3:
        print("用法: python subprocess_worker.py <raw_paragraphs.json> <triple_list_data.json>")
        sys.exit(1)

    raw_file = sys.argv[1]
    triple_file = sys.argv[2]

    try:
        # 读取输入数据
        with open(raw_file, 'r', encoding='utf-8') as f:
            raw_paragraphs = json.load(f)

        with open(triple_file, 'r', encoding='utf-8') as f:
            triple_list_data = json.load(f)

        print(f"开始处理 {len(raw_paragraphs)} 个段落和 {len(triple_list_data)} 个三元组")

        # 执行重建和构建任务
        from src.chat.knowledge.embedding_store import EmbeddingManager
        from src.chat.knowledge.kg_manager import KGManager

        embed_manager = EmbeddingManager()
        try:
            embed_manager.load_from_file()
        except Exception as e:
            print(f"加载EmbeddingManager失败，使用新实例: {e}")

        kg_manager = KGManager()
        try:
            kg_manager.load_from_file()
        except Exception as e:
            print(f"加载KGManager失败，使用新实例: {e}")

        # 存储并重建索引
        print("执行索引重建...")
        embed_manager.store_new_data_set(raw_paragraphs, triple_list_data)
        embed_manager.rebuild_faiss_index()
        embed_manager.save_to_file()

        # 构建 KG 并保存
        print("执行KG构建...")
        kg_manager.build_kg(triple_list_data, embed_manager)
        kg_manager.save_to_file()

        # 清理资源
        try:
            if hasattr(embed_manager, "close"):
                embed_manager.close()
            if hasattr(kg_manager, "close"):
                kg_manager.close()
        except Exception as e:
            print(f"清理资源时出错: {e}")

        print("子进程任务执行成功")
        sys.exit(0)

    except Exception as e:
        import traceback
        print(f"子进程任务执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()