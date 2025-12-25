import json
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

def main():
    if len(sys.argv) < 3:
        print("使用方式：subprocess_worker.py <raw_paragraphs.json> <triple_list_data.json>", file=sys.stderr)
        return 2
    raw_path = sys.argv[1]
    triple_path = sys.argv[2]
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_paragraphs = json.load(f)
        with open(triple_path, "r", encoding="utf-8") as f:
            triple_list_data = json.load(f)
    except Exception as exc:
        print(f"未能读取输入文件{exc}", file=sys.stderr)
        return 3

    try:
        from src.chat.knowledge.embedding_store import EmbeddingManager
        from src.chat.knowledge.kg_manager import KGManager
    except Exception as exc:
        print(f"未能导入lpmm模块 {exc}", file=sys.stderr)
        raise

    try:
        embed_manager = EmbeddingManager()
        try:
            embed_manager.load_from_file()
        except Exception:
            pass

        kg_manager = KGManager()
        try:
            kg_manager.load_from_file()
        except Exception:
            pass

        embed_manager.store_new_data_set(raw_paragraphs, triple_list_data)
        embed_manager.rebuild_faiss_index()
        embed_manager.save_to_file()

        kg_manager.build_kg(triple_list_data, embed_manager)
        kg_manager.save_to_file()

        return 0
    except Exception as exc:
        import traceback
        print(f"执行失败: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 4

if __name__ == "__main__":
    sys.exit(main())


