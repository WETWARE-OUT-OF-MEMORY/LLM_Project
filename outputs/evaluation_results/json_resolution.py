import json
import json

if __name__ == '__main__':
    json_path = 'rag_vs_non_rag_20260216_102458.json'

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for qa in data:
        print("【问题】", qa["question"])
        print("【RAG回答】")
        print(qa["rag_answer"])
        print("【非RAG回答】")
        print(qa["non_rag_answer"])