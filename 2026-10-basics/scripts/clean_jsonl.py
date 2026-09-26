"""JSONL 清洗脚本：去空行、按整行去重、长度统计，输出 clean.jsonl。

用法:
    python scripts/clean_jsonl.py <输入.jsonl> <输出.jsonl>
"""
import json
import sys
from pathlib import Path


def clean(in_path: Path, out_path: Path) -> None:
    seen = set()          # 已见过的整行文本（去重依据）
    stats = {"总行数": 0, "空行": 0, "重复": 0, "非法JSON": 0, "有效": 0}
    length_buckets = {"<100字符": 0, "100—500字符": 0, ">500字符": 0}

    with in_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats["总行数"] += 1
            text = line.strip()
            if not text:                      # ① 去空行
                stats["空行"] += 1
                continue
            if text in seen:                  # ② 去重（按整行内容）
                stats["重复"] += 1
                continue
            try:
                json.loads(text)              # ③ 校验 JSON 合法
            except json.JSONDecodeError:
                stats["非法JSON"] += 1
                continue
            seen.add(text)
            stats["有效"] += 1
            n = len(text)
            if n < 100:
                length_buckets["<100字符"] += 1
            elif n <= 500:
                length_buckets["100—500字符"] += 1
            else:
                length_buckets[">500字符"] += 1
            fout.write(text + "\n")           # ④ 输出 clean.jsonl

    print("统计结果:", json.dumps(stats, ensure_ascii=False))
    print("长度分布:", json.dumps(length_buckets, ensure_ascii=False))
    print(f"输出: {out_path}（有效 {stats['有效']} 条）")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("用法: python scripts/clean_jsonl.py <输入.jsonl> <输出.jsonl>")
    clean(Path(sys.argv[1]), Path(sys.argv[2]))
