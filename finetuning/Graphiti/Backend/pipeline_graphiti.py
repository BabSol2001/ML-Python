import json
import time
from typing import List, Dict, Any
import torch

# ---------------------------------------------------------
# ۱. پیاده‌سازی خط‌لوله حافظه پویای Graphiti
# ---------------------------------------------------------
class TemporalEdge:
    def __init__(self, source: str, relation: str, target: str, created_at: float):
        self.source = source
        self.relation = relation
        self.target = target
        self.created_at = created_at
        self.invalidated_at: float | None = None  # زمان منقضی شدن فکت

class GraphitiMemoryPipeline:
    def __init__(self):
        self.edges: List[TemporalEdge] = []

    def add_episode(self, source: str, relation: str, target: str):
        """افزودن اپیزود جدید و منقضی کردن فکت‌های متناقض قبلی"""
        now = time.time()

        # بررسی تناقض با فکت‌های قبلی (Edge Invalidation)
        for edge in self.edges:
            if edge.source == source and edge.relation == relation and edge.invalidated_at is None:
                # منقضی کردن فکت قدیمی
                edge.invalidated_at = now
                print(f"🔄 فکت قدیمی منقضی شد: {edge.source} {edge.relation} {edge.target}")

        # ثبت فکت جدید
        new_edge = TemporalEdge(source, relation, target, created_at=now)
        self.edges.append(new_edge)

    def save_memory(self):
        data = [
            {
                "source": e.source,
                "relation": e.relation,
                "target": e.target,
                "created_at": e.created_at,
                "invalidated_at": e.invalidated_at
            }
            for e in self.edges
        ]
        with open("graphiti_memory.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("✅ حافظه زمانی Graphiti ذخیره شد.")

if __name__ == "__main__":
    memory = GraphitiMemoryPipeline()
    # روز اول
    memory.add_episode("User", "drinks", "Coffee")
    time.sleep(1)
    # روز دوم (تناقض و به‌روزرسانی حافظه)
    memory.add_episode("User", "drinks", "Green Tea")
    memory.save_memory()