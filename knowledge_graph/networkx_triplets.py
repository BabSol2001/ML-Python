import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ۱. تعریف کلاس سه‌تایی (Triplet)
class Triplet:
    def __init__(self, head: str, relation: str, tail: str):
        self.head = head
        self.relation = relation
        self.tail = tail

    def __repr__(self):
        return f"({self.head}) --[{self.relation}]--> ({self.tail})"

# ۲. تعریف کلاس اصلی گراف دانش (Knowledge Graph Engine)
class KnowledgeGraph:
    def __init__(self):
        self.entities: set = set()
        self.relations: set = set()
        self.triplets: List[Triplet] = []
        self.graph = nx.DiGraph() # گراف جهت‌دار NetworkX

    def add_triplet(self, head: str, relation: str, tail: str):
        """افزودن یک حقیقت جدید به گراف دانش"""
        self.entities.add(head)
        self.entities.add(tail)
        self.relations.add(relation)
        
        triplet = Triplet(head, relation, tail)
        self.triplets.append(triplet)
        
        # اضافه کردن به گراف جهت‌دار همراه با برچسب رابطه
        self.graph.add_edge(head, tail, relation=relation)

    def get_neighbors(self, entity: str) -> List[Tuple[str, str]]:
        """استخراج تمام همسایه‌ها و روابط یک موجودیت"""
        if entity not in self.entities:
            return []
        
        neighbors = []
        # روابط خروجی
        for neighbor in self.graph.successors(entity):
            rel = self.graph[entity][neighbor]['relation']
            neighbors.append((f"OUT: --[{rel}]-->", neighbor))
        # روابط ورودی
        for neighbor in self.graph.predecessors(entity):
            rel = self.graph[neighbor][entity]['relation']
            neighbors.append((f"IN: <--[{rel}]--", neighbor))
            
        return neighbors

        
    def visualize(self, filename: str = "smartbiz_kg.png"):
        """رسم گراف دانش، ذخیره به‌صورت PNG و نمایش آن"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=0.8)
        
        # رسم گره‌ها و یال‌ها
        nx.draw_networkx_nodes(self.graph, pos, node_size=2500, node_color='lightblue')
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
        
        # برچسب روی یال‌ها (روابط)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("SmartBiz-KG: Knowledge Graph MVP", fontsize=14)
        plt.axis('off')
        
        # ۱. ذخیره فایل تصویر با کیفیت بالا (DPI 300) قبل از نمایش
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n[+] تصویر گراف با موفقیت در فایل '{filename}' ذخیره شد.")
        
        # ۲. نمایش پنجره
        plt.show()
    
# ==========================================
# ۳. بارگذاری داده‌های اولیه پروژه (تولید + کسب‌وکار)
# ==========================================
kg = KnowledgeGraph()

# داده‌های بخش مهندسی/تولید
kg.add_triplet("High_Infeed_Rate", "Increases", "Grinding_Temp")
kg.add_triplet("Grinding_Temp", "Causes", "Thermal_Burn")
kg.add_triplet("Thermal_Burn", "Reduces", "Fatigue_Life")
kg.add_triplet("Ti-6Al-4V", "Used_In", "Turbine_Blade")
kg.add_triplet("Turbine_Blade", "Requires_Process", "High_Precision_Grinding")

# داده‌های بخش کسب‌وکار و زنجیره تامین
kg.add_triplet("Supplier_Alpha", "Supplies", "Ti-6Al-4V")
kg.add_triplet("Customer_Boeing", "Orders", "Turbine_Blade")
kg.add_triplet("Supplier_Alpha", "Located_In", "Region_Risk_Zone")

# تست استخراج همسایه‌ها برای یک موجودیت
print("--- همسایه‌ها و روابط موجودیت Turbine_Blade ---")
for rel, entity in kg.get_neighbors("Turbine_Blade"):
    print(f"Turbine_Blade {rel} {entity}")

print(f"\nتعداد کل موجودیت‌ها (Entities): {len(kg.entities)}")
print(f"تعداد کل روابط (Relations): {len(kg.relations)}")
print(f"تعداد کل گزاره‌ها (Triplets): {len(kg.triplets)}")

# تست بصری (باز شدن پنجره گراف)
kg.visualize()