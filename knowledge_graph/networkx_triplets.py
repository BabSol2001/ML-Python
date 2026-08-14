import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import List, Tuple, Dict, Set, Optional

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

    # ==========================================
    # موتور استنتاج و مسیریابی
    # ==========================================
    def find_paths(self, source: str, target: str) -> List[List[str]]:
        """یافتن تمام مسیرهای ساده ممکن بین دو موجودیت"""
        if source not in self.entities or target not in self.entities:
            return []
        return list(nx.all_simple_paths(self.graph, source=source, target=target))

    def analyze_impact(self, source_entity: str) -> Set[str]:
        """تحلیل اثرپذیری: چه موجودیت‌هایی تحت تاثیر این گره قرار دارند؟"""
        if source_entity not in self.entities:
            return set()
        return nx.descendants(self.graph, source_entity)

    def get_subgraph_triplets(self, entities_subset: Set[str]) -> List[Triplet]:
        """استخراج سه‌تایی‌های مربوط به یک زیرگراف خاص (برای استفاده در Graph-RAG)"""
        sub_triplets = []
        for t in self.triplets:
            if t.head in entities_subset or t.tail in entities_subset:
                sub_triplets.append(t)
        return sub_triplets

    # ==========================================
    # رسم تعاملی و داشبورد HTML
    # ==========================================
    def visualize_interactive(self, filename: str = "smartbiz_kg_interactive.html", highlight_risk_from: Optional[str] = None):
        """رسم تعاملی گراف در قالب فایل وب HTML با قابلیت کشیدن، زوم و هایلایت بحران"""
        net = Network(height="750px", width="100%", notebook=False, directed=True)
        
        # محاسبه گره‌های تحت ریسک در صورت فعال بودن هایلایت
        affected_nodes = set()
        if highlight_risk_from and highlight_risk_from in self.entities:
            affected_nodes = self.analyze_impact(highlight_risk_from)

        # افزودن گره‌ها با رنگ‌بندی هوشمند
        for node in self.graph.nodes():
            if node == highlight_risk_from:
                color = "#ff4d4d"  # قرمز: منبع اصلی ریسک/بحران
                title = f"<b>CRITICAL RISK SOURCE</b><br>{node}"
            elif node in affected_nodes:
                color = "#ffa64d"  # نارنجی: گره‌های متأثر از بحران
                title = f"<b>AFFECTED NODE</b><br>{node}"
            else:
                color = "#97c2fc"  # آبی: حالت عادی
                title = f"Node: {node}"
                
            net.add_node(node, label=node, title=title, color=color, shape="ellipse")

        # افزودن یال‌ها و برچسب روابط
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(u, v, title=data.get('relation', ''), label=data.get('relation', ''), color="gray")

        # تنظیمات فیزیک برای حرکت نرم گره‌ها
        net.toggle_physics(True)
        net.write_html(filename)
        print(f"\n[+] داشبورد تعاملی با موفقیت در فایل '{filename}' ذخیره شد.")

    def visualize(self, filename: str = "smartbiz_kg.png"):
        """رسم گراف دانش، ذخیره به‌صورت PNG و نمایش آن"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=0.8)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=2500, node_color='lightblue')
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("SmartBiz-KG: Knowledge Graph MVP", fontsize=14)
        plt.axis('off')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n[+] تصویر گراف با موفقیت در فایل '{filename}' ذخیره شد.")
        plt.show()

# ==========================================
# ۳. عامل Graph-RAG برای تولید گزارش متنی
# ==========================================
class GraphRAGAgent:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def query_impact_report(self, risk_entity: str) -> str:
        """تولید گزارش تحلیلی اثرات بحران بر اساس بازیابی گراف دانش (Graph-RAG)"""
        if risk_entity not in self.kg.entities:
            return f"خطا: موجودیت '{risk_entity}' در گراف دانش یافت نشد."

        affected_nodes = self.kg.analyze_impact(risk_entity)
        related_entities = affected_nodes.union({risk_entity})
        sub_triplets = self.kg.get_subgraph_triplets(related_entities)

        # ساخت کانتکست ساختاریافته (Graph Context Retrieval)
        context_str = "\n".join([f"- {t.head} -> [{t.relation}] -> {t.tail}" for t in sub_triplets])

        # قالب‌بندی گزارش نهایی
        report = f"""
==================================================
📊 گزارش هوشمند Graph-RAG: تحلیل ریسک و اثرات بحران
==================================================
🎯 منبع ریسک بررسی‌شده: {risk_entity}
📉 تعداد موجودیت‌های متأثر مستقیم/غیرمستقیم: {len(affected_nodes)}

🔍 حقایق استخراج‌شده از گراف (Knowledge Graph Context):
{context_str}

💡 تحلیل مدیریتی و زنجیره اثرات:
بر اساس گراف دانش، هرگونه بروز ریسک یا اختلال در '{risk_entity}' به‌صورت زنجیره‌ای موارد زیر را تحت تاثیر قرار می‌دهد:
"""
        for node in affected_nodes:
            paths = self.kg.find_paths(risk_entity, node)
            if paths:
                path_str = " -> ".join(paths[0])
                report += f"  • آسیب به [{node}] از طریق مسیر: ({path_str})\n"

        return report

# ==========================================
# ۴. بارگذاری داده‌های اولیه پروژه (تولید + کسب‌وکار)
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

# اجرا و تولید داشبورد تعاملی با هایلایت کردن ریسک Supplier_Alpha
kg.visualize_interactive(filename="smartbiz_kg_interactive.html", highlight_risk_from="Supplier_Alpha")

# تست استخراج همسایه‌ها برای یک موجودیت
print("--- همسایه‌ها و روابط موجودیت Turbine_Blade ---")
paths = kg.find_paths("Supplier_Alpha", "Ti-6Al-4V")
for p in paths:
    print(" -> ".join(p))

# --- تست ۲: تحلیل اثر بحران (Impact Analysis) ---
risk_node = "Supplier_Alpha"
print(f"\n--- تست استنتاج ۲: تحلیل اثر بحران در '{risk_node}' ---")
affected_entities = kg.analyze_impact(risk_node)
print(f"موجودیت‌هایی که تحت تأثیر بحران این گره قرار می‌گیرند: {affected_entities}")

# --- تست ۳: فراخوانی Graph-RAG Agent و چاپ گزارش متنی ---
rag_agent = GraphRAGAgent(kg)
report_output = rag_agent.query_impact_report(risk_node)
print(report_output)

# رسم گراف نهایی
kg.visualize()