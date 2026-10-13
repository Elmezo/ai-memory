"""Benchmark Evaluation for AI Memory Server.

Measures Recall@1, Recall@3, Recall@5, Recall@10 (R@10), and MRR
across a diverse evaluation suite of Arabic and English memory queries,
including paraphrased questions, distractors, preferences, and technical facts.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp_server import memory_save, memory_search
from service import MemoryService
from store import create_client

logging.basicConfig(level=logging.WARNING)

BENCHMARK_TENANT = "benchmark_eval_tenant"
BENCHMARK_USER = "benchmark_eval_user"

# Corpus of memories to inject (Ground Truth + Distractors)
MEMORIES_CORPUS = [
    # Category: Project & System Facts (English & Arabic)
    ("The active project name is ERP Memory Test A with microservices architecture", "fact", ["project", "name"]),
    ("The backend database runs on PostgreSQL 16 with TimescaleDB extension", "fact", ["database", "postgres"]),
    ("Production Kubernetes cluster namespace deployment requires architect approval", "procedure", ["k8s", "devops"]),
    ("Maximum allowable invoice discount for regular customers is 15%", "constraint", ["finance", "discount"]),
    ("Corporate API rate limit is set to 1000 requests per minute per IP", "constraint", ["api", "security"]),
    ("All microservices authenticate using mutual TLS (mTLS) with internal PKI", "fact", ["security", "mtls"]),
    ("Frontend web application is built with Next.js 15 and Tailwind CSS", "fact", ["frontend", "nextjs"]),
    ("Redis cluster port is 6379 with TLS enabled in production environment", "fact", ["redis", "cache"]),
    ("The daily backup window starts at 02:00 UTC and retention period is 30 days", "procedure", ["backup", "ops"]),
    ("Elasticsearch memory index uses BBQ HNSW quantization for dense vector indexing", "fact", ["elasticsearch", "vector"]),

    # Arabic Project & Business Facts
    ("اسم المشروع الحالي هو ERP Memory Test A ويعمل بتقنية السحابة الهجينة", "fact", ["project", "arabic"]),
    ("الحد الأقصى للائتمان المعتمد للعميل في النظام المالي هو 50,000 ريال سعودي", "constraint", ["مالية", "ائتمان"]),
    ("نظام الدفع يعتمد بوابة ميسر وتابي للمدفوعات الإلكترونية في الشرق الأوسط", "fact", ["دفع", "فواتير"]),
    ("تصدير تقارير المحاسبة الشهرية يتطلب موافقة المدير المالي كتابياً", "procedure", ["محاسبة", "إجراءات"]),
    ("ساعات الصيانة الدورية المجدولة تبدأ يوم الجمعة الساعة 11 مساءً بتوقيت مكة", "procedure", ["صيانة", "جدولة"]),
    ("خادم الذاكرة يدعم موديل bge-m3 بـ 1024 بعداً للبحث الدلالي متعدد اللغات", "fact", ["ذكاء_اصطناعي", "ذاكرة"]),

    # Category: User Preferences (English & Arabic)
    ("The user prefers code samples written in Python 3.11 with strict type annotations", "preference", ["code", "python"]),
    ("The user prefers dark mode theme with JetBrains Mono font size 14", "preference", ["theme", "ide"]),
    ("المستخدم يفضل دائماً قراءة الشروحات التقنية والردود باللغة العربية الفصحى", "preference", ["تفضيل", "لغة"]),
    ("المستخدم يفضل كتابة اختبارات pytest باستخدام fixtures بدلاً من unittest classes", "preference", ["python", "testing"]),
    ("المستخدم يطلب توثيق جميع الـ REST APIs بصيغة OpenAPI / Swagger 3.0", "preference", ["api", "docs"]),
    ("المستخدم يفضل معمارية Clean Architecture مع مجلدات منفصلة للـ domain و infra", "preference", ["architecture", "clean"]),

    # Category: Technical Decisions & Constraints
    ("Decision: Adopt Docker multi-stage builds to reduce image size below 150MB", "decision", ["docker", "build"]),
    ("Decision: Replace Celery with lightweight asyncio background tasks for memory queue", "decision", ["queue", "asyncio"]),
    ("Constraint: Do not expose Elasticsearch port 9200 to public internet without auth", "constraint", ["security", "es"]),
    ("قرار: تم اعتماد نموذج FastEmbed محلياً لتقليل استهلاك الذاكرة وتفادي PyTorch", "decision", ["أداء", "reranker"]),
    ("قرار: تم تفعيل دمج RRF للبحث الهجين لتحسين دقة استرجاع الذكريات", "decision", ["بحث", "rrf"]),

    # Distractors (Similar wording / topics but distinct facts)
    ("Archived project name was Alpha Legacy CRM completed last year", "fact", ["legacy", "project"]),
    ("The staging test database runs on SQLite for fast unit testing", "fact", ["database", "sqlite"]),
    ("Developer sandbox rate limit is set to 50 requests per minute", "constraint", ["api", "sandbox"]),
    ("Legacy customer credit limit was 10,000 dollars before policy update", "constraint", ["finance", "legacy"]),
    ("Previous frontend version used Angular 12 with Bootstrap 4", "fact", ["frontend", "angular"]),
    ("تم إيقاف مشروع المحاسبة القديم وإحالته للأرشفة في مطلع عام 2024", "fact", ["أرشيف", "قديم"]),
    ("المستخدم لا يفضل استخدام مكتبة moment.js في مشاريع جافاسكريبت", "preference", ["frontend", "js"]),
]

# Evaluation Queries Suite: (Query, Expected Substring or Keyword in Target Document)
EVAL_QUERIES = [
    # Paraphrased queries in Arabic
    ("ماذا تعرف عن اسم المشروع الذي نعمل عليه حالياً؟", "ERP Memory Test A"),
    ("ما هو الحد الائتماني المسموح به للعميل في النظام المالي؟", "50,000 ريال"),
    ("هل تتذكر ما هي بوابات الدفع المعتمدة في النظام؟", "ميسر وتابي"),
    ("ما هي الإجراءات المطلوبة لاعتماد تقارير المحاسبة الشهرية؟", "موافقة المدير المالي"),
    ("متى موعد الصيانة الدورية الأسبوعية للنظام؟", "الجمعة الساعة 11"),
    ("ما هو نموذج الـ Embeddings المستخدم في خادم الذاكرة وكم عدد أبعاده؟", "bge-m3"),
    ("ما هي لغة البرمجة وتفضيلات الـ Type Annotations المفضلة للمستخدم؟", "Python 3.11"),
    ("ما هي لغة الردود المفضلة لدى المستخدم؟", "العربية الفصحى"),
    ("ما هو إطار العمل المفضل للمستخدم في كتابة الاختبارات؟", "pytest"),
    ("ما هو القرار المتخذ بخصوص بديل Celery للمهام الخلفية؟", "asyncio background tasks"),
    ("ما هو القرار المتخذ بخصوص تفعيل RRF؟", "دمج RRF للبحث الهجين"),
    ("ما هو القرار المتعلق بمكتبة الـ reranker والأداء؟", "FastEmbed محلياً"),

    # Paraphrased queries in English
    ("What is the current project name we are developing?", "ERP Memory Test A"),
    ("Which relational database and version does the backend rely on?", "PostgreSQL 16"),
    ("What are the deployment approval requirements for production Kubernetes?", "architect approval"),
    ("What is the maximum allowed discount rate for customer invoices?", "15%"),
    ("What is the rate limit configured for the public API?", "1000 requests per minute"),
    ("How do backend microservices handle internal service-to-service authentication?", "mTLS"),
    ("Which frontend framework and CSS library are we using?", "Next.js 15"),
    ("What is the Redis port and security configuration?", "6379"),
    ("When does the automated database backup occur and what is retention?", "02:00 UTC"),
    ("What vector quantization technique does Elasticsearch index utilize?", "BBQ HNSW"),
    ("What theme and font size does the user prefer in their code editor?", "dark mode theme"),
    ("What architectural pattern does the user prefer for backend structuring?", "Clean Architecture"),
    ("Why did the team decide to use Docker multi-stage builds?", "below 150MB"),
    ("What is the security constraint regarding Elasticsearch port exposure?", "Do not expose Elasticsearch port 9200"),

    # Conversational questions with stop words / conversational fluff
    ("من فضلك ابحث في الذاكرة عن اسم المشروع النشط", "ERP Memory Test A"),
    ("Can you please check memory and tell me what our project name is?", "ERP Memory Test A"),
    ("ما هو الحد الأقصى للخصم التجاري المسموح به للعميل؟", "15%"),
    ("What web framework is our client dashboard built with?", "Next.js 15"),
    ("ما هي شروط النشر على بيئة الإنتاج في كوبرنيتس؟", "architect approval"),
    ("Could you remind me which database version we selected for backend?", "PostgreSQL 16"),
    ("أخبرني عن تفضيلات المستخدم بخصوص التوثيق والـ Swagger", "OpenAPI / Swagger"),
    ("What were the user instructions regarding API documentation format?", "OpenAPI / Swagger 3.0"),
    ("ما هي بوابة الدفع المستخدمة للعملاء في الشرق الأوسط؟", "ميسر وتابي"),
    ("Tell me about the backup schedule and retention window", "02:00 UTC"),
]


def setup_benchmark_corpus() -> None:
    print(f"Injecting {len(MEMORIES_CORPUS)} test memories into tenant '{BENCHMARK_TENANT}'...")
    for content, mtype, tags in MEMORIES_CORPUS:
        memory_save(
            content=content,
            user_id=BENCHMARK_USER,
            tenant_id=BENCHMARK_TENANT,
            type=mtype,
            tags=tags,
            importance=0.8,
            confidence=1.0,
            source="benchmark_suite",
        )
    # Allow indexing refresh
    time.sleep(1.0)
    print("Corpus injection complete!\n")


def run_benchmark() -> None:
    print("==========================================================================")
    print("   📊 RUNNING AI MEMORY RETRIEVAL BENCHMARK (RRF + BGE-RERANKER + HYBRID)  ")
    print("==========================================================================\n")

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    total_queries = len(EVAL_QUERIES)

    print(f"{'#':<3} | {'Query':<52} | {'Rank':<5} | {'Latency':<8} | {'Status'}")
    print("-" * 82)

    for idx, (query, expected_substring) in enumerate(EVAL_QUERIES, 1):
        t0 = time.perf_counter()
        res = memory_search(
            query=query,
            user_id=BENCHMARK_USER,
            tenant_id=BENCHMARK_TENANT,
            mode="hybrid",
            limit=10,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        memories = res.get("memories", [])
        matched_rank = None

        for rank_idx, m in enumerate(memories, 1):
            content = m.get("content", "")
            if expected_substring.lower() in content.lower():
                matched_rank = rank_idx
                break

        query_preview = (query[:49] + "...") if len(query) > 52 else query

        if matched_rank is not None:
            if matched_rank == 1:
                top1_hits += 1
            if matched_rank <= 3:
                top3_hits += 1
            if matched_rank <= 5:
                top5_hits += 1
            if matched_rank <= 10:
                top10_hits += 1
            reciprocal_ranks.append(1.0 / matched_rank)
            rank_str = f"#{matched_rank}"
            status = "HIT"
        else:
            reciprocal_ranks.append(0.0)
            rank_str = "N/A"
            status = "MISS"

        print(f"{idx:<3} | {query_preview:<52} | {rank_str:<5} | {elapsed_ms:>6.1f}ms | {status}")

    # Metrics computation
    r1 = top1_hits / total_queries
    r3 = top3_hits / total_queries
    r5 = top5_hits / total_queries
    r10 = top10_hits / total_queries
    mrr = sum(reciprocal_ranks) / total_queries
    avg_latency = sum(latencies) / total_queries

    print("\n" + "=" * 70)
    print("                     🏆 FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  • Total Evaluated Queries  : {total_queries}")
    print(f"  • Recall@1  (Top-1 Accuracy): {r1:.4f}  ({r1 * 100:.1f}%)")
    print(f"  • Recall@3  (Top-3 Recall)  : {r3:.4f}  ({r3 * 100:.1f}%)")
    print(f"  • Recall@5  (Top-5 Recall)  : {r5:.4f}  ({r5 * 100:.1f}%)")
    print(f"  • Recall@10 (R@10 Target)   : {r10:.4f}  ({r10 * 100:.1f}%)  <-- [Atlas Target: 0.8900]")
    print(f"  • MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"  • Avg Latency per Query     : {avg_latency:.1f} ms")
    print("=" * 70)

    if r10 >= 0.89:
        print(f"\n  🎉 SUCCESS: Your AI Memory Server achieved {r10:.2f} Recall@10, MATCHING OR BEATING ATLAS (0.89)!\n")
    else:
        print(f"\n  📈 Solid performance: {r10:.2f} Recall@10. (Atlas: 0.89).\n")


if __name__ == "__main__":
    setup_benchmark_corpus()
    run_benchmark()
