import httpx
import json
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

def test_multi_agent_scenarios():
    print("=== Testing Multi-Agent & Multi-Project Context Isolation ===")
    
    # 1. Agent A (Finance Agent) in Project 'erp_finance'
    print("\n1. Saving Memory for Finance Agent (Arabic)...")
    res_a = httpx.post(f"{BASE_URL}/memories", json={
        "tenant_id": "enterprise_corp",
        "user_id": "manager_01",
        "agent_id": "finance_agent",
        "project_id": "erp_finance",
        "type": "constraint",
        "content": "الحد الائتماني المعتمد للعميل في النظام المالي هو 50,000 ريال سعودي.",
        "tags": ["ائتمان", "مالية"]
    })
    assert res_a.status_code in (200, 201), res_a.text
    data_a = res_a.json()
    mem_a_id = data_a.get("memory", {}).get("memory_id")
    print(f"Finance Memory Saved: ID={mem_a_id}, Status={data_a.get('status')}")

    # 2. Agent B (DevOps Agent) in Project 'infra_cloud'
    print("\n2. Saving Memory for DevOps Agent (English - Different Project)...")
    res_b = httpx.post(f"{BASE_URL}/memories", json={
        "tenant_id": "enterprise_corp",
        "user_id": "manager_01",
        "agent_id": "devops_agent",
        "project_id": "infra_cloud",
        "type": "procedure",
        "content": "Kubernetes cluster production namespace deployment requires approval from lead architect.",
        "tags": ["k8s", "devops"]
    })
    assert res_b.status_code in (200, 201), res_b.text
    data_b = res_b.json()
    mem_b_id = data_b.get("memory", {}).get("memory_id")
    print(f"DevOps Memory Saved: ID={mem_b_id}, Status={data_b.get('status')}")

    # 3. Querying context as Finance Agent (Must retrieve ONLY erp_finance context)
    print("\n3. Querying context filtered by Finance Agent & Project 'erp_finance'...")
    search_finance = httpx.post(f"{BASE_URL}/memories/search", json={
        "tenant_id": "enterprise_corp",
        "user_id": "manager_01",
        "agent_id": "finance_agent",
        "project_id": "erp_finance",
        "query": "الحد الائتماني والاعتماد"
    })
    assert search_finance.status_code == 200, search_finance.text
    hits_finance = search_finance.json().get("memories", [])
    print(f"Finance Agent Hits Count: {len(hits_finance)}")
    for h in hits_finance:
        print(f"-> Agent: {h.get('agent_id')} | Project: {h.get('project_id')} | Content: {h.get('content')}")
        assert h.get("agent_id") == "finance_agent"
        assert h.get("project_id") == "erp_finance"

    # 4. Querying context as DevOps Agent (Must retrieve ONLY infra_cloud context)
    print("\n4. Querying context filtered by DevOps Agent & Project 'infra_cloud'...")
    search_devops = httpx.post(f"{BASE_URL}/memories/search", json={
        "tenant_id": "enterprise_corp",
        "user_id": "manager_01",
        "agent_id": "devops_agent",
        "project_id": "infra_cloud",
        "query": "Kubernetes deployment approval"
    })
    assert search_devops.status_code == 200, search_devops.text
    hits_devops = search_devops.json().get("memories", [])
    print(f"DevOps Agent Hits Count: {len(hits_devops)}")
    for h in hits_devops:
        print(f"-> Agent: {h.get('agent_id')} | Project: {h.get('project_id')} | Content: {h.get('content')}")
        assert h.get("agent_id") == "devops_agent"
        assert h.get("project_id") == "infra_cloud"

    # 5. Cross-Agent Verification: Ensure Finance Agent CANNOT see DevOps memory even if queried
    print("\n5. Cross-Verification: Finance Agent searches for 'Kubernetes'...")
    cross_search = httpx.post(f"{BASE_URL}/memories/search", json={
        "tenant_id": "enterprise_corp",
        "user_id": "manager_01",
        "agent_id": "finance_agent",
        "project_id": "erp_finance",
        "query": "Kubernetes"
    })
    assert cross_search.status_code == 200
    cross_hits = cross_search.json().get("memories", [])
    print(f"Cross Search Hits Count: {len(cross_hits)} (Expected: 0)")
    assert len(cross_hits) == 0, "Security isolation violation: DevOps context leaked to Finance Agent!"

    print("\n=== SUCCESS: Each agent strictly retrieves only its authorized project context! ===")

if __name__ == "__main__":
    test_multi_agent_scenarios()
