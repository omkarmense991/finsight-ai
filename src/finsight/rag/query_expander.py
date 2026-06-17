# src/finsight/rag/query_expander.py


def generate_search_queries(question: str) -> list[str]:
    lower_question = question.lower()

    queries = [question]

    # =========================================
    # Query Intent Flags
    # =========================================

    is_geography_query = (
        "geograph" in lower_question
        or "geographical" in lower_question
        or "geography" in lower_question
        or "region" in lower_question
        or "regional" in lower_question
        or "country" in lower_question
        or "north america" in lower_question
        or "europe" in lower_question
        or "india" in lower_question
        or "rest of the world" in lower_question
    )

    is_segment_query = (
        "business segment" in lower_question
        or "segments" in lower_question
        or "segmental" in lower_question
        or "segment reporting" in lower_question
        or "operating segment" in lower_question
        or "reportable segment" in lower_question
    )

    is_risk_query = (
        "risk" in lower_question
        or "risks" in lower_question
        or "risk management" in lower_question
        or "emerging risks" in lower_question
    )

    is_revenue_query = (
        "revenue" in lower_question
        or "growth" in lower_question
        or "income" in lower_question
        or "sales" in lower_question
    )

    is_margin_query = (
        "margin" in lower_question
        or "profitability" in lower_question
        or "operating income" in lower_question
        or "operating profit" in lower_question
    )

    is_dividend_query = (
        "dividend" in lower_question
        or "dividends" in lower_question
        or "final dividend" in lower_question
        or "interim dividend" in lower_question
        or "total dividend" in lower_question
    )

    # =========================================
    # Geography Query Expansion
    # =========================================
    # Important:
    # Geography queries should not trigger business-segment expansion,
    # even if the user says "geographical revenue segments".

    if is_geography_query:
        queries.extend(
            [
                "revenue distribution by geographical segments",
                "geographical segments north america europe india rest of the world",
                "revenue by geography north america europe india rest of the world",
                "geographical revenue is based on the domicile of customer north america europe india rest of the world",
                "north america europe rest of the world india revenue distribution",
            ]
        )

    # =========================================
    # Business Segment Query Expansion
    # =========================================
    # Only trigger this if it is not a geography query.

    if is_segment_query and not is_geography_query:
        queries.extend(
            [
                "business segments",
                "operating segments reportable segments segment reporting",
                "revenue distribution by business segments",
                "business segments consolidated segmental revenues segmental operating income segment profit",
                "segmental revenues segmental operating income segment profit",
                "financial services manufacturing energy utilities resources retail communication hi-tech life sciences all other segments",
            ]
        )

    # =========================================
    # Risk Query Expansion
    # =========================================

    if is_risk_query:
        queries.extend(
            [
                "risk management key risks emerging risks",
                "geopolitical risk cybersecurity risk regulatory risk",
                "contractual liabilities risk mitigation",
                "cybersecurity data protection privacy ESG talent availability regulatory environment",
            ]
        )

    # =========================================
    # Revenue Query Expansion
    # =========================================

    if is_revenue_query and not is_geography_query:
        queries.extend(
            [
                "revenue from operations revenue growth",
                "financial performance year on year growth",
                "consolidated revenue standalone revenue",
                "revenue fiscal 2026 fiscal 2025 percentage growth",
            ]
        )

    # =========================================
    # Margin / Profitability Query Expansion
    # =========================================

    if is_margin_query:
        queries.extend(
            [
                "operating margin",
                "segmental operating margin",
                "overall segment profitability",
                "operating margin percentage",
                "segmental operating income segmental operating margin",
                "revenue operating income operating margin",
                "operating margin fiscal 2026 fiscal 2025",
            ]
        )

    if is_dividend_query:
        queries.extend(
            [
                "dividend final dividend interim dividend total dividend per share",
                "announced total dividend per share",
                "dividend of rupees per equity share",
                "recommended final dividend declared interim dividend",
                "dividend distribution shareholders per share fiscal 2026",
            ]
        )

    # =========================================
    # Remove Duplicate Queries
    # =========================================

    unique_queries = []

    for query in queries:
        query = query.strip()

        if query and query not in unique_queries:
            unique_queries.append(query)

    return unique_queries
