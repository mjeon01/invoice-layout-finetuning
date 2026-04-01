"""
BIO label schema for invoice token classification.

Entity types derived from actual GT field names in the dataset.
22 entity types → 45 labels: O + B-X + I-X for each entity.
"""

import re

# ── GT key → entity type 매핑 ─────────────────────────────────────────────
# line_items[N].* 패턴은 N을 무시하고 entity type으로만 매핑
GT_KEY_TO_ENTITY: dict[str, str] = {
    "invoice_number":               "INVOICE_NUMBER",
    "invoice_date":                 "INVOICE_DATE",
    "exporter.name":                "EXPORTER_NAME",
    "exporter.address":             "EXPORTER_ADDRESS",
    "importer.name":                "IMPORTER_NAME",
    "importer.address":             "IMPORTER_ADDRESS",
    "shipment.port_of_loading":     "PORT_LOADING",
    "shipment.port_of_destination": "PORT_DESTINATION",
    "shipment.incoterms":           "INCOTERMS",
    # line_items[N].* → 아래 패턴 함수로 처리
    "line_items.description":       "ITEM_DESC",
    "line_items.quantity":          "ITEM_QTY",
    "line_items.quantity_unit":     "ITEM_QTY_UNIT",
    "line_items.net_weight":        "ITEM_NET_WEIGHT",
    "line_items.net_weight_unit":   "ITEM_NET_WEIGHT_UNIT",
    "line_items.unit_price":        "ITEM_UNIT_PRICE",
    "line_items.currency":          "ITEM_CURRENCY",
    "line_items.amount":            "ITEM_AMOUNT",
    "line_items.size_or_grade":     "ITEM_SIZE_GRADE",
    "totals.quantity_total":        "TOTAL_QTY",
    "totals.net_weight_total":      "TOTAL_NET_WEIGHT",
    "totals.final_total":           "TOTAL_AMOUNT",
    "totals.currency":              "TOTAL_CURRENCY",
}

_LINE_ITEM_RE = re.compile(r"^line_items\[\d+\]\.(.+)$")


def gt_key_to_entity(key: str) -> str | None:
    """
    Convert a GT JSON key to its entity type string.
    Returns None for keys not in the schema (e.g. 'payments[*]').

    Examples:
        "invoice_number"          → "INVOICE_NUMBER"
        "line_items[0].quantity"  → "ITEM_QTY"
        "payments[0].amount"      → None
    """
    if key in GT_KEY_TO_ENTITY:
        return GT_KEY_TO_ENTITY[key]
    m = _LINE_ITEM_RE.match(key)
    if m:
        sub = "line_items." + m.group(1)
        return GT_KEY_TO_ENTITY.get(sub)
    return None


# ── BIO 레이블 정의 ────────────────────────────────────────────────────────
ENTITY_TYPES: list[str] = [
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "EXPORTER_NAME",
    "EXPORTER_ADDRESS",
    "IMPORTER_NAME",
    "IMPORTER_ADDRESS",
    "PORT_LOADING",
    "PORT_DESTINATION",
    "INCOTERMS",
    "ITEM_DESC",
    "ITEM_QTY",
    "ITEM_QTY_UNIT",
    "ITEM_NET_WEIGHT",
    "ITEM_NET_WEIGHT_UNIT",
    "ITEM_UNIT_PRICE",
    "ITEM_CURRENCY",
    "ITEM_AMOUNT",
    "ITEM_SIZE_GRADE",
    "TOTAL_QTY",
    "TOTAL_NET_WEIGHT",
    "TOTAL_AMOUNT",
    "TOTAL_CURRENCY",
]

LABELS: list[str] = (
    ["O"]
    + [f"B-{e}" for e in ENTITY_TYPES]
    + [f"I-{e}" for e in ENTITY_TYPES]
)

label2id: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
id2label: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}

NUM_LABELS: int = len(LABELS)  # 45


def bio_tags(entity_type: str, n_tokens: int) -> list[str]:
    """Return BIO tag sequence for n_tokens belonging to entity_type."""
    if n_tokens <= 0:
        return []
    return [f"B-{entity_type}"] + [f"I-{entity_type}"] * (n_tokens - 1)


def bio_ids(entity_type: str, n_tokens: int) -> list[int]:
    return [label2id[t] for t in bio_tags(entity_type, n_tokens)]
