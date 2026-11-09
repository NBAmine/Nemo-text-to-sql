"""Resilient evaluation with the bundled Spider scoring rules.

Run from the repository root: python -m scripts.evaluation --help.
See docs/spider-evaluation.md for split paths, formats, and metric limitations.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
from typing import Annotated, Optional

import typer

from utils import process_sql
from utils.sql_execution import QueryLimits, SQLComparisonWorker, compare_spider_queries


logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False, help=__doc__)
LEVELS = ("easy", "medium", "hard", "extra", "all")
PARTIAL_TYPES = (
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
)
PARSE_ERRORS = (
    AssertionError,
    KeyError,
    IndexError,
    ValueError,
    TypeError,
    RecursionError,
)
EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


# Spider component scoring and normalization adapted from Tao Yu et al.:
# https://github.com/taoyds/spider
# Kept locally so the original evaluation.py is no longer required.
DISABLE_VALUE = True


DISABLE_DISTINCT = True


WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)


AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}


def has_agg(unit):
    return unit[0] != AGG_OPS.index("none")


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def eval_sel(pred, label):
    pred_sel = pred["select"][1]
    label_sel = label["select"][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred["where"][::2]]
    label_conds = [unit for unit in label["where"][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred["groupBy"]]
    label_cols = [unit[1] for unit in label["groupBy"]]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [
        label.split(".")[1] if "." in label else label for label in label_cols
    ]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred["groupBy"]) > 0:
        pred_total = 1
    if len(label["groupBy"]) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred["groupBy"]]
    label_cols = [unit[1] for unit in label["groupBy"]]
    if (
        pred_total == label_total == 1
        and pred_cols == label_cols
        and pred["having"] == label["having"]
    ):
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred["orderBy"]) > 0:
        pred_total = 1
    if len(label["orderBy"]) > 0:
        label_total = 1
    if (
        len(label["orderBy"]) > 0
        and pred["orderBy"] == label["orderBy"]
        and (
            (pred["limit"] is None and label["limit"] is None)
            or (pred["limit"] is not None and label["limit"] is not None)
        )
    ):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred["where"][1::2]
    label_ao = label["where"][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += Evaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred["intersect"], label["intersect"])
    lt2, pt2, cnt2 = eval_nested(pred["except"], label["except"])
    lt3, pt3, cnt3 = eval_nested(pred["union"], label["union"])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql["where"]) > 0:
        res.add("where")
    if len(sql["groupBy"]) > 0:
        res.add("group")
    if len(sql["having"]) > 0:
        res.add("having")
    if len(sql["orderBy"]) > 0:
        res.add(sql["orderBy"][0])
        res.add("order")
    if sql["limit"] is not None:
        res.add("limit")
    if sql["except"] is not None:
        res.add("except")
    if sql["union"] is not None:
        res.add("union")
    if sql["intersect"] is not None:
        res.add("intersect")

    # or keyword
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    if len([token for token in ao if token == "or"]) > 0:
        res.add("or")

    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add("not")

    # in keyword
    if (
        len(
            [
                cond_unit
                for cond_unit in cond_units
                if cond_unit[1] == WHERE_OPS.index("in")
            ]
        )
        > 0
    ):
        res.add("in")

    # like keyword
    if (
        len(
            [
                cond_unit
                for cond_unit in cond_units
                if cond_unit[1] == WHERE_OPS.index("like")
            ]
        )
        > 0
    ):
        res.add("like")

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len(
        [
            cond_unit
            for cond_unit in cond_units
            if cond_unit[1] == WHERE_OPS.index("like")
        ]
    )

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql["select"][1])
    agg_count += count_agg(sql["where"][::2])
    agg_count += count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql["select"][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql["where"]) > 1:
        count += 1

    # number of group by clauses
    if len(sql["groupBy"]) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""

    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
        ):
            return "medium"
        elif (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
        ):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score["f1"] != 1:
                return 0
        if len(label["from"]["table_units"]) > 0:
            label_tables = sorted(label["from"]["table_units"])
            pred_tables = sorted(pred["from"]["table_units"])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["select"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res["select(no AGG)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["where"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res["where(no OP)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["group(no Having)"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["group"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["order"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["and/or"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["IUEN"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res["keywords"] = {
            "acc": acc,
            "rec": rec,
            "f1": f1,
            "label_total": label_total,
            "pred_total": pred_total,
        }

        return res


def print_scores(scores, etype):
    levels = ["easy", "medium", "hard", "extra", "all"]
    partial_types = [
        "select",
        "select(no AGG)",
        "where",
        "where(no OP)",
        "group(no Having)",
        "group",
        "order",
        "and/or",
        "IUEN",
        "keywords",
    ]

    logger.info("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]["count"] for level in levels]
    logger.info(
        "{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts)
    )

    if etype in ["all", "exec"]:
        logger.info(
            "=====================   EXECUTION ACCURACY     ====================="
        )
        this_scores = [scores[level]["exec"] for level in levels]
        logger.info(
            "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                "execution", *this_scores
            )
        )

    if etype in ["all", "match"]:
        logger.info(
            "\n====================== EXACT MATCHING ACCURACY ====================="
        )
        exact_scores = [scores[level]["exact"] for level in levels]
        logger.info(
            "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                "exact match", *exact_scores
            )
        )
        logger.info(
            "\n---------------------PARTIAL MATCHING ACCURACY----------------------"
        )
        for type_ in partial_types:
            this_scores = [scores[level]["partial"][type_]["acc"] for level in levels]
            logger.info(
                "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                    type_, *this_scores
                )
            )

        logger.info(
            "---------------------- PARTIAL MATCHING RECALL ----------------------"
        )
        for type_ in partial_types:
            this_scores = [scores[level]["partial"][type_]["rec"] for level in levels]
            logger.info(
                "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                    type_, *this_scores
                )
            )

        logger.info(
            "---------------------- PARTIAL MATCHING F1 --------------------------"
        )
        for type_ in partial_types:
            this_scores = [scores[level]["partial"][type_]["f1"] for level in levels]
            logger.info(
                "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                    type_, *this_scores
                )
            )


def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql["from"]["conds"] = rebuild_condition_val(sql["from"]["conds"])
    sql["having"] = rebuild_condition_val(sql["having"])
    sql["where"] = rebuild_condition_val(sql["where"])
    sql["intersect"] = rebuild_sql_val(sql["intersect"])
    sql["except"] = rebuild_sql_val(sql["except"])
    sql["union"] = rebuild_sql_val(sql["union"])

    return sql


def build_valid_col_units(table_units, schema):
    col_ids = [
        table_unit[1]
        for table_unit in table_units
        if table_unit[0] == TABLE_TYPE["table_unit"]
    ]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units = []
    for value in schema.idMap.values():
        if "." in value and value[: value.index(".")] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(
                valid_col_units, condition[idx], kmap
            )
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_["table_units"] = [
        rebuild_table_unit_col(valid_col_units, table_unit, kmap)
        for table_unit in from_["table_units"]
    ]
    from_["conds"] = rebuild_condition_col(valid_col_units, from_["conds"], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [
        rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by
    ]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [
        rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units
    ]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    if sql is None:
        return sql

    sql["select"] = rebuild_select_col(valid_col_units, sql["select"], kmap)
    sql["from"] = rebuild_from_col(valid_col_units, sql["from"], kmap)
    sql["where"] = rebuild_condition_col(valid_col_units, sql["where"], kmap)
    sql["groupBy"] = rebuild_group_by_col(valid_col_units, sql["groupBy"], kmap)
    sql["orderBy"] = rebuild_order_by_col(valid_col_units, sql["orderBy"], kmap)
    sql["having"] = rebuild_condition_col(valid_col_units, sql["having"], kmap)
    sql["intersect"] = rebuild_sql_col(valid_col_units, sql["intersect"], kmap)
    sql["except"] = rebuild_sql_col(valid_col_units, sql["except"], kmap)
    sql["union"] = rebuild_sql_col(valid_col_units, sql["union"], kmap)

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


class EvaluationError(ValueError):
    """Invalid inputs or references prevent a trustworthy evaluation result."""


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise EvaluationError(f"Cannot read {path}: {error}") from error


def _validate_db_id(db_id):
    if (
        not isinstance(db_id, str)
        or not db_id.strip()
        or db_id != db_id.strip()
        or db_id in {".", ".."}
        or any(c in db_id for c in "/\\:\0\r\n\t")
    ):
        raise EvaluationError(f"Invalid database ID: {db_id!r}")


def _load_pairs(gold, predict):
    text = _read_text(gold)
    if Path(gold).suffix.lower() == ".json":
        try:
            rows = json.loads(text)
        except ValueError as error:
            raise EvaluationError(f"Invalid gold JSON {gold}: {error}") from error
        if not isinstance(rows, list):
            raise EvaluationError(
                "Gold JSON must be a Spider array of query/db_id objects"
            )
        references = []
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise EvaluationError(f"Expected gold object at example {number}")
            references.append((row.get("query"), row.get("db_id")))
    else:
        references = []
        for number, line in enumerate(text.splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 2:
                raise EvaluationError(f"Expected SQL<TAB>db_id at gold line {number}")
            references.append((fields[0], fields[1].strip()))
    if not references:
        raise EvaluationError("Gold input is empty")
    for number, (sql, db_id) in enumerate(references, 1):
        if not isinstance(sql, str) or not sql.strip():
            raise EvaluationError(f"Missing gold SQL at example {number}")
        _validate_db_id(db_id)
    # splitlines retains internal blank records and ignores one final newline.
    predictions = _read_text(predict).splitlines()
    if len(predictions) != len(references):
        raise EvaluationError(
            f"Prediction count {len(predictions)} differs from gold count {len(references)}; "
            "blank prediction lines count as examples"
        )
    pairs = []
    for number, ((sql, db_id), prediction) in enumerate(
        zip(references, predictions), 1
    ):
        fields = prediction.split("\t")
        if len(fields) > 2 or (len(fields) == 2 and fields[1].strip() != db_id):
            raise EvaluationError(
                f"Invalid prediction database at line {number}: expected {db_id}"
            )
        pairs.append((sql, db_id, fields[0]))
    return pairs


def build_foreign_key_map_from_json(table):
    """Load the legacy foreign-key maps, rejecting malformed schema metadata."""
    try:
        entries = json.loads(_read_text(table))
        if not isinstance(entries, list) or not entries:
            raise EvaluationError("Schema metadata must be a nonempty JSON array")
        result = {}
        for entry in entries:
            db_id = entry["db_id"]
            _validate_db_id(db_id)
            if db_id in result:
                raise EvaluationError(f"Duplicate database in schema metadata: {db_id}")
            columns = entry["column_names_original"]
            tables = entry["table_names_original"]
            if (
                not isinstance(columns, list)
                or not columns
                or not isinstance(tables, list)
                or not tables
                or not all(isinstance(name, str) and name for name in tables)
            ):
                raise EvaluationError(f"Invalid schema columns/tables for {db_id}")
            for index, name in columns:
                if (
                    type(index) is not int
                    or not -1 <= index < len(tables)
                    or not isinstance(name, str)
                    or not name
                ):
                    raise EvaluationError(f"Invalid schema column for {db_id}")
            for left, right in entry["foreign_keys"]:
                if any(
                    type(i) is not int or not 0 <= i < len(columns)
                    for i in (left, right)
                ):
                    raise EvaluationError(f"Invalid foreign-key index for {db_id}")
            result[db_id] = build_foreign_key_map(entry)
        return result
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        raise EvaluationError(
            f"Cannot load schema metadata {table}: {error}"
        ) from error


def _database(root, db_id):
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EvaluationError(f"Missing database or path outside database root: {path}")
    return path


def _schema(path):
    """Read schemas once, closing connections and never creating absent files."""
    with closing(
        sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1)
    ) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        schema = {}
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            schema[name.lower()] = [
                row[1].lower()
                for row in connection.execute(f"PRAGMA table_info({quoted})")
            ]
    if not schema:
        raise EvaluationError(f"Database has no tables: {path}")
    return process_sql.Schema(schema)


def _parse(schema, query, *, strict=True, warnings=None):
    if len(query.encode("utf-8")) > 1024 * 1024:
        raise ValueError("SQL exceeds 1 MiB")
    tokens = process_sql.tokenize(query)
    aliases = process_sql.get_tables_with_alias(schema.schema, tokens)
    end, parsed = process_sql.parse_sql(tokens, 0, aliases, schema)
    if end != len(tokens):
        message = f"Unparsed SQL tokens after position {end}"
        if strict:
            raise ValueError(message)
        if warnings is not None:
            warnings.append(message)
    return parsed


def _normalize(sql, schema, kmap):
    sql = deepcopy(sql)
    valid_columns = build_valid_col_units(sql["from"]["table_units"], schema)
    return rebuild_sql_col(valid_columns, rebuild_sql_val(sql), kmap)


def _new_scores():
    return {
        level: {
            "count": 0,
            "exec": 0.0,
            "exact": 0.0,
            "partial": {
                clause: {
                    "acc": 0.0,
                    "rec": 0.0,
                    "f1": 0.0,
                    "acc_count": 0,
                    "rec_count": 0,
                }
                for clause in PARTIAL_TYPES
            },
        }
        for level in LEVELS
    }


def _add_partial(bucket, partial):
    for clause, result in partial.items():
        aggregate = bucket["partial"][clause]
        for metric, total in (("acc", "pred_total"), ("rec", "label_total")):
            if result[total] > 0:
                aggregate[metric] += result[metric]
                aggregate[metric + "_count"] += 1


def _finish_scores(scores, etype):
    for bucket in scores.values():
        if not bucket["count"]:
            continue
        if etype in {"all", "exec"}:
            bucket["exec"] /= bucket["count"]
        if etype in {"all", "match"}:
            bucket["exact"] /= bucket["count"]
            for clause in bucket["partial"].values():
                for metric in ("acc", "rec"):
                    count = clause[metric + "_count"]
                    clause[metric] = clause[metric] / count if count else 0.0
                acc, rec = clause["acc"], clause["rec"]
                # Preserve the bundled evaluator's aggregate F1 convention.
                clause["f1"] = 2 * acc * rec / (acc + rec) if acc + rec else 1.0


def evaluate(
    gold,
    predict,
    db_dir,
    etype="all",
    kmaps=None,
    *,
    table_path=None,
    limits=QueryLimits(),
    return_details=False,
    strict_parsing=True,
):
    """Evaluate aligned SQL predictions using the bundled Spider scoring rules.

    The five positional arguments and default return value match evaluation.py.
    Gold accepts SQL<TAB>db_id lines or Spider JSON; predictions accept one SQL
    per line with an optional matching db_id. Blank predictions score zero.
    Pass table_path instead of kmaps to load schema metadata here.

    Invalid predictions are recorded and evaluation continues. Missing inputs,
    mismatched counts, broken databases, and failed references raise
    EvaluationError with context; no incomplete score is returned. Execution
    reuses one lightweight worker per pass and is read-only and bounded by
    QueryLimits. return_details adds per-example
    diagnostics around the same score dictionary. Exact match ignores literal
    values and DISTINCT, as in the legacy evaluator; this is not test-suite EX.
    strict_parsing=False explicitly permits the legacy parser to ignore trailing
    tokens, recording warnings for every affected gold or prediction query.
    """
    if etype not in {"all", "exec", "match"}:
        raise EvaluationError(f"Unknown evaluation method: {etype!r}")
    if not isinstance(limits, QueryLimits):
        raise EvaluationError("limits must be a QueryLimits instance")
    if type(strict_parsing) is not bool:
        raise EvaluationError("strict_parsing must be a boolean")
    pairs = _load_pairs(gold, predict)
    root = Path(db_dir).resolve()
    if not root.is_dir():
        raise EvaluationError(f"Database root does not exist: {root}")
    if kmaps is not None and table_path is not None:
        raise EvaluationError("Supply either kmaps or table_path")
    if kmaps is None:
        if table_path is None:
            raise EvaluationError(
                "Supply kmaps or table_path for foreign-key normalization"
            )
        kmaps = build_foreign_key_map_from_json(table_path)
    if not isinstance(kmaps, dict):
        raise EvaluationError("kmaps must map database IDs to foreign-key dictionaries")
    databases, schemas = {}, {}
    for _, db_id, _ in pairs:
        if db_id in databases:
            continue
        if db_id not in kmaps or not isinstance(kmaps[db_id], dict):
            raise EvaluationError(
                f"Missing or invalid foreign-key map for database {db_id}"
            )
        databases[db_id] = _database(root, db_id)
        try:
            schemas[db_id] = _schema(databases[db_id])
        except (sqlite3.Error, OSError, ValueError) as error:
            raise EvaluationError(f"Cannot read database {db_id}: {error}") from error
        column_ids = {
            schemas[db_id].idMap[f"{table}.{column}"]
            for table, columns in schemas[db_id].schema.items()
            for column in columns
        }
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or key not in column_ids
            or value not in column_ids
            for key, value in kmaps[db_id].items()
        ):
            raise EvaluationError(
                f"Invalid foreign-key column mapping for database {db_id}"
            )
    scores, examples = _new_scores(), []
    evaluator = Evaluator()
    with SQLComparisonWorker() as worker:
        for number, (gold_sql, db_id, prediction) in enumerate(pairs, 1):
            context = f"example {number} (database {db_id})"
            schema = schemas[db_id]
            gold_warnings, pred_warnings = [], []
            try:
                gold_ast = _parse(
                    schema, gold_sql, strict=strict_parsing, warnings=gold_warnings
                )
                hardness = evaluator.eval_hardness(gold_ast)
                gold_ast = _normalize(gold_ast, schema, kmaps[db_id])
            except PARSE_ERRORS as error:
                raise EvaluationError(
                    f"Gold parse/normalization failed at {context}: {type(error).__name__}: {error}"
                ) from error
            errors = []
            pred_ast = None
            try:
                if not prediction.strip():
                    raise ValueError("Empty prediction")
                pred_ast = _normalize(
                    _parse(
                        schema, prediction, strict=strict_parsing, warnings=pred_warnings
                    ),
                    schema,
                    kmaps[db_id],
                )
            except PARSE_ERRORS as error:
                errors.append(
                    {
                        "stage": "prediction_parse",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            outcome = None
            if etype in {"all", "exec"}:
                try:
                    outcome = compare_spider_queries(
                        databases[db_id],
                        gold_sql,
                        prediction if pred_ast is not None else None,
                        [unit[1] for unit in gold_ast["select"][1]],
                        [unit[1] for unit in pred_ast["select"][1]]
                        if pred_ast is not None
                        else [],
                        limits,
                        worker=worker,
                    )
                except (OSError, RuntimeError) as error:
                    raise EvaluationError(
                        f"Execution infrastructure failed at {context}: {error}"
                    ) from error
                if outcome.get("reference_error"):
                    raise EvaluationError(
                        f"Gold execution failed at {context}: {outcome['status']}: {outcome.get('error', '')}"
                    )
                if outcome.get("error"):
                    errors.append(
                        {
                            "stage": "prediction_execution",
                            "error": outcome["error"],
                            "status": outcome["status"],
                        }
                    )
            exact, partial = 0, None
            if etype in {"all", "match"}:
                try:
                    exact = int(
                        evaluator.eval_exact_match(
                            deepcopy(pred_ast if pred_ast is not None else EMPTY_SQL),
                            deepcopy(gold_ast),
                        )
                    )
                    partial = evaluator.partial_scores
                except PARSE_ERRORS as error:
                    errors.append(
                        {
                            "stage": "prediction_match",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    evaluator.eval_exact_match(deepcopy(EMPTY_SQL), deepcopy(gold_ast))
                    partial = evaluator.partial_scores
                    exact = 0
            for level in (hardness, "all"):
                scores[level]["count"] += 1
                scores[level]["exact"] += exact
                scores[level]["exec"] += int(bool(outcome and outcome["correct"]))
                if partial is not None:
                    _add_partial(scores[level], partial)
            example = {
                "index": number,
                "db_id": db_id,
                "hardness": hardness,
                "gold_sql": gold_sql,
                "prediction_sql": prediction,
                "exact": exact if etype in {"all", "match"} else None,
                "exec": bool(outcome["correct"]) if outcome else None,
                "status": "prediction_error" if errors else "ok",
                "errors": errors,
                "execution": outcome,
                "partial": partial,
                "warnings": [
                    {"stage": stage, "warning": warning}
                    for stage, messages in (
                        ("gold_parse", gold_warnings),
                        ("prediction_parse", pred_warnings),
                    )
                    for warning in messages
                ] + (outcome.get("warnings", []) if outcome else []),
            }
            examples.append(example)
            if errors:
                logger.warning("Prediction failed at %s: %s", context, errors)
            if example["warnings"]:
                logger.warning("Evaluation warning at %s: %s", context, example["warnings"])
    _finish_scores(scores, etype)
    print_scores(scores, etype)
    report = {
        "protocol": "bundled-spider-resilient-v1",
        "scores": scores,
        "policy": {
            "strict_parsing": strict_parsing,
            "limits": asdict(limits),
            "disable_value": DISABLE_VALUE,
            "disable_distinct": DISABLE_DISTINCT,
            "execution_comparison": "legacy Spider selected-expression mapping with row order",
            "result_text_decoding": "utf-8 with surrogateescape",
        },
        "summary": {
            "count": len(examples),
            "prediction_error_count": sum(bool(e["errors"]) for e in examples),
            "error_counts": dict(
                Counter(error["stage"] for e in examples for error in e["errors"])
            ),
            "warning_counts": dict(
                Counter(warning["stage"] for e in examples for warning in e["warnings"])
            ),
        },
        "examples": examples,
    }
    return report if return_details else scores


class EvaluationType(str, Enum):
    ALL = "all"
    EXEC = "exec"
    MATCH = "match"


@app.command()
def main(
    gold: Annotated[
        Path, typer.Option("--gold", help="Spider gold SQL file or JSON split")
    ],
    pred: Annotated[Path, typer.Option("--pred", help="One SQL prediction per line")],
    db: Annotated[
        Path, typer.Option("--db", help="Root of the Spider SQLite databases")
    ],
    table: Annotated[Path, typer.Option("--table", help="Spider schema metadata JSON")],
    etype: Annotated[EvaluationType, typer.Option("--etype")] = EvaluationType.ALL,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds per SQL query")
    ] = 5.0,
    max_rows: Annotated[int, typer.Option("--max-rows", min=1)] = 100_000,
    max_bytes: Annotated[int, typer.Option("--max-bytes", min=1)] = 8 * 1024 * 1024,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", help="Optional JSON report, created exclusively"),
    ] = None,
    allow_unparsed_tokens: Annotated[
        bool,
        typer.Option(
            "--allow-unparsed-tokens",
            help="Permit legacy partial parses and record warnings (default: reject)",
        ),
    ] = False,
):
    """Evaluate Spider predictions and optionally save the full JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if output and output.exists():
            raise EvaluationError(f"Report already exists: {output}")
        report = evaluate(
            gold,
            pred,
            db,
            etype.value,
            table_path=table,
            limits=QueryLimits(timeout, max_rows, max_bytes),
            return_details=True,
            strict_parsing=not allow_unparsed_tokens,
        )
        if output:
            encoded = (
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        json.dumps(
            {"summary": report["summary"], "all": report["scores"]["all"]}, indent=2
        )
    )
    return report


if __name__ == "__main__":
    app()
