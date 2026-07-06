# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:29 2026

@author: Z01788
"""

import pandas as pd
import numpy as np

# =========================
# 0. 讀取資料
# =========================
visit_df = pd.read_csv("data/input/visit_sample.csv", encoding="utf-8-sig")
policy_df = pd.read_csv("data/input/policy_sample.csv", encoding="utf-8-sig")

# =========================
# 1. 建立統一的客戶ID
#    visit_df: 優先用 客戶UUID，空值再補 客戶UUID-2
#    policy_df: 用 客戶UUID
# =========================
visit_df["客戶UUID"] = visit_df["客戶UUID"].astype(str).str.strip()
policy_df["客戶UUID"] = policy_df["客戶UUID"].astype(str).str.strip()


visit_df["客戶ID"] = visit_df["客戶UUID"].fillna(visit_df["客戶UUID"])
policy_df["客戶ID"] = policy_df["客戶UUID"]

# =========================
# 2. 日期欄位轉型
# =========================
def parse_tw_datetime(series):
    s = series.astype(str).str.strip()
    s = s.replace(["nan", "None", "", "NaT"], np.nan)

    # 將中文上午/下午轉成 AM/PM
    s = s.str.replace("上午", "AM", regex=False)
    s = s.str.replace("下午", "PM", regex=False)

    # 去除多餘空白
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # 例如: 2014/12/29 AM 12:00:00
    dt = pd.to_datetime(s, format="%Y/%m/%d %p %I:%M:%S", errors="coerce")

    # 如果有少數格式不同，再補一次泛解析
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], errors="coerce")
        dt.loc[mask] = dt2

    return dt

visit_df["拜訪日"] = parse_tw_datetime(visit_df["拜訪時間"])
policy_df["投保日"] = parse_tw_datetime(policy_df["投保日"])




# =========================
# 1. 拜訪資料（每筆拜訪一列）
# =========================


# 標籤轉 0/1
def to_binary(x):
    if pd.isna(x):
        return 0
    
    # 先試數字
    try:
        if float(x) == 1:
            return 1
    except:
        pass
    
    # 再處理文字
    x = str(x).strip().lower()
    if x in ["1", "true", "t", "y", "yes", "有", "是"]:
        return 1
    
    return 0

tag_cols = [
    "2026(上)儲蓄險最後繳費",
    "2027(上)儲蓄險最後繳費",
    "2026(下)儲蓄險最後繳費",
    "2027(下)儲蓄險最後繳費",
    "投資型潛在客"
]

for col in tag_cols:
    visit_df[col] = visit_df[col].apply(to_binary)


    
visit_level_df = visit_df[[
    "客戶ID",
    "拜訪日",
    "拜訪紀錄UUID_visit",
    "2026(上)儲蓄險最後繳費",
    "2027(上)儲蓄險最後繳費",
    "2026(下)儲蓄險最後繳費",
    "2027(下)儲蓄險最後繳費",
    "投資型潛在客", 
    "標籤(多)"
]].dropna(subset=["客戶ID", "拜訪日"]).copy()

# =========================
# 2. 保單資料（只留必要欄位）
# =========================
policy_match_df = policy_df[[
    "客戶ID",
    "投保日",
    "保單申請案號"
]].dropna(subset=["客戶ID", "投保日", "保單申請案號"]).copy()

# =========================
# 3. 拜訪 × 保單（判斷拜訪後成交）
# =========================
visit_policy_df = visit_level_df.merge(
    policy_match_df,
    on="客戶ID",
    how="left"
)

# 只保留拜訪後成交
visit_policy_df = visit_policy_df[
    visit_policy_df["投保日"] >= visit_policy_df["拜訪日"]
].copy()

# =========================
# 4. 每筆拜訪是否有成交
# =========================
visit_post_summary_df = (
    visit_policy_df.groupby(["客戶ID", "拜訪日", "拜訪紀錄UUID_visit"], as_index=False)
    .agg(
        拜訪後最近成交日=("投保日", "min"),
        拜訪後成交件數=("保單申請案號", "nunique")
    )
)

visit_post_summary_df["是否拜訪後成交"] = 1

# =========================
# 5. 合併回拜訪資料
# =========================
visit_level_df = visit_level_df.merge(
    visit_post_summary_df,
    on=["客戶ID", "拜訪日", "拜訪紀錄UUID_visit"],
    how="left"
)

visit_level_df["是否拜訪後成交"] = visit_level_df["是否拜訪後成交"].fillna(0).astype(int)
visit_level_df["拜訪後成交件數"] = visit_level_df["拜訪後成交件數"].fillna(0).astype(int)

# 成交天數
visit_level_df["拜訪到成交天數"] = np.where(
    visit_level_df["是否拜訪後成交"] == 1,
    (visit_level_df["拜訪後最近成交日"] - visit_level_df["拜訪日"]).dt.days,
    np.nan
)

# =========================
# 6. 輸出
# =========================
visit_level_df.to_csv("data/output/visit_level_mmdd.csv", index=False, encoding="utf-8-sig")



