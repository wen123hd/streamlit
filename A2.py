# -*- coding: utf-8 -*-
"""
Streamlit网页版：A2 反洗钱交易网络：闭环识别与最小中断系统
功能：上传csv / 生成示例数据 → 自动计算SCC、风险评分、最小中断集、证据链 → 网页可视化三张图
启动命令：streamlit run app.py
"""
import csv, os, random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO

random.seed(2025)

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 数据准备
# =========================================================
def gen_sample_transactions():
    rows = []
    accts = [f"A{i:02d}" for i in range(40)]
    for _ in range(300):
        f, t = random.sample(accts, 2)
        rows.append([f, t, round(random.uniform(50, 5000), 2),
                     random.randint(0, 71), random.choice(["APP", "柜台", "POS", "网银"])])
    ring1 = [f"A{i:02d}" for i in range(30, 38)]
    for i in range(30, 60):
        f = ring1[i % 8]; t = ring1[(i + 1) % 8]
        rows.append([f, t, round(random.uniform(990, 1010), 2),
                     random.randint(10, 12), "APP"])
    ring2 = [f"B{i:02d}" for i in range(10, 16)]
    for i in range(30, 55):
        f = ring2[i % 6]; t = ring2[(i + 1) % 6]
        rows.append([f, t, round(random.uniform(490, 510), 2),
                     random.randint(30, 32), random.choice(["网银", "APP"])])
    for k in range(4):
        rows.append([f"X0{k}", random.choice(["B10", "B12"]),
                     round(random.uniform(480, 520), 2), random.randint(30, 32), "柜台"])
        rows.append([random.choice(["B14", "B15"]), f"Y0{k}",
                     round(random.uniform(480, 520), 2), random.randint(30, 32), "柜台"])
    df = pd.DataFrame(rows, columns=["from","to","amount","ts","channel"])
    return df

def load_csv(upload_file):
    df = pd.read_csv(upload_file, encoding="utf-8")
    df["amount"] = df["amount"].astype(float)
    df["ts"] = df["ts"].astype(int)
    return df

# =========================================================
# 2. 建图
# =========================================================
def build_graph_from_df(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        f, t, amt, ts, ch = row["from"], row["to"], row["amount"], row["ts"], row["channel"]
        G.add_edge(f, t, amount=amt, ts=ts, channel=ch)
    return G

# =========================================================
# 3. Tarjan SCC
# =========================================================
def tarjan_scc(G):
    index, low, stack, on = {}, {}, [], set()
    idx, sccs = 0, []
    def dfs(v):
        nonlocal idx
        index[v] = low[v] = idx; idx += 1
        stack.append(v); on.add(v)
        for w in G.successors(v):
            if w not in index:
                dfs(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                u = stack.pop(); on.discard(u); comp.append(u)
                if u == v: break
            sccs.append(comp)
    for v in G.nodes:
        if v not in index: dfs(v)
    return sccs

# =========================================================
# 4. 画像打分
# =========================================================
def net_flow(G, comp):
    sin = sum(G.in_degree(v, weight="amount") for v in comp)
    sout = sum(G.out_degree(v, weight="amount") for v in comp)
    return abs(sin - sout), sin + sout

def time_window_ratio(sub, hours=6):
    by_h = defaultdict(float)
    for u, v, d in sub.edges(data=True):
        by_h[d["ts"]] += d["amount"]
    total = sum(by_h.values())
    if total == 0: return 0.0
    keys = sorted(by_h)
    best = 0.0
    for i in range(len(keys)):
        win = sum(by_h[k] for k in keys if keys[i] <= k < keys[i] + hours)
        best = max(best, win)
    return best / total

def channel_switch(sub):
    return sum(1 for _, _, d in sub.edges(data=True) if d["channel"] in ("APP", "网银"))

def risk_score(G, comp, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    sub = G.subgraph(comp)
    n = len(comp)
    m = sub.number_of_edges()
    den = m / (n * (n - 1)) if n > 1 else 0.0
    net, total = net_flow(G, comp)
    net_ratio = net / total if total > 0 else 0.0
    tw = time_window_ratio(sub)
    ch = channel_switch(sub)
    score = alpha * den + beta * tw + gamma * (1 - net_ratio) + delta * min(ch, 10) / 10.0
    return score, dict(ring_density=den, net_ratio=net_ratio, time_win=tw, ch_sw=ch)

# =========================================================
# 5. 点分裂最小割
# =========================================================
def min_interrupt_set(G, comp):
    sub = G.subgraph(comp)
    s = comp[0]
    dist = nx.single_source_shortest_path_length(sub, s)
    cands = [n for n, d in dist.items() if d >= 2]
    if not cands:
        return list(comp), len(comp)
    t = cands[0]
    H = nx.DiGraph()
    for v in comp:
        H.add_edge(f"{v}_in", f"{v}_out", capacity=1 if v not in (s, t) else float("inf"))
    for u, v in sub.edges():
        H.add_edge(f"{u}_out", f"{v}_in", capacity=float("inf"))
    cut_value, partition = nx.minimum_cut(H, f"{s}_out", f"{t}_in")
    reachable, _ = partition
    X = [v for v in comp if f"{v}_in" in reachable and f"{v}_out" not in reachable]
    return X, cut_value

# =========================================================
# 6. 绘图函数，返回图片buffer
# =========================================================
def plot_fig1(G, top, X, entry, exit_, chain, big):
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12,8))
    colors, sizes = [], []
    for n in G.nodes:
        if n in X:
            colors.append("red"); sizes.append(700)
        elif n in exit_:
            colors.append("purple"); sizes.append(550)
        elif n in entry:
            colors.append("limegreen"); sizes.append(550)
        elif n in top:
            colors.append("orange"); sizes.append(550)
        elif any(n in c for c in big):
            colors.append("yellow"); sizes.append(450)
        else:
            colors.append("lightblue"); sizes.append(400)
    nx.draw_networkx(G, pos, node_color=colors, node_size=sizes, with_labels=True,
                     arrows=True, arrowstyle="-|>", font_size=8, ax=ax)
    if chain:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, *_ in chain],
                               edge_color="red", width=3, arrowstyle="-|>", ax=ax)
    in_edges = [(u, v) for v in entry for u in G.predecessors(v) if u not in top]
    out_edges = [(u, v) for u in exit_ for v in G.successors(u) if v not in top]
    if in_edges:
        nx.draw_networkx_edges(G, pos, edgelist=in_edges, edge_color="limegreen",
                               width=2, style="dashed", arrowstyle="-|>", ax=ax)
    if out_edges:
        nx.draw_networkx_edges(G, pos, edgelist=out_edges, edge_color="purple",
                               width=2, style="dashed", arrowstyle="-|>", ax=ax)
    ax.set_title("A2 反洗钱：SCC(橙) + 入口(绿) + 出口(紫) + 最小中断集合X(红) + 证据链环(红边)")
    ax.axis("off")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_fig2(df, sub):
    by_h = defaultdict(float)
    for _, row in df.iterrows():
        by_h[row["ts"]] += row["amount"]
    hours = sorted(by_h); vals = [by_h[h] for h in hours]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(hours, vals, color="steelblue", label="全部交易金额/小时")
    comp_h = defaultdict(float)
    for u, v, d in sub.edges(data=True):
        comp_h[d["ts"]] += d["amount"]
    ck = list(comp_h.keys())
    if ck:
        ax.scatter(ck, [comp_h[k] for k in ck], color="red", zorder=5, label="候选SCC交易(密集段)")
    ax.set_xlabel("时间(小时)"); ax.set_ylabel("金额")
    ax.set_title("图2 时间轴：候选闭环的短时高密度转账")
    ax.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_fig3(scored):
    names = []
    scores = []
    for (s, _), c in scored:
        srt = sorted(c)
        if len(srt) > 4:
            names.append(f"{srt[0]}…{srt[-1]}（{len(srt)}节点）")
        else:
            names.append("-".join(srt))
        scores.append(s)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(range(len(scores)), scores, color="tomato")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("风险评分")
    ax.set_title("图3 各候选SCC风险评分对比")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# =========================================================
# Streamlit主页面
# =========================================================
st.set_page_config(page_title="反洗钱闭环识别系统", layout="wide")
st.title("A2 反洗钱交易网络：闭环识别与最小中断系统")
st.markdown("""
功能说明：
1. 上传 `transactions.csv`（表头：from,to,amount,ts,channel），或者使用内置示例数据
2. 自动 Tarjan SCC 强连通分量识别、风险画像打分、最小顶点割（最小中断账户集）、闭环证据链提取
3. 输出三张可视化图表、结果明细，支持图片下载
""")

with st.sidebar:
    st.header("参数设置")
    use_sample = st.checkbox("使用内置示例数据", value=True)
    uploaded_file = st.file_uploader("上传 transactions.csv", type=["csv"])
    alpha = st.slider("环密度权重 α", 0.0, 1.0, 0.4, 0.05)
    beta = st.slider("短窗占比权重 β", 0.0, 1.0, 0.3, 0.05)
    gamma = st.slider("净流量比权重 γ", 0.0, 1.0, 0.2, 0.05)
    delta = st.slider("渠道切换权重 δ", 0.0, 1.0, 0.1, 0.05)
    run_btn = st.button("🚀 开始分析")

if run_btn:
    with st.spinner("正在构建交易图、SCC计算、风险打分、最小割求解..."):
        if use_sample or uploaded_file is None:
            df = gen_sample_transactions()
            st.info("✅ 使用内置示例数据")
        else:
            df = load_csv(uploaded_file)
            st.success("✅ 已上传用户CSV数据")

        st.subheader("原始交易数据预览")
        st.dataframe(df.head(15), use_container_width=True)

        G = build_graph_from_df(df)
        sccs = tarjan_scc(G)
        big = sorted([c for c in sccs if len(c) >= 3], key=lambda c: min(c))
        st.write(f"总SCC数量：{len(sccs)} | 节点≥3的候选闭环：{len(big)}")

        # 打分
        scored = []
        for c in big:
            s, detail = risk_score(G, c, alpha, beta, gamma, delta)
            scored.append(((s, detail), c))
        scored = sorted(scored, key=lambda x: x[0][0], reverse=True)

        st.subheader("候选SCC风险评分表")
        table_rows = []
        for (s, detail), c in scored:
            table_rows.append({
                "账户集合": sorted(c),
                "风险总分": round(s, 3),
                "环密度": round(detail["ring_density"],2),
                "净流量比": round(detail["net_ratio"],2),
                "短时窗口占比": round(detail["time_win"],2),
                "渠道切换数": detail["ch_sw"]
            })
        st.dataframe(table_rows, use_container_width=True)

        top_detail, top = scored[0]
        X, cut = min_interrupt_set(G, top)
        entry = [v for v in top if any(u not in top for u in G.predecessors(v))]
        exit_ = [v for v in top if any(w not in top for w in G.successors(v))]
        sub = G.subgraph(top)

        # 证据链
        chain = []
        if nx.is_strongly_connected(sub) and len(top) > 1:
            cy = nx.find_cycle(sub)
            for u, v in cy:
                d = sub[u][v]
                chain.append((u, v, d["amount"], d["ts"], d["channel"]))

        st.subheader("最高风险闭环详情")
        st.write(f"闭环账户：{sorted(top)}")
        st.write(f"最小中断账户集（需要阻断）：**{sorted(X)}**，最小割容量：{cut}")
        st.write(f"资金入口账户：{sorted(entry)}")
        st.write(f"资金出口账户：{sorted(exit_)}")
        if chain:
            chain_text = " -> ".join(f"{u}->{v}(¥{a:.2f},t={t},{c})" for u, v, a, t, c in chain)
            st.write("闭环证据链：", chain_text)

        # 绘图
        buf1 = plot_fig1(G, top, X, entry, exit_, chain, big)
        buf2 = plot_fig2(df, sub)
        buf3 = plot_fig3(scored)

        col1, col2 = st.columns(2)
        with col1:
            st.image(buf1, caption="图1 交易网络图", use_container_width=True)
            st.download_button("下载图1", buf1, file_name="fig1_网络图.png")
        with col2:
            st.image(buf2, caption="图2 时间轴", use_container_width=True)
            st.download_button("下载图2", buf2, file_name="fig2_时间轴.png")

        st.image(buf3, caption="图3 风险评分对比", use_container_width=True)
        st.download_button("下载图3", buf3, file_name="fig3_评分.png")

        st.success("✅ 全部分析完成，图片可直接下载给老师")

else:
    st.info("👈 左侧选择数据，点击【开始分析】运行")
