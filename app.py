import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 系统配置 =================
st.set_page_config(page_title="多空轮动系统", layout="wide", page_icon="🚀")

# 获取当前脚本所在目录（兼容 GitHub/Streamlit Cloud 部署）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 字体适配 ---
# 将字体文件也放在项目根目录下
FONT_FILE = os.path.join(BASE_DIR, "SimHei.ttf")
if os.path.exists(FONT_FILE):
    my_font = fm.FontProperties(fname=FONT_FILE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:
    my_font = fm.FontProperties(family='SimHei')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# --- 板块成分股映射 ---
SECTOR_CONSTITUENTS = {
    '农产品': ['白糖', '菜籽油', '菜籽粕', '豆粕', '豆油', '棕榈油', '玉米', '淀粉', '鸡蛋', '棉花', '生猪'],
    '化工': ['甲醇', 'PTA', '纯碱', '尿素', 'PVC', '塑料', '聚丙烯', '苯乙烯', '乙二醇', '橡胶'],
    '有色': ['沪铜', '沪铝', '沪锌', '沪铅', '沪镍', '沪锡'],
    '黑': ['铁矿石', '焦炭', '焦煤', '螺纹钢', '热卷', '硅铁', '锰硅'],
    '贵金属': ['沪金', '沪银']
}

SECTOR_NAME_MAP = {
    '黑链': '黑', '有色板块': '有色', '化工板块': '化工', '农产品': '农产品', '贵金属': '贵金属'
}

# --- 品种映射与乘数 ---
CN_NAME_MAP = {
    '沪金': 'au', '沪银': 'ag', '沪铜': 'cu', '沪铝': 'al',
    '沪锌': 'zn', '沪铅': 'pb', '沪镍': 'ni', '沪锡': 'sn',
    '螺纹钢': 'rb', '热卷': 'hc', '铁矿石': 'i',
    '焦炭': 'j', '焦煤': 'jm', '硅铁': 'sf', '锰硅': 'sm',
    '原油': 'sc', '燃油': 'fu', '低硫燃油': 'lu', '沥青': 'bu',
    '橡胶': 'ru', '20号胶': 'nr', '塑料': 'l', 'PVC': 'v', '聚丙烯': 'pp',
    '苯乙烯': 'eb', '乙二醇': 'eg', '甲醇': 'ma', 'PTA': 'ta',
    '纯碱': 'sa', '玻璃': 'fg', '尿素': 'ur', '豆一': 'a', '豆粕': 'm',
    '豆油': 'y', '棕榈油': 'p', '玉米': 'c', '淀粉': 'cs', '鸡蛋': 'jd',
    '生猪': 'lh', '白糖': 'sr', '棉花': 'cf', '菜籽油': 'oi', '菜籽粕': 'rm'
}

CONTRACT_MULTIPLIERS = {
    'rb': 10, 'hc': 10, 'i': 100, 'j': 100, 'jm': 60, 'sf': 5, 'sm': 5,
    'cu': 5, 'al': 5, 'zn': 5, 'pb': 5, 'ni': 1, 'sn': 1,
    'au': 1000, 'ag': 15, 'ru': 10, 'bu': 10, 'fu': 10, 'sc': 1000,
    'l': 5, 'pp': 5, 'v': 5, 'eg': 10, 'ta': 5, 'ma': 10, 'ur': 20,
    'sa': 20, 'eb': 5, 'c': 10, 'cs': 10, 'a': 10, 'm': 10, 'y': 10,
    'p': 10, 'oi': 10, 'rm': 10, 'cf': 5, 'sr': 10, 'jd': 10, 'lh': 16
}


def get_multiplier(asset_name):
    import re
    match = re.match(r"([a-zA-Z]+)", asset_name)
    if match:
        code = match.group(1).lower()
        return CONTRACT_MULTIPLIERS.get(code, 1)
    clean_name = asset_name.replace("主连", "").replace("指数", "").replace("连续", "").replace("日线", "").replace(
        ".csv", "").strip()
    code = CN_NAME_MAP.get(clean_name)
    if code:
        return CONTRACT_MULTIPLIERS.get(code, 1)
    return 1


# ================= 2. 数据处理 =================
def read_robust_csv(f):
    for enc in ['gbk', 'utf-8', 'gb18030', 'cp936']:
        try:
            df = pd.read_csv(f, encoding=enc, engine='python')
            rename_map = {}
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in ['日期', '日期/时间', 'date', 'Date']: rename_map[c] = 'date'
                if c_str in ['收盘价', '收盘', 'close', 'price', 'Close']: rename_map[c] = 'close'
                if c_str in ['最高价', '最高', 'high', 'High']: rename_map[c] = 'high'
                if c_str in ['最低价', '最低', 'low', 'Low']: rename_map[c] = 'low'
                if c_str in ['开盘价', '开盘', 'open', 'Open']: rename_map[c] = 'open'
                if c_str in ['成交量', 'volume', 'Volume', 'vol']: rename_map[c] = 'volume'
                if c_str in ['成交额', 'amount', 'Amount']: rename_map[c] = 'amount'
                if c_str in ['持仓量', 'open_interest', 'oi']: rename_map[c] = 'open_interest'
            df.rename(columns=rename_map, inplace=True)
            if 'date' in df.columns and 'close' in df.columns:
                return df
        except:
            continue
    return None


@st.cache_data(ttl=3600)
def load_directory_data(folder, is_index=False):
    if not os.path.exists(folder):
        return None, None, None, None, None, f"路径不存在: {folder}"
    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    if not files:
        return None, None, None, None, None, f"[{folder}] 中无CSV文件"

    price_dict, low_dict, open_dict, high_dict, amount_dict = {}, {}, {}, {}, {}

    for file in files:
        name = file.split('.')[0].replace("主连", "").replace("日线", "").replace("指数", "").strip()
        path = os.path.join(folder, file)
        df = read_robust_csv(path)
        if df is None: continue
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            df.set_index('date', inplace=True)

            if 'amount' not in df.columns:
                multiplier = get_multiplier(name)
                df['amount'] = df['close'] * df['volume'] * multiplier if 'volume' in df.columns else 1.0

            price_dict[name] = df['close']
            low_dict[name] = df['low'] if 'low' in df.columns else df['close']
            open_dict[name] = df['open'] if 'open' in df.columns else df['close'].shift(1).fillna(df['close'])
            high_dict[name] = df['high'] if 'high' in df.columns else df['close']
            amount_dict[name] = df['amount']
        except:
            continue

    return (pd.DataFrame(price_dict).ffill(), pd.DataFrame(low_dict).ffill(),
            pd.DataFrame(open_dict).ffill(), pd.DataFrame(high_dict).ffill(),
            pd.DataFrame(amount_dict).ffill(), None)


# ================= 3. 因子与回测引擎 =================
class FactorEngine:
    @staticmethod
    def calculate_netpower_factor(df_c, df_h, df_l, df_amt, win_long=40, win_short=10):
        hl_range = df_h - df_l
        hl_range = hl_range.replace(0, np.nan)
        net_power = df_amt * ((df_c - df_l) - (df_h - df_c)) / hl_range
        net_power = net_power.fillna(0)

        ewma_long = net_power.ewm(span=win_long, adjust=False).mean()
        ewma_short = net_power.ewm(span=win_short, adjust=False).mean()
        factor = ewma_long - ewma_short

        factor_vol = factor.rolling(20).std().replace(0, np.nan)
        return factor / factor_vol


def get_target_constituents(sector_name, available_assets):
    std_name = SECTOR_NAME_MAP.get(sector_name, sector_name)
    for key, const_list in SECTOR_CONSTITUENTS.items():
        if key == std_name:
            return [c for c in const_list if c in available_assets]
    return []


def run_hybrid_strategy(idx_p, idx_o, idx_h, idx_l, idx_amt, ast_p, ast_o, params):
    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])
    comm_rate = params.get('commission', 0.001)
    target_weekday = params.get('rebalance_weekday', 3)  # 默认3是周四
    bench_keywords = ['文华', '综合', 'NH0100', params.get('bench_name', '')]

    idx_factors = FactorEngine.calculate_netpower_factor(idx_p, idx_h, idx_l, idx_amt, params['win_long'],
                                                         params['win_short'])
    valid_sectors = [c for c in idx_factors.columns if not any(k in c for k in bench_keywords if k)]

    dates = ast_p.index.intersection(idx_p.index)
    if len(dates) == 0: return pd.DataFrame(), [], {}, {}

    start_idx = max(dates.get_indexer([start_date], method='bfill')[0], 1)

    cash = 1.0
    positions = {}
    nav_record = []
    logs = []
    current_sector = None

    asset_trade_pnls = {a: [] for const in SECTOR_CONSTITUENTS.values() for a in const}
    sector_trade_pnls = {s: [] for s in SECTOR_CONSTITUENTS.keys()}

    # 智能调仓日控制变量
    current_week = dates[start_idx].isocalendar().week
    week_rebalanced = False

    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        if curr_date > end_date: break
        prev_date = dates[i - 1]

        # 检查是否进入新的一周
        if curr_date.isocalendar().week != current_week:
            current_week = curr_date.isocalendar().week
            week_rebalanced = False

        # --- 核心：极度鲁棒的调仓日判定逻辑 ---
        is_rebalance_day = False
        if not week_rebalanced:
            # 如果今天等于或超过了设定的星期几，触发调仓
            if curr_date.dayofweek >= target_weekday:
                is_rebalance_day = True
                week_rebalanced = True
            else:
                # 极端容错：如果今天还没到设定的日子，但明天就跨周了(比如目标周五，但周三是本周最后一天)
                if i < len(dates) - 1 and curr_date.isocalendar().week != dates[i + 1].isocalendar().week:
                    is_rebalance_day = True
                    week_rebalanced = True
                elif i == len(dates) - 1:
                    is_rebalance_day = True
        # --------------------------------------

        executed_rotation = False
        today_factors = idx_factors.loc[prev_date, valid_sectors].dropna()

        if is_rebalance_day and not today_factors.empty:
            target_sector = today_factors.idxmin()
            min_factor_val = today_factors[target_sector]

            if target_sector != current_sector or i == start_idx:
                assets_to_sell = list(positions.keys())
                current_round_pnls = []

                for asset in assets_to_sell:
                    sell_price = ast_o.loc[curr_date, asset] if pd.notna(ast_o.loc[curr_date, asset]) else ast_p.loc[
                        prev_date, asset]
                    entry_price = positions[asset]['entry_price']

                    trade_pnl = (sell_price - entry_price) / entry_price
                    current_round_pnls.append(trade_pnl)

                    if asset in asset_trade_pnls:
                        asset_trade_pnls[asset].append(trade_pnl)

                    prev_close = ast_p.loc[prev_date, asset]
                    sell_value = positions[asset]['value'] * (sell_price / prev_close) if prev_close > 0 else 0

                    cash += sell_value * (1 - comm_rate / 2)
                    logs.append(
                        f"🔄 【清仓成分】 {curr_date.date()} ({curr_date.day_name()}) | 平仓: {asset:4s} | 买入: {entry_price:7.2f} | 卖出: {sell_price:7.2f} | 收益: {trade_pnl * 100:>+6.2f}%")
                    del positions[asset]

                if current_round_pnls:
                    avg_pnl = sum(current_round_pnls) / len(current_round_pnls)
                    if current_sector:
                        std_sector = SECTOR_NAME_MAP.get(current_sector, current_sector)
                        if std_sector in sector_trade_pnls:
                            sector_trade_pnls[std_sector].append(avg_pnl)

                    logs.append(
                        f"📊 【本周收益】 {curr_date.date()} | 平仓板块: {current_sector} | 平均收益: {avg_pnl * 100:>+6.2f}%\n" + "-" * 50)

                if cash > 0:
                    constituents = get_target_constituents(target_sector, ast_p.columns)
                    valid_constituents = [c for c in constituents if pd.notna(ast_o.loc[curr_date, c])]

                    if valid_constituents:
                        cash_per_asset = cash / len(valid_constituents)
                        buy_details = []
                        for asset in valid_constituents:
                            buy_price = ast_o.loc[curr_date, asset]
                            positions[asset] = {
                                'value': cash_per_asset * (1 - comm_rate / 2),
                                'entry_price': buy_price,
                                'sector': target_sector
                            }
                            buy_details.append(asset)

                        cash = 0.0
                        logs.append(
                            f"🚀 【板块发车】 {curr_date.date()} ({curr_date.day_name()}) | 强势板块: {target_sector} | 均分买入: {', '.join(buy_details)}")
                    else:
                        logs.append(f"⚠️ 【买入失败】 {curr_date.date()} | 板块 {target_sector} 无可用数据，转为空仓。")
                        target_sector = None

                current_sector = target_sector
                executed_rotation = True

        assets_in_hand = list(positions.keys())
        for asset in assets_in_hand:
            prev_close = ast_p.loc[prev_date, asset]
            today_close = ast_p.loc[curr_date, asset]
            base_price = ast_o.loc[curr_date, asset] if executed_rotation else prev_close

            if base_price > 0 and pd.notna(today_close):
                positions[asset]['value'] *= (today_close / base_price)

        total_nav = cash + sum(pos['value'] for pos in positions.values())
        nav_record.append(
            {'date': curr_date, 'nav': total_nav, 'sector': current_sector if current_sector else "空仓(现金)"})

    if positions:
        last_date = dates[-1]
        mtm_sector_pnls = {}
        for asset, pos in positions.items():
            curr_price = ast_p.loc[last_date, asset]
            if pd.notna(curr_price):
                trade_pnl = (curr_price - pos['entry_price']) / pos['entry_price']
                if asset in asset_trade_pnls:
                    asset_trade_pnls[asset].append(trade_pnl)
                s = pos['sector']
                std_s = SECTOR_NAME_MAP.get(s, s)
                if std_s not in mtm_sector_pnls:
                    mtm_sector_pnls[std_s] = []
                mtm_sector_pnls[std_s].append(trade_pnl)

        for s, pnls in mtm_sector_pnls.items():
            if s in sector_trade_pnls:
                sector_trade_pnls[s].append(sum(pnls) / len(pnls))

    def calc_cumulative_return(pnls_list):
        if not pnls_list: return 0.0
        cum = 1.0
        for p in pnls_list: cum *= (1 + p)
        return (cum - 1) * 100

    sector_stats = {s: calc_cumulative_return(pnls) for s, pnls in sector_trade_pnls.items() if pnls}
    asset_stats = {a: calc_cumulative_return(pnls) for a, pnls in asset_trade_pnls.items() if pnls}

    nav_df = pd.DataFrame(nav_record).set_index('date')
    return nav_df, logs, sector_stats, asset_stats


# ================= 4. UI 主程序 =================
with st.sidebar:
    st.header("双轨制：指数信号 -> 单品种执行")

    # 设定默认的数据相对路径 (指向 GitHub 仓库中的 data/index_data 和 data/asset_data 文件夹)
    default_index_dir = os.path.join(BASE_DIR, "data", "index_data")
    default_asset_dir = os.path.join(BASE_DIR, "data", "asset_data")

    index_folder = st.text_input("1. 指数数据目录 (相对路径)", value=default_index_dir)
    asset_folder = st.text_input("2. 单品种数据目录 (相对路径)", value=default_asset_dir)
    bench_name_input = st.text_input("基准识别名", value="文华商品")

    col1, col2 = st.columns(2)
    start_d = col1.date_input("开始日期", value=pd.to_datetime("2020-01-01"))
    end_d = col2.date_input("结束日期", value=pd.to_datetime("2026-12-31"))

    st.subheader("🛠️ 因子与交易参数")
    c1, c2 = st.columns(2)
    win_long = c1.number_input("EWMA长期", 10, 100, 40)
    win_short = c2.number_input("EWMA短期", 2, 50, 10)

    # --- 调仓日选择控件 ---
    day_options = {"周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4}
    rebalance_day_name = st.selectbox("🎯 指定调仓日", list(day_options.keys()), index=4)  # 默认周五
    rebalance_weekday = day_options[rebalance_day_name]

    comm_bp = st.number_input("双边换手(bp)", 0.0, 50.0, 0.0)

    st.info("📌 **系统提示**：\n你现在可以自由选择任意交易日调仓啦！如果该日放假，系统会自动寻找前后最近的交易日代替。")
    run_btn = st.button("🚀 运行双轨轮动", type="primary", use_container_width=True)

st.title("板块轮动策略 (任意周频·后复权版)")

if run_btn:
    with st.spinner("正在加载两路数据 (指数信号轨 + 单品种交易轨)..."):
        idx_p, idx_l, idx_o, idx_h, idx_amt, err1 = load_directory_data(index_folder, is_index=True)
        ast_p, ast_l, ast_o, ast_h, ast_amt, err2 = load_directory_data(asset_folder, is_index=False)

    if err1 or err2:
        st.error(f"数据加载错误：\n{err1 or ''}\n{err2 or ''}")
    else:
        params = {
            'start_date': start_d, 'end_date': end_d,
            'commission': comm_bp / 10000.0,
            'win_long': win_long, 'win_short': win_short,
            'bench_name': bench_name_input,
            'rebalance_weekday': rebalance_weekday  # 传入目标调仓日
        }

        with st.spinner("双轨回测撮合中，处理成分股资金分配..."):
            res_nav, res_logs, res_sector_stats, res_asset_stats = run_hybrid_strategy(idx_p, idx_o, idx_h, idx_l,
                                                                                       idx_amt, ast_p, ast_o, params)

        if res_nav.empty:
            st.warning("区间内未产生交易结果，请检查日期范围")
        else:
            tot_ret = res_nav['nav'].iloc[-1] - 1
            days = (res_nav.index[-1] - res_nav.index[0]).days
            ann_ret = (1 + tot_ret) ** (365 / days) - 1 if days > 0 else 0
            peak = res_nav['nav'].cummax()
            max_dd = ((res_nav['nav'] - peak) / peak).min()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总收益", f"{tot_ret * 100:.2f}%")
            col2.metric("年化收益", f"{ann_ret * 100:.2f}%")
            col3.metric("最大回撤", f"{max_dd * 100:.2f}%")
            col4.metric("卡玛比率", f"{ann_ret / abs(max_dd) if max_dd != 0 else 0:.2f}")

            bench_nav_series = None
            actual_bench_name = next((c for c in idx_p.columns if bench_name_input in c), None)
            if actual_bench_name:
                try:
                    bench_slice = idx_p.loc[res_nav.index[0]:res_nav.index[-1], actual_bench_name]
                    bench_nav_series = bench_slice / bench_slice.iloc[0]
                except:
                    pass

            t1, t2, t3 = st.tabs(["📈 净值曲线", "📝 品种调仓日志", "🏆 策略盈亏贡献榜"])

            with t1:
                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                ax1.plot(res_nav.index, res_nav['nav'], color='#d62728', lw=2,
                         label=f'成分股落地策略 (最终: {res_nav["nav"].iloc[-1]:.2f})')
                if bench_nav_series is not None:
                    ax1.plot(bench_nav_series.index, bench_nav_series, color='#1f77b4', lw=1.5, alpha=0.8,
                             label=f'基准 ({actual_bench_name})')
                ax1.fill_between(res_nav.index, res_nav['nav'], 1, color='#d62728', alpha=0.1)
                ax1.legend(prop=my_font)
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig)

            with t2:
                log_text = "\n".join(res_logs) if res_logs else "暂无调仓记录"
                st.text_area(f"详细调仓日志 (已设定为每周 {rebalance_day_name} 调仓)", log_text, height=600)

            with t3:
                all_sectors = list(SECTOR_CONSTITUENTS.keys())
                sector_vals = [res_sector_stats.get(s, 0.0) for s in all_sectors]
                sector_series = pd.Series(sector_vals, index=all_sectors).sort_values(ascending=False)

                fig_sec, ax_sec = plt.subplots(figsize=(10, 4.5))
                colors_sec = ['#d62728' if x > 0 else '#2ca02c' for x in sector_series.values]
                bars_sec = ax_sec.bar(range(len(sector_series)), sector_series.values, color=colors_sec)
                ax_sec.set_ylabel("策略累计收益贡献 (%)", fontproperties=my_font)
                ax_sec.set_xticks(range(len(sector_series)))
                ax_sec.set_xticklabels(sector_series.index, fontproperties=my_font, rotation=0)
                ax_sec.grid(axis='y', alpha=0.3)
                for bar in bars_sec:
                    yval = bar.get_height()
                    offset = 1 if yval >= 0 else -1
                    ax_sec.text(bar.get_x() + bar.get_width() / 2.0, yval + offset, f'{yval:.1f}%', va='center',
                                ha='center', fontproperties=my_font)
                st.pyplot(fig_sec)
