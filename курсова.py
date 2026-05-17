import os
import ssl
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
ssl._create_default_https_context = ssl._create_unverified_context

DATASET_NAME = "computingvictor/transactions-fraud-datasets"
AGE_BINS = [0, 18, 25, 35, 50, 120]
AGE_LABELS = ["<18", "18-25", "25-35", "35-50", "50+"]
WEEKDAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def extract_and_transform():
    print("[1/2] Завантаження та очищення повного масиву даних...")
    path = kagglehub.dataset_download(DATASET_NAME)

    df_users = pd.read_csv(os.path.join(path, "users_data.csv"))
    
    df_tx = pd.read_csv(
        os.path.join(path, "transactions_data.csv"), 
        usecols=lambda x: x.strip() in ["client_id", "date", "amount", "merchant_city"],
        low_memory=False
    )
    df_users.columns = df_users.columns.str.strip()
    df_tx.columns = df_tx.columns.str.strip()

    if "id" in df_users.columns: df_users.rename(columns={"id": "User_id"}, inplace=True)
    if "client_id" in df_tx.columns: df_tx.rename(columns={"client_id": "User_id"}, inplace=True)

    df_users["User_id"] = df_users["User_id"].astype(str)
    df_tx["User_id"] = df_tx["User_id"].astype(str)

    df = df_tx.merge(df_users, on="User_id", how="left")

    df = df_tx.merge(df_users, on="User_id", how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["amount"].dtype == object:
        df["amount"] = df["amount"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)

    # Категоризація за реальними сумами
    
    bins = [0, 20, 50, 100, float('inf')]
    labels = ["До $20", "$20 - $50", "$50 - $100", "Понад $100"]
    df["tx_quality"] = pd.cut(df["amount"], bins=bins, labels=labels, right=False)

    city_col = "merchant_city" if "merchant_city" in df.columns else "city"
    if city_col in df.columns:
        df[city_col] = df[city_col].fillna("Unknown")
        if city_col != "city": df.rename(columns={city_col: "city"}, inplace=True)

    age_col = "current_age" if "current_age" in df.columns else "age"
    if age_col in df.columns:
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce').fillna(30)
        df["age_group"] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    gender_cols = [c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
    if gender_cols:
        df['gender'] = df[gender_cols[0]].fillna('Unknown')

    print(f"  ✔ Готово. Оброблено рядків: {len(df):,}")
    return df

def prepare_analytics(df):
    user_col = "User_ID" if "User_ID" in df.columns else "User_id"
    unique_users_df = df.drop_duplicates(subset=[user_col])
    
    analysis_date = df['date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby(user_col).agg(
        Recency=('date', lambda x: (analysis_date - x.max()).days),
        Frequency=('amount', 'count'),
        Monetary=('amount', 'sum')
    ).reset_index()

    r_med, f_med = rfm['Recency'].median(), rfm['Frequency'].median()

    def get_segment(row):
        if row['Recency'] <= r_med and row['Frequency'] > f_med: return "VIP (Активні)"
        elif row['Recency'] <= r_med and row['Frequency'] <= f_med: return "Перспективні"
        elif row['Recency'] > r_med and row['Frequency'] > f_med: return "Сплячі (Лояльні)"
        else: return "Втрачені (Ризик)"
            
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    
    return unique_users_df, rfm

def render_dashboards(df, unique_users_df, rfm):
    print("[2/2] Генерація Дашбордів...")
    sns.set_theme(style="whitegrid", palette="Blues_d")
    
    total_tx = len(df)
    total_revenue = df["amount"].sum()
    unique_users = len(unique_users_df)

    #  ДАШБОРД 1 
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.canvas.manager.set_window_title('Дашборд 1: Профіль Клієнтської Бази')
    fig1.suptitle("Частина 1: Загальний Аналіз Структури Транзакцій та Клієнтів", fontsize=18, fontweight="bold", y=0.96)

    # 1.1 Стать (Donut)
    ax_gender = axes1[0, 0]
    gender_counts = unique_users_df["gender"].value_counts()
    ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, 
                  wedgeprops=dict(width=0.45, edgecolor='white'))
    ax_gender.set_title("Розподіл за статтю", fontweight="bold")

    # 1.2 KPI BOARD
    ax_kpi = axes1[0, 1]
    ax_kpi.axis('off')
    kpi_text = (
        f"ЦЕНТРАЛЬНІ ПОКАЗНИКИ\n\n"
        f"Клієнти: {unique_users:,}\n"
        f"Транзакції: {total_tx:,}\n"
        f"Обіг: ${total_revenue:,.0f}\n"
        f"Сер. чек: ${df['amount'].mean():.2f}"
    )
    ax_kpi.text(0.5, 0.5, kpi_text, fontsize=15, va='center', ha='center', fontweight='bold',
                bbox=dict(facecolor='#f8fbff', edgecolor='#004c99', boxstyle='round,pad=1.2'))

    # 1.3 ТОП міст
    ax_cities = axes1[0, 2]
    top_cities = df[df["city"].str.upper() != "ONLINE"]["city"].value_counts().head(10)
    bars_c = ax_cities.barh(top_cities.index[::-1], top_cities.values[::-1], color="#2b7fb8")
    ax_cities.set_title("Географія: ТОП-10 міст", fontweight="bold")

    # 1.4 Розподіл чеків за реальними сумами
    ax_qual = axes1[1, 0]
    qual_labels = ["До $20", "$20 - $50", "$50 - $100", "Понад $100"]
    qual_counts = df["tx_quality"].value_counts().reindex(qual_labels)
    bars_q = sns.barplot(x=qual_counts.index, y=qual_counts.values, ax=ax_qual, palette="Blues")
    ax_qual.set_title("Розподіл транзакцій за розміром чека", fontweight="bold")
    ax_qual.set_xlabel("") 
    ax_qual.set_ylabel("")

    # 1.5 Кількість клієнтів за віком
    ax_age = axes1[1, 1]
    age_dist = unique_users_df["age_group"].value_counts().sort_index()
    sns.barplot(x=age_dist.index.astype(str), y=age_dist.values, ax=ax_age, color="#41b6c4")
    ax_age.set_title("Кількість клієнтів за віком", fontweight="bold")

    # 1.6 Середній чек за віком
    ax_avg_age = axes1[1, 2]
    avg_age_val = df.groupby("age_group", observed=False)["amount"].mean().reset_index()
    sns.barplot(data=avg_age_val, x="age_group", y="amount", ax=ax_avg_age, palette="GnBu")
    ax_avg_age.set_title("Середній чек за віком", fontweight="bold")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # ДАШБОРД 2 
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    fig2.canvas.manager.set_window_title('Дашборд 2: RFM Сегментація')
    fig2.suptitle("Частина 2: RFM Сегментація та Клієнтська Поведінка", fontsize=18, fontweight="bold", y=0.96)

    rfm_colors = {'VIP (Активні)': '#084594', 'Перспективні': '#2171b5', 'Сплячі (Лояльні)': '#6baed6', 'Втрачені (Ризик)': '#c6dbef'}
    seg_counts = rfm['Segment'].value_counts()
    seg_money = rfm.groupby('Segment')['Monetary'].sum().reindex(seg_counts.index)
    seg_avg_check = rfm.groupby('Segment').apply(lambda x: x['Monetary'].sum() / x['Frequency'].sum()).reindex(seg_counts.index)
    seg_freq = rfm.groupby('Segment')['Frequency'].mean().reindex(seg_counts.index)

    # 2.1 RFM Дохід
    ax_rev = axes2[0, 0]
    bars_rev = ax_rev.barh(seg_money.index[::-1], seg_money.values[::-1], color="#084594", alpha=0.8)
    ax_rev.set_title("Сумарний дохід від сегментів", fontweight="bold")

    # 2.2 Online vs Offline
    ax_donut = axes2[0, 1]
    is_online = df["city"].str.upper() == "ONLINE"
    online_offline = pd.Series({"Online": is_online.sum(), "Офлайн": (~is_online).sum()})
    ax_donut.pie(online_offline, labels=online_offline.index, autopct='%1.1f%%', 
                 startangle=140, wedgeprops=dict(width=0.4, edgecolor='white'))
    ax_donut.set_title("Тип операцій: Online vs Офлайн", fontweight="bold")

    # 2.3 Середній чек по RFM
    ax_check = axes2[1, 0]
    sns.barplot(x=seg_avg_check.index, y=seg_avg_check.values, ax=ax_check, palette="Blues_r")
    ax_check.set_title("Середній чек сегмента ($)", fontweight="bold")
    ax_check.tick_params(axis='x', rotation=15)

    # 2.4 Частота покупок
    ax_freq = axes2[1, 1]
    ax_freq.vlines(x=seg_freq.index, ymin=0, ymax=seg_freq.values, color='#084594', linewidth=3, alpha=0.6)
    ax_freq.plot(seg_freq.index, seg_freq.values, "o", markersize=12, color='#084594')
    ax_freq.set_title("Середня кількість покупок на клієнта", fontweight="bold")
    ax_freq.tick_params(axis='x', rotation=15)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    df_clean = extract_and_transform()
    unique_users_df, rfm_data = prepare_analytics(df_clean)
    render_dashboards(df_clean, unique_users_df, rfm_data)