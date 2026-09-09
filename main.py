import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

FLOOR = 1000  # 예측치가 이보다 작으면 그래프 바닥에 붙여 표시

# 체크박스로 고를 수 있는 변수 목록 (기술 컬럼명 -> 화면에 보여줄 한글 이름)
NUMERIC_FEATURES = {
    "first_scrn": "개봉 첫 관측일 스크린수",
    "first_show": "개봉 첫 관측일 상영횟수",
    "first_week_audi": "개봉 첫 주 관객수",
    "days_in_top10": "박스오피스 10위권 체류일수",
    "peak": "성수기(1·7·12월) 개봉 여부",
    "daily_days": "일별 순위표 등장 일수",
    "daily_avg_audi": "일별 평균 관객수",
    "daily_max_audi": "일별 최고 관객수",
    "daily_avg_rank": "일별 평균 순위",
    "daily_best_rank": "일별 최고 순위(최소값)",
    "daily_avg_screen": "일별 평균 스크린수",
    "daily_max_screen": "일별 최고 스크린수",
    "daily_avg_show": "일별 평균 상영횟수",
    "daily_max_show": "일별 최고 상영횟수",
}
CATEGORICAL_FEATURES = {
    "genre": "장르",
    "nation": "국가",
}
# 기본으로 체크되어 있을 변수
DEFAULT_ON = {"first_week_audi", "days_in_top10", "daily_avg_audi", "peak"}


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    daily = daily.rename(columns={"영화코드": "movieCd"})
    daily["날짜"] = pd.to_datetime(daily["날짜"], format="%Y%m%d")
    return daily, movies


@st.cache_data(show_spinner="영화 정보와 박스오피스 데이터를 합치는 중입니다...")
def build_dataset(daily: pd.DataFrame, movies: pd.DataFrame):
    # 일별 데이터를 영화코드 기준으로 요약(집계)
    agg = daily.groupby("movieCd").agg(
        daily_days=("날짜", "count"),
        daily_avg_audi=("일관객", "mean"),
        daily_max_audi=("일관객", "max"),
        daily_avg_rank=("순위", "mean"),
        daily_best_rank=("순위", "min"),
        daily_avg_screen=("스크린수", "mean"),
        daily_max_screen=("스크린수", "max"),
        daily_avg_show=("상영횟수", "mean"),
        daily_max_show=("상영횟수", "max"),
    ).reset_index()

    # 영화별 표에 있는 영화를 모두 사용 (왼쪽 기준 병합)
    merged = movies.merge(agg, on="movieCd", how="left")

    numeric_cols = list(NUMERIC_FEATURES.keys())
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    for col in CATEGORICAL_FEATURES:
        merged[col] = merged[col].fillna("unknown").astype(str)

    # 영화코드 순으로 정렬 후, 열 편마다 앞의 세 편을 시험용으로 지정
    merged = merged.sort_values("movieCd").reset_index(drop=True)
    merged["position_in_10"] = merged.index % 10
    merged["is_test"] = merged["position_in_10"] < 3

    period_start = daily["날짜"].min().strftime("%Y-%m-%d")
    period_end = daily["날짜"].max().strftime("%Y-%m-%d")

    return merged, period_start, period_end


def build_feature_matrix(df: pd.DataFrame, selected_numeric, selected_categorical):
    parts = []
    if selected_numeric:
        parts.append(df[selected_numeric].reset_index(drop=True))
    if selected_categorical:
        dummies = pd.get_dummies(df[selected_categorical], prefix=selected_categorical)
        parts.append(dummies.reset_index(drop=True))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def main():
    st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
    st.title("🎬 영화 흥행 예측기")
    st.caption("KOBIS 박스오피스 일별 데이터와 영화 정보를 합쳐서, 다중 회귀로 총 관객 수를 예측합니다.")

    daily, movies = load_data()
    merged, period_start, period_end = build_dataset(daily, movies)

    st.subheader("① 예측에 사용할 변수 선택")
    selected_numeric = []
    selected_categorical = []

    all_items = list(NUMERIC_FEATURES.items()) + list(CATEGORICAL_FEATURES.items())
    n_cols = 3
    cols = st.columns(n_cols)
    for i, (col_name, label) in enumerate(all_items):
        with cols[i % n_cols]:
            checked = st.checkbox(label, value=(col_name in DEFAULT_ON), key=f"chk_{col_name}")
            if checked:
                if col_name in NUMERIC_FEATURES:
                    selected_numeric.append(col_name)
                else:
                    selected_categorical.append(col_name)

    if not selected_numeric and not selected_categorical:
        st.warning("변수를 최소 1개 이상 선택해 주세요.")
        st.stop()

    # 학습/시험 분할
    train_df = merged[~merged["is_test"]].reset_index(drop=True)
    test_df = merged[merged["is_test"]].reset_index(drop=True)

    X_all = build_feature_matrix(merged, selected_numeric, selected_categorical)
    X_train = X_all.loc[~merged["is_test"].values].reset_index(drop=True)
    X_test = X_all.loc[merged["is_test"].values].reset_index(drop=True)
    y_train = train_df["total_audi"].values
    y_test = test_df["total_audi"].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("② 학습 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("학습에 쓴 영화 편수", f"{len(train_df):,}편")
    c2.metric("예측 점수를 잰 영화 편수", f"{len(test_df):,}편")
    c3.metric("기준 기간", f"{period_start} ~ {period_end}")

    c4, c5, c6 = st.columns(3)
    c4.metric("예측 점수 (R²)", f"{r2:.3f}")
    c5.metric("평균 오차 (MAE)", f"{mae:,.0f}명")
    c6.metric("평균 오차 (RMSE)", f"{rmse:,.0f}명")

    st.subheader("③ 실제 총 관객 수 vs 예측 총 관객 수 (시험용 영화)")

    actual = y_test.astype(float)
    predicted = y_pred.astype(float)
    actual_safe = np.clip(actual, 1, None)  # 로그축 안전 처리
    is_floored = predicted < FLOOR
    predicted_clamped = np.where(is_floored, FLOOR, predicted)
    predicted_clamped = np.clip(predicted_clamped, 1, None)
    n_floored = int(is_floored.sum())

    names = test_df["movieNm"].values

    low_bound = FLOOR / 2
    high_bound = max(actual_safe.max(), predicted_clamped.max()) * 1.5

    fig = go.Figure()

    normal_mask = ~is_floored
    fig.add_trace(go.Scatter(
        x=actual_safe[normal_mask],
        y=predicted_clamped[normal_mask],
        mode="markers",
        name="예측값",
        marker=dict(color="royalblue", size=8, opacity=0.75),
        text=names[normal_mask],
        hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
    ))

    if n_floored > 0:
        fig.add_trace(go.Scatter(
            x=actual_safe[is_floored],
            y=predicted_clamped[is_floored],
            mode="markers",
            name=f"예측 1,000명 미만 ({n_floored}편, 바닥에 표시)",
            marker=dict(color="crimson", size=9, symbol="triangle-down"),
            text=names[is_floored],
            hovertemplate="%{text}<br>실제: %{x:,.0f}명<br>예측(원래값 1,000명 미만): %{y:,.0f}명<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[low_bound, high_bound],
        y=[low_bound, high_bound],
        mode="lines",
        name="실제 = 예측 기준선",
        line=dict(color="gray", dash="dash"),
    ))

    fig.update_xaxes(type="log", title="실제 총 관객 수 (명, 로그축)",
                      range=[np.log10(low_bound), np.log10(high_bound)])
    fig.update_yaxes(type="log", title="예측 총 관객 수 (명, 로그축)",
                      range=[np.log10(low_bound), np.log10(high_bound)])
    fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02))

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"⚠️ 예측이 1,000명보다 작게 나온 영화는 그래프 바닥(1,000명 위치)에 붙여 표시했습니다. 해당 영화: **{n_floored}편**")

    with st.expander("시험용 영화 상세 결과 보기"):
        detail = pd.DataFrame({
            "영화명": names,
            "실제 총 관객 수": actual.astype(int),
            "예측 총 관객 수": predicted.astype(int),
            "오차(예측-실제)": (predicted - actual).astype(int),
        })
        st.dataframe(detail, use_container_width=True)


if __name__ == "__main__":
    main()
