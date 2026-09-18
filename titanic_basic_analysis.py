import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 데이터 불러오기
# =========================================================

file_path = "data/titanic/train_and_test2.csv"

df = pd.read_csv(file_path)


# =========================================================
# 2. 데이터 기본 정보 확인
# =========================================================

print("=" * 60)
print("1. 데이터 기본 정보")
print("=" * 60)

print("\n데이터 크기:")
print(df.shape)

print("\n컬럼 이름:")
print(df.columns.tolist())

print("\n앞의 5개 데이터:")
print(df.head())

print("\n데이터 타입:")
print(df.dtypes)


# =========================================================
# 3. 결측치 확인
# =========================================================

print("\n" + "=" * 60)
print("2. 결측치 확인")
print("=" * 60)

missing_count = df.isnull().sum()

missing_percent = (
    df.isnull().sum() / len(df) * 100
)

missing_df = pd.DataFrame({
    "결측치 개수": missing_count,
    "결측치 비율(%)": missing_percent
})

# 결측치가 있는 컬럼만 출력
missing_df = missing_df[missing_df["결측치 개수"] > 0]

print(missing_df)


# =========================================================
# 4. 결측치 시각화
# =========================================================

if len(missing_df) > 0:

    plt.figure(figsize=(10, 5))

    plt.bar(
        missing_df.index,
        missing_df["결측치 개수"]
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Number of Missing Values")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

else:
    print("\n결측치가 없습니다.")


# =========================================================
# 5. 중복값 확인
# =========================================================

print("\n" + "=" * 60)
print("3. 중복값 확인")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("전체 행 수 :", len(df))
print("중복 행 수 :", duplicate_count)


# =========================================================
# 6. 중복값 시각화
# =========================================================

normal_count = len(df) - duplicate_count

plt.figure(figsize=(6, 5))

plt.bar(
    ["Normal", "Duplicate"],
    [normal_count, duplicate_count]
)

plt.title("Duplicate Rows")
plt.ylabel("Number of Rows")

plt.tight_layout()

plt.show()


# =========================================================
# 7. 숫자형 데이터 확인
# =========================================================

print("\n" + "=" * 60)
print("4. 숫자형 변수 확인")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(numeric_columns.tolist())


# =========================================================
# 8. 이상치 확인 - IQR 방법
# =========================================================

print("\n" + "=" * 60)
print("5. 이상치 확인 (IQR 방법)")
print("=" * 60)

outlier_result = {}

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    ]

    outlier_result[column] = len(outliers)


outlier_df = pd.DataFrame.from_dict(
    outlier_result,
    orient="index",
    columns=["이상치 개수"]
)

outlier_df = outlier_df.sort_values(
    "이상치 개수",
    ascending=False
)

print(outlier_df)


# =========================================================
# 9. 이상치 개수 시각화
# =========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    outlier_df.index,
    outlier_df["이상치 개수"]
)

plt.title("Number of Outliers by Column")
plt.xlabel("Column")
plt.ylabel("Number of Outliers")

plt.xticks(rotation=60)

plt.tight_layout()

plt.show()


# =========================================================
# 10. 숫자형 변수 Box Plot
# =========================================================

for column in numeric_columns:

    plt.figure(figsize=(6, 3))

    plt.boxplot(
        df[column].dropna(),
        vert=False
    )

    plt.title(f"Boxplot - {column}")

    plt.xlabel(column)

    plt.tight_layout()

    plt.show()
