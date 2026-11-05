import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_colwidth', None) # Show full cell contents
pd.set_option('display.width', None)        # Adjust display width
#block 0
# Load the original dataset without automatically interpreting
# strings such as "NULL" as missing values.
df = pd.read_csv(
    "C:\\Users\\malak\\Downloads\\H2.csv",
    keep_default_na=False,
    low_memory=False
)

# Keep the original dataset unchanged throughout the audit.
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
#block 1
# Count actual pandas missing values.
actual_missing = df.isna().sum()

# Count empty or whitespace-only strings.
blank_strings = pd.Series({
    col: df[col].astype(str).str.strip().eq("").sum()
    for col in df.columns
})

# Count common missing-value placeholders.
null_tokens = {"NULL", "NA", "N/A", "NONE", "NAN", "<NA>"}

hidden_nulls = pd.Series({
    col: df[col].astype(str).str.strip().str.upper()
          .isin(null_tokens).sum()
    for col in df.columns
})

missing_audit = pd.DataFrame({
    "Actual_Missing": actual_missing,
    "Blank_Strings": blank_strings,
    "Hidden_Null_Tokens": hidden_nulls
})

missing_audit["Total_Suspicious"] = (
    missing_audit["Actual_Missing"]
    + missing_audit["Blank_Strings"]
    + missing_audit["Hidden_Null_Tokens"]
)

missing_audit = missing_audit.sort_values(
    "Total_Suspicious",
    ascending=False
)

print(missing_audit[missing_audit["Total_Suspicious"] > 0])
#block 1B
columns_to_inspect = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate"
]

for col in columns_to_inspect:
    print(f"\n{'=' * 50}")
    print(f"Column: {col}")
    print(f"{'=' * 50}")

    values = df[col].astype(str).str.strip()

    print("Number of distinct values:", values.nunique())

    print("\nMost frequent values:")
    print(values.value_counts(dropna=False).head(15))
#block 2
# Identify rows that are exact copies of earlier rows.
duplicate_mask = df.duplicated(keep="first")

print("Total rows:", len(df))
print("Exact duplicate rows:", duplicate_mask.sum())

print(
    "Percentage of rows that are duplicates:",
    round(duplicate_mask.mean() * 100, 2),
    "%"
)

print(
    "Rows remaining if all later exact copies were removed:",
    len(df) - duplicate_mask.sum()
)
#block 2B
# Retrieve all copies belonging to duplicated groups.
all_duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df.loc[all_duplicate_mask].copy()

print("Rows belonging to duplicate groups:", len(duplicate_rows))

print(duplicate_rows.head(20))
#block 2C
comparison = pd.DataFrame({
    "Group": ["All rows", "Duplicate copies", "Non-duplicate rows"],
    "Rows": [
        len(df),
        duplicate_mask.sum(),
        (~all_duplicate_mask).sum()
    ],
    "Mean_ADR": [
        pd.to_numeric(df["ADR"], errors="coerce").mean(),
        pd.to_numeric(
            df.loc[duplicate_mask, "ADR"],
            errors="coerce"
        ).mean(),
        pd.to_numeric(
            df.loc[~all_duplicate_mask, "ADR"],
            errors="coerce"
        ).mean()
    ],
    "Median_ADR": [
        pd.to_numeric(df["ADR"], errors="coerce").median(),
        pd.to_numeric(
            df.loc[duplicate_mask, "ADR"],
            errors="coerce"
        ).median(),
        pd.to_numeric(
            df.loc[~all_duplicate_mask, "ADR"],
            errors="coerce"
        ).median()
    ]
})
print(comparison)
#block 3A
# Convert the guest-count columns temporarily to numeric values.
guest_columns = ["Adults", "Children", "Babies"]

guest_counts = df[guest_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Identify records with zero recorded guests.
no_guests_mask = guest_counts.sum(axis=1).eq(0)

print("Bookings with zero recorded guests:", no_guests_mask.sum())

# print(
#     df.loc[no_guests_mask].head(20)
# )
#block 3B
children_without_adults_mask = (
    guest_counts["Adults"].eq(0)
    & (
        guest_counts["Children"].gt(0)
        | guest_counts["Babies"].gt(0)
    )
)

print(
    "Bookings with children/babies but no adults:",
    children_without_adults_mask.sum()
)

# print(
#     df.loc[children_without_adults_mask].head(20)
# )
#block 3C
negative_guests_mask = guest_counts.lt(0).any(axis=1)

print("Rows with negative guest counts:", negative_guests_mask.sum())

# print(
#     df.loc[negative_guests_mask]
# )
#block 3D
# Convert ADR to numeric temporarily.
adr = pd.to_numeric(df["ADR"], errors="coerce")

# Calculate the number of nights.
weekend_nights = pd.to_numeric(
    df["StaysInWeekendNights"],
    errors="coerce"
)

weekday_nights = pd.to_numeric(
    df["StaysInWeekNights"],
    errors="coerce"
)

total_nights = weekend_nights + weekday_nights

zero_adr_mask = adr.eq(0)

print("Rows with ADR = 0:", zero_adr_mask.sum())

print(
    "Zero-ADR rows with positive nights:",
    (zero_adr_mask & total_nights.gt(0)).sum()
)

print(
    "Zero-ADR rows with zero nights:",
    (zero_adr_mask & total_nights.eq(0)).sum()
)

print(
    "Zero-ADR rows with negative nights:",
    (zero_adr_mask & total_nights.lt(0)).sum()
)

print(
    "Zero-ADR rows with missing ADR or nights:",
    (adr.isna() | total_nights.isna()).sum()
)

# print(
#     df.loc[zero_adr_mask].head(20)
# )
#block 3E
zero_nights_mask = total_nights.eq(0)

print("Rows with zero total nights:", zero_nights_mask.sum())

print(
    "Zero-night rows with ADR = 0:",
    (zero_nights_mask & adr.eq(0)).sum()
)

print(
    "Zero-night rows with positive ADR:",
    (zero_nights_mask & adr.gt(0)).sum()
)

print(
    "Zero-night rows with negative ADR:",
    (zero_nights_mask & adr.lt(0)).sum()
)

# print(
#     df.loc[zero_nights_mask].head(20)
# )
#block 3F
nonnegative_columns = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
]

for col in nonnegative_columns:
    values = pd.to_numeric(df[col], errors="coerce")

    negative_mask = values.lt(0)

    print(f"{col}: {negative_mask.sum()} negative values")

    if negative_mask.any():
        print(df.loc[negative_mask, [col, "ADR"]].head(10))

#block 4A
print("ADR summary statistics:")
print(adr.describe(percentiles=[
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99
]))

print("\nADR skewness:", adr.skew())

print("\nADR quantiles:")
print(
    adr.quantile([
        0,
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        1.00
    ]).to_frame("ADR")
)
#block 4B
Q1 = adr.quantile(0.25)
Q3 = adr.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outlier_mask = (
    adr.lt(lower_bound)
    | adr.gt(upper_bound)
)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

print("Number of IQR outliers:", iqr_outlier_mask.sum())

# print(
#     df.loc[iqr_outlier_mask]
#       .assign(ADR_numeric=adr.loc[iqr_outlier_mask])
#       .sort_values("ADR_numeric", ascending=False)
#       .head(20)
# )
#block 4C
numeric_columns = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ADR"
]

summary = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
).describe().T

summary["Skewness"] = df[numeric_columns].apply(
    lambda col: pd.to_numeric(
        col,
        errors="coerce"
    ).skew()
)
print(summary)
#block 5
categorical_columns = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
    "ReservationStatus",
    "PreviousBookingsNotCanceled"
]

# Create a temporary analysis copy.
categorical_audit = df[categorical_columns].copy()

for col in categorical_columns:
    categorical_audit[col] = (
        categorical_audit[col]
        .astype(str)
        .str.strip()
    )

print("Distinct category counts:")

print(
    categorical_audit.nunique(dropna=False)
      .sort_values(ascending=False)
      .to_frame("Distinct_Categories")
)

for col in categorical_columns:
    print(f"\n{'=' * 60}")
    print(f"Column: {col}")
    print(f"{'=' * 60}")

    # Count the frequency of every category
    counts = categorical_audit[col].value_counts(dropna=False)

    # Calculate the percentage of each category
    percentages = (
        categorical_audit[col]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    # Combine the results into one table
    category_audit = pd.DataFrame({
        "Frequency": counts,
        "Percentage": percentages
    })

    # Display every category, from most frequent to least frequent
    print(category_audit)
# #block 6A
dtype_audit = pd.DataFrame({
    "Data_Type": df.dtypes.astype(str),
    "Unique_Values": df.nunique(dropna=False),
    "Missing_Values": df.isna().sum()
})

print(dtype_audit)
# #block 6B
expected_numeric_columns = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
]

for col in expected_numeric_columns:
    converted = pd.to_numeric(df[col], errors="coerce")

    original_values = df[col].astype(str).str.strip()

    # Values that are non-empty but cannot be converted to numbers.
    invalid_mask = (
        original_values.ne("")
        & converted.isna()
    )

    print(f"\nColumn: {col}")
    print("Invalid numeric values:", invalid_mask.sum())

    if invalid_mask.any():
        print("Examples:")
        print(df.loc[invalid_mask, [col]].head(10))
# #block 6C
binary_columns = [
    "IsCanceled",
    "IsRepeatedGuest"
]

for col in binary_columns:
    values = pd.to_numeric(df[col], errors="coerce")

    invalid_mask = ~values.isin([0, 1])

    print(f"\n{col}")
    print("Value counts:")
    print(values.value_counts(dropna=False))

    print("Invalid binary values:", invalid_mask.sum())

    if invalid_mask.any():
        print(df.loc[invalid_mask, [col]])
#block 6D
status_dates = pd.to_datetime(
    df["ReservationStatusDate"].astype(str).str.strip(),
    errors="coerce"
)

invalid_dates_mask = status_dates.isna()

print("Unparseable reservation status dates:", invalid_dates_mask.sum())

print("\nExamples of invalid dates:")
print(
    df.loc[invalid_dates_mask, ["ReservationStatusDate"]].head(20)
)

print("\nEarliest parsed date:", status_dates.min())
print("Latest parsed date:", status_dates.max())
#block 7A
print("block 7")
numeric_features = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
]

numeric_data = df[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

numeric_data["ADR"] = adr

pearson_corr = numeric_data.corr(method="pearson")["ADR"].drop("ADR")

spearman_corr = numeric_data.corr(method="spearman")["ADR"].drop("ADR")

correlation_audit = pd.DataFrame({
    "Pearson_Correlation": pearson_corr,
    "Spearman_Correlation": spearman_corr
})

correlation_audit["Abs_Pearson"] = (
    correlation_audit["Pearson_Correlation"].abs()
)

correlation_audit["Abs_Spearman"] = (
    correlation_audit["Spearman_Correlation"].abs()
)

correlation_audit = correlation_audit.sort_values(
    "Abs_Spearman",
    ascending=False
)

print(correlation_audit)
#block 7B
categorical_features = [
    "MarketSegment",
    "DistributionChannel",
    "Meal",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "CustomerType"
]

for col in categorical_features:
    print(f"\n{'=' * 60}")
    print(f"Feature: {col}")
    print(f"{'=' * 60}")

    temp = pd.DataFrame({
        "Category": (
            df[col]
            .astype(str)
            .str.strip()
        ),
        "ADR": adr
    })

    stats = (
        temp.groupby("Category")["ADR"]
        .agg(
            Count="count",
            Mean_ADR="mean",
            Median_ADR="median",
            Std_ADR="std"
        )
        .sort_values("Median_ADR", ascending=False)
    )

    print(stats)
#block 7C
stay_analysis = pd.DataFrame({
    "Weekend_Nights": weekend_nights,
    "Weekday_Nights": weekday_nights,
    "Total_Nights": total_nights,
    "ADR": adr
})

stay_analysis = stay_analysis.dropna(
    subset=["Total_Nights", "ADR"]
)

stay_summary = (
    stay_analysis.groupby("Total_Nights")["ADR"]
    .agg(
        Count="count",
        Mean_ADR="mean",
        Median_ADR="median"
    )
    .sort_index()
)

print(stay_summary.head(30))
#block 7D
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

month_analysis = pd.DataFrame({
    "ArrivalMonth": (
        df["ArrivalDateMonth"]
        .astype(str)
        .str.strip()
    ),
    "ADR": adr
})

monthly_adr = (
    month_analysis.groupby("ArrivalMonth")["ADR"]
    .agg(
        Count="count",
        Mean_ADR="mean",
        Median_ADR="median",
        Std_ADR="std"
    )
)

monthly_adr = monthly_adr.reindex(month_order)

print(monthly_adr)
#block 8A
leakage_columns = [
    "ReservationStatus",
    "ReservationStatusDate"
]

for col in leakage_columns:
    print(f"\n{'=' * 60}")
    print(f"Column: {col}")
    print(f"{'=' * 60}")

    print("Distinct values:", df[col].nunique(dropna=False))

    print("\nMost frequent values:")
    print(df[col].value_counts(dropna=False).head(20))
#block 8B
status_adr = pd.DataFrame({
    "ReservationStatus": (
        df["ReservationStatus"]
        .astype(str)
        .str.strip()
    ),
    "ADR": adr
})

status_summary = (
    status_adr.groupby("ReservationStatus")["ADR"]
    .agg(
        Count="count",
        Mean_ADR="mean",
        Median_ADR="median"
    )
    .sort_values("Count", ascending=False)
)

print(status_summary)
#block 9
adr_audit = pd.DataFrame({
    "ADR": adr
})

print("Total rows:", len(adr_audit))

print("Missing or unparseable ADR:", adr.isna().sum())

print("Negative ADR:", adr.lt(0).sum())

print("Zero ADR:", adr.eq(0).sum())

print("Positive ADR:", adr.gt(0).sum())

print("\nADR summary:")
print(adr.describe())

print("\nADR value counts for the most frequent values:")
print(adr.value_counts().head(20))
#block 10
columns_to_check = [
    "Children",
    "Country",
    "Agent",
    "Company"
]

missingness_results = []

for col in columns_to_check:

    values = df[col].astype(str).str.strip()

    is_missing = (
        values.eq("")
        | values.str.upper().isin(
            {"NULL", "NA", "N/A", "NONE", "NAN", "<NA>"}
        )
    )

    temp = pd.DataFrame({
        "Missingness": np.where(
            is_missing,
            "Missing",
            "Present"
        ),
        "ADR": adr
    })

    summary = (
        temp.groupby("Missingness")["ADR"]
        .agg(
            Count="count",
            Mean_ADR="mean",
            Median_ADR="median"
        )
        .reset_index()
    )

    summary["Feature"] = col

    missingness_results.append(summary)

missingness_report = pd.concat(
    missingness_results,
    ignore_index=True
)

print(
    missingness_report[
        ["Feature", "Missingness", "Count", "Mean_ADR", "Median_ADR"]
    ]
)

# Investigate records where the reserved or assigned room type is P.

room_p_mask = (
    categorical_audit["ReservedRoomType"].eq("P")
    | categorical_audit["AssignedRoomType"].eq("P")
)

columns_to_inspect = [
    "ADR",
    "Adults",
    "Children",
    "Babies",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "ReservedRoomType",
    "AssignedRoomType",
    "MarketSegment",
    "CustomerType",
    "DepositType"
]
print(df.loc[room_p_mask, columns_to_inspect])