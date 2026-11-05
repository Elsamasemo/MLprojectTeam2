from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"
OUT_DIR = PROJECT_ROOT / "outputs"

SEED = 42
TARGET = "IsCanceled"

EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]


ID_AND_TEXT_COLUMNS = ["Children", "Country", "Agent", "Company"]

PLACEHOLDERS = {
    "Children": {"NA"},
    "Country": {"NULL"},
    "Agent": {"NULL"},
    "Company": {"NULL"},
}
NULL_TOKENS = {"NULL", "NA", "N/A", "NONE", "NAN", "<NA>"}

MODE_COLUMNS = ["MarketSegment", "DistributionChannel", "Country"]
NO_AGENT = "No Agent"

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_NUMBER = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

NUMERIC_FOR_SUMMARY = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
    "Babies", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "BookingChanges", "DaysInWaitingList", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ADR",
]
CATEGORICAL_COLUMNS = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "AssignedRoomType", "DepositType", "Agent", "Company",
    "CustomerType", "ReservationStatus",
]


DIRECT_LEAKAGE = ["ReservationStatus", "ReservationStatusDate"]
INDIRECT_LEAKAGE = ["AssignedRoomType", "BookingChanges", "DaysInWaitingList"]
BORDERLINE = ["ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]


CANDIDATE_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights", "Adults",
    "Children", "Babies", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ReservedRoomType", "DepositType", "Agent", "Company", "CustomerType",
]


DROP_EXACT_DUPLICATES = True
DROP_BORDERLINE = True
DROP_NO_GUEST_ROWS = True

TEST_SIZE = 0.20
N_SPLITS = 5
COUNTRY_THRESHOLDS = [100, 1000]
SEASONS = {
    "December": "winter", "January": "winter", "February": "winter",
    "March": "spring", "April": "spring", "May": "spring",
    "June": "summer", "July": "summer", "August": "summer",
    "September": "fall", "October": "fall", "November": "fall",
}
WEAK_GROUP = ["ArrivalDateDayOfMonth", "ArrivalDateWeekNumber", "PreviousBookingsNotCanceled"]
DECISION_THRESHOLD = 0.5