import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "overtime": ["Yes", "No", "Yes", "No"],
            "monthly_income": [4000, 6000, 5000, 7000],
            "job_satisfaction": [1, 3, 2, 4],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_mixed(basic_df):
    assert attrition_rate(basic_df) == 50.0


def test_attrition_rate_all_leavers(basic_df):
    basic_df["attrition"] = "Yes"
    assert attrition_rate(basic_df) == 100.0


def test_attrition_rate_no_leavers(basic_df):
    basic_df["attrition"] = "No"
    assert attrition_rate(basic_df) == 0.0


def test_attrition_rate_empty_dataframe():
    empty = pd.DataFrame(columns=["employee_id", "attrition"])
    assert attrition_rate(empty) == 0.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_columns(basic_df):
    result = attrition_by_department(basic_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(basic_df):
    result = attrition_by_department(basic_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending(basic_df):
    # Sales: 1/2 = 50%, HR: 1/2 = 50% — add a third dept with 0% to check sort
    extra = pd.DataFrame(
        {
            "employee_id": [5, 6],
            "department": ["IT", "IT"],
            "overtime": ["No", "No"],
            "monthly_income": [8000, 9000],
            "job_satisfaction": [3, 4],
            "attrition": ["No", "No"],
        }
    )
    df = pd.concat([basic_df, extra], ignore_index=True)
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


def test_attrition_by_department_zero_leavers():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "department": ["IT", "IT"],
            "attrition": ["No", "No"],
        }
    )
    result = attrition_by_department(df)
    assert result.iloc[0]["attrition_rate"] == 0.0


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_columns(basic_df):
    result = attrition_by_overtime(basic_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(basic_df):
    result = attrition_by_overtime(basic_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    # Employees 1 and 3 work overtime, both left → 100%
    assert yes_row["employees"] == 2
    assert yes_row["leavers"] == 2
    assert yes_row["attrition_rate"] == 100.0
    # Employees 2 and 4 do not work overtime, none left → 0%
    assert no_row["employees"] == 2
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_columns(basic_df):
    result = average_income_by_attrition(basic_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(basic_df):
    result = average_income_by_attrition(basic_df)
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    # Leavers: employees 1 (4000) and 3 (5000) → mean 4500
    assert yes_avg == 4500.0
    # Stayers: employees 2 (6000) and 4 (7000) → mean 6500
    assert no_avg == 6500.0


def test_average_income_leavers_earn_less_than_stayers(basic_df):
    result = average_income_by_attrition(basic_df)
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_avg < no_avg


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_columns(basic_df):
    result = satisfaction_summary(basic_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_attrition_rate_per_group(basic_df):
    result = satisfaction_summary(basic_df)
    # Score 1: employee 1, left → 1/1 = 100%
    row1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert row1["total_employees"] == 1
    assert row1["leavers"] == 1
    assert row1["attrition_rate"] == 100.0
    # Score 3: employee 2, stayed → 0/1 = 0%
    row3 = result[result["job_satisfaction"] == 3].iloc[0]
    assert row3["total_employees"] == 1
    assert row3["leavers"] == 0
    assert row3["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_ascending(basic_df):
    result = satisfaction_summary(basic_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_rate_not_share_of_total_leavers(basic_df):
    # Regression test for the fixed denominator bug.
    # Before the fix, attrition_rate was leavers / total_leavers * 100.
    # With 2 total leavers (employees 1 and 3), score-1 would show 50.0, not 100.0.
    result = satisfaction_summary(basic_df)
    row1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert row1["attrition_rate"] == 100.0
