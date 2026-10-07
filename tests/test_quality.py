import pytest

from bench.quality import agreement, extract_answer, gsm8k_score, normalize_number


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so 3 + 4 = 7.\nThe answer is 7", "7"),
        ("The answer is: $1,250.", "1250"),
        ("first 5 then 6. The answer is 12.50", "12.5"),
        ("We get \\boxed{42} in the end, 3 apples", "42"),
        ("no explicit phrase, total 18 dollars", "18"),
        ("The answer is -3", "-3"),
        ("nothing numeric", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_normalize_number():
    assert normalize_number("1,000.00") == "1000"
    assert normalize_number("0.50") == "0.5"
    assert normalize_number("7.") == "7"


def test_gsm8k_score():
    rows = [{"id": "a", "answer": "7"}, {"id": "b", "answer": "1000"}, {"id": "c", "answer": "3"}]
    outputs = {"a": "The answer is 7", "b": "The answer is 1,000", "c": "The answer is 4"}
    s = gsm8k_score(rows, outputs)
    assert s == {"n": 3, "correct": 2, "missing": 0, "accuracy": pytest.approx(2 / 3)}
    assert gsm8k_score(rows, {"a": "The answer is 7"})["missing"] == 2


def test_agreement():
    ref = {"1": "hello world", "2": "abcd", "3": "only in ref"}
    cand = {"1": "hello world", "2": "abXX", "4": "only in cand"}
    a = agreement(cand, ref)
    assert a["n"] == 2
    assert a["exact_match_rate"] == 0.5
    assert a["mean_prefix_fraction"] == pytest.approx((1.0 + 0.5) / 2)
    assert agreement({}, ref)["exact_match_rate"] is None
