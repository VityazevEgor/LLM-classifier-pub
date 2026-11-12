from llm_classifier.parse import parse_response, vote


def test_valid_json():
    assert parse_response('{"label": 3, "explanation": "ok"}') == (3, "ok")


def test_markdown_and_text():
    c = 'Рассуждаю...\n```json\n{"label": 4, "explanation": "hostPID"}\n```\nГотово'
    assert parse_response(c) == (4, "hostPID")


def test_invalid_cases():
    for c in [None, "", "мусор", '{"label": 7, "explanation": "x"}', '{"label": "2", "explanation": "x"}',
              '{"label": true, "explanation": "x"}', '{"label": 2, "explanation": "  "}', '{"label": 2}',
              '{"label": 2, "explanation": "x"', "[1,2,3]"]:
        assert parse_response(c) == (0, None), c


def test_vote():
    assert vote([2, 3, 2]) == 2
    assert vote([1, 2, 3]) == 1
    assert vote([0, 0, 3]) == 0
    assert vote([4, 4, 4]) == 4
