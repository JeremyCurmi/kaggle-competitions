from utils import snake_case_transformation


def test_snake_case_transformation():
    data = [["Test1", "TestTest1", "#T#Test1"], ["test_1", ""]]
    expected = [["test1", "test_test1", "#_t#_test1"], ["test_1", ""]]

    for t, e in zip(data, expected):
        assert snake_case_transformation(t) == e