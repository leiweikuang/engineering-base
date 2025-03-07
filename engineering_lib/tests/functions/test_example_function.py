# This is an example test file for the example_function.py file

from engineering_lib.functions.example_function import calc_bearing_pressure


def test_calc_bearing_pressure():
    # Test case 1: Cohesion = 50 kPa
    input_cohesion = 50
    expected_output = 257.0
    output = calc_bearing_pressure(input_cohesion)

    assert output == expected_output


if __name__ == "__main__":
    test_calc_bearing_pressure()
