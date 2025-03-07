# This script call a function and runs in a workflow.
import pandas as pd
from engineering_lib.functions import example_function

if __name__ == "__main__":
    # Example function call
    cohesion = 50
    bearing_pressure = example_function.calc_bearing_pressure(cohesion)
    print(f"The bearing pressure of the soil with cohesion {cohesion} kPa is {bearing_pressure} kPa.")