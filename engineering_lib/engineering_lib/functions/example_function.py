def calc_bearing_pressure(cohesion: float) -> float:
    """
    Calculates the bearing capacity of a structure based on its cohesion.

    Args:
        cohesion (float): The cohesion of the soil.

    Returns:
        float: The bearing pressure of the soil.
    """

    return cohesion * 5.14
