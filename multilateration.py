import pandas as pd
import math
from scipy.optimize import least_squares

AP_POSITIONS_CSV = "data/samplesF5-multilayer.csv"

P0 = -40    # TODO: tune - RSSI (dBm) at 1 meter from an AP.
N = 3       # TODO: tune - Path loss exponent, how fast the signal gets weaker with distance.
MAX_APS = 5 # TODO: tune - How many of the strongest APs to use per scan.

def load_ap_positions(csv_path=AP_POSITIONS_CSV):
    """
    Returns a dict from AP name to its (x, z) position on the floor, in Unity coordinates (meters).
    The samples csv has NU-APxxxxx_x, _y and _z columns with the AP positions, these are the same in every row.
    """
    first_row = pd.read_csv(csv_path, nrows=1).iloc[0]

    ap_positions = {}
    for column in first_row.index:
        if column.startswith("NU-AP") and column.endswith("_x"):
            ap_name = column[:-2]
            ap_positions[ap_name] = (float(first_row[ap_name + "_x"]), float(first_row[ap_name + "_z"]))

    return ap_positions

def rssi_to_distance(rssi_dbm, p0=P0, n=N):
    """
    Estimates the distance (meters) to an AP from its RSSI, using the log-distance path loss model:
    rssi = p0 - 10 * n * log10(distance)
    """
    return 10 ** ((p0 - rssi_dbm) / (10 * n))

def ap_floor(ap_name):
    """'NU-AP05130' -> 5. Returns None for APs without a floor number, like the basement ones (NU-APK3404)."""
    floor_digits = ap_name[5:7]
    if not floor_digits.isdigit():
        return None
    return int(floor_digits)

def select_aps(ap_rssi, ap_positions, floor, max_aps=MAX_APS):
    """
    Returns the strongest APs of a scan as a list of (ap_name, rssi_dbm), strongest first.
    Only APs on the given floor and with a known position are used.
    """
    usable_aps = [(ap, rssi) for ap, rssi in ap_rssi.items() if ap_floor(ap) == floor and ap in ap_positions]
    usable_aps.sort(key=lambda ap_and_rssi: ap_and_rssi[1], reverse=True)
    return usable_aps[:max_aps]

def weighted_centroid(selected_aps, ap_positions):
    """Returns the average (x, z) position of the selected APs, where closer APs count more."""
    total_weight = 0
    weighted_x = 0
    weighted_z = 0

    for ap, rssi in selected_aps:
        x, z = ap_positions[ap]
        weight = 1 / (rssi_to_distance(rssi))

        total_weight += weight
        weighted_x += x * weight
        weighted_z += z * weight

    return (weighted_x / total_weight, weighted_z / total_weight)


def residuals(guess_position, selected_aps, ap_positions):
    """
    Returns for every selected AP how far off the guess is: the distance from the guess minus the distance from the rssi.
    Positive means the guess is too far from the AP, negative means too close.
    """
    errors = []

    for ap, rssi in selected_aps:
        distance_from_guess = math.dist(guess_position, ap_positions[ap])
        distance_from_rssi = rssi_to_distance(rssi)
        errors.append(distance_from_guess-distance_from_rssi)

    return errors

def estimate_position(selected_aps, ap_positions):
    """Estimates the (x, z) position of a scan: starts at the weighted centroid and improves it with least squares."""
    start = weighted_centroid(selected_aps, ap_positions)
    result = least_squares(residuals, start, args=(selected_aps, ap_positions))
    return (result.x[0], result.x[1])
