import csv

DEVICES_CSV = "data/reference/Devices-03-03-2026.csv"

def mac_without_last_digit(mac):
    return mac.strip().lower()[:-1]

def load_bssid_to_ap(devices_csv=DEVICES_CSV):
    """
    Returns a dict from MAC address (without its last hex digit) to AP name.
    An AP has a BSSID for every SSID and band, these only differ from its MAC address in the last hex digit.
    """
    bssid_to_ap = {}

    # utf-8-sig: the Cisco export starts with an invisible BOM character
    with open(devices_csv, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            bssid_to_ap[mac_without_last_digit(row["MAC Address"])] = row["Device Name"]

    return bssid_to_ap

def scan_to_ap_rssi(scan, bssid_to_ap):
    """Returns the strongest RSSI of each AP in a scan. BSSIDs not belonging to an AP are skipped."""
    ap_rssi = {}

    for reading in scan.readings:
        ap = bssid_to_ap.get(mac_without_last_digit(reading.bssid))
        if ap is None:
            continue

        if ap not in ap_rssi or reading.rssi_dbm > ap_rssi[ap]:
            ap_rssi[ap] = reading.rssi_dbm

    return ap_rssi
