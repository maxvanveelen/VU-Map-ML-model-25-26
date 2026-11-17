import csv
import os
import sys
from datetime import datetime
from statistics import mean

# Reads the csv files made by rssi_scan.py.
# Every row is one AP reading, rows with the same timestamp belong to the same scan.
# There are two formats:
#   old (nmcli): timestamp, ssid, bssid, rssi_dbm (nmcli % - 110), frequency_ghz ("2437 MHz"), signal_quality
#   new (iw):    timestamp (with ms), ssid, bssid, rssi_dbm (real dBm), frequency_mhz, last_seen_ms
#                + a <name>_checkpoints.csv file with: checkpoint, timestamp

def parse_timestamp(text):
    """New files have milliseconds in the timestamp, old ones don't."""
    if "." in text:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")

def nmcli_to_dbm(stored_rssi):
    """
    rssi_scan.py stored nmcli's signal percentage minus 110, this converts it back to real dBm.
    nmcli (NetworkManager 1.36) calculates the percentage as a straight line from -100 dBm (0%) to -40 dBm (100%),
    so anything stronger than -40 dBm is capped at -40.
    """
    percent = stored_rssi + 110
    return 0.6 * percent - 100

class ApReading:
    def __init__(self, ssid, bssid, rssi_dbm, frequency_mhz, last_seen_ms):
        self.ssid = ssid
        self.bssid = bssid
        self.rssi_dbm = rssi_dbm
        self.frequency_mhz = frequency_mhz
        self.last_seen_ms = last_seen_ms     # how old the reading was when it was scanned, None for old (nmcli) files

    @classmethod
    def from_row(cls, row):
        if "last_seen_ms" in row:
            return cls(
                ssid=row["ssid"],
                bssid=row["bssid"],
                rssi_dbm=float(row["rssi_dbm"]),
                frequency_mhz=int(row["frequency_mhz"]),
                last_seen_ms=int(row["last_seen_ms"]),
            )

        return cls(
            ssid=row["ssid"],
            bssid=row["bssid"].replace("\\:", ":"),             # nmcli escapes the colons in a BSSID as \:
            rssi_dbm=nmcli_to_dbm(int(row["rssi_dbm"])),
            frequency_mhz=int(row["frequency_ghz"].split()[0]),  # "2437 MHz" -> 2437
            last_seen_ms=None,
        )

    @property
    def band(self):
        if 2400 <= self.frequency_mhz < 2500:
            return "2.4GHz"
        if 4900 <= self.frequency_mhz < 5900:
            return "5GHz"
        if 5900 <= self.frequency_mhz < 7200:
            return "6GHz"
        return "unknown"

    def __str__(self):
        return f"ApReading(ssid={self.ssid}, bssid={self.bssid}, rssi_dbm={self.rssi_dbm}, frequency_mhz={self.frequency_mhz})"


class RssiScan:
    def __init__(self, timestamp, readings):
        self.timestamp = timestamp
        self.readings = readings

    def __len__(self):
        return len(self.readings)

    def __iter__(self):
        return iter(self.readings)

    @property
    def strongest(self):
        return max(self.readings, key=lambda r: r.rssi_dbm, default=None)

    @property
    def weakest(self):
        return min(self.readings, key=lambda r: r.rssi_dbm, default=None)

    @property
    def average_rssi(self):
        if not self.readings:
            return None
        return mean(r.rssi_dbm for r in self.readings)

    def ssids(self):
        return {r.ssid for r in self.readings}

    def readings_for_ssid(self, ssid):
        return [r for r in self.readings if r.ssid == ssid]

    def by_band(self):
        bands = {}
        for r in self.readings:
            bands.setdefault(r.band, []).append(r)
        return bands


def load_checkpoints(scan_csv_path):
    """
    Returns the checkpoints of a recording as a list of (checkpoint number, timestamp).
    They are in <name>_checkpoints.csv next to <name>.csv, old recordings don't have this file and get an empty list.
    """
    checkpoint_path = scan_csv_path[:-len(".csv")] + "_checkpoints.csv"
    if not os.path.isfile(checkpoint_path):
        return []

    with open(checkpoint_path, newline="", encoding="utf-8") as file:
        return [(int(row["checkpoint"]), parse_timestamp(row["timestamp"])) for row in csv.DictReader(file)]


class RssiScanSession:
    def __init__(self, scans, checkpoints=None):
        self.scans = sorted(scans, key=lambda s: s.timestamp)
        self.checkpoints = checkpoints or []    # list of (checkpoint number, timestamp)

    @classmethod
    def from_csv(cls, path):
        readings_per_timestamp = {}

        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                timestamp = parse_timestamp(row["timestamp"])
                readings_per_timestamp.setdefault(timestamp, []).append(ApReading.from_row(row))

        scans = [RssiScan(timestamp, readings) for timestamp, readings in readings_per_timestamp.items()]
        return cls(scans, load_checkpoints(path))

    @classmethod
    def from_csvs(cls, paths):
        scans = []
        checkpoints = []
        for path in paths:
            session = cls.from_csv(path)
            scans = scans + session.scans
            checkpoints = checkpoints + session.checkpoints
        return cls(scans, checkpoints)

    def __len__(self):
        return len(self.scans)

    def __iter__(self):
        return iter(self.scans)

    def __getitem__(self, index):
        return self.scans[index]

    @property
    def start_time(self):
        return self.scans[0].timestamp if self.scans else None

    @property
    def end_time(self):
        return self.scans[-1].timestamp if self.scans else None

    def all_ssids(self):
        ssids = set()
        for scan in self.scans:
            ssids = ssids | scan.ssids()
        return ssids

    def rssi_timeseries(self, bssid):
        """Returns the rssi over time of one BSSID, as a list of (timestamp, rssi_dbm)."""
        series = []
        for scan in self.scans:
            for r in scan.readings:
                if r.bssid == bssid:
                    series.append((scan.timestamp, r.rssi_dbm))
        return series

    def summary(self):
        lines = [
            f"RssiScanSession: {len(self.scans)} scans from {self.start_time} to {self.end_time}",
            f"  Unique SSIDs: {sorted(self.all_ssids())}",
        ]
        for scan in self.scans:
            lines.append(f"  [{scan.timestamp}] {len(scan)} APs, strongest={scan.strongest}, avg_rssi={scan.average_rssi:.1f} dBm")
        return "\n".join(lines)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/rssi_scan_logs/rssi_scan_F5_R1_1.csv"
    session = RssiScanSession.from_csv(csv_path)

    print(session.summary())
    print()

    first_scan = session[0]
    print(f"First scan strongest AP: {first_scan.strongest}")
    print(f"First scan bands: { {band: len(readings) for band, readings in first_scan.by_band().items()} }")
