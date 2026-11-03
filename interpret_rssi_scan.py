import csv
import sys
from datetime import datetime
from statistics import mean

# Reads the csv files made by rssi_scan.py.
# Every row is one AP reading, rows with the same timestamp belong to the same scan.

class ApReading:
    def __init__(self, ssid, bssid, rssi_dbm, frequency_mhz, signal_quality):
        self.ssid = ssid
        self.bssid = bssid
        self.rssi_dbm = rssi_dbm
        self.frequency_mhz = frequency_mhz
        self.signal_quality = signal_quality

    @classmethod
    def from_row(cls, row):
        return cls(
            ssid=row["ssid"],
            bssid=row["bssid"].replace("\\:", ":"),             # nmcli escapes the colons in a BSSID as \:
            rssi_dbm=int(row["rssi_dbm"]),
            frequency_mhz=int(row["frequency_ghz"].split()[0]),  # "2437 MHz" -> 2437
            signal_quality=row["signal_quality"],
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


class RssiScanSession:
    def __init__(self, scans):
        self.scans = sorted(scans, key=lambda s: s.timestamp)

    @classmethod
    def from_csv(cls, path):
        readings_per_timestamp = {}

        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                readings_per_timestamp.setdefault(timestamp, []).append(ApReading.from_row(row))

        scans = [RssiScan(timestamp, readings) for timestamp, readings in readings_per_timestamp.items()]
        return cls(scans)

    @classmethod
    def from_csvs(cls, paths):
        scans = []
        for path in paths:
            scans = scans + cls.from_csv(path).scans
        return cls(scans)

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
