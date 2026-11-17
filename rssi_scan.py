import subprocess
import re
import time
import os
import csv
import sys
import select
from datetime import datetime

LOG_DIR = 'data/rssi_scan_logs'
INTERFACE = 'wlo1'
INTERVAL = 1                                # seconds between scans
SCAN_FREQUENCIES = ['2412', '2437', '2462']   # VU's 2.4 GHz channels (MHz), use [] to scan all channels
SCANS_PER_INTERVAL = 3                        # short scans combined into one, a single short scan only hears part of the networks
CSV_FIELDS = ['timestamp', 'ssid', 'bssid', 'rssi_dbm', 'frequency_mhz', 'last_seen_ms']

def scan_wifi(interface):
    """Does a few short scans and combines them: for every network the most recent reading."""
    # flush: throw away the kernel's cache first, so a scan only returns networks it actually heard
    command = ['sudo', 'iw', 'dev', interface, 'scan', 'flush']
    if SCAN_FREQUENCIES:
        command = command + ['freq'] + SCAN_FREQUENCIES

    latest_per_bssid = {}
    scan_end_per_bssid = {}
    for _ in range(SCANS_PER_INTERVAL):
        result = subprocess.run(command, capture_output=True, text=True)
        scan_end = time.time()
        for network in parse_iw_scan(result.stdout):
            latest_per_bssid[network['bssid']] = network
            scan_end_per_bssid[network['bssid']] = scan_end

    # last_seen_ms is relative to the end of its own scan, add the time since then so it is relative to now
    now = time.time()
    for bssid, network in latest_per_bssid.items():
        network['last_seen_ms'] += int((now - scan_end_per_bssid[bssid]) * 1000)

    return list(latest_per_bssid.values())

def parse_iw_scan(output):
    """
    Turns the text output of 'iw scan' into a list of networks.
    Every network starts with a line 'BSS <bssid>(on wlo1)', followed by indented lines like
    'freq: 2437', 'signal: -54.00 dBm', 'last seen: 120 ms ago' and 'SSID: eduroam'.
    """
    networks = []

    for line in output.split('\n'):
        if line.startswith('BSS '):
            networks.append({'ssid': 'Hidden', 'bssid': line[4:21], 'rssi_dbm': None, 'frequency_mhz': None, 'last_seen_ms': None})
            continue
        if not networks:
            continue

        network = networks[-1]
        line = line.strip()
        if line.startswith('freq:'):
            network['frequency_mhz'] = int(float(line.split()[1]))
        elif line.startswith('signal:'):
            network['rssi_dbm'] = float(line.split()[1])
        elif line.startswith('last seen:') and line.endswith('ms ago'):
            network['last_seen_ms'] = int(line.split()[2])
        elif line.startswith('SSID:') and network['ssid'] == 'Hidden' and line[5:].strip():
            network['ssid'] = line[5:].strip()

    return [network for network in networks if network['rssi_dbm'] is not None]

def signal_quality(rssi):
    if rssi >= -50:
        return 'Excellent'
    elif rssi >= -60:
        return 'Good'
    elif rssi >= -70:
        return 'Fair'
    elif rssi >= -80:
        return 'Weak'
    else:
        return 'Very Poor'

def signal_bar(rssi):
    bars = {'Excellent': '█████', 'Good': '████░', 'Fair': '███░░', 'Weak': '██░░░', 'Very Poor': '█░░░░'}
    quality = signal_quality(rssi)
    return f'{bars[quality]} {quality}'

def display(networks, scan_count, checkpoints, failed_scans):
    os.system('clear')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"  WiFi RSSI Monitor  |  Scan #{scan_count}  |  {timestamp}  |  Interval: {INTERVAL}s")
    print(f"  Press Enter at a checkpoint  |  Press Ctrl+C to stop and save")
    print(f"  Checkpoints: {len(checkpoints)}" + (f" (last at {checkpoints[-1].strftime('%H:%M:%S')})" if checkpoints else "")
          + (f"  |  Failed scans: {failed_scans}" if failed_scans else ""))
    print('=' * 82)
    print(f"  {'SSID':<28} {'RSSI':>9}  {'Signal':<16}  {'Age':>7}  {'BSSID'}")
    print('-' * 82)

    for n in sorted(networks, key=lambda n: n['rssi_dbm'], reverse=True):
        print(f"  {n['ssid'][:28]:<28} {n['rssi_dbm']:>6.1f}dBm  {signal_bar(n['rssi_dbm']):<16}  {n['last_seen_ms']:>5}ms  {n['bssid']}")

    print('=' * 82)
    print(f"  {len(networks)} access point(s) found")

def wait_for_next_scan(seconds, checkpoints):
    """Waits until the next scan. Every time Enter is pressed during the wait, the current time is saved as a checkpoint."""
    end = time.time() + seconds
    while time.time() < end:
        # select waits until there is input (Enter) or the time is up, whichever comes first
        enter_pressed, _, _ = select.select([sys.stdin], [], [], max(0, end - time.time()))
        if enter_pressed:
            # an empty string means the input was closed (Ctrl+D), not Enter: stop checking and just wait
            if sys.stdin.readline() == '':
                time.sleep(max(0, end - time.time()))
                return
            checkpoints.append(datetime.now())

def run_scan():
    """Scans until Ctrl+C is pressed, returns a list of (networks, timestamp) and a list of checkpoint times."""
    all_results = []
    checkpoints = []
    failed_scans = 0

    try:
        while True:
            scan_start = time.time()
            networks = scan_wifi(INTERFACE)

            # iw sometimes fails, e.g. when NetworkManager is scanning at the same moment
            if networks:
                all_results.append((networks, datetime.now()))
            else:
                failed_scans += 1

            display(networks, len(all_results), checkpoints, failed_scans)
            wait_for_next_scan(INTERVAL - (time.time() - scan_start), checkpoints)
    except KeyboardInterrupt:
        pass

    return all_results, checkpoints

def ask_filename():
    print('\n' + '=' * 75)
    print("  Scan stopped. Enter a name for the output files, starting with the floor, e.g. F5_lap01")
    print("  Files will be saved as:  rssi_scan_<your name>.csv / rssi_scan_<your name>_checkpoints.csv")
    print("  Leave blank to use a timestamp instead.")
    print('=' * 75)

    name = input("  Name: ").strip()
    if not name:
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(f"  Using timestamp: {name}")

    # only keep letters, digits, hyphens, underscores and dots
    safe_name = re.sub(r'[^\w\-.]', '_', name)
    if safe_name != name:
        print(f"  (Special characters replaced: '{name}' → '{safe_name}')")

    return f'rssi_scan_{safe_name}'

def format_timestamp(timestamp):
    """'2026-11-17 14:40:01.123', with milliseconds because there can be more than one scan per second."""
    return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def save_results(all_results, checkpoints, basename):
    os.makedirs(LOG_DIR, exist_ok=True)
    csvfile = os.path.join(LOG_DIR, f'{basename}.csv')
    checkpoint_file = os.path.join(LOG_DIR, f'{basename}_checkpoints.csv')

    with open(csvfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for networks, timestamp in all_results:
            for n in networks:
                writer.writerow(dict(n, timestamp=format_timestamp(timestamp)))

    with open(checkpoint_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['checkpoint', 'timestamp'])
        for number, timestamp in enumerate(checkpoints, start=1):
            writer.writerow([number, format_timestamp(timestamp)])

    return csvfile, checkpoint_file

def main():
    os.system('clear')
    print('=' * 75)
    print('  WiFi RSSI Monitor')
    print(f'  Interface: {INTERFACE}  |  Interval: {INTERVAL}s  |  Channels: {SCAN_FREQUENCIES or "all"}')
    print('=' * 75)
    print()

    # ask for the sudo password now, so it doesn't interrupt the scanning later
    subprocess.run(['sudo', '-v'])
    input('  Press Enter to start scanning...')

    all_results, checkpoints = run_scan()

    if not all_results:
        print('\n  No scans completed. Exiting.')
        return

    basename = ask_filename()
    csvfile, checkpoint_file = save_results(all_results, checkpoints, basename)

    print()
    print(f'  Saved {len(all_results)} scan(s) and {len(checkpoints)} checkpoint(s) to:')
    print(f'    {csvfile}')
    print(f'    {checkpoint_file}')
    print()

if __name__ == '__main__':
    main()
