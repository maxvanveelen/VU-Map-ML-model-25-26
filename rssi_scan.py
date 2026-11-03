import subprocess
import re
import time
import os
import csv
from datetime import datetime

LOG_DIR = 'data/rssi_scan_logs'
INTERFACE = 'wlo1'
INTERVAL = 3    # seconds between scans
CSV_FIELDS = ['timestamp', 'ssid', 'bssid', 'rssi_dbm', 'frequency_ghz', 'signal_quality']

def scan_wifi(interface):
    # ask for a fresh scan, then give the wifi card a moment to finish it
    subprocess.run(['sudo', 'nmcli', 'device', 'wifi', 'rescan', 'ifname', interface], capture_output=True)
    time.sleep(1)

    result = subprocess.run(
        ['nmcli', '--terse', '--fields', 'SSID,BSSID,FREQ,SIGNAL', 'device', 'wifi', 'list', 'ifname', interface],
        capture_output=True, text=True
    )

    # a line looks like: eduroam:48\:00\:B3\:5E\:3D\:61:2437 MHz:94
    # splitting on ':' also splits the BSSID, so it ends up in parts[1:7]
    networks = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split(':')
        if len(parts) < 9:
            continue

        try:
            networks.append({
                'ssid':      parts[0] or 'Hidden',
                'address':   ':'.join(parts[1:7]),
                'frequency': parts[7],
                'rssi':      int(parts[8]) - 110,   # nmcli gives signal as 0-100%, not dBm, this is an estimate
            })
        except ValueError:
            continue

    return networks

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

def display(networks, scan_count):
    os.system('clear')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"  WiFi RSSI Monitor  |  Scan #{scan_count}  |  {timestamp}  |  Interval: {INTERVAL}s")
    print(f"  Press Ctrl+C to stop and save")
    print('=' * 75)
    print(f"  {'SSID':<28} {'RSSI':>6}  {'Signal':<20}  {'BSSID'}")
    print('-' * 75)

    for n in sorted(networks, key=lambda n: n['rssi'], reverse=True):
        print(f"  {n['ssid'][:28]:<28} {n['rssi']:>5}dBm  {signal_bar(n['rssi']):<20}  {n['address']}")

    print('=' * 75)
    print(f"  {len(networks)} access point(s) found")

def run_scan():
    """Scans until Ctrl+C is pressed, returns a list of (networks, timestamp)."""
    all_results = []

    try:
        while True:
            networks = scan_wifi(INTERFACE)
            all_results.append((networks, datetime.now()))
            display(networks, len(all_results))
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        pass

    return all_results

def ask_filename():
    print('\n' + '=' * 75)
    print("  Scan stopped. Enter a name for the output files.")
    print("  Files will be saved as:  rssi_scan_<your name>.log / .csv")
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

def save_results(all_results, basename):
    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(LOG_DIR, f'{basename}.log')
    csvfile = os.path.join(LOG_DIR, f'{basename}.csv')
    csv_exists = os.path.isfile(csvfile)

    with open(logfile, 'a') as log, open(csvfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()

        for networks, timestamp in all_results:
            ts = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            log.write(f'\n--- Scan at {ts} ---\n')

            for n in networks:
                log.write(f"SSID: {n['ssid']:<30} RSSI: {n['rssi']:>4} dBm  BSSID: {n['address']}\n")
                writer.writerow({
                    'timestamp':      ts,
                    'ssid':           n['ssid'],
                    'bssid':          n['address'],
                    'rssi_dbm':       n['rssi'],
                    'frequency_ghz':  n['frequency'],
                    'signal_quality': signal_quality(n['rssi']),
                })

    return logfile, csvfile

def main():
    os.system('clear')
    print('=' * 75)
    print('  WiFi RSSI Monitor')
    print(f'  Interface: {INTERFACE}  |  Interval: {INTERVAL}s')
    print('=' * 75)
    print()
    input('  Press Enter to start scanning...')

    all_results = run_scan()

    if not all_results:
        print('\n  No scans completed. Exiting.')
        return

    basename = ask_filename()
    logfile, csvfile = save_results(all_results, basename)

    print()
    print(f'  Saved {len(all_results)} scan(s) to:')
    print(f'    {logfile}')
    print(f'    {csvfile}')
    print()

if __name__ == '__main__':
    main()
