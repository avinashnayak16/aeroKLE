import os
import subprocess
import time

PASSWORD = "aerokle2"  # The password for all aerokle2 networks
SSID_PREFIX = "aerokle2"  # The common prefix for matching SSIDs

def get_wifi_networks():
    # Scan for available Wi-Fi networks and return a list of tuples (SSID, Signal Strength)
    networks = []
    ssid = None
    signal = None
    
    try:
        output = subprocess.check_output(["sudo", "iwlist", "wlan0", "scan"])
        output = output.decode('utf-8')

        for line in output.split("\n"):
            line = line.strip()
            if "ESSID:" in line:
                ssid = line.split("ESSID:")[1].strip().replace('"', '')
            elif "Signal level=" in line:
                signal = int(line.split("Signal level=")[1].split(" ")[0])
            
            # If both SSID and signal level are found, append them to the list
            if ssid and signal:
                networks.append((ssid, signal))
                ssid = None
                signal = None
    except Exception as e:
        print(f"Error scanning for networks: {e}")
    
    return networks

def get_best_wifi():
    # Get all available Wi-Fi networks
    networks = get_wifi_networks()

    best_ssid = None
    best_signal = float('-inf')

    # Filter networks that match the SSID pattern (aerokle2 or aerokle2-*)
    for ssid, signal in networks:
        if ssid == SSID_PREFIX or ssid.startswith(SSID_PREFIX + "-"):
            if signal > best_signal:
                best_signal = signal
                best_ssid = ssid
    
    return best_ssid, best_signal

def get_current_wifi():
    # Get the currently connected Wi-Fi network using 'iwgetid -r'
    try:
        output = subprocess.check_output(["iwgetid", "-r"])
        current_ssid = output.decode('utf-8').strip()
        return current_ssid
    except Exception as e:
        return None

def connect_to_wifi(ssid):
    # Connect to a Wi-Fi network using 'nmcli'
    try:
        print(f"Connecting to Wi-Fi network '{ssid}'...")
        # Connect using nmcli and the shared password 'aerokle2'
        result = subprocess.check_call(['sudo', 'nmcli', 'device', 'wifi', 'connect', ssid, 'password', PASSWORD])
        print(f"Successfully connected to '{ssid}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to connect to '{ssid}': {e}")
        return False

def main():
    try:
        while True:
            # Scan for the best available Wi-Fi network with SSID matching the pattern
            best_ssid, best_signal = get_best_wifi()

            if best_ssid is not None:
                print(f"Best Wi-Fi network found: '{best_ssid}' with signal strength {best_signal} dBm.")
                current_ssid = get_current_wifi()

                if current_ssid == best_ssid:
                    print(f"Already connected to the best Wi-Fi network '{current_ssid}'. Waiting 5 seconds before next scan...")
                    time.sleep(5)
                    continue
                else:
                    # Try to connect to the best Wi-Fi network until successful
                    while not connect_to_wifi(best_ssid):
                        print(f"Retrying connection to '{best_ssid}' in 5 seconds...")
                        time.sleep(5)
            else:
                print("No matching Wi-Fi networks found. Retrying in 5 seconds...")

            # Delay before the next scan
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
        return

if __name__ == "__main__":
    main()

