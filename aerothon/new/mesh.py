import subprocess
import time
import iwlib

# Set the interface name for your Wi-Fi (commonly 'wlan0')
INTERFACE = 'wlan0'
# Set the threshold for signal strength (in dBm) to switch connections
THRESHOLD = -70  # You can adjust this value based on your requirements

def get_best_wifi():
    # Scan for available Wi-Fi networks
    scan_output = subprocess.check_output(["sudo", "iwlist", INTERFACE, "scan"]).decode('utf-8')
    
    # Parse the output to find SSID and signal strength
    networks = []
    
    for line in scan_output.splitlines():
        if 'ESSID' in line:
            ssid = line.split('"')[1]  # Extract SSID
        elif 'Signal level' in line:
            signal = int(line.split('Signal level=')[1].split(' ')[0])  # Extract signal level
            networks.append((ssid, signal))
    
    # Find the best network based on signal strength
    best_network = max(networks, key=lambda x: x[1], default=(None, None))
    return best_network

def connect_to_wifi(ssid, password):
    # Connect to the specified Wi-Fi network using nmcli
    try:
        subprocess.call(["nmcli", "d", "wifi", "connect", ssid, "password", password])
        print(f"Connected to {ssid}")
    except Exception as e:
        print(f"Failed to connect to {ssid}: {e}")

def main():
    while True:
        best_ssid, best_signal = get_best_wifi()
        
        if best_signal is not None and best_signal > THRESHOLD:
            print(f"Best Wi-Fi: {best_ssid} with signal {best_signal} dBm")
            # Replace with the actual password of your networks
            connect_to_wifi(best_ssid, "your_password")  
        else:
            print("No suitable Wi-Fi networks found")

        # Check every 10 seconds
        time.sleep(10)

if _name_ == "_main_":
    main()