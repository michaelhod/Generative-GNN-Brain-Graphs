import csv
import time
import psutil

import psutil
import time
import csv

# Set the target PID
PID = 3972692
LOG_FILE = "memory_usage_log_2.csv"

def log_memory_usage(pid, log_file, interval=2):
    """
    Continuously logs memory usage of a process to a CSV file.

    Params:
    - pid: Process ID to monitor.
    - log_file: Path to the CSV file to save logs.
    - interval: Time interval (in seconds) between measurements.
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found.")
        return

    # Open CSV file and write headers
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "memory_MB"])  # Header row

        print(f"Logging memory usage for PID {pid} every {interval} seconds...")
        try:
            while True:
                mem_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                writer.writerow([timestamp, mem_usage])
                file.flush()  # Ensure data is written immediately
                
                print(f"[{timestamp}] Memory: {mem_usage:.2f} MB")
                time.sleep(interval)  # Wait before next measurement
        except psutil.NoSuchProcess:
            print(f"Process {pid} ended. Stopping logging.")
        except KeyboardInterrupt:
            print("\nLogging stopped manually.")

log_memory_usage(PID, LOG_FILE)