import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371000  # Earth's radius in meters
# Example usage:
image_width = 640  # pixels
image_height = 480  # pixels
altitude = 15  # meters
hfov = 53  # degrees
vfov = 41  # degrees
mid_lat = 15.3677175  # Midpoint latitude
mid_lon = 75.1254693  # Midpoint longitude

def getLatLon(x_pixel,y_pixel):

    x_c = image_width // 2
    y_c = image_height // 2

    # Angle per pixel
    hpp = hfov / image_width  # Horizontal angle per pixel
    vpp = vfov / image_height  # Vertical angle per pixel

    # Pixel displacement from center
    delta_x = x_pixel - x_c
    delta_y = y_pixel - y_c

    # Angle displacement
    theta_x = delta_x * hpp  # in degrees
    theta_y = delta_y * vpp  # in degrees

    # Convert angles from degrees to radians
    theta_x_rad = np.radians(theta_x)
    theta_y_rad = np.radians(theta_y)

    # Distance calculation
    Dx = abs(np.tan(theta_x_rad) * vehicle.globalrelativeframe.alt)
    Dy = abs(np.tan(theta_y_rad) * vehicle.globalrelativeframe.alt)

    mid_lat = vehicle.globalrelativeframe.lat
    mid_lon = vehicle.globalrelativeframe.lon

    # Latitude change
    delta_lat = Dy / R
    target_lat = mid_lat + np.degrees(delta_lat)

    # Longitude change (adjusting for latitude)
    delta_lon = Dx / (R * np.cos(np.radians(mid_lat)))
    target_lon = mid_lon + np.degrees(delta_lon)

    if target_lat ==0 and  target_lon != 0:
        return mid_lat, target_lon

    elif target_lat !=0 and  target_lon == 0:
        return target_lat, mid_lon

    return target_lat, target_lon

def calculate_target_distance(image_width, image_height, altitude, hfov, vfov, x_t, y_t):
    # Center of the image
    x_c = image_width // 2
    y_c = image_height // 2

    # Angle per pixel
    hpp = hfov / image_width  # Horizontal angle per pixel
    vpp = vfov / image_height  # Vertical angle per pixel

    # Pixel displacement from center
    delta_x = x_t - x_c
    delta_y = y_t - y_c

    # Angle displacement
    theta_x = delta_x * hpp  # in degrees
    theta_y = delta_y * vpp  # in degrees

    # Convert angles from degrees to radians
    theta_x_rad = np.radians(theta_x)
    theta_y_rad = np.radians(theta_y)

    # Distance calculation
    Dx = abs(np.tan(theta_x_rad) * altitude)
    Dy = abs(np.tan(theta_y_rad) * altitude)

    return Dx, Dy

def calculate_lat_lon_from_distance(mid_lat, mid_lon, Dx, Dy):
    """
    Calculate the latitude and longitude of a target given the midpoint coordinates and distances.
    
    Parameters:
    - mid_lat: Latitude of the midpoint
    - mid_lon: Longitude of the midpoint
    - Dx: Horizontal distance from the center to the target (meters)
    - Dy: Vertical distance from the center to the target (meters)
    
    Returns:
    - target_lat: Latitude of the target
    - target_lon: Longitude of the target
    """
    # Latitude change
    delta_lat = Dy / R
    target_lat = mid_lat + np.degrees(delta_lat)

    # Longitude change (adjusting for latitude)
    delta_lon = Dx / (R * np.cos(np.radians(mid_lat)))
    target_lon = mid_lon + np.degrees(delta_lon)

    return target_lat, target_lon

def visualize_multiple_targets(image_width, image_height, target_positions, distances, mid_lat, mid_lon):
    """
    Visualize multiple targets' positions relative to the image center and display the calculated distances
    and latitude/longitude coordinates.
    
    Parameters:
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    - target_positions: List of (x_t, y_t) tuples for each target's coordinates in pixels
    - distances: List of (Dx, Dy) tuples for each target's distance in meters
    - mid_lat: Latitude of the midpoint
    - mid_lon: Longitude of the midpoint
    """
    # Center of the image
    x_c = image_width // 2
    y_c = image_height // 2

    plt.figure(figsize=(8, 6))

    for (x_t, y_t), (Dx, Dy) in zip(target_positions, distances):
        target_lat, target_lon = calculate_lat_lon_from_distance(mid_lat, mid_lon, Dx, Dy)
        
        plt.scatter([x_c, x_t], [y_c, y_t], color=['blue', 'red'])
        plt.plot([x_c, x_t], [y_c, y_t], 'r--')
        plt.text(x_t, y_t, f'Target\n({Dx:.2f}m, {Dy:.2f}m)\nLat: {target_lat:.6f}\nLon: {target_lon:.6f}', 
                 fontsize=12, ha='left', color='red')

    plt.text(x_c, y_c, 'Center', fontsize=12, ha='right', color='blue')
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.title('Distance from Center to Targets')
    plt.xlabel('X-axis (pixels)')
    plt.ylabel('Y-axis (pixels)')
    plt.grid(True)
    plt.show()

# # Example usage:
# image_width = 640  # pixels
# image_height = 480  # pixels
# altitude = 15  # meters
# hfov = 53  # degrees
# vfov = 41  # degrees
# mid_lat = 15.3677175  # Midpoint latitude
# mid_lon = 75.1254693  # Midpoint longitude

# Define multiple targets
# target_positions = [(320, 480), (0, 0)]  # (x_t, y_t) for each target
# distances = [calculate_target_distance(image_width, image_height, altitude, hfov, vfov, x_t, y_t) for x_t, y_t in target_positions]
distances = getLatLon(320,480)

# Visualize the targets with latitude and longitude
visualize_multiple_targets(image_width, image_height, target_positions, distances, mid_lat, mid_lon)

