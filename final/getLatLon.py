import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371000  # Earth's radius in meters

# Image and camera parameters
image_width = 640  # pixels
image_height = 480  # pixels
altitude = 15  # meters
hfov = 53  # degrees
vfov = 41  # degrees
mid_lat = 15.3677175  # Midpoint latitude
mid_lon = 75.1254693  # Midpoint longitude

def getLatLon(x_pixel, y_pixel):
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
    Dx = abs(np.tan(theta_x_rad) * altitude)
    Dy = abs(np.tan(theta_y_rad) * altitude)

    # Latitude change
    delta_lat = Dy / R
    target_lat = mid_lat + np.degrees(delta_lat)

    # Longitude change (adjusting for latitude)
    delta_lon = Dx / (R * np.cos(np.radians(mid_lat)))
    target_lon = mid_lon + np.degrees(delta_lon)

    # Handling edge cases where target_lat or target_lon is zero
    if target_lat == 0 and target_lon != 0:
        return mid_lat, target_lon
    elif target_lat != 0 and target_lon == 0:
        return target_lat, mid_lon

    return target_lat, target_lon

# Example pixel positions to convert to latitude and longitude
example_pixels = [
    (320, 240),  # Center of the image (should return midpoint latitude and longitude)
    (640, 480),  # Bottom-right corner of the image
    (0, 0),      # Top-left corner of the image
    (320, 0),    # Middle of the top edge
    (0, 240),    # Middle of the left edge
]

# Calculate the latitude and longitude for each pixel position
lat_lon_positions = [getLatLon(x_pixel, y_pixel) for (x_pixel, y_pixel) in example_pixels]

# Visualization
plt.figure(figsize=(10, 8))
for (x_pixel, y_pixel), (lat, lon) in zip(example_pixels, lat_lon_positions):
    plt.scatter(x_pixel, y_pixel, color='red')
    plt.text(x_pixel, y_pixel, f'({lat:.6f}, {lon:.6f})', fontsize=9, ha='right', color='blue')

# Plot image boundaries for context
plt.plot([0, image_width, image_width, 0, 0], [0, 0, image_height, image_height, 0], 'k-')

plt.title('Pixel Positions and Corresponding Lat/Lon Coordinates')
plt.xlabel('X-axis (pixels)')
plt.ylabel('Y-axis (pixels)')
plt.xlim(0, image_width)
plt.ylim(image_height, 0)
plt.grid(True)
plt.show()
