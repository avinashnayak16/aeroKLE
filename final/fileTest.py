def clearCoordintesFile():
    with open("coordinates.txt", "a") as f:
        f.write(f"")  # Add f-string to format the string properly

def readingLastCoordinates():
    with open("coordinates.txt", "r") as file:
        lines = file.readlines()
        if not lines:
            return None, None  # Return None if the file is empty
        last_line = lines[-1].strip()
        x_pix, y_pix, hotspotNum, typeDetect = last_line.split(", ")
        x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
        return x_pix ,y_pix,hotspotNum,typeDetect

def readingStepCoordinfates(countFile):
    try:
        with open("coordinates.txt", "r") as file:
            lines = file.readlines()
            print(len(lines))
            if not lines:
                return None, None, None, None  # Return None if the file is empty

            if countFile >= len(lines) or countFile < 0:
                return None, None, None, None  # Return None if countFile is out of bounds

            last_line = lines[countFile].strip()
            x_pix, y_pix, hotspotNum, typeDetect = last_line.split(", ")
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix, hotspotNum, typeDetect
            
    except FileNotFoundError:
        print("Error: The file coordinates.txt does not exist.")
        return None, None, None, None
    except ValueError:
        print("Error: The line does not contain the expected number of elements.")
        return None, None, None, None

# x_pixel,y_pixel,detctNum,typeDetect = readingStepCoordinates(0)
# print(x_pixel,y_pixel,detctNum,typeDetect)
def readingAllCoordinates():
    try:
        with open("coordinates.txt", "r") as file:
            for line in file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:  # Ensure the line is not empty
                    print(f"Processing line: '{line}'")  # Debugging: show the raw line content
                    parts = line.split(", ")
                    if len(parts) != 4:
                        print(f"Error: The line does not contain exactly 4 elements. Found {len(parts)} elements.")
                        continue
                    x_pix, y_pix, hotspotNum, typeDetect = parts
                    x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
                    print(f"x: {x_pix}, y: {y_pix}, hotspotNum: {hotspotNum}, typeDetect: {typeDetect}")
                    
    except FileNotFoundError:
        print("Error: The file coordinates.txt does not exist.")
    except ValueError as ve:
        print(f"Error: {ve}")

def get_coordinates_from_file(filename, line_number):
    try:
        with open("coordinates.txt", "r") as file:
            lines = file.readlines()
            if line_number >= len(lines) or line_number < 0:
                raise IndexError("Error: Line number out of range.")

            line = lines[line_number].strip()  # Remove any leading/trailing whitespace
            print(f"Raw line content: '{line}'")  # Debug: show the raw line content

            # Split using just a comma as the delimiter
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Error: The line does not contain exactly 4 elements after splitting. Found {len(parts)} element(s).")

            x_pix, y_pix, hotspotNum, typeDetect = parts
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix, hotspotNum, typeDetect

    except FileNotFoundError:
        print("Error: The file does not exist.")
    except IndexError as ie:
        print(ie)
    except ValueError as ve:
        print(ve)


def clearCoordinatesFile():
    with open("coordinates.txt", "w") as f:
        pass  # Opening in write mode with 'w' will clear the file content


def readingLastCoordinates():
    with open("coordinates.txt", "r") as file:
            lines = file.readlines()
           
            line = lines[-1].strip()  # Remove any leading/trailing whitespace
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Error: The line does not contain exactly 4 elements after splitting. Found {len(parts)} element(s).")

            x_pix, y_pix, hotspotNum, typeDetect = parts
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix, hotspotNum, typeDetect
        

def readingStepCoordinates(countFile):
    with open("coordinates.txt", "r") as file:
            lines = file.readlines()
            if countFile >= len(lines) or countFile < 0:
                raise IndexError("Error: Line number out of range.")

            line = lines[countFile].strip()  # Remove any leading/trailing whitespace
            #print(f"Raw line content: '{line}'")  # Debug: show the raw line content
            # Split using just a comma as the delimiter
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Error: The line does not contain exactly 4 elements after splitting. Found {len(parts)} element(s).")

            x_pix, y_pix, hotspotNum, typeDetect = parts
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix, hotspotNum, typeDetect

print(readingLastCoordinates())
print(readingStepCoordinates(0))
clearCoordinatesFile()