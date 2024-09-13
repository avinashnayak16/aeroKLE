- **This repository provides a comprehensive setup guide for integrating a Raspberry Pi with a Pixhawk flight controller. It includes detailed instructions on configuring the hardware connections, setting up the software environment, and troubleshooting common issues.** 
- **Additionally, the repository features sample code for problem statements, demonstrating how to handle specific scenarios and challenges related to the integration. Whether you're working on autonomous flight systems or experimenting with UAV technology, this repository offers valuable resources and code snippets to streamline your project.**


###  [Dronekit library github](https://github.com/dronekit/dronekit-python)

# Pixhawk and Raspberrypi Integration
  - [**Previous version of raspbery pi**](https://downloads.raspberrypi.org/raspios_full_armhf/images/raspios_full_armhf-2021-05-28/) 

<details>
  <summary>Raspberry Pi-4 Setup</summary>
  
  ## Raspberry Pi OS setup
  For Raspberry pi-4 
  Raspberry Pi OS (64-bit)
  A port of Debian Bookworm with the Raspberry Pi Desktop (Recommended)
  - Install RPI software using “Imager” to SD card or Pen Drive.
    - [Imager Software](https://www.raspberrypi.com/software/)
      
    ![image](https://github.com/user-attachments/assets/90ce022a-ab6f-4ffb-a4e1-61166b89bd36)


  - Connect Pi SSH using Wi-Fi.
    - Enable VNC using putty or Windows PowerShell ( "ssh @piexample"    ip_address or host_name)
    - add below two lines at bottom of file `sudo nano /boot/config.txt` ,if VNC not working
      ```python
      hdmi_force_hotplug=1
      hdmi_group=2
      hdmi_mode=9
      ```
  - Power the RPI using BEC module.
      - Check port
      
      >  ```bash
      >  ls /dev/ttyAMA0
      >  ```
  - add below two lines at bottom of file  `sudo nano /boot/config.txt` ,if not there
        
      >  ```bash
      >  enable_uart=1
      >  dtoverlay=disable-bt
      >  ```
       [Youtube Link](https://youtu.be/hA9r13ZUS08?si=trx05AKz2boaaN3q)
</details>

<details>
  <summary>Raspberry Pi-5 Setup</summary>
  
  ## Raspberry Pi OS setup
  Using Raspberry Pi Imager, flash the Raspberry Pi OS compatible with Raspberry Pi 5     
  (Recommended: Rasperry Pi OS (Debian Bookworm) Full 64-bit with Desktop Environment and 
   Recommended applications) on a SD Card .
  - Install RPI software using “Imager” to SD card or Pen Drive.
    - [Imager Software](https://www.raspberrypi.com/software/)
      
    ![image](https://github.com/user-attachments/assets/5ee760ea-91b6-4d39-85e0-94de57e17ef9)

  - Connect Pi SSH using Wi-Fi.
    - Enable VNC using putty or Windows PowerShell ( "ssh @piexample"    ip_address or host_name)
    - Set up serial connection and type the following in SSH:
      > ```bash
      > sudo raspi-config
      > ```
    - Change the folowing settings:
        a) Go to interface settings
        
        b) Enable SSH
        
        c) Enable VNC
        
        d) Go to serial
        
        e) When prompted, select no to 'Would you like a login shell to be accessible over serial?'
        
        f) When prompted, select yes to 'Would you like the serial port hardware to be enabled?'.
        
        g) Reboot the Raspberry Pi using sudo reboot when you are done.
        
        h) The Raspberry Pi’s serial port will now be usable on /dev/serial0.
  - Run the following commands:
    > ```bash
    > sudo apt-get update
    > sudo apt-get upgrade
    > ```
  - Create a virtual environment to install any external packages:
    > ```python
    > python3 -m venv myenv
    > source myenv/bin/activate
    > ```
  - Install required Python packages:
      - (picamera2 library is required in Debian Bookworm if you are using a Raspberry Pi Camera)
    > ```python
    > pip install future
    > pip install lxml
    > pip install picamera2
    > ```
  - If you want to activate the virtual environment everytime the terminal is opened, go to nano ~/.bashrc and add the following line at the end:
     > ```bash
     > source ~/myenv/bin/activate
     > ```
     > Save the file and exit the text editor (in nano, you do this by pressing CTRL + X, then Y, and Enter).
     > To apply the changes immediately without needing to restart the terminal, run:
     > ```bash
     > source ~/.bashrc
     > ```
  - To deactivate the virtual environment when not required, run:
      > ```bash
      > deactivate
      > ```

  - Power the RPI using BEC module.
    - **Power the Raspberry Pi using BEC module. Make sure that the power supply used is atleast 5V/3A (Recommended: 5V/5A (25 W to 27W)). Power supply less than 5V/3A may cause performance issues or the Pi may end up abruptly crashing or shutting down.**
      - Check port        
      >  ```bash
      >  ls /dev/ttyAMA0
      >  ```
      - add below two lines at bottom of file  `sudo nano /boot/firmware/config.txt` ,if not there
        
      >  ```bash
      >  enable_uart=1
      >  dtoverlay=disable-bt
      >  ```

</details>


## 1.Connect to Raspberry pi
  - Connect Pi VNC using Wi-Fi.
  > - Run the Following commands,
  > ```python
  >  sudo apt-get update
  >  sudo apt-get upgrade
  >  sudo pip3 install pyserial
  >  sudo pip3 install dronekit
  >  sudo pip3 install geopy
  >  sudo pip3 install MAVProxy
  > ```

  - Set up serial connection, type following in ssh
  > ```bash
  > sudo raspi-config
  - After opens setting follow these step
    a. goto interface options,
    
    b. go to serial,
    
    c. When prompted, select no to “Would you like a login shell to be accessible over serial?” 
    
    d. When prompted, select yes to “Would you like the serial port hardware to be enabled?”.    
    e. Reboot the Raspberry Pi when you are done.
    
    f. The Raspberry Pi’s serial port will now be usable on `/dev/serial0.`

- Set following parameters in mission planner,
  
  `SERIAL2_PROTOCOL = 2`
  
  `SERIAL2_BAUD = 921`
  
    if required do following also,
  
    `LOG_BACKEND_TYPE = 3`

      
- Now connect Pixhawk and Raspberry pi, as shown in,
  ![image](https://github.com/user-attachments/assets/56a0fee3-f292-4f83-a284-e47ca6003ab8)


  - Now type the following to get the telemetry data of pixhawk,
  >```bash
  > mavproxy.py --master=/dev/serial0 --baudrate 921600
  >           (or)
  > mavproxy.py --master=/dev/ttyAMA0 --baudrate 921600
 - Type the following if you want telemetry data to be displayed in mission planner,
  >  ```bash
  >  mavproxy.py --master=/dev/serial0 --baudrate 921600 --out udp:127.0.0.1:14552
  >  
  >  /*Here,
  >   '127.0.0.1' Your PC's IP Adress, Obtained by typing 'ipconfig' in command prompt
  >   '14552' is the port to which you need to connect to mission planner using UDP
  >  */

## 2.To Run Python code type,
>  ```bash
>  python3 demo.py --connect /dev/ttyAMA0
