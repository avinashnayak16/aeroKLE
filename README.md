# Pixhawk and Raspberrypi Integration
  - [Previous version of raspbery pi](https://downloads.raspberrypi.org/raspios_full_armhf/images/raspios_full_armhf-2021-05-28/) 

<details>
  <summary>Raspberry Pi-4 Setup</summary>
  
  ## Raspberry Pi OS setup
  - Install RPI software using “Imager” to SD card or Pen Drive.
    - [Imager Software](https://www.raspberrypi.com/software/)
      
    ![image](https://github.com/user-attachments/assets/349d6c34-065e-426c-b4d7-142feadab6cc)

  - Connect Pi SSH using Wi-Fi.
    - Enable VNC using putty or Windows PowerShell ( "ssh @piexample"    ip_address or host_name)
    - add below two lines at bottom of file `sudo nano /boot/config.txt` ,if VNC not working
      ```python
      hdmi_force_hotplug=1
      hdmi_group=2
      hdmi_mode=9
      ```
       [Youtube Link](https://youtu.be/hA9r13ZUS08?si=trx05AKz2boaaN3q)
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
  >  

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
