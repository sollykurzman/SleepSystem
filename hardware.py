#!/usr/bin/python3

import threading
import time
import os
import math
from dotenv import load_dotenv
from requests import Session, RequestException

try:
    from gpiozero import PWMOutputDevice, Button
except ImportError:
    print("Running on mac, using Mock Hardware")
    # define mock classes for non-pi environments
    class PWMOutputDevice:
        def __init__(self, *args, **kwargs): self.value = 0
        def off(self): pass
    class Button:
        def __init__(self, *args, **kwargs): pass
        @property
        def is_pressed(self): return False

# load environment variables
load_dotenv()

# hardware configuration
BUZZER_PIN = 23
BUTTON_PIN = 24
HA_URL = "http://192.168.1.39:8123/api/services" 
HA_TOKEN = os.environ.get("HA_AUTH_TOKEN")

class HardwareController:
    def __init__(self):
        # initialize threading control
        self._stop_event = threading.Event()
        self._alarm_thread = None
        self._fade_thread = None
        
        # setup persistent http session
        self.session = Session()
        self.headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }

    def stop_all(self):
        print("Hardware: Stopping all actions.")
        
        # signal all threads to stop
        self._stop_event.set()
        
        # wait for threads to finish
        if self._alarm_thread: self._alarm_thread.join()
        if self._fade_thread: self._fade_thread.join()
        
        # reset signal for next use
        self._stop_event.clear() 

    def trigger_alarm(self):
        # check if alarm is already active
        if self._alarm_thread and self._alarm_thread.is_alive():
            return 

        # start buzzer logic in background
        self._alarm_thread = threading.Thread(target=self._run_buzzer, daemon=True)
        self._alarm_thread.start()

    def trigger_fade(self, duration=1800):
        # check if fade is already active
        if self._fade_thread and self._fade_thread.is_alive():
            return 

        # start light fade logic in background
        self._fade_thread = threading.Thread(target=self._run_lights, args=(duration,), daemon=True)
        self._fade_thread.start()

    def _run_buzzer(self):
        print("Hardware: Buzzer Started")
        
        # initialize hardware devices
        buzzer = PWMOutputDevice(BUZZER_PIN, frequency=2000)
        button = Button(BUTTON_PIN)
        
        # define beep pattern (on_time, off_time)
        pattern = [(0.1, 0.1), (0.1, 0.6)] 
        
        try:
            # loop until stopped externally
            while not self._stop_event.is_set():
                # check for manual stop via physical button
                if button.is_pressed:
                    print("Hardware: Button pressed, stopping.")
                    break
                
                # cycle through beep pattern
                for on_t, off_t in pattern:
                    if self._stop_event.is_set() or button.is_pressed: break
                    
                    # turn buzzer on
                    buzzer.value = 1.0
                    time.sleep(on_t)
                    
                    # turn buzzer off
                    buzzer.off()
                    time.sleep(off_t)
        finally:
            # ensure buzzer is off on exit
            buzzer.off()
            self._stop_event.set() 

    def _run_lights(self, duration):
        print("Hardware: Fading lights")
        
        # configure fade parameters
        steps = 255
        step_time = duration / steps
        entities = ["light.bed_light", "light.desk_overhead_light", "light.main_bedroom_light"]
        
        start_time = time.monotonic()
        
        for i in range(steps + 1):
            # stop if interrupted
            if self._stop_event.is_set(): break
            
            # calculate brightness using cosine curve for smoothness
            x = i / steps
            brightness = 78 * (1 - math.cos(x * math.pi)) / 2
            
            # send command to home assistant
            self._set_ha_light(entities, brightness)
            
            # drift correction loop sleep
            next_target = start_time + (i + 1) * step_time
            sleep_for = next_target - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _set_ha_light(self, entities, brightness):
        payload = {"entity_id": entities, "brightness": int(brightness)}
        try:
            # post request to home assistant api
            self.session.post(
                f"{HA_URL}/light/turn_on",
                headers=self.headers,
                json=payload,
                timeout=2
            )
        except Exception as e:
            print(f"HA Error: {e}")

# create singleton instance
hw = HardwareController()