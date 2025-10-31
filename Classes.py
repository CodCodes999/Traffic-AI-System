import pygame as p
import random as r
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import os
import subprocess
import pickle

class Car(p.sprite.Sprite):
        def __init__(self, x, y, angle, road, opposite_road, signal, dir_to_turn, roads):
            super().__init__()
            self.direction_turning = dir_to_turn
            self.image_dir = p.image.load("imgs/" + self.direction_turning + ".png")
            self.image = p.transform.rotate(self.image_dir, angle)
            self.rect = self.image.get_rect(topleft=(x, y))
            self.rect.x = x
            self.rect.y = y
            self.angle_facing_initial = angle
            self.DIR = angle
            self.speed = 4
            self.turn_progress = 0
            self.TURN_RADIUS_RIGHT = 59
            self.TURN_RADIUS_LEFT = 36
            self.SAFE_DISTANCE_TO_TURN = 50
            self.road = road
            self.opposite_road = opposite_road
            self.signal = signal
            self.should_move = False
            self.has_turned = False
            self.should_turn = False
            self.in_junction = False
            self.complete_turn = True
        
        def rotate_image(self, theta_anticlockwise):
            """Roatets car image anticlockwise by theta degrees"""
            rotated_image = p.transform.rotate(self.image, theta_anticlockwise)
            self.rect = rotated_image.get_rect(center=self.rect.center)
            self.image = rotated_image

        def draw(self, screen):
            """Displayes car"""
            screen.blit(self.image, self.rect)
        
        def move(self):
            """Moves the car forward or turns"""
            if self.angle_facing_initial == 0:
                self.rect.y -= self.speed
            elif self.angle_facing_initial == 90:
                self.rect.x -= self.speed
            elif self.angle_facing_initial == 180:
                self.rect.y += self.speed
            elif self.angle_facing_initial == 270:
                self.rect.x += self.speed

        def check_move_and_turn(self):
            """Checks if the car should move or turn"""
            if (self.angle_facing_initial == 0 and self.rect.y > 305) or \
               (self.angle_facing_initial == 90 and self.rect.x > 305) or \
               (self.angle_facing_initial == 180 and self.rect.y < 185) or \
               (self.angle_facing_initial == 270 and self.rect.x < 185):
                self.in_junction = False
            else:
                self.in_junction = True

            car_ahead = self.road.cars[self.road.cars.index(self) - 1] if self.road.cars.index(self) > 0 else None

            if car_ahead:
                if self.DIR == 0:
                    self.should_move = (self.rect.y > (car_ahead.rect.y + (self.road.gap_between_cars + 24)))
                elif self.DIR == 90:
                    self.should_move = (self.rect.x > (car_ahead.rect.x + (self.road.gap_between_cars + 24)))
                elif self.DIR == 180:
                    self.should_move = (self.rect.y < (car_ahead.rect.y - (self.road.gap_between_cars + 24)))
                elif self.DIR == 270:
                    self.should_move = (self.rect.x < (car_ahead.rect.x - (self.road.gap_between_cars + 24)))
            else:
                self.should_move = True
            self.should_turn = self.should_move and self.in_junction

            if not self.in_junction:

                if self.signal.colour == "red" or (self.signal.colour == "yellow" and self.signal.previous_colour == "red"):
                    if not car_ahead or car_ahead.in_junction:
                        if self.DIR == 0:
                            self.should_move = (self.rect.y > self.road.first_car_pos[1]+4)
                        elif self.DIR == 90:
                            self.should_move = (self.rect.x > self.road.first_car_pos[0]+4)
                        elif self.DIR == 180:
                            self.should_move = (self.rect.y < self.road.first_car_pos[1]-4)
                        elif self.DIR == 270:
                            self.should_move = (self.rect.x < self.road.first_car_pos[0]-4)
                    elif not car_ahead.in_junction:
                        if self.DIR == 0:
                            self.should_move = (self.rect.y > car_ahead.rect.y + (self.road.gap_between_cars + 24))
                        elif self.DIR == 90:
                            self.should_move = (self.rect.x >  car_ahead.rect.x + (self.road.gap_between_cars + 24))
                        elif self.DIR == 180:
                            self.should_move = (self.rect.y < car_ahead.rect.y - (self.road.gap_between_cars + 24))
                        elif self.DIR == 270:
                            self.should_move = (self.rect.x < car_ahead.rect.x - (self.road.gap_between_cars + 24))

                elif self.signal.colour == "green" or (self.signal.colour == "yellow" and self.signal.previous_colour == "green"):
                    if car_ahead:
                        if self.DIR == 0:
                            self.should_move = (self.rect.y > (car_ahead.rect.y + (self.road.gap_between_cars + 24)))
                        elif self.DIR == 90:
                            self.should_move = (self.rect.x > (car_ahead.rect.x + (self.road.gap_between_cars + 24)))
                        elif self.DIR == 180:
                            self.should_move = (self.rect.y < (car_ahead.rect.y - (self.road.gap_between_cars + 24)))
                        elif self.DIR == 270:
                            self.should_move = (self.rect.x < (car_ahead.rect.x - (self.road.gap_between_cars + 24)))
                    else:
                        self.should_move = True
                    self.should_turn = self.should_move and self.in_junction

            if self.in_junction:
                if self.direction_turning != 'right':
                    if car_ahead:
                        self.should_move = car_ahead.should_move
                    else:
                        self.should_move = True
                    self.should_turn = True
                else:
                    for car in self.opposite_road.cars:
                        if self.turn_progress == 10:
                            car_ahead = self.road.cars[self.road.cars.index(self) - 1] if self.road.cars.index(self) > 0 else None    
                            if car_ahead:
                                if (self.rect.y > (car_ahead.rect.y + (self.road.gap_between_cars + 24))) and self.DIR == 0 or \
                                   (self.rect.x > (car_ahead.rect.x + (self.road.gap_between_cars + 24))) and self.DIR == 90 or \
                                   (self.rect.y < (car_ahead.rect.y - (self.road.gap_between_cars + 24))) and self.DIR == 180 or \
                                   (self.rect.x < (car_ahead.rect.x - (self.road.gap_between_cars + 24))) and self.DIR == 270:     
                                    self.should_move = True
                                else:
                                    self.should_move = False
                            else:
                                self.should_move = True
             
                        elif (25 < self.turn_progress <= 40) and (self.direction_turning == 'right'):
                            if car.direction_turning == 'forward':
                                if (car.DIR == 180 and 181 < car.rect.y < 303 and car.should_move) or \
                                    (car.DIR == 270 and 181 < car.rect.x < 303 and car.should_move) or \
                                    (car.DIR == 90 and 210 < car.rect.x < 332 and car.should_move) or \
                                    (car.DIR == 0 and 210 < car.rect.y < 332 and car.should_move):
                                    self.should_move = False
                                    self.should_turn = False
                            elif car.direction_turning == 'left' and car.in_junction:
                                self.should_move = False
                                self.should_turn = False
                            elif car.direction_turning == 'right' and car.in_junction:
                                self.should_move = True
                                self.should_turn = True
                        else:
                            self.should_move = True
                            self.should_turn = True
                
                if (self.angle_facing_initial == 0 and self.rect.y < 185) or \
                   (self.angle_facing_initial == 90 and self.rect.x < 185) or \
                   (self.angle_facing_initial == 180 and self.rect.y > 305) or \
                   (self.angle_facing_initial == 270 and self.rect.x > 305) or self.has_turned: # Has passed junction
                    self.should_move = True
                    self.should_turn = False

        def turn(self):
            """Turns the car in the direction it is supposed to turn"""
            if self.direction_turning == "right":
                self.turn_right()
            elif self.direction_turning == "left":
                self.turn_left()

        def turn_right(self):
            """Quarter-circle right turn — robust start/end by computing start angle & radius from rect.center"""
            if self.turn_progress == 0:
                # set the correct centre for this approach (your original values)
                if self.angle_facing_initial == 0:       # Up → Right
                    self.center_x, self.center_y = 304, 304
                elif self.angle_facing_initial == 90:    # Left → Up
                    self.center_x, self.center_y = 304, 209
                elif self.angle_facing_initial == 180:   # Down → Left
                    self.center_x, self.center_y = 209, 209
                elif self.angle_facing_initial == 270:   # Right → Down
                    self.center_x, self.center_y = 209, 304

                self.original_image = p.transform.rotate(self.image_dir, self.angle_facing_initial)

                cx, cy = self.center_x, self.center_y
                x0, y0 = self.rect.centerx, self.rect.centery
                dx = x0 - cx
                dy = cy - y0
                self.TURN_RADIUS_RIGHT = math.hypot(dx, dy) or 1.0
                self.start_angle = math.atan2(dy, dx)
             
            if self.should_turn:
                self.turn_progress += 5
                theta = math.radians(self.turn_progress)

                angle = self.start_angle - theta

                self.rect.centerx = self.center_x + self.TURN_RADIUS_RIGHT * math.cos(angle)
                self.rect.centery = self.center_y - self.TURN_RADIUS_RIGHT * math.sin(angle)

                rotated_image = p.transform.rotate(self.original_image, -self.turn_progress)
                self.image = rotated_image
                self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.centery))

                if self.turn_progress >= 90: # stop at 35 degrees if there is a car turning
                    self.direction_turning = None
                    self.turn_progress = 0
                    self.has_turned = True
                    self.angle_facing_initial = (self.angle_facing_initial - 90) % 360
                    self.image = p.transform.rotate(self.original_image, -90)
                    self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.centery))

        def turn_left(self):
            """Quarter-circle left turn"""
            if self.turn_progress == 0:
                if self.angle_facing_initial == 0:
                    self.center_x, self.center_y = 209, 304
                elif self.angle_facing_initial == 90:
                    self.center_x, self.center_y = 304, 304
                elif self.angle_facing_initial == 180:
                    self.center_x, self.center_y = 304, 209
                elif self.angle_facing_initial == 270:
                    self.center_x, self.center_y = 209, 209
                self.start_angle = math.radians(self.angle_facing_initial)
                self.original_image = p.transform.rotate(self.image_dir, self.angle_facing_initial)

            self.turn_progress += 5  # step size in degrees
            theta = math.radians(self.turn_progress)

            # Left turn: add theta to start_angle
            angle = self.start_angle + theta
            self.rect.centerx = self.center_x + self.TURN_RADIUS_LEFT * math.cos(angle)
            self.rect.centery = self.center_y - self.TURN_RADIUS_LEFT * math.sin(angle)

            rotated_image = p.transform.rotate(self.original_image, self.turn_progress)
            self.image = rotated_image
            self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.centery))

            if self.turn_progress >= 90:
                self.direction_turning = None
                self.turn_progress = 0
                self.has_turned = True
                self.angle_facing_initial = (self.angle_facing_initial + 90) % 360
                self.image = p.transform.rotate(self.original_image, 90)
                self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.centery))


class Road():
    def __init__(self, first_car_pos, gap_between_cars, angle, signal, opposite_road=None, roads=None):
        self.dir = angle
        self.roads = roads
        self.opposite_road = opposite_road
        self.first_car_pos = first_car_pos
        self.gap_between_cars = gap_between_cars
        self.cars = []
        self.init_num_cars = 5
        self.car_spawn_rate = 2
        l = 30
        r = 30
        f = 40
        self.dirs_to_turn = ['left'] * l + ['forward'] * f + ['right'] * r
        self.car_length = 24
        self.signal = signal

    def get_data(self):
        """Gets user input for road parameters"""
        print(f"-------------------------------------------------------------------------------------------- Lane {self.dir}: --------------------------------------------------------------------------------------------")
        while True:
            self.init_num_cars = int(input(f"Number of cars initially - {self.dir}: ")) # max 50 cars
            if self.init_num_cars <= 50:
                break
            print("Too many cars! Please enter a number less than or equal to 50.")
        
        while True:
            self.car_spawn_rate = int(input(f"Percentage Chance of Car Spawning / Second - {self.dir}: "))
            if self.car_spawn_rate <= 100:
                break
            print("Too many cars spawning! Please enter a number less than or equal to 100.")
        
        while True:
            car_going_l = int(input(f"Left - {self.dir}: (out of 100) "))
            car_going_f = int(input(f"Right - {self.dir}: (out of 100) "))
            car_going_r = int(input(f"Forward - {self.dir}: (out of 100) "))
            if car_going_l + car_going_f + car_going_r == 100:
                self.dirs_to_turn = ['left'] * car_going_l + ['forward'] * car_going_f + ['right'] * car_going_r
                break
            print("Percentages must add up to 100! Please try again.")

    def get_spawn_offset(self, n):
        """Get the (x, y) coords for the nth car to be spawned on this road"""
        if self.dir == 0:
            dx = 0
            dy = n * (self.car_length + self.gap_between_cars)
        elif self.dir == 90:
            dx = n * (self.car_length + self.gap_between_cars)
            dy = 0
        elif self.dir == 180:
            dx = 0
            dy = -n * (self.car_length + self.gap_between_cars)
        elif self.dir == 270:
            dx = -n * (self.car_length + self.gap_between_cars)
            dy = 0
        return (dx, dy)

    def init_cars(self):
        """Spawns the Initial cars on this road"""
        for i in range(self.init_num_cars):
            dx, dy = self.get_spawn_offset(i)
            x = self.first_car_pos[0] + dx
            y = self.first_car_pos[1] + dy
            new_car = Car(x, y, self.dir, self, self.opposite_road, self.signal, r.choice(self.dirs_to_turn), self.roads)
            self.cars.append(new_car)

    def spawn_car(self):
        """Always spawns the car right behind the last one. If it's the first car, spawn at the fixed starting coords (to be filled)."""
        if r.randint(1, 100) <= self.car_spawn_rate:
            last_car_in_junc = None
            for car in self.cars[-1::-1]:
                    if not car.in_junction:
                        last_car_in_junc = car
                        break
                    else:
                        last_car_in_junc = None
            if last_car_in_junc == None:
                if self.dir == 0:
                    x, y = 238, 390
                elif self.dir == 90:
                    x, y = 390, 260
                elif self.dir == 180:
                    x, y = 260, 100
                elif self.dir == 270:
                    x, y = 100, 238
            else:
                if self.dir == 0:
                    x = last_car_in_junc.rect.x
                    y = last_car_in_junc.rect.y + self.car_length + self.gap_between_cars
                elif self.dir == 90:
                    x = last_car_in_junc.rect.x + self.car_length + self.gap_between_cars
                    y = last_car_in_junc.rect.y
                elif self.dir == 180:
                    x = last_car_in_junc.rect.x
                    y = last_car_in_junc.rect.y - (self.car_length + self.gap_between_cars)
                elif self.dir == 270:
                    x = last_car_in_junc.rect.x - (self.car_length + self.gap_between_cars)
                    y = last_car_in_junc.rect.y

            new_car = Car(x, y, self.dir, self, self.opposite_road, self.signal, r.choice(self.dirs_to_turn), self.roads)
            self.cars.append(new_car)
    
    def remove(self):
        """Removes cars that have exited the screen"""
        for car in self.cars:
            if (car.angle_facing_initial == 0 and car.rect.y < 173) or \
               (car.angle_facing_initial == 90 and car.rect.x < 173) or \
               (car.angle_facing_initial == 180 and car.rect.y > 330) or \
               (car.angle_facing_initial == 270 and car.rect.x > 330):
                self.cars.remove(car)


class Signal(p.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.image = p.Surface((2, 2))
        self.previous_colour = "white"
        self.colour = "white"
        self.image.fill(self.colour)
        self.rect = self.image.get_rect(topleft=(x, y))
    
    def become_green(self):
        self.previous_colour = "green"
        self.colour = "green"
    
    def become_red(self):
        self.previous_colour = "red"
        self.colour = "red"
    
    def become_yellow(self):
        self.colour = "yellow"
    
    def draw(self, screen):
        self.image.fill(self.colour)
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        screen.blit(self.image, self.rect)


class AIAccurate:
    def __init__(self, signals, roads, alpha=0.1, gamma=0.9, epsilon=0.1,
                 eval_window_seconds=3.0, qtable_file="qtable_acc.pkl",
                 time_scale=0.2, min_green_time=3.0):
        """
        Q-learning traffic controller using only total accurate car distance as input.
        - signals: list of signal objects
        - roads: list of road objects (each has .cars, each car has .length)
        - time_scale: multiplier to adjust how quickly the AI evaluates actions
        - min_green_time: minimum time (s) a signal stays before switching
        """
        self.signals = signals
        self.roads = roads
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_window = eval_window_seconds * time_scale  # faster evaluation
        self.qtable_file = qtable_file
        self.min_green_time = min_green_time

        # Q-table: {state_tuple: [Q_no_switch, Q_switch]}
        self.q_table = {}

        # Tracking last decision
        self.last_state = None
        self.last_action = None
        self.last_total_dist = None
        self.last_action_time = None
        self.last_switch_time = time.time()  # 🕒 track last switch
        self.total_updates = 0

        with open("Data Analysis/Run Number.txt", "a+") as f:
            f.seek(0)
            lines = f.readlines()
            if lines:
                i = int(lines[-1].strip()) + 1
            else:
                i = 1
            f.write(f"{i}\n")
        self.file_path = f"Data Analysis/Run {i} Acc.xlsx"

        # Load existing Q-table if available
        self._load_qtable()

    # -------------------- State & Metric helpers --------------------
    def _get_state(self):
        """State = tuple of bucketed total distances for each road."""
        state = []
        for road in self.roads:
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = max(last_car.rect.y + 24, 305) - min(last_car.rect.y + 24, 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = max(last_car.rect.x + 24, 305) - min(last_car.rect.x + 24, 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = max(205, last_car.rect.y) - min(205, last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = max(205, last_car.rect.x) - min(205, last_car.rect.x)
            else:
                total_dist = 0
            bucket = min(10, int(total_dist // 100))  # 0–10 buckets
            state.append(bucket)
        return tuple(state)

    def _total_distance_metric(self):
        """Total congestion = sum of all car distances across all roads."""
        total = 0
        for road in self.roads:
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = max(last_car.rect.y + 24, 305) - min(last_car.rect.y + 24, 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = max(last_car.rect.x + 24, 305) - min(last_car.rect.x + 24, 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = max(205, last_car.rect.y) - min(205, last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = max(205, last_car.rect.x) - min(205, last_car.rect.x)
            else:
                total_dist = 0
            total += total_dist
        return total

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]

    # -------------------- Decision logic --------------------
    def calculate_switch(self):
        """Decide whether to switch lights this frame."""
        now = time.time()
        state = self._get_state()
        self._ensure_state(state)

        # 🕒 Enforce minimum green time between switches
        if now - self.last_switch_time < self.min_green_time:
            return False  # too soon — don’t switch yet

        # Evaluate previous decision if enough time passed
        if self.last_action is not None and (now - self.last_action_time >= self.eval_window):
            self._evaluate_and_update(state)
            self.last_action = None
            self.last_state = None
            self.last_action_time = None
            self.last_total_dist = None

        # Epsilon-greedy decision
        if r.random() < self.epsilon:
            action = r.choice([0, 1])  # explore
        else:
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))  # exploit

        # Log decision data
        self._log_data(stage="decision")

        # Record data if we switch
        if action == 1:
            self.last_state = state
            self.last_action = action
            self.last_total_dist = self._total_distance_metric()
            self.last_action_time = now
            self.last_switch_time = now  # ✅ reset the cooldown timer

        return bool(action)

    # -------------------- Learning update --------------------
    def _evaluate_and_update(self, next_state):
        """Compute reward and update Q-table."""
        self._ensure_state(next_state)
        prev_dist = self.last_total_dist
        curr_dist = self._total_distance_metric()

        # Reward: positive if total distance (congestion) decreases
        reward = float(prev_dist - curr_dist)
        reward = max(-1.0, min(1.0, reward / (abs(prev_dist) + 1e-6)))

        s, a = self.last_state, self.last_action
        old_value = self.q_table[s][a]
        next_max = max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[s][a] = new_value

        self._log_data(stage="result")
        self.total_updates += 1

        # Gradual epsilon decay
        if self.total_updates % 50 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.98)

        # Periodic saving
        if self.total_updates % 100 == 0:
            self._save_qtable()

    # -------------------- Data Logging --------------------
    def _log_data(self, stage):
        """Store total distance + number of cars for each road."""
        data = {"time": [time.strftime("%H:%M:%S")], "stage": [stage]}

        for i, road in enumerate(self.roads):
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = max(last_car.rect.y + 24, 305) - min(last_car.rect.y + 24, 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = max(last_car.rect.x + 24, 305) - min(last_car.rect.x + 24, 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = max(205, last_car.rect.y) - min(205, last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = max(205, last_car.rect.x) - min(205, last_car.rect.x)
            else:
                total_dist = 0
            num_cars = len(road.cars)
            data[f"road{i*90}_dist"] = [total_dist]
            data[f"road{i*90}_cars"] = [num_cars]

        df_new = pd.DataFrame(data)

        if not os.path.exists(self.file_path):
            df_new.to_excel(self.file_path, index=False)
        else:
            df_existing = pd.read_excel(self.file_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(self.file_path, index=False)

    def open_data_file(self):
        """Open Excel file when simulation ends."""
        self._save_qtable()
        subprocess.run(["open", self.file_path])

    # -------------------- Q-table persistence --------------------
    def _save_qtable(self):
        with open(self.qtable_file, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[INFO] Q-table saved ({len(self.q_table)} states).")

    def _load_qtable(self):
        if os.path.exists(self.qtable_file):
            with open(self.qtable_file, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"[INFO] Loaded Q-table from {self.qtable_file} ({len(self.q_table)} states).")
        else:
            print("[INFO] No existing Q-table found. Starting fresh.")


class AIInaccurate:
    def __init__(self, signals, roads, alpha=0.1, gamma=0.9, epsilon=0.1,
                 eval_window_seconds=3.0, qtable_file="qtable_inacc.pkl",
                 time_scale=0.2, min_green_time=3.0):
        import time, os, pickle, random as r, subprocess
        import numpy as np
        import pandas as pd

        self.signals = signals
        self.roads = roads
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_window = eval_window_seconds * time_scale
        self.qtable_file = qtable_file
        self.min_green_time = min_green_time

        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_total_dist = None
        self.last_action_time = None
        self.last_switch_time = time.time()
        self.total_updates = 0

        with open("Data Analysis/Run Number.txt", "a+") as f:
            f.seek(0)
            lines = f.readlines()
            if lines:
                i = int(lines[-1].strip()) + 1
            else:
                i = 1
            f.write(f"{i}\n")
        self.file_path = f"Data Analysis/Run {i} Inacc.xlsx"

        self._load_qtable()

    # -------------------- State & Metric helpers --------------------
    def _get_state(self):
        """State = tuple of rounded total distances (nearest 100 px) for each road."""
        state = []
        for road in self.roads:
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = abs((last_car.rect.y + 24) - 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = abs((last_car.rect.x + 24) - 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = abs(205 - last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = abs(205 - last_car.rect.x)
            else:
                total_dist = 0

            # 🔹 Round to nearest 100 px
            rounded_dist = int(round(total_dist / 100.0) * 100)
            state.append(rounded_dist)
        return tuple(state)

    def _total_distance_metric(self):
        """Total congestion = sum of all rounded car distances."""
        total = 0
        for road in self.roads:
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = abs((last_car.rect.y + 24) - 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = abs((last_car.rect.x + 24) - 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = abs(205 - last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = abs(205 - last_car.rect.x)
            else:
                total_dist = 0

            # 🔹 Round to nearest 100 px before summing
            total += int(round(total_dist / 100.0) * 100)
        return total

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]

    # -------------------- Decision logic --------------------
    def calculate_switch(self):
        import time, random as r, numpy as np

        now = time.time()
        state = self._get_state()
        self._ensure_state(state)

        if now - self.last_switch_time < self.min_green_time:
            return False

        if self.last_action is not None and (now - self.last_action_time >= self.eval_window):
            self._evaluate_and_update(state)
            self.last_action = None
            self.last_state = None
            self.last_action_time = None
            self.last_total_dist = None

        if r.random() < self.epsilon:
            action = r.choice([0, 1])
        else:
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))

        self._log_data(stage="decision")

        if action == 1:
            self.last_state = state
            self.last_action = action
            self.last_total_dist = self._total_distance_metric()
            self.last_action_time = now
            self.last_switch_time = now

        return bool(action)

    # -------------------- Learning update --------------------
    def _evaluate_and_update(self, next_state):
        import numpy as np

        self._ensure_state(next_state)
        prev_dist = self.last_total_dist
        curr_dist = self._total_distance_metric()

        reward = float(prev_dist - curr_dist)
        reward = max(-1.0, min(1.0, reward / (abs(prev_dist) + 1e-6)))

        s, a = self.last_state, self.last_action
        old_value = self.q_table[s][a]
        next_max = max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[s][a] = new_value

        self._log_data(stage="result")
        self.total_updates += 1

        if self.total_updates % 50 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.98)
        if self.total_updates % 100 == 0:
            self._save_qtable()

    # -------------------- Data Logging --------------------
    def _log_data(self, stage):
        import pandas as pd, os, time

        data = {"time": [time.strftime("%H:%M:%S")], "stage": [stage]}
        for i, road in enumerate(self.roads):
            last_car = road.cars[-1] if road.cars else None
            if last_car:
                if last_car.angle_facing_initial == 0:
                    total_dist = abs((last_car.rect.y + 24) - 305)
                elif last_car.angle_facing_initial == 90:
                    total_dist = abs((last_car.rect.x + 24) - 305)
                elif last_car.angle_facing_initial == 180:
                    total_dist = abs(205 - last_car.rect.y)
                elif last_car.angle_facing_initial == 270:
                    total_dist = abs(205 - last_car.rect.x)
            else:
                total_dist = 0

            # 🔹 Round before logging
            total_dist = int(round(total_dist / 100.0) * 100)
            num_cars = len(road.cars)

            data[f"road{i*90}_dist"] = [total_dist]
            data[f"road{i*90}_cars"] = [num_cars]

        df_new = pd.DataFrame(data)

        if not os.path.exists(self.file_path):
            df_new.to_excel(self.file_path, index=False)
        else:
            df_existing = pd.read_excel(self.file_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(self.file_path, index=False)

    def open_data_file(self):
        import subprocess
        self._save_qtable()
        subprocess.run(["open", self.file_path])

    # -------------------- Q-table persistence --------------------
    def _save_qtable(self):
        import pickle
        with open(self.qtable_file, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[INFO] Q-table saved ({len(self.q_table)} states).")

    def _load_qtable(self):
        import os, pickle
        if os.path.exists(self.qtable_file):
            with open(self.qtable_file, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"[INFO] Loaded Q-table from {self.qtable_file} ({len(self.q_table)} states).")
        else:
            print("[INFO] No existing Q-table found. Starting fresh.")