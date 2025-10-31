import pygame
import time
import json
from Classes import *

FPS = 10
YELLOW_TIME_1 = 2.0 # Green -> Red yellow time
YELLOW_TIME_2 = 3.0 # Red -> Green yellow time
MINIMUM_GREEN = YELLOW_TIME_1 + 5.0
pygame.init()
window = pygame.display.set_mode((512, 512))
pygame.display.set_caption("Traffic Simulation")
clock = pygame.time.Clock()

# Create instances anticlockwise from going up
signal_0 = Signal(231, 305) # Up
signal_0.become_red()
signal_90 = Signal(305, 279) # Left
signal_90.become_green()
signal_180 = Signal(279, 205) # Down
signal_180.become_red()
signal_270 = Signal(205, 231) # Right
signal_270.become_green()
signals = [signal_0, signal_90, signal_180, signal_270]

road_0 = Road((238, 311), 10, 0, signal_0) # Up
road_90 = Road((311, 260), 10, 90, signal_90) # Left
road_180 = Road((260, 179), 10, 180, signal_180, road_0) # Down
road_270 = Road((179, 238), 10, 270, signal_270, road_90) # Right
road_0.opposite_road = road_180
road_90.opposite_road = road_270
roads = [road_0, road_90, road_180, road_270]
road_0.roads = roads
road_90.roads = roads
road_180.roads = roads
road_270.roads = roads

AI = input("Choose AI (accurate - 1/inaccurate - 2): ").strip().lower()
if "in" in AI or AI == "2":
    My_Guy = AIInaccurate(signals, roads, min_green_time=MINIMUM_GREEN)
else:
    My_Guy = AIAccurate(signals, roads, min_green_time=MINIMUM_GREEN)

for road in roads:
    road.get_data()

SIM_DURATION = input("Enter simulation duration in seconds (or press Enter for infinite): ").strip()
if SIM_DURATION.isdigit():
    SIM_DURATION = int(SIM_DURATION) * 1000  # Convert to milliseconds
    print(f"Simulation will run for {SIM_DURATION / 1000} seconds.")
else:
    SIM_DURATION = None
    print("Simulation will run for a looong time.")

def draw(screen, roads):
    window.blit(pygame.image.load("imgs/bg.png"), (0, 0))
    for road in roads:
        for car in road.cars:
            car.draw(screen)
    for signal in signals:
        signal.draw(screen)


def update(roads):
    for road in roads:
        road.spawn_car()
        road.remove()
        for car in road.cars:
                car.check_move_and_turn()

    for road in roads:
        try:
            for car in road.cars:
                if car.should_turn:
                    car.turn()
                if car.should_move:
                    car.move()
                if (car.angle_facing_initial == 0 and car.rect.y < 173) or \
                   (car.angle_facing_initial == 90 and car.rect.x < 173) or \
                   (car.angle_facing_initial == 180 and car.rect.y > 330) or \
                   (car.angle_facing_initial == 270 and car.rect.x > 330):
                    road.cars.remove(car)
        except IndexError:
            pass

for road in roads:
    road.init_cars()
switching_signals = False
switching_1 = False

sim_start_time = time.time()

runtime_data = {
    "AI_Type": "Accurate" if isinstance(My_Guy, AIAccurate) else "Inaccurate",
    "Road_Data": {},
    "Simulation_Duration_s": None,
    "Start_Timestamp": sim_start_time
}

for road in roads:
    runtime_data["Road_Data"][road.dir] = {
        "Initial_Cars": road.init_num_cars,
        "Car_Spawn_Rate": road.car_spawn_rate,
        "Turning_Distribution": {
            "Left": road.dirs_to_turn.count('left'),
            "Forward": road.dirs_to_turn.count('forward'),
            "Right": road.dirs_to_turn.count('right')
        }
    }

with open(f"Runtime Data {My_Guy.i}.json", "w") as f:
    json.dump(runtime_data, f, indent=4)

while True:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            My_Guy.open_data_file()
            exit()
    
    if SIM_DURATION and (time.time() - sim_start_time) * 1000 >= SIM_DURATION:
        pygame.quit()
        My_Guy.open_data_file()
        exit()

    if My_Guy.calculate_switch() and not switching_signals:
        for signal in signals:
            if signal.colour == "green":
                signal.become_yellow()
        switching_signals = True
        switching_1 = True
        switch_start_time = pygame.time.get_ticks()  # Start timer

    if switching_signals:
        if switching_1 and pygame.time.get_ticks() - switch_start_time >= YELLOW_TIME_1 * 1000:
            for signal in signals:
                if signal.colour == "yellow":
                    signal.become_red()
                elif signal.colour == "red":
                    signal.become_yellow()
            switching_1 = False
            switching_2 = True
            switch_start_time = pygame.time.get_ticks()
        if not switching_1 and switching_2 and pygame.time.get_ticks() - switch_start_time >= YELLOW_TIME_2 * 1000:
            for signal in signals:
                if signal.colour == "yellow":
                    signal.become_green()
            switching_2 = False
            switching_signals = False

    update(roads)
    draw(window, roads)

    pygame.display.update()
