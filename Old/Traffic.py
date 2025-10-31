import pygame as p
import random as r
import os
import json

screen = p.display.set_mode((1024, 1024))
p.display.set_caption("4-Way Junction")

bg = p.image.load("imgs/bg.png")
clock = p.time.Clock()
FPS = 4
p.font.init()


# == Car Class ==
class Car(p.sprite.Sprite):
    def __init__(self, x, y, angle):
        super().__init__()
        image = r.choice(['red', 'blue', 'orange'])
        self.image = p.image.load("imgs/" + image + str(angle) + ".png")
        self.rect = self.image.get_rect(topleft=(x, y))
        self.rect.x = x
        self.rect.y = y
        self.angle = angle
        self.speed = 15
        self.delete = False
        self.move = False
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)
    
    def update(self):
        if self.angle == 0:
            self.rect.y -= self.speed
        elif self.angle == 90:
            self.rect.x += self.speed
        elif self.angle == 180:
            self.rect.y += self.speed
        elif self.angle == 270:
            self.rect.x -= self.speed
    
    def check_move(self, signals, data):
        '''Checks whether car can move based on traffic light and position compared to traffic light'''
        signal_allows_movement = (
            (self.angle == 0 and signals[1].state == 'green') or
            (self.angle == 90 and signals[2].state == 'green') or
            (self.angle == 180 and signals[3].state == 'green') or
            (self.angle == 270 and signals[0].state == 'green'))

        passed_traffic_light = (
            (self.angle == 0 and self.rect.y < 142*4) or
            (self.angle == 90 and self.rect.x > 116*4) or
            (self.angle == 180 and self.rect.y > 116*4) or
            (self.angle == 270 and self.rect.x < 142*4)
        )
        try: # 121*4, 140*4, 0), (110*4, 121*4, 90), (131*4, 110*4, 180), (140*4, 131*4, 270
            cars_ahead = data[self.angle // 90][2].index(self)
            if self.angle == 0:
                position = self.rect.y > 146*4 + (28 * cars_ahead)
            elif self.angle == 90:
                position = self.rect.x < 112*4 - (28 * cars_ahead)
            elif self.angle == 180:
                position = self.rect.y < 112*4 - (28 * cars_ahead)
            elif self.angle == 270:
                position = self.rect.x > 146*4 + (28 * cars_ahead)

        except ValueError:
            cars_ahead = -1

        self.move = signal_allows_movement or passed_traffic_light or position

    def check_deletion(self):
        self.delete = (
            (self.rect.y < 110 * 4 and self.angle == 0) or        # Car moving up (0) exits top of screen
            (self.rect.x > 140 * 4 and self.angle == 90) or       # Car moving right (90) exits right of screen
            (self.rect.y > 140 * 4 and self.angle == 180) or      # Car moving down (180) exits bottom of screen
            (self.rect.x < 110 * 4 and self.angle == 270)         # Car moving left (270) exits left of screen
        )


# == Signal Class ==
class Signal(p.sprite.Sprite):
    def __init__(self, x, y, state='green'):
        super().__init__()
        self.image = p.Surface((12, 12))
        self.state = state
        self.prev_state = None
        if state == 'green':
            self.image.fill((0, 255, 0))
        elif state == 'yellow':
            self.image.fill((255, 255, 0))
        elif state == 'red':
            self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.rect.x = x
        self.rect.y = y

    def change_state(self, state):
        self.prev_state = self.state
        self.state = state
        if state == 'green':
            self.image.fill((0, 255, 0))
        elif state == 'yellow':
            self.image.fill((255, 255, 0))
        elif state == 'red':
            self.image.fill((255, 0, 0))
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)
    


# == Initialization of Objects ==
signals = [Signal(113*4, 116*4, 'red'), Signal(137*4, 113*4), Signal(140*4, 137*4, 'red'), Signal(116*4, 140*4)]

data = []
cars = []
for i in [(121*4, 140*4, 0), (110*4, 121*4, 90), (131*4, 110*4, 180), (140*4, 131*4, 270)]:
    inital_cars = int(input(f"Enter the number of cars for road {i[2]}: "))
    car_rate = int(input(f"Enter the rate of car generation per minute for road {i[2]}: "))    
    queue = []
    o = 28
    if i[2] == 0:
        x, y = 0, o
    elif i[2] == 90:
        x, y = -o, 0
    elif i[2] == 180:
        x, y = 0, -o
    elif i[2] == 270:
        x, y = o, 0

    for j in range(inital_cars):
        car = Car(i[0] + (j * x), i[1] + (j * y), i[2])
        queue.append(car)
        cars.append(car)
    
    data.append([inital_cars, car_rate, queue]) # data structure = [[initial cars, car rate, queue], [initial cars, car rate, queue], [initial cars, car rate, queue], [initial cars, car rate, queue]]



def draw_queue_info(screen, data):
    font = p.font.SysFont('Arial', 20)
    
    # Display queue lengths
    queue_text = f"Queue 0: {len(data[0][2])} | Queue 90: {len(data[1][2])} | Queue 180: {len(data[2][2])} | Queue 270: {len(data[3][2])}"
    queue_surface = font.render(queue_text, True, (255, 255, 255))
    screen.blit(queue_surface, (10, 10))



def draw(cars, signals, bg, screen):
    screen.blit(bg, (0, 0))
    for signal in signals:
        signal.draw(screen)
    for car in cars:
        if not car.delete:
            car.draw(screen)
    # Add this line to display queue info
    draw_queue_info(screen, data)



def update(cars, signals, bg, screen):
    draw(cars, signals, bg, screen)
    for car in cars[:]:  # Create a copy to safely iterate while removing
        car.check_move(signals, data)
        if car.move:
            car.update()
        car.check_deletion()
        if car.delete:
            cars.remove(car)
            # Use integer division to get the correct index
            index = car.angle // 90
            if car in data[index][2]:  # Check if car is still in the queue
                data[index][2].remove(car)



while True:
    for event in p.event.get():
        if event.type == p.QUIT:
            p.quit()
            exit()

    # == Car Spawning Logic ==
    h = r.random()
    if h < data[0][1] / (60.0 * FPS):
        car = Car(121*4, 140*4 + (len(data[0][2]) * 28), 0)
        data[0][2].append(car)
        cars.append(car)
    
    h = r.random()
    if h < data[1][1] / (60.0 * FPS):
        car = Car(110*4 - (len(data[1][2]) * 28), 121*4, 90)
        data[1][2].append(car)
        cars.append(car)
    
    h = r.random()
    if h < data[2][1] / (60.0 * FPS):
        car = Car(131*4, 110*4 - (len(data[2][2]) * 28), 180)
        data[2][2].append(car)
        cars.append(car)
    
    h = r.random()
    if h < data[3][1] / (60.0 * FPS):
        car = Car(140*4 + (len(data[3][2]) * 28), 131*4, 270)
        data[3][2].append(car)
        cars.append(car)

    update(cars, signals, bg, screen)
    p.display.update()
    clock.tick(FPS)