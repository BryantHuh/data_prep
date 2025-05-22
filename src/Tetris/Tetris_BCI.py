import pygame
from grid import Grid
from blocks import *
pygame.init()

screen_width = 600
screen_height = 1200

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tetris_BCI_Version")

game_grid = Grid()

block = TBlock()

clock = pygame.time.Clock()
FPS = 60

# Define colors

BACKGROUND_COLOR = (44, 44, 127)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill(BACKGROUND_COLOR)
    game_grid.draw(screen)
    block.draw(screen)

    pygame.display.update()
    clock.tick(FPS)

