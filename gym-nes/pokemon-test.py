from pyboy import PyBoy, WindowEvent
import random
import time

# Replace this with the path to your ROM file
rom_path = r"C:\projects\personal_projects\RL-projects\ROMs\pokemonRed.gb"

# Initialize PyBoy with the ROM
pyboy = PyBoy(rom_path, game_wrapper=True)  # Use "headless" for no GUI
pyboy.set_emulation_speed(0)  # 0 for unbounded speed, 1 for 100%

# Get the game wrapper
pyboy.cartridge_title() == "POKEMON BLUE"
game_wrapper = pyboy.game_wrapper()

# Start the game using the wrapper
game_wrapper.start_game()

press_actions = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_SELECT,
    WindowEvent.PRESS_BUTTON_START,
]

release_actions = {
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
    WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
}

# Start the emulator
try:
    while not pyboy.tick():
        print(game_wrapper)

        # Random press action
        action = random.choice(press_actions)
        pyboy.send_input(action)
        # Simulate a delay of some ticks (e.g., 10)
        for _ in range(10):
            pyboy.tick()
        # Release the button
        release_action = release_actions[action]
        pyboy.send_input(release_action)

except KeyboardInterrupt:
    pass

# Close the emulator when done
pyboy.stop()