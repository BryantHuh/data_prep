#!/usr/bin/env python3
"""
BCI Marker Stream Creator

Creates a marker stream for BCI experiments with OpenBCI setup.
This script provides a Pygame interface for creating marker streams that can
be synchronized with EEG data in LabRecorder for BCI experiments.

Features:
- Pygame interface for easy marker creation
- LSL marker stream output
- Configurable experiment parameters
- Visual feedback for experiment flow
- Synchronization with EEG recordings
"""

import pygame
import sys
import time
import random
import argparse
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, cf_string

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('marker_stream', log_dir='logs', level='INFO')

class BCIMarkerStream:
    """BCI marker stream creator with pygame interface"""

    def __init__(self, stream_name='MarkerStream', stream_id='bci_marker_stream_uid'):
        self.stream_name = stream_name
        self.stream_id = stream_id

        # LSL outlet
        self.outlet = None

        # Pygame setup
        self.screen = None
        self.font = None
        self.clock = None

        # Experiment state
        self.experiment_state = 'idle'
        self.marker_sent = False
        self.directions = []
        self.current_index = 0
        self.state_start_time = 0

        # UI elements
        self.input_box = None
        self.button_box = None
        self.user_text = ''
        self.start_experiment = False

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Screen center
        self.cx, self.cy = 400, 300

        # Arrow points
        self._setup_arrows()

        logger.info(f"Initialized BCI marker stream: {stream_name}")

    def _setup_arrows(self):
        """Setup arrow points for different directions"""
        # Base arrow
        base = [(-100, 0), (0, -50), (0, -20), (100, -20), (100, 20), (0, 20), (0, 50)]

        def rotate(pt, angle):
            from math import cos, sin, radians
            x, y = pt
            a = radians(angle)
            return (x * cos(a) - y * sin(a), x * sin(a) + y * cos(a))

        self.points_left = [(self.cx + x, self.cy + y) for x, y in base]
        self.points_up = [(self.cx + x, self.cy + y) for x, y in (rotate(p, 90) for p in base)]
        self.points_right = [(self.cx + x, self.cy + y) for x, y in (rotate(p, 180) for p in base)]
        self.points_down = [(self.cx + x, self.cy + y) for x, y in (rotate(p, -90) for p in base)]

    def setup_lsl_stream(self):
        """Setup LSL marker stream"""
        try:
            # Create LSL stream info
            info = StreamInfo(
                name=self.stream_name,
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format=cf_string,
                source_id=self.stream_id
            )

            # Create outlet
            self.outlet = StreamOutlet(info)
            logger.info(f"Created LSL marker stream: {self.stream_name}")

        except Exception as e:
            logger.error(f"Failed to setup LSL stream: {e}")
            raise

    def setup_pygame(self):
        """Setup pygame window and components"""
        try:
            # Initialize pygame
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("BCI Motor Imagery Paradigm")
            self.font = pygame.font.SysFont(None, 48)
            self.clock = pygame.time.Clock()

            # Setup UI elements
            self.input_box = pygame.Rect(300, 200, 200, 50)
            self.button_box = pygame.Rect(350, 300, 100, 50)

            logger.info("Pygame setup completed")

        except Exception as e:
            logger.error(f"Failed to setup pygame: {e}")
            raise

    def draw_text_center(self, text, y):
        """Draw centered text"""
        rendered = self.font.render(text, True, self.WHITE)
        rect = rendered.get_rect(center=(400, y))
        self.screen.blit(rendered, rect)

    def draw_fixation_cross(self):
        """Draw fixation cross"""
        pygame.draw.line(self.screen, self.WHITE, (200, 300), (600, 300), 2)
        pygame.draw.line(self.screen, self.WHITE, (400, 100), (400, 500), 2)

    def draw_arrow(self, direction):
        """Draw arrow in specified direction"""
        if direction == 'left':
            pygame.draw.polygon(self.screen, self.WHITE, self.points_left)
        elif direction == 'right':
            pygame.draw.polygon(self.screen, self.WHITE, self.points_right)
        elif direction == 'up':
            pygame.draw.polygon(self.screen, self.WHITE, self.points_up)
        elif direction == 'down':
            pygame.draw.polygon(self.screen, self.WHITE, self.points_down)

    def send_marker(self, marker):
        """Send marker through LSL stream"""
        try:
            if self.outlet:
                self.outlet.push_sample([marker])
                logger.info(f"Marker sent: {marker}")
            else:
                logger.warning("LSL outlet not available")
        except Exception as e:
            logger.error(f"Failed to send marker: {e}")

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if self.experiment_state == 'idle':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        self.user_text = self.user_text[:-1]
                    else:
                        self.user_text += event.unicode

                if event.type == pygame.MOUSEBUTTONDOWN and self.button_box.collidepoint(event.pos):
                    try:
                        runs_left = int(self.user_text)
                        if runs_left % 4 != 0:
                            self.user_text = ''
                        else:
                            # Create balanced directions
                            self.directions = ['left', 'right', 'up', 'down'] * (runs_left // 4)
                            random.shuffle(self.directions)
                            self.current_index = 0
                            self.experiment_state = 'fixation'
                            self.state_start_time = pygame.time.get_ticks()
                            logger.info(f"Starting experiment with {runs_left} trials")
                    except ValueError:
                        self.user_text = ''

        return True

    def update_experiment(self):
        """Update experiment state"""
        now = pygame.time.get_ticks()

        if self.experiment_state == 'fixation':
            self.draw_fixation_cross()
            if now - self.state_start_time >= 2000:  # 2 seconds
                self.experiment_state = 'arrow'
                self.state_start_time = now
                self.marker_sent = False

        elif self.experiment_state == 'arrow':
            direction = self.directions[self.current_index]
            self.draw_arrow(direction)

            # Send marker only once per arrow
            if not self.marker_sent:
                self.send_marker(direction)
                self.marker_sent = True

            if now - self.state_start_time >= 4000:  # 4 seconds
                self.current_index += 1
                if self.current_index < len(self.directions):
                    self.experiment_state = 'fixation'
                    self.state_start_time = now
                    self.marker_sent = False
                else:
                    self.experiment_state = 'idle'
                    self.user_text = ''
                    logger.info("Experiment completed")

    def draw_ui(self):
        """Draw user interface"""
        if self.experiment_state == 'idle':
            # Draw input box
            pygame.draw.rect(self.screen, self.WHITE, self.input_box, 2)
            text_surface = self.font.render(self.user_text, True, self.WHITE)
            self.screen.blit(text_surface, (self.input_box.x + 10, self.input_box.y + 10))
            self.draw_text_center("Number of trials (divisible by 4):", 150)

            # Draw start button
            pygame.draw.rect(self.screen, self.WHITE, self.button_box, 2)
            self.draw_text_center("Start", 325)

    def run(self):
        """Main experiment loop"""
        try:
            logger.info("Starting BCI marker stream experiment")

            # Setup components
            self.setup_lsl_stream()
            self.setup_pygame()

            logger.info("Experiment ready. Enter number of trials and click Start.")

            running = True
            while running:
                self.screen.fill(self.BLACK)

                # Handle events
                running = self.handle_events()

                # Update experiment
                self.update_experiment()

                # Draw UI
                self.draw_ui()

                pygame.display.flip()
                self.clock.tick(30)

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            pygame.quit()
            logger.info("Experiment stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='BCI Marker Stream Creator')
    parser.add_argument('--stream-name', type=str, default='MarkerStream',
                       help='LSL stream name for markers')
    parser.add_argument('--stream-id', type=str, default='bci_marker_stream_uid',
                       help='LSL stream source ID')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BCI Marker Stream Creator")
    logger.info("=" * 60)
    logger.info(f"Stream name: {args.stream_name}")
    logger.info(f"Stream ID: {args.stream_id}")

    try:
        # Create and run marker stream
        marker_stream = BCIMarkerStream(
            stream_name=args.stream_name,
            stream_id=args.stream_id
        )

        marker_stream.run()

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()