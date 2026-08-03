import time
import logging
from enum import Enum, auto

logger = logging.getLogger("ParkingFSM")


class ParkingState(Enum):
    IDLE = auto()
    SEARCHING_SPOT = auto()
    SPOT_IDENTIFIED = auto()
    MANEUVERING = auto()
    PARKED = auto()
    SAFETY_STOP = auto()


class ParkingFSM:
    """
    Finite State Machine (FSM) controlling JetBot autonomous driving and parking.
    Translates perception model output (probabilities) into discrete motor control actions.
    """
    def __init__(self, motor_controller, base_speed=0.20, confidence_threshold=0.60):
        self.motors = motor_controller
        self.base_speed = base_speed
        self.confidence_threshold = confidence_threshold

        self.state = ParkingState.IDLE
        self.spot_side = None  # 'left' or 'right'
        self.state_start_time = time.time()

    def set_state(self, new_state):
        if self.state != new_state:
            logger.info(f"FSM Transition: {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_start_time = time.time()

    def update(self, class_idx, class_name, confidence):
        """
        Main control update loop called on every camera frame.
        :param class_idx: Predicted class index from model
        :param class_name: String name of predicted class
        :param confidence: Softmax probability (0.0 to 1.0)
        """
        elapsed = time.time() - self.state_start_time

        # Safety Override: If obstacle is detected with high confidence, stop immediately
        if class_name == "obstacle_blocked" and confidence >= self.confidence_threshold:
            self.motors.stop()
            self.set_state(ParkingState.SAFETY_STOP)
            return

        if self.state == ParkingState.IDLE:
            self.motors.stop()

        elif self.state == ParkingState.SEARCHING_SPOT:
            if class_name == "path_free" and confidence >= self.confidence_threshold:
                # Normal forward drive along lane
                self.motors.drive_vector(linear_vel=self.base_speed, steering=0.0)

            elif class_name in ["parking_spot_left", "parking_spot_right"] and confidence >= self.confidence_threshold:
                self.spot_side = "left" if class_name == "parking_spot_left" else "right"
                self.motors.stop()
                self.set_state(ParkingState.SPOT_IDENTIFIED)

            else:
                # Creep forward cautiously if low confidence
                self.motors.drive_vector(linear_vel=self.base_speed * 0.5, steering=0.0)

        elif self.state == ParkingState.SPOT_IDENTIFIED:
            # Hold position for 0.5s to stabilize before starting maneuver
            self.motors.stop()
            if elapsed > 0.5:
                self.set_state(ParkingState.MANEUVERING)

        elif self.state == ParkingState.MANEUVERING:
            # Phase 1: Turn into spot (first 1.5 seconds)
            if elapsed <= 1.5:
                steer = -0.5 if self.spot_side == "left" else 0.5
                self.motors.drive_vector(linear_vel=self.base_speed * 0.7, steering=steer)
            # Phase 2: Straighten up into spot (next 1.0 seconds)
            elif elapsed <= 2.5:
                self.motors.drive_vector(linear_vel=self.base_speed * 0.5, steering=0.0)
            # Phase 3: Maneuver complete
            else:
                self.motors.stop()
                self.set_state(ParkingState.PARKED)

        elif self.state == ParkingState.PARKED:
            self.motors.stop()
            logger.info("JetBot successfully parked!")

        elif self.state == ParkingState.SAFETY_STOP:
            self.motors.stop()
            if class_name != "obstacle_blocked":
                # Clear safety stop if path clears up
                self.set_state(ParkingState.SEARCHING_SPOT)

    def start_parking_search(self):
        """Trigger search for parking spot."""
        self.set_state(ParkingState.SEARCHING_SPOT)

    def stop(self):
        self.set_state(ParkingState.IDLE)
        self.motors.stop()
