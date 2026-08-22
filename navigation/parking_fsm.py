import time
import logging
from collections import deque
from enum import Enum, auto

logger = logging.getLogger("ParkingFSM")


class SlidingWindowVoter:
    """
    Replaces the brittle consecutive-count debouncer.

    Keeps a fixed-length window of (class_name, confidence) observations.
    The winner is the class with the highest total confidence-weighted vote
    across the window, provided it exceeds a dominance threshold.
    """

    def __init__(self, window_size: int = 7, dominance_ratio: float = 0.50):
        self.window = deque(maxlen=window_size)
        self.dominance_ratio = dominance_ratio

    def observe(self, class_name: str, confidence: float):
        self.window.append((class_name, confidence))

    def winner(self):
        if not self.window:
            return None, 0.0

        votes = {}
        for cls, conf in self.window:
            votes[cls] = votes.get(cls, 0.0) + conf

        total = sum(votes.values())
        if total == 0:
            return None, 0.0

        best_cls = max(votes, key=lambda c: votes[c])
        best_score = votes[best_cls] / total

        if best_score >= self.dominance_ratio:
            return best_cls, best_score
        return None, best_score

    def reset(self):
        self.window.clear()


class ParkingState(Enum):
    IDLE        = auto()
    SEARCHING   = auto()   # driving forward, looking for a spot
    APPROACHING = auto()   # spot tentatively detected — slow down for confirmation
    MANEUVERING = auto()   # committed to parking maneuver
    PARKED      = auto()
    SAFETY_STOP = auto()


class ParkingFSM:
    """
    Redesigned FSM for 4-class model (lane_clear / spot_left / spot_right / blocked).
    """

    CLS_CLEAR   = "lane_clear"
    CLS_LEFT    = "spot_left"
    CLS_RIGHT   = "spot_right"
    CLS_BLOCKED = "blocked"

    def __init__(
        self,
        motor_controller,
        base_speed: float = 0.20,
        confidence_threshold: float = 0.55,
        voter_window: int = 7,
        park_side_override: str = None,   # None | 'left' | 'right'
        side_detector=None,
    ):
        self.motors               = motor_controller
        self.base_speed           = base_speed
        self.confidence_threshold = confidence_threshold
        self.park_side_override   = park_side_override
        self.side_detector        = side_detector

        self.state            = ParkingState.IDLE
        self.spot_side        = None
        self.state_start_time = time.time()

        self._voter          = SlidingWindowVoter(window_size=voter_window, dominance_ratio=0.50)
        self._approach_voter = SlidingWindowVoter(window_size=5, dominance_ratio=0.60)

    def start_parking_search(self):
        self._transition(ParkingState.SEARCHING)

    def stop(self):
        self._transition(ParkingState.IDLE)
        self.motors.stop()

    def update(self, class_name: str, confidence: float, raw_frame=None):
        elapsed = time.time() - self.state_start_time

        self._voter.observe(class_name, confidence)
        voted_class, vote_score = self._voter.winner()

        if self.state not in (ParkingState.MANEUVERING, ParkingState.PARKED):
            if (class_name == self.CLS_BLOCKED
                    and confidence >= self.confidence_threshold + 0.10):
                self.motors.stop()
                self._transition(ParkingState.SAFETY_STOP)
                return

        if self.state == ParkingState.IDLE:
            self.motors.stop()
        elif self.state == ParkingState.SEARCHING:
            self._state_searching(class_name, confidence, voted_class, vote_score, raw_frame)
        elif self.state == ParkingState.APPROACHING:
            self._state_approaching(class_name, confidence, elapsed, raw_frame)
        elif self.state == ParkingState.MANEUVERING:
            self._state_maneuvering(elapsed)
        elif self.state == ParkingState.PARKED:
            self.motors.stop()
        elif self.state == ParkingState.SAFETY_STOP:
            self.motors.stop()
            if voted_class == self.CLS_CLEAR and vote_score >= 0.55:
                logger.info("Path cleared — resuming search.")
                self._transition(ParkingState.SEARCHING)

    def _state_searching(self, class_name, confidence, voted_class, vote_score, raw_frame):
        spot_classes = (self.CLS_LEFT, self.CLS_RIGHT)

        if voted_class in spot_classes and vote_score >= 0.50:
            self._approach_voter.reset()
            self._transition(ParkingState.APPROACHING)
            return

        if voted_class == self.CLS_CLEAR:
            self.motors.drive_vector(linear_vel=self.base_speed, steering=0.0)
        else:
            self.motors.drive_vector(linear_vel=self.base_speed * 0.5, steering=0.0)

    def _state_approaching(self, class_name, confidence, elapsed, raw_frame):
        self.motors.drive_vector(linear_vel=self.base_speed * 0.30, steering=0.0)
        self._approach_voter.observe(class_name, confidence)
        confirmed_side, side_score = self._approach_voter.winner()

        spot_classes = (self.CLS_LEFT, self.CLS_RIGHT)

        if confirmed_side in spot_classes:
            if self.side_detector is not None and raw_frame is not None:
                cv_side = self.side_detector.detect(raw_frame)
                if cv_side is not None and cv_side != confirmed_side:
                    logger.info(f"Side override: ML={confirmed_side}, CV={cv_side} -> using CV result")
                    confirmed_side = cv_side

            if self.park_side_override is not None:
                side = self.park_side_override
            else:
                side = "left" if confirmed_side == self.CLS_LEFT else "right"

            self.spot_side = side
            self.motors.stop()
            logger.info(f"Spot confirmed: {side} side (score={side_score:.2f}) — starting maneuver")
            self._transition(ParkingState.MANEUVERING)

        elif elapsed > 3.0:
            logger.info("Approach timeout — resuming search.")
            self._voter.reset()
            self._transition(ParkingState.SEARCHING)

        elif confirmed_side == self.CLS_CLEAR and self._approach_voter.winner()[1] >= 0.65:
            logger.info("Spot signal lost during approach — resuming search.")
            self._voter.reset()
            self._transition(ParkingState.SEARCHING)

    def _state_maneuvering(self, elapsed):
        """
        Open-loop timed parking maneuver.
        TUNE THESE VALUES for your specific bot and table setup.

        Phase 1 (0 → 1.2 s): Arc into spot
        Phase 2 (1.2 → 2.2 s): Straighten and push in
        Phase 3 (> 2.2 s):    Stop, declare parked
        """
        turn_steering = -0.55 if self.spot_side == "left" else +0.55

        if elapsed <= 1.2:
            self.motors.drive_vector(linear_vel=self.base_speed * 0.65, steering=turn_steering)
        elif elapsed <= 2.2:
            self.motors.drive_vector(linear_vel=self.base_speed * 0.50, steering=0.0)
        else:
            self.motors.stop()
            self._transition(ParkingState.PARKED)
            logger.info("Parking maneuver complete.")

    def _transition(self, new_state: ParkingState):
        if self.state != new_state:
            logger.info(f"FSM: {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_start_time = time.time()
