from dataclasses import dataclass

@dataclass
class State:
    isFishing: bool = False
    fish_in_greenbar: bool = False

    fish_y: int = 0
    greenbar_center_y: int = 0
    greenbar_height: int = 0
    distance: int = 0

    fish_velocity_y: int = 0
    greenbar_velocity_y: int = 0