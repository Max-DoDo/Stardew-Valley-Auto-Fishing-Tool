from __future__ import annotations

from pynput.mouse import Button, Controller


class InputController:
    def __init__(self):
        self.mouse = Controller()
        self.is_holding = False

    def hold(self) -> None:
        if self.is_holding:
            return

        self.mouse.press(Button.left)
        self.is_holding = True

    def release(self) -> None:
        if not self.is_holding:
            return

        self.mouse.release(Button.left)
        self.is_holding = False

    def apply_action(self, action: int) -> None:
        if action == 1:
            self.hold()
        else:
            self.release()

    def reset(self) -> None:
        self.release()